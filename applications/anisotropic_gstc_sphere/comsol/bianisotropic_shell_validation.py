#!/usr/bin/env python3
"""COMSOL Multiphysics Validation Pipeline for Bianisotropic Spherical Metasurfaces (Case B).

Executes a full 3D FEM validation of the Case B volumetric shell representation against
SIE-BEM solver results:
  - Universal input loading: .bin (optimized_thetas.bin or face_tensors.bin) or .csv
  - Mesh-centroid-based adaptive 2D (theta, phi) periodic interpolation to COMSOL '% Grid' tables
  - COMSOL session management supporting COMSOL 5.4 (Wave Optics ewfd) and 6.1 (RF emw)
  - Concentric geometry construction with refinement bands and octant partitioning
  - Swept prism or conformal tetrahedral meshing
  - Equation View weak-form injection of bianisotropic constitutive relations
  - Direct linear solver (Pardiso/MUMPS) solve
  - Extraction and export of physical near-field (radial cut, zx-plane) and far-field cuts

Usage Examples:
  # COMSOL 5.4 swept mesh validation on optimized_thetas.bin:
  python applications/anisotropic_gstc_sphere/comsol/bianisotropic_shell_validation.py \\
      --thetas applications/anisotropic_gstc_sphere/results/dipole_rotation/orthogonal_90deg_er7_offset_1m_phase_aligned_smooth002_norm_400iter/optimized_thetas.bin \\
      --mesh applications/anisotropic_gstc_sphere/meshes/reflectionless_100mhz_radius2m_ppw10/sphere.obj \\
      --config applications/anisotropic_gstc_sphere/configurations/moment_rot/rot90_eps7.json \\
      --version 5.4 \\
      --shell-mesh swept \\
      --output-dir applications/anisotropic_gstc_sphere/results/dipole_rotation/comsol/orthogonal_90deg_er7_offset_1m_swept

  # With CSV parameters:
  python applications/anisotropic_gstc_sphere/comsol/bianisotropic_shell_validation.py \\
      --csv path/to/face_parameters.csv \\
      --version 5.4 \\
      --output-dir results/comsol_validation_csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import c


def get_default_cores() -> int:
    """Return default CPU core allocation based on SLURM environment or system hardware."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    total = os.cpu_count() or 4
    # Leave 2 cores for OS/desktop responsiveness on high-core workstations
    return max(1, total - 2) if total > 4 else total

# Add repo root to sys.path so modules resolve whether executed as package or script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from applications.anisotropic_gstc_sphere.comsol.comsol_helpers import (
    compute_dipole_normalization,
    extract_far_field_cuts,
    extract_near_field_radial,
    extract_near_field_zx,
    save_far_field_cuts_csv,
    save_physical_radial_csv,
    save_physical_zx_csv,
    setup_physics_comsol_54,
    setup_physics_comsol_61,
    setup_spherical_shell_geometry,
    setup_spherical_shell_mesh,
    setup_study_and_solver,
    start_comsol,
    stop_comsol,
)
from applications.anisotropic_gstc_sphere.comsol.interpolation import (
    ETA_0,
    compute_adaptive_grid_resolution,
    export_comsol_grid_files,
    interpolate_thetas_to_grid,
    load_face_parameters,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the COMSOL validation pipeline."""
    parser = argparse.ArgumentParser(
        description="COMSOL FEM Validation for Bianisotropic Spherical Metasurfaces (Case B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Input Data Sources
    # -------------------------------------------------------------------------
    input_group = parser.add_argument_group("Input GSTC Metasurface Parameters")
    input_group.add_argument(
        "--thetas",
        type=Path,
        default=None,
        help="Path to optimized_thetas.bin or thetas.bin file",
    )
    input_group.add_argument(
        "--tensors",
        type=Path,
        default=None,
        help="Path to face_tensors.bin file",
    )
    input_group.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to face_parameters.csv file",
    )
    input_group.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to sphere.obj mesh file (required when loading .bin files)",
    )
    input_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to case workflow configuration JSON (e.g. configuration.json)",
    )
    input_group.add_argument(
        "--grid-oversample",
        type=float,
        default=3.0,
        help="Oversampling factor for mesh-centroid-to-grid (theta, phi) interpolation",
    )
    input_group.add_argument(
        "--grid-method",
        choices=["cubic", "linear"],
        default="cubic",
        help="Interpolation method from unstructured mesh centroids to (theta, phi) grid",
    )

    # -------------------------------------------------------------------------
    # Geometry and Media Specifications
    # -------------------------------------------------------------------------
    geom_group = parser.add_argument_group("Geometry and Media Parameters")
    geom_group.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Interior sphere radius a in meters (overrides config if given)",
    )
    geom_group.add_argument(
        "--frequency",
        type=float,
        default=None,
        help="Operational frequency f0 in Hz (overrides config if given)",
    )
    geom_group.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Metasurface shell thickness tau in meters (default: 0.01 * lambda0)",
    )
    geom_group.add_argument(
        "--air-extent",
        type=float,
        default=None,
        help="Exterior air box half-width b in meters (default: 2.0 * lambda0)",
    )
    geom_group.add_argument(
        "--pml-thickness",
        type=float,
        default=None,
        help="PML absorbing boundary layer thickness in meters (default: 0.25 * lambda0)",
    )
    geom_group.add_argument(
        "--er-interior",
        type=float,
        default=None,
        help="Relative permittivity of interior medium (default: 7.0 or from config)",
    )
    geom_group.add_argument(
        "--mur-interior",
        type=float,
        default=None,
        help="Relative permeability of interior medium (default: 1.0 or from config)",
    )
    geom_group.add_argument(
        "--er-exterior",
        type=float,
        default=None,
        help="Relative permittivity of exterior medium (default: 1.0 or from config)",
    )
    geom_group.add_argument(
        "--mur-exterior",
        type=float,
        default=None,
        help="Relative permeability of exterior medium (default: 1.0 or from config)",
    )

    # -------------------------------------------------------------------------
    # Excitation
    # -------------------------------------------------------------------------
    exc_group = parser.add_argument_group("Excitation Parameters")
    exc_group.add_argument(
        "--dipole-type",
        choices=["electric", "magnetic"],
        default="electric",
        help="Central point dipole source type",
    )
    exc_group.add_argument(
        "--dipole-direction",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Point dipole orientation Cartesian unit vector [dx, dy, dz]",
    )

    # -------------------------------------------------------------------------
    # Meshing Strategy & Concentric Refinement Bands
    # -------------------------------------------------------------------------
    mesh_group = parser.add_argument_group("Meshing Strategy & Concentric Refinements")
    mesh_group.add_argument(
        "--shell-mesh",
        choices=["swept", "tetra"],
        default="swept",
        help="Meshing strategy for thin bianisotropic shell (swept prisms or tetrahedra)",
    )
    mesh_group.add_argument(
        "--shell-layers",
        "--shell-radial-layers",
        dest="shell_layers",
        type=int,
        default=2,
        help="Number of radial prism element layers across shell thickness (for swept mesh)",
    )
    mesh_group.add_argument(
        "--shell-tangential-divisor",
        type=float,
        default=20.0,
        help="Shell surface triangle maximum size as lambda0 / divisor",
    )
    mesh_group.add_argument(
        "--global-mesh-divisor",
        "--global-hmax-divisor",
        dest="global_mesh_divisor",
        type=float,
        default=6.0,
        help="Global background/exterior air maximum element size as lambda0 / divisor",
    )
    mesh_group.add_argument(
        "--interior-hmax-divisor",
        type=float,
        default=None,
        help=(
            "Dielectric interior maximum element size as lambda0 / divisor "
            "(default: auto-scaled by refractive index as global_mesh_divisor * sqrt(er_int * mur_int))"
        ),
    )
    mesh_group.add_argument(
        "--source-ball-radius",
        type=float,
        default=0.15,
        help="Radius of conforming refinement ball around the point dipole in lambda0 (0 to disable)",
    )
    mesh_group.add_argument(
        "--source-ball-hmax-divisor",
        type=float,
        default=30.0,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the point-source ball",
    )
    mesh_group.add_argument(
        "--inner-shell-band-width",
        type=float,
        default=None,
        help="Width of optional dielectric refinement band inside the GSTC shell in lambda0",
    )
    mesh_group.add_argument(
        "--inner-shell-band-hmax-divisor",
        type=float,
        default=None,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the inner-shell band",
    )
    mesh_group.add_argument(
        "--outer-shell-band-width",
        type=float,
        default=None,
        help="Width of optional air refinement band outside the GSTC shell in lambda0",
    )
    mesh_group.add_argument(
        "--outer-shell-band-hmax-divisor",
        type=float,
        default=None,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the outer-shell band",
    )
    mesh_group.add_argument(
        "--discretization",
        choices=["linear", "quadratic"],
        default="linear",
        help="Finite element curl-element discretization order for the electric field",
    )

    # -------------------------------------------------------------------------
    # COMSOL Execution and Linear Solver
    # -------------------------------------------------------------------------
    solver_group = parser.add_argument_group("COMSOL Execution & Solver")
    solver_group.add_argument(
        "--version",
        choices=["5.4", "6.1"],
        default="5.4",
        help="Target COMSOL Multiphysics version",
    )
    solver_group.add_argument(
        "--cores",
        type=int,
        default=get_default_cores(),
        help="Number of CPU cores allocated for COMSOL server (default: auto-detected)",
    )
    solver_group.add_argument(
        "--linear-solver",
        choices=["pardiso", "mumps", "spooles"],
        default="pardiso",
        help="Direct linear solver engine",
    )
    solver_group.add_argument(
        "--save-mph",
        action="store_true",
        default=True,
        help="Save the solved COMSOL model file (.mph) to the output directory",
    )
    parser.add_argument(
        "--no-save-mph",
        action="store_false",
        dest="save_mph",
        help="Do not save the solved COMSOL model file (.mph)",
    )

    # -------------------------------------------------------------------------
    # Output and Sampling Cuts
    # -------------------------------------------------------------------------
    out_group = parser.add_argument_group("Output & Extraction Cuts")
    out_group.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for CSV artifacts, solved .mph, and metrics",
    )
    out_group.add_argument(
        "--radial-points",
        type=int,
        default=None,
        help="Number of sample points along 1D physical radial cut (default: from config or 200)",
    )
    out_group.add_argument(
        "--radial-min-radii",
        type=float,
        default=None,
        help="Minimum radial cut radius in units of sphere radius a (default: from config or 0.05)",
    )
    out_group.add_argument(
        "--radial-max-radii",
        type=float,
        default=None,
        help="Maximum radial cut radius in units of sphere radius a (default: from config or 2.0)",
    )
    out_group.add_argument(
        "--radial-theta",
        type=float,
        default=None,
        help="Radial cut polar angle theta in degrees (default: from config or 45.0)",
    )
    out_group.add_argument(
        "--radial-phi",
        type=float,
        default=None,
        help="Radial cut azimuthal angle phi in degrees (default: from config or 0.0)",
    )
    out_group.add_argument(
        "--zx-nx",
        type=int,
        default=None,
        help="Number of x grid points for 2D zx-plane cut (default: from config or 200)",
    )
    out_group.add_argument(
        "--zx-nz",
        type=int,
        default=None,
        help="Number of z grid points for 2D zx-plane cut (default: from config or 200)",
    )
    out_group.add_argument(
        "--zx-span-radii",
        type=float,
        default=None,
        help="Half-span of 2D zx-plane cut in units of sphere radius a (default: from config or 2.0)",
    )
    out_group.add_argument(
        "--far-field-points",
        type=int,
        default=361,
        help="Number of angular points (0 to 360 deg) for far-field principal cuts",
    )
    out_group.add_argument(
        "--disable-dipole-scale",
        action="store_true",
        default=False,
        help="Do not apply physical dipole normalization scale factor",
    )

    return parser.parse_args()


def load_config_defaults(args: argparse.Namespace) -> dict[str, Any]:
    """Extract default parameters from workflow configuration JSON (Schema v1 or v2)."""
    defaults: dict[str, Any] = {
        "frequency": 100e6,
        "radius": 2.0,
        "er_interior": 7.0,
        "mur_interior": 1.0,
        "er_exterior": 1.0,
        "mur_exterior": 1.0,
        "dipole_type": "electric",
        "dipole_direction": (0.0, 0.0, 1.0),
        "radial_points": 200,
        "radial_min_radii": 0.05,
        "radial_max_radii": 2.0,
        "radial_theta_deg": 45.0,
        "radial_phi_deg": 0.0,
        "zx_grid_size": 200,
        "zx_span_radii": 2.0,
    }

    if args.config is not None:
        if not args.config.is_file():
            raise FileNotFoundError(
                f"Configuration JSON not found: {args.config}\n"
                "Check the path passed to --config."
            )
        print(f"[Config] Loading workflow configuration: {args.config}")
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        defaults["raw_config"] = cfg

        # Schema v2 ('physical')
        phys_cfg = cfg.get("physical", {})
        if "frequency_hz" in phys_cfg:
            defaults["frequency"] = float(phys_cfg["frequency_hz"])
        if "radius_m" in phys_cfg:
            defaults["radius"] = float(phys_cfg["radius_m"])
        if "interior" in phys_cfg:
            defaults["er_interior"] = float(phys_cfg["interior"].get("epsilon_r", defaults["er_interior"]))
            defaults["mur_interior"] = float(phys_cfg["interior"].get("mu_r", defaults["mur_interior"]))
        if "exterior" in phys_cfg:
            defaults["er_exterior"] = float(phys_cfg["exterior"].get("epsilon_r", defaults["er_exterior"]))
            defaults["mur_exterior"] = float(phys_cfg["exterior"].get("mu_r", defaults["mur_exterior"]))

        # Schema v1 fallback ('case')
        case_cfg = cfg.get("case", {})
        if "frequency" in case_cfg:
            defaults["frequency"] = float(case_cfg["frequency"])
        if "radius" in case_cfg:
            defaults["radius"] = float(case_cfg["radius"])
        if "epsilon_interior" in case_cfg:
            defaults["er_interior"] = float(case_cfg["epsilon_interior"])
        if "mu_interior" in case_cfg:
            defaults["mur_interior"] = float(case_cfg["mu_interior"])
        if "epsilon_exterior" in case_cfg:
            defaults["er_exterior"] = float(case_cfg["epsilon_exterior"])
        if "mu_exterior" in case_cfg:
            defaults["mur_exterior"] = float(case_cfg["mu_exterior"])

        # Sampling configuration
        samp_cfg = cfg.get("sampling", {})
        if "radial_points" in samp_cfg:
            defaults["radial_points"] = int(samp_cfg["radial_points"])
        if "radial_min_radii" in samp_cfg:
            defaults["radial_min_radii"] = float(samp_cfg["radial_min_radii"])
        if "radial_max_radii" in samp_cfg:
            defaults["radial_max_radii"] = float(samp_cfg["radial_max_radii"])
        if "radial_theta_degrees" in samp_cfg:
            defaults["radial_theta_deg"] = float(samp_cfg["radial_theta_degrees"])
        if "radial_phi_degrees" in samp_cfg:
            defaults["radial_phi_deg"] = float(samp_cfg["radial_phi_degrees"])
        if "zx_grid_size" in samp_cfg:
            defaults["zx_grid_size"] = int(samp_cfg["zx_grid_size"])
        if "zx_span_radii" in samp_cfg:
            defaults["zx_span_radii"] = float(samp_cfg["zx_span_radii"])

    return defaults


def run_validation_pipeline(args: argparse.Namespace) -> None:
    """Execute the end-to-end COMSOL validation pipeline."""
    total_start_time = time.time()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_defaults = load_config_defaults(args)
    f0 = args.frequency if args.frequency is not None else cfg_defaults["frequency"]
    a = args.radius if args.radius is not None else cfg_defaults["radius"]
    er_int = args.er_interior if args.er_interior is not None else cfg_defaults["er_interior"]
    mur_int = args.mur_interior if args.mur_interior is not None else cfg_defaults["mur_interior"]
    er_ext = args.er_exterior if args.er_exterior is not None else cfg_defaults["er_exterior"]
    mur_ext = args.mur_exterior if args.mur_exterior is not None else cfg_defaults["mur_exterior"]
    dipole_type = args.dipole_type or cfg_defaults["dipole_type"]
    dipole_dir = tuple(args.dipole_direction) if args.dipole_direction else cfg_defaults["dipole_direction"]

    radial_points = args.radial_points if args.radial_points is not None else cfg_defaults["radial_points"]
    radial_min_radii = args.radial_min_radii if args.radial_min_radii is not None else cfg_defaults["radial_min_radii"]
    radial_max_radii = args.radial_max_radii if args.radial_max_radii is not None else cfg_defaults["radial_max_radii"]
    radial_theta_deg = args.radial_theta if args.radial_theta is not None else cfg_defaults["radial_theta_deg"]
    radial_phi_deg = args.radial_phi if args.radial_phi is not None else cfg_defaults["radial_phi_deg"]
    zx_nx = args.zx_nx if args.zx_nx is not None else cfg_defaults["zx_grid_size"]
    zx_nz = args.zx_nz if args.zx_nz is not None else cfg_defaults["zx_grid_size"]
    zx_span_radii = args.zx_span_radii if args.zx_span_radii is not None else cfg_defaults["zx_span_radii"]

    lam0 = c / f0
    tau = args.tau if args.tau is not None else 0.01 * lam0
    b = args.air_extent if args.air_extent is not None else 2.0 * lam0
    t_pml = args.pml_thickness if args.pml_thickness is not None else 0.25 * lam0

    # Auto-scale interior dielectric mesh divisor by refractive index if not explicitly set
    n_int = math.sqrt(er_int * mur_int)
    interior_hmax_divisor = (
        args.interior_hmax_divisor
        if args.interior_hmax_divisor is not None
        else round(args.global_mesh_divisor * n_int, 2)
    )

    source_ball_radius_m = (
        args.source_ball_radius * lam0
        if args.source_ball_radius is not None and args.source_ball_radius > 0.0
        else None
    )
    source_ball_hmax_divisor = (
        args.source_ball_hmax_divisor if source_ball_radius_m is not None else None
    )
    inner_band_width_m = (
        args.inner_shell_band_width * lam0
        if args.inner_shell_band_width is not None and args.inner_shell_band_width > 0.0
        else None
    )
    inner_band_hmax_divisor = (
        args.inner_shell_band_hmax_divisor if inner_band_width_m is not None else None
    )
    outer_band_width_m = (
        args.outer_shell_band_width * lam0
        if args.outer_shell_band_width is not None and args.outer_shell_band_width > 0.0
        else None
    )
    outer_band_hmax_divisor = (
        args.outer_shell_band_hmax_divisor if outer_band_width_m is not None else None
    )

    print("=" * 80)
    print("BIANISOTROPIC SPHERICAL METASURFACE COMSOL VALIDATION (CASE B)")
    print("=" * 80)
    print(f"Target COMSOL Version : COMSOL {args.version} (Discretization: {args.discretization})")
    print(f"Frequency (f0)        : {f0 / 1e6:.3f} MHz (lambda0 = {lam0:.3f} m)")
    print(f"Interior Radius (a)   : {a:.3f} m (lambda_int = {lam0 / n_int:.3f} m)")
    print(f"Shell Thickness (tau) : {tau:.4f} m ({tau / lam0:.3f} lambda0)")
    print(f"Air Extent (b)        : {b:.3f} m ({b / lam0:.2f} lambda0)")
    print(f"PML Thickness (t_pml) : {t_pml:.3f} m ({t_pml / lam0:.2f} lambda0)")
    print(f"Media Parameters      : Interior er={er_int:.2f}, mur={mur_int:.2f} | Exterior er={er_ext:.2f}, mur={mur_ext:.2f}")
    print(f"Dipole Source         : {dipole_type.capitalize()} point dipole along {dipole_dir}")
    print(f"Shell Mesh Strategy   : {args.shell_mesh.capitalize()} ({args.shell_layers} radial layers, tan_div={args.shell_tangential_divisor})")
    print(f"Volume Mesh Divisors  : Air={args.global_mesh_divisor}, Core={interior_hmax_divisor}, SourceBall={source_ball_hmax_divisor}")
    print(f"Output Directory      : {out_dir}")
    print("=" * 80)

    # =========================================================================
    # Phase 1: Load GSTC Metasurface Parameters & Mesh Centroids
    # =========================================================================
    t0 = time.time()
    print("\n[Phase 1] Loading GSTC Metasurface Parameters & Mesh Face Centroids...")
    param_path = args.thetas or args.tensors or args.csv
    if param_path is None:
        raise ValueError("Must specify at least one parameter input file via --thetas, --tensors, or --csv")

    thetas, theta_c, phi_c, centroids = load_face_parameters(
        param_path,
        mesh_path=args.mesh,
    )
    n_faces = len(thetas)
    delta_centroid_deg = math.degrees(math.sqrt(4.0 * math.pi / max(1, n_faces)))
    print(f"  Loaded {n_faces} face parameters from: {param_path}")
    if args.mesh is not None:
        print(f"  Extracted {len(centroids)} triangle centroids from mesh: {args.mesh}")
    print(f"  Mean centroid angular spacing: Delta_theta_centroid = {delta_centroid_deg:.2f} deg")
    print(f"  Phase 1 completed in {time.time() - t0:.2f} s")

    # =========================================================================
    # Phase 2: Mesh-Centroid-Based 2D (theta, phi) Periodic Interpolation
    # =========================================================================
    t0 = time.time()
    print("\n[Phase 2] Mesh-Centroid-to-Grid (theta, phi) Periodic Interpolation...")
    n_theta, n_phi = compute_adaptive_grid_resolution(
        n_faces, oversample_factor=args.grid_oversample
    )
    delta_grid_deg = 180.0 / (n_theta - 1)
    print(
        f"  Centroid-Adaptive Grid Resolution: {n_theta} x {n_phi} points "
        f"(Delta_theta_grid = {delta_grid_deg:.2f} deg, oversample = {args.grid_oversample:.1f}x, method = {args.grid_method})"
    )

    grid_theta, grid_phi, grid_values = interpolate_thetas_to_grid(
        theta_c,
        phi_c,
        thetas,
        n_theta=n_theta,
        n_phi=n_phi,
        method=args.grid_method,
    )

    # Save interpolation grid files permanently in out_dir / "interpolation_grids"
    # so the saved .mph model remains self-contained when opened in COMSOL Desktop
    grids_dir = out_dir / "interpolation_grids"
    grid_files = export_comsol_grid_files(grids_dir, grid_theta, grid_phi, grid_values)
    print(f"  Exported {len(grid_files)} COMSOL '% Grid' files to: {grids_dir}")
    print(f"  Phase 2 completed in {time.time() - t0:.2f} s")

    # =========================================================================
    # Phase 3 & 4: Start COMSOL Session, Build Geometry, and Create Mesh
    # =========================================================================
    runtime_dir = out_dir / ".comsol_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    server = None
    client = None
    try:
        t0 = time.time()
        print(f"\n[Phase 3] Launching COMSOL {args.version} Server ({args.cores} cores)...")
        server, client = start_comsol(
            version=args.version,
            cores=args.cores,
            runtime_dir=runtime_dir,
        )
        model = client.create(f"bianisotropic_validation_{args.shell_mesh}")
        java = model.java

        # Set Global Model Parameters
        java.param().set("f0", f"{f0:.17g}[Hz]")
        java.param().set("lambda0", "c_const/f0")
        java.param().set("a", f"{a:.17g}[m]")
        java.param().set("tau", f"{tau:.17g}[m]")
        java.param().set("b", f"{b:.17g}[m]")
        java.param().set("t_pml", f"{t_pml:.17g}[m]")
        java.param().set("er_int", f"{er_int:.17g}")
        java.param().set("mur_int", f"{mur_int:.17g}")
        java.param().set("er_ext", f"{er_ext:.17g}")
        java.param().set("mur_ext", f"{mur_ext:.17g}")
        java.param().set("eta0", f"{ETA_0:.17g}[ohm]")
        java.param().set("pI", "1.0[A*m]")
        java.param().set("mI", "1.0[A*m^2]")
        if source_ball_radius_m is not None:
            java.param().set("r_src", f"{source_ball_radius_m:.17g}[m]")
        if inner_band_width_m is not None:
            java.param().set("w_in", f"{inner_band_width_m:.17g}[m]")
        if outer_band_width_m is not None:
            java.param().set("w_out", f"{outer_band_width_m:.17g}[m]")

        comp = java.component().create("comp1", True)

        print("  Constructing Concentric Geometry & Refinement Bands...")
        selections = setup_spherical_shell_geometry(
            comp,
            wavelength_m=lam0,
            radius_m=a,
            tau_m=tau,
            air_extent_m=b,
            inner_band_width_m=inner_band_width_m,
            outer_band_width_m=outer_band_width_m,
            source_ball_radius_m=source_ball_radius_m,
            shell_mesh=args.shell_mesh,
        )
        print(f"  Phase 3 completed in {time.time() - t0:.2f} s")

        # =====================================================================
        # Phase 4: Bianisotropic Physics & Equation View Overrides
        # =====================================================================
        t0 = time.time()
        print("\n[Phase 4] Setting Up Bianisotropic Physics & Equation View Overrides...", flush=True)
        if args.version == "5.4":
            phys_tag = setup_physics_comsol_54(
                java,
                comp,
                selections,
                grid_files=grid_files,
                discretization=args.discretization,
                dipole_type=dipole_type,
                dipole_direction=dipole_dir,
                enable_far_field=True,
            )
        else:
            phys_tag = setup_physics_comsol_61(
                java,
                comp,
                selections,
                grid_files=grid_files,
                discretization=args.discretization,
                dipole_type=dipole_type,
                dipole_direction=dipole_dir,
                enable_far_field=True,
            )
        print(f"  Activated Physics Interface: {phys_tag} (discretization={args.discretization})", flush=True)
        print(f"  Phase 4 completed in {time.time() - t0:.2f} s", flush=True)

        # =====================================================================
        # Phase 5: Meshing Strategy (Swept Prism or Tetrahedral)
        # =====================================================================
        t0 = time.time()
        print(f"\n[Phase 5] Meshing Shell and Volume Domains ({args.shell_mesh.capitalize()})...", flush=True)
        mesh_stats = setup_spherical_shell_mesh(
            comp,
            selections,
            shell_mesh=args.shell_mesh,
            global_hmax_divisor=args.global_mesh_divisor,
            interior_hmax_divisor=interior_hmax_divisor,
            source_ball_hmax_divisor=source_ball_hmax_divisor,
            inner_shell_band_hmax_divisor=inner_band_hmax_divisor,
            outer_shell_band_hmax_divisor=outer_band_hmax_divisor,
            shell_tangential_divisor=args.shell_tangential_divisor,
            shell_radial_layers=args.shell_layers,
        )
        print(f"  Mesh Generated: {mesh_stats.total_elements:,} total elements", flush=True)
        if args.shell_mesh == "swept":
            print(f"    - Swept Shell Prisms : {mesh_stats.prisms:,} ({mesh_stats.source_triangles:,} source triangles x {args.shell_layers} layers)", flush=True)
            print(f"    - Volume Tetrahedra  : {mesh_stats.tetrahedra:,}", flush=True)
        else:
            print(f"    - Total Tetrahedra   : {mesh_stats.tetrahedra:,}", flush=True)
        print(f"  Phase 5 completed in {time.time() - t0:.2f} s", flush=True)

        # =====================================================================
        # Phase 6: Study & Direct Linear Solver Configuration
        # =====================================================================
        t0 = time.time()
        print(f"\n[Phase 6] Setting Up Frequency Study & Direct Solver ({args.linear_solver.upper()})...")
        study_tag = setup_study_and_solver(
            java,
            linear_solver=args.linear_solver,
        )
        print(f"  Configured Study '{study_tag}' with {args.linear_solver.upper()} direct solver")

        print("  Solving COMSOL FEM Model...")
        solve_start = time.time()
        model.solve("Study 1")
        solve_duration = time.time() - solve_start
        print(f"  Model Solved Successfully in {solve_duration:.2f} s!")

        # Save Solved Model (.mph) immediately after solving so it is never lost
        mph_path: Path | None = None
        if args.save_mph:
            mph_path = out_dir / f"bianisotropic_shell_{args.shell_mesh}_solved.mph"
            print(f"  Saving Solved COMSOL Model to: {mph_path}...")
            model.save(str(mph_path))
            print("    Model saved successfully.")

        print(f"  Phase 6 completed in {time.time() - t0:.2f} s")

        # =====================================================================
        # Phase 7: Post-Processing, Field Extraction, and CSV Generation
        # =====================================================================
        t0 = time.time()
        print("\n[Phase 7] Post-Processing Field Extractions & Validation Artifacts...")

        # Physical dipole scaling
        if not args.disable_dipole_scale:
            dipole_scale = compute_dipole_normalization(f0, er_interior=er_int, mur_interior=mur_int)
            print(f"  Applying Dipole Normalization Scale Factor: {dipole_scale}")
        else:
            dipole_scale = None
            print("  Dipole normalization disabled (raw COMSOL fields)")

        # 1. 1D Physical Radial Cut
        r_min_m = radial_min_radii * a
        r_max_m = min(radial_max_radii * a, b)
        print(
            f"  Extracting Physical Radial Cut ({radial_points} points, "
            f"r in [{r_min_m:.2f}, {r_max_m:.2f}] m at theta={radial_theta_deg:.1f} deg, phi={radial_phi_deg:.1f} deg)..."
        )
        radial_data = extract_near_field_radial(
            model,
            phys_tag,
            r_min=r_min_m,
            r_max=r_max_m,
            num_points=radial_points,
            theta_rad=math.radians(radial_theta_deg),
            phi_rad=math.radians(radial_phi_deg),
            a_radius=a,
            tau_m=tau,
            dipole_scale_factor=dipole_scale,
        )
        radial_csv_path = out_dir / "comsol_physical_radial.csv"
        save_physical_radial_csv(radial_data, radial_csv_path)
        print(f"    Saved: {radial_csv_path.name}")

        # 2. 2D Physical ZX-Plane Near-Field Cut
        zx_extent = min(zx_span_radii * a, b)
        print(
            f"  Extracting 2D Physical ZX Cut ({zx_nx} x {zx_nz} = {zx_nx * zx_nz:,} points, "
            f"span=[-{zx_extent:.2f}, {zx_extent:.2f}] m)..."
        )
        zx_data = extract_near_field_zx(
            model,
            phys_tag,
            x_min=-zx_extent,
            x_max=zx_extent,
            z_min=-zx_extent,
            z_max=zx_extent,
            nx=zx_nx,
            nz=zx_nz,
            a_radius=a,
            tau_m=tau,
            dipole_scale_factor=dipole_scale,
        )
        zx_csv_path = out_dir / "comsol_physical_zx.csv"
        save_physical_zx_csv(zx_data, zx_csv_path)
        print(f"    Saved: {zx_csv_path.name}")

        # 3. Stratton-Chu Far-Field Principal Cuts
        print(f"  Extracting Stratton-Chu Far-Field Cuts ({args.far_field_points} angles)...")
        ff_cuts = extract_far_field_cuts(
            model,
            phys_tag,
            num_points=args.far_field_points,
            dipole_scale_factor=dipole_scale,
        )
        ff_csv_path = out_dir / "comsol_far_field_cuts.csv"
        save_far_field_cuts_csv(ff_cuts, ff_csv_path)
        print(f"    Saved: {ff_csv_path.name}")
        print(f"    Radiated Power P_rad : {ff_cuts['p_rad_w']:.4f} W")
        print(f"    E-Plane Max D_dbi    : {np.max(ff_cuts['E_plane']['directivity_dbi']):.2f} dBi")
        print(f"    H-Plane Max D_dbi    : {np.max(ff_cuts['H_plane']['directivity_dbi']):.2f} dBi")
        print(f"    XY-Plane Max D_dbi   : {np.max(ff_cuts['XY_plane']['directivity_dbi']):.2f} dBi")

        # 4. Save Metrics Summary JSON
        metrics = {
            "validation_case": {
                "comsol_version": args.version,
                "physics_interface": phys_tag,
                "discretization": args.discretization,
                "frequency_hz": f0,
                "wavelength_m": lam0,
                "radius_m": a,
                "tau_m": tau,
                "air_extent_m": b,
                "pml_thickness_m": t_pml,
                "er_interior": er_int,
                "mur_interior": mur_int,
                "er_exterior": er_ext,
                "mur_exterior": mur_ext,
                "dipole_type": dipole_type,
                "dipole_direction": dipole_dir,
                "dipole_normalization_scale": str(dipole_scale) if dipole_scale else None,
            },
            "interpolation": {
                "num_mesh_centroids": n_faces,
                "mean_centroid_spacing_deg": delta_centroid_deg,
                "oversample_factor": args.grid_oversample,
                "interpolation_method": args.grid_method,
                "grid_n_theta": n_theta,
                "grid_n_phi": n_phi,
                "grid_step_deg": delta_grid_deg,
                "grids_dir": str(grids_dir),
            },
            "mesh": {
                "strategy": args.shell_mesh,
                "shell_layers": args.shell_layers if args.shell_mesh == "swept" else 1,
                "global_hmax_divisor": args.global_mesh_divisor,
                "interior_hmax_divisor": interior_hmax_divisor,
                "shell_tangential_divisor": args.shell_tangential_divisor,
                "source_ball_radius_m": source_ball_radius_m,
                "source_ball_hmax_divisor": source_ball_hmax_divisor,
                "inner_band_width_m": inner_band_width_m,
                "inner_band_hmax_divisor": inner_band_hmax_divisor,
                "outer_band_width_m": outer_band_width_m,
                "outer_band_hmax_divisor": outer_band_hmax_divisor,
                "total_elements": mesh_stats.total_elements,
                "prisms": mesh_stats.prisms,
                "tetrahedra": mesh_stats.tetrahedra,
                "source_triangles": mesh_stats.source_triangles,
            },
            "solver": {
                "linear_solver": args.linear_solver,
                "solve_duration_seconds": solve_duration,
                "cores": args.cores,
            },
            "far_field_metrics": {
                "p_rad_w": ff_cuts["p_rad_w"],
                "e_plane_max_dbi": float(np.max(ff_cuts["E_plane"]["directivity_dbi"])),
                "h_plane_max_dbi": float(np.max(ff_cuts["H_plane"]["directivity_dbi"])),
                "xy_plane_max_dbi": float(np.max(ff_cuts["XY_plane"]["directivity_dbi"])),
            },
            "files": {
                "mph_model": mph_path.name if mph_path else None,
                "radial_csv": radial_csv_path.name,
                "zx_csv": zx_csv_path.name,
                "far_field_cuts_csv": ff_csv_path.name,
            },
            "total_execution_seconds": time.time() - total_start_time,
        }
        metrics_path = out_dir / "comsol_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"    Saved: {metrics_path.name}")

        # Also export standard schema_version 2 run.json for plot_fields.py integration
        raw_cfg = cfg_defaults.get("raw_config", {})
        case_id = raw_cfg.get("case_id", "comsol_validation")
        sim_cases = raw_cfg.get("simulation", {}).get("cases", [])
        sim_case_id = (
            sim_cases[0].get("id", "electric_incidence") if sim_cases else "electric_incidence"
        )
        sim_excitations = (
            sim_cases[0].get("excitations", [{"field": "electric_z_interior", "region": "interior"}])
            if sim_cases
            else [{"field": "electric_z_interior", "region": "interior"}]
        )

        run_manifest = {
            "schema_version": 2,
            "application": "anisotropic_gstc_sphere",
            "case_id": case_id,
            "backend": "comsol",
            "status": "completed",
            "phasor_convention": "exp(+j omega t)",
            "tangent_basis": "theta, phi; outward radial normal",
            "frequency_hz": f0,
            "radius_m": a,
            "sampling": {
                "zx_grid_size": zx_nx,
                "zx_span_radii": zx_span_radii,
                "radial_points": radial_points,
                "radial_theta_degrees": radial_theta_deg,
                "radial_phi_degrees": radial_phi_deg,
                "radial_min_radii": radial_min_radii,
                "radial_max_radii": radial_max_radii,
            },
            "field_data": {
                "coordinate_system": "Cartesian",
                "coordinate_unit": "m",
                "electric_field_unit": "V/m",
                "magnetic_field_unit": "A/m",
                "regions": {
                    "1": "interior",
                    "2": "exterior",
                    "3": "shell",
                },
                "physical_construction": "interior for r<a; exterior for r>=a+tau",
            },
            "simulation": {
                "cases": [
                    {
                        "id": sim_case_id,
                        "status": "completed",
                        "excitations": sim_excitations,
                        "files": {
                            "computed_radial": radial_csv_path.name,
                            "computed_zx": zx_csv_path.name,
                            "plots": "plots",
                        },
                    }
                ]
            },
            "files": {
                "configuration": "configuration.json",
            },
        }
        run_manifest_path = out_dir / "run.json"
        with run_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)
        print(f"    Saved: {run_manifest_path.name}")

        if args.config and args.config.is_file():
            shutil.copyfile(args.config, out_dir / "configuration.json")
            print(f"    Saved: configuration.json (snapshot)")

        print(f"  Phase 7 completed in {time.time() - t0:.2f} s")

    finally:
        if server is not None and client is not None:
            print("\n[Shutdown] Stopping COMSOL Server...")
            stop_comsol(server, client)
        # Clean up temporary COMSOL server runtime directory (keep interpolation_grids/)
        shutil.rmtree(runtime_dir, ignore_errors=True)

    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"COMSOL VALIDATION COMPLETE IN {total_time:.2f} s ({total_time / 60:.2f} min)")
    print(f"Artifacts Directory: {out_dir}")
    print("=" * 80)


def main() -> None:
    """Main CLI entry point."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True, errors="replace")
    except Exception:
        pass
    args = parse_arguments()
    run_validation_pipeline(args)


if __name__ == "__main__":
    main()
