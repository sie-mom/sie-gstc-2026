#!/usr/bin/env python3
"""COMSOL Multiphysics Validation Pipeline for Bianisotropic Spherical Metasurfaces (Case A).

Executes a full 3D FEM validation of the Case A zero-thickness sheet GSTC formulation
against SIE-BEM solver results:
  - Universal input loading: .bin (optimized_thetas.bin or face_tensors.bin) or .csv
  - Mesh-centroid-based adaptive 2D (theta, phi) periodic interpolation to COMSOL '% Grid' tables
  - COMSOL session management supporting COMSOL 5.4 and COMSOL 6.1
  - Concentric zero-thickness metasurface sphere (r = a) + optional detached Huygens far-field sphere
  - Direct two-sided GSTC jump enforcement via SurfaceCurrent (scu1) and
    SurfaceMagneticCurrentDensity (smcd1) with broken-H(curl) up()/down() field trace averages
  - Surface-refined conformal tetrahedral meshing (no thin volumetric shell or prism layers)
  - MUMPS direct linear solver for the nonsymmetric constrained system
  - Extraction and export of physical near-field (radial cut, zx-plane) and Stratton-Chu far-field cuts

Usage Example:
  python applications/anisotropic_gstc_sphere/comsol/bianisotropic_sheet_validation.py \\
      --tensors applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7/face_tensors.bin \\
      --mesh applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere_ppw6.obj \\
      --config applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot/e2m_eps7.json \\
      --version 6.1 \\
      --linear-solver mumps \\
      --output-dir applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7/comsol/case_a_test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Mapping
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
    return max(1, total - 2) if total > 4 else total


# Add repo root to sys.path so modules resolve whether executed as package or script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from applications.anisotropic_gstc_sphere.comsol.comsol_helpers import (
    DipoleExcitationCase,
    compute_dipole_normalization,
    extract_far_field_cuts,
    extract_near_field_radial,
    extract_near_field_zx,
    save_far_field_cuts_csv,
    save_physical_radial_csv,
    save_physical_zx_csv,
    setup_sheet_physics_comsol_54,
    setup_sheet_physics_comsol_61,
    setup_spherical_sheet_geometry,
    setup_spherical_sheet_mesh,
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
    """Parse command line arguments for the Case A COMSOL validation pipeline."""
    parser = argparse.ArgumentParser(
        description="COMSOL FEM Validation for Bianisotropic Spherical Metasurfaces (Case A Zero-Thickness Sheet)",
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
        help="Metasurface sphere radius a in meters (overrides config if given)",
    )
    geom_group.add_argument(
        "--frequency",
        type=float,
        default=None,
        help="Operational frequency f0 in Hz (overrides config if given)",
    )
    geom_group.add_argument(
        "--air-extent",
        type=float,
        default=None,
        help="Exterior air box half-width b in meters (default: max(2.0 * lambda0, a + 0.35 * lambda0))",
    )
    geom_group.add_argument(
        "--pml-thickness",
        type=float,
        default=None,
        help="PML absorbing boundary layer thickness in meters (default: 0.25 * lambda0)",
    )
    geom_group.add_argument(
        "--far-field-offset-lambda",
        type=float,
        default=0.10,
        help=(
            "Radial offset (in lambda0) of the exterior Huygens sphere r_ff = a + offset*lambda0 "
            "for Stratton-Chu far-field integration (0 to integrate directly on r = a^+)"
        ),
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
        "--case-id",
        "--case",
        dest="case_id",
        default=None,
        help="Select a specific simulation case ID from configuration simulation.cases (e.g. 'electric_incidence')",
    )
    exc_group.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Select one or more simulation case IDs from configuration simulation.cases",
    )
    exc_group.add_argument(
        "--dipole-type",
        choices=["electric", "magnetic"],
        default=None,
        help="Central point dipole source type (overrides config simulation.cases if explicitly passed)",
    )
    exc_group.add_argument(
        "--dipole-direction",
        type=float,
        nargs=3,
        default=None,
        help="Point dipole orientation Cartesian unit vector [dx, dy, dz] (overrides config if explicitly passed)",
    )

    # -------------------------------------------------------------------------
    # Meshing Strategy & Concentric Refinements
    # -------------------------------------------------------------------------
    mesh_group = parser.add_argument_group("Meshing Strategy & Concentric Refinements")
    mesh_group.add_argument(
        "--sheet-tangential-divisor",
        "--shell-tangential-divisor",
        dest="sheet_tangential_divisor",
        type=float,
        default=25.0,
        help="Metasurface boundary (r = a) triangle maximum size as lambda0 / divisor",
    )
    mesh_group.add_argument(
        "--global-mesh-divisor",
        "--global-hmax-divisor",
        dest="global_mesh_divisor",
        type=float,
        default=15.0,
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
        default=0.10,
        help="Radius of conforming refinement ball around the point dipole in lambda0 (0 to disable)",
    )
    mesh_group.add_argument(
        "--source-ball-hmax-divisor",
        type=float,
        default=60.0,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the point-source ball",
    )
    mesh_group.add_argument(
        "--inner-sheet-band-width",
        type=float,
        default=None,
        help="Width of optional dielectric refinement band inside r = a in lambda0",
    )
    mesh_group.add_argument(
        "--inner-sheet-band-hmax-divisor",
        type=float,
        default=None,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the inner-sheet band",
    )
    mesh_group.add_argument(
        "--outer-sheet-band-width",
        type=float,
        default=None,
        help="Width of optional air refinement band outside r = a in lambda0",
    )
    mesh_group.add_argument(
        "--outer-sheet-band-hmax-divisor",
        type=float,
        default=None,
        help="Divisor N for maximum tetrahedron size lambda0 / N in the outer-sheet band",
    )
    mesh_group.add_argument(
        "--far-field-surface-divisor",
        type=float,
        default=None,
        help="Divisor N for surface triangles on the Huygens far-field sphere (default: max(global, 15))",
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
        default="6.1",
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
        choices=["mumps", "pardiso", "spooles"],
        default="mumps",
        help="Direct linear solver engine (MUMPS strongly recommended for Case A nonsymmetric constraints)",
    )
    solver_group.add_argument(
        "--mesh-only",
        action="store_true",
        default=False,
        help="Build geometry, Case A physics, mesh, and solver configuration without solving",
    )
    solver_group.add_argument(
        "--save-mph",
        action="store_true",
        default=True,
        help="Save the COMSOL model file (.mph) to the output directory",
    )
    parser.add_argument(
        "--no-save-mph",
        action="store_false",
        dest="save_mph",
        help="Do not save the COMSOL model file (.mph)",
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


def _decompose_linear_moment(
    moment_cfg: Mapping[str, Any] | list[float] | tuple[float, ...],
) -> tuple[tuple[float, float, float], complex]:
    """Decompose a linearly polarized complex dipole moment into (unit_direction, complex_amplitude)."""
    if isinstance(moment_cfg, Mapping):
        real_part = np.asarray(moment_cfg.get("real", [0.0, 0.0, 1.0]), dtype=float)
        imag_part = np.asarray(moment_cfg.get("imag", [0.0, 0.0, 0.0]), dtype=float)
        vec = real_part + 1j * imag_part
    else:
        vec = np.asarray(moment_cfg, dtype=complex)

    if vec.shape != (3,):
        raise ValueError(f"Dipole effective_moment must have 3 components, got shape {vec.shape}")
    abs_vec = np.abs(vec)
    max_idx = int(np.argmax(abs_vec))
    if abs_vec[max_idx] == 0.0:
        raise ValueError("Dipole effective_moment must be nonzero")

    phase_ref = vec[max_idx] / abs_vec[max_idx]
    aligned = vec / phase_ref
    if np.linalg.norm(aligned.imag) > 1e-12 * np.linalg.norm(aligned.real):
        raise ValueError(
            "Elliptically polarized point dipoles are not supported by MagnitudeAndDirection; "
            f"got effective_moment={vec.tolist()}"
        )

    real_vec = aligned.real
    norm_val = float(np.linalg.norm(real_vec))
    unit_dir = (
        float(real_vec[0] / norm_val),
        float(real_vec[1] / norm_val),
        float(real_vec[2] / norm_val),
    )
    amplitude = complex(norm_val * phase_ref)
    return unit_dir, amplitude


def resolve_excitation_cases(
    args: argparse.Namespace,
    cfg_defaults: dict[str, Any],
) -> list[DipoleExcitationCase]:
    """Resolve one or more central point-dipole excitation cases from CLI or Schema v2 JSON."""
    raw_cfg = cfg_defaults.get("raw_config", {})
    sim_cases = list(raw_cfg.get("simulation", {}).get("cases", []))
    fields_catalog = raw_cfg.get("fields", {})

    selected_cases: list[str] | None = None
    cli_cases = getattr(args, "cases", None)
    cli_case_id = getattr(args, "case_id", None)
    if cli_cases:
        selected_cases = [str(c) for c in cli_cases]
    elif cli_case_id:
        selected_cases = [str(cli_case_id)]

    if selected_cases is not None and sim_cases:
        case_map = {str(c["id"]): c for c in sim_cases}
        for sc in selected_cases:
            if sc not in case_map:
                available = list(case_map.keys())
                raise ValueError(
                    f"Selected case '{sc}' not found in configuration simulation.cases. "
                    f"Available cases: {available}"
                )
        sim_cases = [case_map[sc] for sc in selected_cases]

    if args.dipole_type is None and args.dipole_direction is None and sim_cases and fields_catalog:
        resolved: list[DipoleExcitationCase] = []
        for sim_case in sim_cases:
            case_id = str(sim_case["id"])
            excitations = list(sim_case.get("excitations", []))
            if len(excitations) != 1:
                raise ValueError(
                    f"COMSOL Case A currently requires exactly 1 excitation per simulation case; "
                    f"case '{case_id}' has {len(excitations)}"
                )
            exc = excitations[0]
            if exc.get("region") != "interior":
                raise ValueError(
                    f"COMSOL central point-dipole validation requires interior excitation; "
                    f"case '{case_id}' has region={exc.get('region')!r}"
                )
            field_id = exc["field"]
            if field_id not in fields_catalog:
                raise KeyError(f"Case '{case_id}' references unknown field '{field_id}'")
            field_def = fields_catalog[field_id]
            if field_def.get("type") != "dipole":
                raise ValueError(
                    f"COMSOL point-dipole validation requires field.type == 'dipole'; "
                    f"field '{field_id}' has type={field_def.get('type')!r}"
                )
            src_pos = np.asarray(field_def.get("source_position", [0.0, 0.0, 0.0]), dtype=float)
            if float(np.linalg.norm(src_pos)) > 1e-12:
                raise ValueError(
                    f"COMSOL central point-dipole geometry requires source_position == [0, 0, 0]; "
                    f"field '{field_id}' has source_position={src_pos.tolist()}"
                )
            unit_dir, amp = _decompose_linear_moment(
                field_def.get("effective_moment", [0.0, 0.0, 1.0])
            )
            resolved.append(
                DipoleExcitationCase(
                    case_id=case_id,
                    dipole_type=str(field_def["dipole_type"]),
                    dipole_direction=unit_dir,
                    effective_moment_amplitude=amp,
                    raw_excitations=excitations,
                )
            )
        return resolved

    dipole_type = args.dipole_type or cfg_defaults["dipole_type"]
    raw_dir = tuple(args.dipole_direction) if args.dipole_direction else cfg_defaults["dipole_direction"]
    unit_dir, amp = _decompose_linear_moment(raw_dir)
    if sim_cases:
        default_case_id = str(sim_cases[0].get("id", f"{dipole_type}_incidence"))
        default_excitations = list(
            sim_cases[0].get(
                "excitations", [{"field": f"{dipole_type}_z_interior", "region": "interior"}]
            )
        )
    else:
        default_case_id = f"{dipole_type}_incidence"
        default_excitations = [{"field": f"{dipole_type}_z_interior", "region": "interior"}]

    return [
        DipoleExcitationCase(
            case_id=default_case_id,
            dipole_type=dipole_type,
            dipole_direction=unit_dir,
            effective_moment_amplitude=amp,
            raw_excitations=default_excitations,
        )
    ]


def _extract_case_artifacts(
    *,
    model: Any,
    phys_tag: str,
    exc_case: DipoleExcitationCase,
    case_out_dir: Path,
    root_out_dir: Path,
    solnum: int | None,
    f0: float,
    er_int: float,
    mur_int: float,
    disable_dipole_scale: bool,
    a: float,
    b: float,
    radial_points: int,
    radial_min_radii: float,
    radial_max_radii: float,
    radial_theta_deg: float,
    radial_phi_deg: float,
    zx_nx: int,
    zx_nz: int,
    zx_span_radii: float,
    far_field_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract radial, 2D ZX, and Stratton-Chu far-field CSVs for one solved excitation case."""
    case_out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"  [{exc_case.case_id}] " if solnum is not None else "  "

    if not disable_dipole_scale:
        dipole_scale = compute_dipole_normalization(
            f0,
            er_interior=er_int,
            mur_interior=mur_int,
            dipole_type=exc_case.dipole_type,
            effective_moment_amplitude=exc_case.effective_moment_amplitude,
        )
        print(
            f"{prefix}Applying {exc_case.dipole_type} dipole normalization scale factor: {dipole_scale}"
        )
    else:
        dipole_scale = None
        print(f"{prefix}Dipole normalization disabled (raw COMSOL fields)")

    # 1. 1D Physical Radial Cut (tau_m = 0.0 for Case A zero-thickness sheet)
    r_min_m = radial_min_radii * a
    r_max_m = min(radial_max_radii * a, b)
    print(
        f"{prefix}Extracting Physical Radial Cut ({radial_points} points, "
        f"r in [{r_min_m:.4f}, {r_max_m:.4f}] m at theta={radial_theta_deg:.1f} deg, phi={radial_phi_deg:.1f} deg)..."
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
        tau_m=0.0,
        dipole_scale_factor=dipole_scale,
        solnum=solnum,
    )
    radial_csv_path = case_out_dir / "comsol_physical_radial.csv"
    save_physical_radial_csv(radial_data, radial_csv_path)
    rel_radial = radial_csv_path.relative_to(root_out_dir).as_posix()
    print(f"{prefix}  Saved: {rel_radial}")

    # 2. 2D Physical ZX-Plane Near-Field Cut
    zx_extent = min(zx_span_radii * a, b)
    print(
        f"{prefix}Extracting 2D Physical ZX Cut ({zx_nx} x {zx_nz} = {zx_nx * zx_nz:,} points, "
        f"span=[-{zx_extent:.4f}, {zx_extent:.4f}] m)..."
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
        tau_m=0.0,
        dipole_scale_factor=dipole_scale,
        solnum=solnum,
    )
    zx_csv_path = case_out_dir / "comsol_physical_zx.csv"
    save_physical_zx_csv(zx_data, zx_csv_path)
    rel_zx = zx_csv_path.relative_to(root_out_dir).as_posix()
    print(f"{prefix}  Saved: {rel_zx}")

    # 3. Stratton-Chu Far-Field Principal Cuts
    print(f"{prefix}Extracting Stratton-Chu Far-Field Cuts ({far_field_points} angles)...")
    ff_cuts = extract_far_field_cuts(
        model,
        phys_tag,
        num_points=far_field_points,
        dipole_scale_factor=dipole_scale,
        solnum=solnum,
    )
    ff_csv_path = case_out_dir / "comsol_far_field_cuts.csv"
    save_far_field_cuts_csv(ff_cuts, ff_csv_path)
    rel_ff = ff_csv_path.relative_to(root_out_dir).as_posix()
    print(f"{prefix}  Saved: {rel_ff}")
    print(f"{prefix}  Radiated Power P_rad : {ff_cuts['p_rad_w']:.4f} W")
    print(f"{prefix}  E-Plane Max D_dbi    : {np.max(ff_cuts['E_plane']['directivity_dbi']):.2f} dBi")
    print(f"{prefix}  H-Plane Max D_dbi    : {np.max(ff_cuts['H_plane']['directivity_dbi']):.2f} dBi")
    print(f"{prefix}  XY-Plane Max D_dbi   : {np.max(ff_cuts['XY_plane']['directivity_dbi']):.2f} dBi")

    rel_plots = (case_out_dir / "plots").relative_to(root_out_dir).as_posix()
    case_metrics = {
        "case_id": exc_case.case_id,
        "solnum": solnum,
        "dipole_type": exc_case.dipole_type,
        "dipole_direction": exc_case.dipole_direction,
        "effective_moment_amplitude": str(exc_case.effective_moment_amplitude),
        "dipole_normalization_scale": str(dipole_scale) if dipole_scale is not None else None,
        "far_field_metrics": {
            "p_rad_w": ff_cuts["p_rad_w"],
            "e_plane_max_dbi": float(np.max(ff_cuts["E_plane"]["directivity_dbi"])),
            "h_plane_max_dbi": float(np.max(ff_cuts["H_plane"]["directivity_dbi"])),
            "xy_plane_max_dbi": float(np.max(ff_cuts["XY_plane"]["directivity_dbi"])),
        },
        "files": {
            "radial_csv": rel_radial,
            "zx_csv": rel_zx,
            "far_field_cuts_csv": rel_ff,
            "plots": rel_plots,
        },
    }
    manifest_case_entry = {
        "id": exc_case.case_id,
        "status": "completed",
        "excitations": exc_case.raw_excitations
        or [{"field": f"{exc_case.dipole_type}_z_interior", "region": "interior"}],
        "files": {
            "computed_radial": rel_radial,
            "computed_zx": rel_zx,
            "plots": rel_plots,
        },
    }
    return case_metrics, manifest_case_entry


def run_validation_pipeline(args: argparse.Namespace) -> None:
    """Execute the end-to-end Case A zero-thickness sheet COMSOL validation pipeline."""
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

    excitation_cases = resolve_excitation_cases(args, cfg_defaults)
    is_multi_case = len(excitation_cases) > 1
    primary_case = excitation_cases[0]
    dipole_type = primary_case.dipole_type
    dipole_dir = primary_case.dipole_direction

    radial_points = args.radial_points if args.radial_points is not None else cfg_defaults["radial_points"]
    radial_min_radii = args.radial_min_radii if args.radial_min_radii is not None else cfg_defaults["radial_min_radii"]
    radial_max_radii = args.radial_max_radii if args.radial_max_radii is not None else cfg_defaults["radial_max_radii"]
    radial_theta_deg = args.radial_theta if args.radial_theta is not None else cfg_defaults["radial_theta_deg"]
    radial_phi_deg = args.radial_phi if args.radial_phi is not None else cfg_defaults["radial_phi_deg"]
    zx_nx = args.zx_nx if args.zx_nx is not None else cfg_defaults["zx_grid_size"]
    zx_nz = args.zx_nz if args.zx_nz is not None else cfg_defaults["zx_grid_size"]
    zx_span_radii = args.zx_span_radii if args.zx_span_radii is not None else cfg_defaults["zx_span_radii"]

    lam0 = c / f0
    b = args.air_extent if args.air_extent is not None else max(2.0 * lam0, a + 0.35 * lam0)
    t_pml = args.pml_thickness if args.pml_thickness is not None else 0.25 * lam0

    far_field_offset_m = max(0.0, float(args.far_field_offset_lambda) * lam0)
    if far_field_offset_m > 0.0 and (a + far_field_offset_m >= b - 0.02 * lam0):
        clamped = max(0.0, 0.5 * (b - a))
        print(
            f"[Warning] Requested far-field offset {far_field_offset_m:.4e} m reaches outer box b={b:.4e} m; "
            f"clamping to {clamped:.4e} m."
        )
        far_field_offset_m = clamped

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
        args.inner_sheet_band_width * lam0
        if args.inner_sheet_band_width is not None and args.inner_sheet_band_width > 0.0
        else None
    )
    inner_band_hmax_divisor = (
        args.inner_sheet_band_hmax_divisor if inner_band_width_m is not None else None
    )
    outer_band_width_m = (
        args.outer_sheet_band_width * lam0
        if args.outer_sheet_band_width is not None and args.outer_sheet_band_width > 0.0
        else None
    )
    outer_band_hmax_divisor = (
        args.outer_sheet_band_hmax_divisor if outer_band_width_m is not None else None
    )
    far_field_surface_divisor = (
        args.far_field_surface_divisor
        if args.far_field_surface_divisor is not None
        else (max(args.global_mesh_divisor, 15.0) if far_field_offset_m > 0.0 else None)
    )

    print("=" * 80)
    print("BIANISOTROPIC SPHERICAL METASURFACE COMSOL VALIDATION (CASE A: ZERO-THICKNESS SHEET)")
    print("=" * 80)
    print(f"Target COMSOL Version : COMSOL {args.version} (Discretization: {args.discretization})")
    print(f"Linear Solver         : {args.linear_solver.upper()}")
    print(f"Frequency (f0)        : {f0 / 1e6:.3f} MHz (lambda0 = {lam0:.4f} m)")
    print(f"Metasurface Radius (a): {a:.4f} m ({a / lam0:.3f} lambda0, lambda_int = {lam0 / n_int:.4f} m)")
    print(f"Huygens FF Radius     : {a + far_field_offset_m:.4f} m (offset = {far_field_offset_m / lam0:.3f} lambda0)")
    print(f"Air Extent (b)        : {b:.4f} m ({b / lam0:.2f} lambda0)")
    print(f"PML Thickness (t_pml) : {t_pml:.4f} m ({t_pml / lam0:.2f} lambda0)")
    print(f"Media Parameters      : Interior er={er_int:.2f}, mur={mur_int:.2f} | Exterior er={er_ext:.2f}, mur={mur_ext:.2f}")
    if is_multi_case:
        cases_desc = ", ".join(f"{ec.case_id} ({ec.dipole_type})" for ec in excitation_cases)
        print(f"Excitation Cases (LU) : {len(excitation_cases)} RHS cases [{cases_desc}]")
    else:
        print(f"Dipole Source         : {dipole_type.capitalize()} point dipole along {dipole_dir}")
    print(f"Sheet Mesh Strategy   : Conformal FreeTri on r=a (tan_div={args.sheet_tangential_divisor}) + FreeTet")
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

    grids_dir = out_dir / "interpolation_grids"
    grid_files = export_comsol_grid_files(grids_dir, grid_theta, grid_phi, grid_values)
    print(f"  Exported {len(grid_files)} COMSOL '% Grid' files to: {grids_dir}")
    print(f"  Phase 2 completed in {time.time() - t0:.2f} s")

    # =========================================================================
    # Phase 3–7: COMSOL Session, Geometry, Case A Sheet Physics, Mesh & Solve
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
        model = client.create("bianisotropic_sheet_validation_case_a")
        java = model.java

        # Set Global Model Parameters
        java.param().set("f0", f"{f0:.17g}[Hz]")
        java.param().set("lambda0", "c_const/f0")
        java.param().set("a", f"{a:.17g}[m]")
        java.param().set("b", f"{b:.17g}[m]")
        java.param().set("t_pml", f"{t_pml:.17g}[m]")
        java.param().set("er_int", f"{er_int:.17g}")
        java.param().set("mur_int", f"{mur_int:.17g}")
        java.param().set("er_ext", f"{er_ext:.17g}")
        java.param().set("mur_ext", f"{mur_ext:.17g}")
        java.param().set("eta0", f"{ETA_0:.17g}[ohm]")
        java.param().set("pI", "1.0[A*m]")
        java.param().set("mI", "1.0[A*m^2]")
        if far_field_offset_m > 0.0:
            java.param().set("w_ff", f"{far_field_offset_m:.17g}[m]")
        if source_ball_radius_m is not None:
            java.param().set("r_src", f"{source_ball_radius_m:.17g}[m]")
        if inner_band_width_m is not None:
            java.param().set("w_in", f"{inner_band_width_m:.17g}[m]")
        if outer_band_width_m is not None:
            java.param().set("w_out", f"{outer_band_width_m:.17g}[m]")

        comp = java.component().create("comp1", True)

        print("  Constructing Zero-Thickness Sheet Geometry & Refinement Spheres...")
        selections = setup_spherical_sheet_geometry(
            comp,
            wavelength_m=lam0,
            radius_m=a,
            air_extent_m=b,
            far_field_offset_m=far_field_offset_m,
            inner_band_width_m=inner_band_width_m,
            outer_band_width_m=outer_band_width_m,
            source_ball_radius_m=source_ball_radius_m,
        )
        print(f"  Phase 3 completed in {time.time() - t0:.2f} s")

        # =====================================================================
        # Phase 4: Case A Zero-Thickness Sheet Physics (scu1 + smcd1)
        # =====================================================================
        t0 = time.time()
        print("\n[Phase 4] Setting Up Case A Zero-Thickness Sheet GSTC Physics...", flush=True)
        physics_setup_fn = (
            setup_sheet_physics_comsol_54 if args.version == "5.4" else setup_sheet_physics_comsol_61
        )
        phys_tag = physics_setup_fn(
            java,
            comp,
            selections,
            grid_files=grid_files,
            discretization=args.discretization,
            dipole_type=dipole_type,
            dipole_direction=dipole_dir,
            excitation_cases=excitation_cases,
            enable_far_field=True,
        )
        print(f"  Activated Physics Interface: {phys_tag} (discretization={args.discretization})", flush=True)
        print(f"  Phase 4 completed in {time.time() - t0:.2f} s", flush=True)

        # =====================================================================
        # Phase 5: Surface-Refined Conformal Tetrahedral Meshing
        # =====================================================================
        t0 = time.time()
        print("\n[Phase 5] Meshing Metasurface Boundary (FreeTri) and Volume Domains (FreeTet)...", flush=True)
        mesh_stats = setup_spherical_sheet_mesh(
            comp,
            selections,
            global_hmax_divisor=args.global_mesh_divisor,
            interior_hmax_divisor=interior_hmax_divisor,
            source_ball_hmax_divisor=source_ball_hmax_divisor,
            inner_sheet_band_hmax_divisor=inner_band_hmax_divisor,
            outer_sheet_band_hmax_divisor=outer_band_hmax_divisor,
            sheet_tangential_divisor=args.sheet_tangential_divisor,
            far_field_surface_divisor=far_field_surface_divisor,
        )
        print(f"  Mesh Generated: {mesh_stats.total_elements:,} total elements", flush=True)
        print(f"    - Metasurface Sheet Triangles (r = a) : {mesh_stats.source_triangles:,}", flush=True)
        print(f"    - Volume Tetrahedra                   : {mesh_stats.tetrahedra:,}", flush=True)
        print(f"  Phase 5 completed in {time.time() - t0:.2f} s", flush=True)

        # =====================================================================
        # Phase 6: Study & Direct Linear Solver Configuration (MUMPS)
        # =====================================================================
        t0 = time.time()
        print(f"\n[Phase 6] Setting Up Frequency Study & Direct Solver ({args.linear_solver.upper()})...")
        aux_param_name = "case_idx" if is_multi_case else None
        aux_param_values = list(range(1, len(excitation_cases) + 1)) if is_multi_case else None
        study_tag = setup_study_and_solver(
            java,
            linear_solver=args.linear_solver,
            aux_param_name=aux_param_name,
            aux_param_values=aux_param_values,
        )
        if is_multi_case:
            print(
                f"  Configured Study '{study_tag}' with {args.linear_solver.upper()} direct solver "
                f"and LU factorization reuse across {len(excitation_cases)} RHS excitations"
            )
        else:
            print(f"  Configured Study '{study_tag}' with {args.linear_solver.upper()} direct solver")

        if args.mesh_only:
            mph_path: Path | None = None
            if args.save_mph:
                mph_path = out_dir / "bianisotropic_sheet_case_a_mesh_only.mph"
                print(f"  Saving Mesh-Only COMSOL Model to: {mph_path}...")
                model.save(str(mph_path))
            metrics_mesh_only = {
                "mode": "mesh_only",
                "formulation": "case_a_zero_thickness_sheet",
                "physics_interface": phys_tag,
                "case_count": len(excitation_cases),
                "cases": [ec.case_id for ec in excitation_cases],
                "mesh": {
                    "strategy": "sheet_freetri_conformal_tet",
                    "global_hmax_divisor": args.global_mesh_divisor,
                    "interior_hmax_divisor": interior_hmax_divisor,
                    "sheet_tangential_divisor": args.sheet_tangential_divisor,
                    "source_ball_radius_m": source_ball_radius_m,
                    "source_ball_hmax_divisor": source_ball_hmax_divisor,
                    "total_elements": mesh_stats.total_elements,
                    "tetrahedra": mesh_stats.tetrahedra,
                    "sheet_triangles": mesh_stats.source_triangles,
                },
            }
            with (out_dir / "comsol_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_mesh_only, f, indent=2)
            print("  Mesh-only verification completed successfully.")
            return

        if is_multi_case:
            print(
                f"  Solving COMSOL Case A FEM Model ({len(excitation_cases)} RHS cases via single LU factorization)..."
            )
        else:
            print("  Solving COMSOL Case A FEM Model...")
        solve_start = time.time()
        model.solve("Study 1")
        solve_duration = time.time() - solve_start
        print(f"  Model Solved Successfully in {solve_duration:.2f} s!")

        mph_path = None
        if args.save_mph:
            mph_path = out_dir / "bianisotropic_sheet_case_a_solved.mph"
            print(f"  Saving Solved COMSOL Model to: {mph_path}...")
            model.save(str(mph_path))
            print("    Model saved successfully.")

        print(f"  Phase 6 completed in {time.time() - t0:.2f} s")

        # =====================================================================
        # Phase 7: Post-Processing, Field Extraction, and CSV Generation
        # =====================================================================
        t0 = time.time()
        print("\n[Phase 7] Post-Processing Field Extractions & Validation Artifacts...")

        per_case_metrics: dict[str, dict[str, Any]] = {}
        manifest_cases: list[dict[str, Any]] = []

        raw_sim_cases = cfg_defaults.get("raw_config", {}).get("simulation", {}).get("cases", [])
        use_case_subfolder = (
            is_multi_case
            or len(raw_sim_cases) > 1
            or bool(getattr(args, "case_id", None))
            or bool(getattr(args, "cases", None))
        )

        for idx, exc_case in enumerate(excitation_cases, start=1):
            solnum = idx if is_multi_case else None
            case_out_dir = (out_dir / "cases" / exc_case.case_id) if use_case_subfolder else out_dir
            case_m, manifest_c = _extract_case_artifacts(
                model=model,
                phys_tag=phys_tag,
                exc_case=exc_case,
                case_out_dir=case_out_dir,
                root_out_dir=out_dir,
                solnum=solnum,
                f0=f0,
                er_int=er_int,
                mur_int=mur_int,
                disable_dipole_scale=args.disable_dipole_scale,
                a=a,
                b=b,
                radial_points=radial_points,
                radial_min_radii=radial_min_radii,
                radial_max_radii=radial_max_radii,
                radial_theta_deg=radial_theta_deg,
                radial_phi_deg=radial_phi_deg,
                zx_nx=zx_nx,
                zx_nz=zx_nz,
                zx_span_radii=zx_span_radii,
                far_field_points=args.far_field_points,
            )
            per_case_metrics[exc_case.case_id] = case_m
            manifest_cases.append(manifest_c)

        primary_metrics = per_case_metrics[primary_case.case_id]

        # Save Metrics Summary JSON (preserving all top-level keys for single-case compatibility)
        metrics: dict[str, Any] = {
            "validation_case": {
                "formulation": "case_a_zero_thickness_sheet",
                "comsol_version": args.version,
                "physics_interface": phys_tag,
                "discretization": args.discretization,
                "frequency_hz": f0,
                "wavelength_m": lam0,
                "radius_m": a,
                "tau_m": 0.0,
                "far_field_offset_m": far_field_offset_m,
                "air_extent_m": b,
                "pml_thickness_m": t_pml,
                "er_interior": er_int,
                "mur_interior": mur_int,
                "er_exterior": er_ext,
                "mur_exterior": mur_ext,
                "case_count": len(excitation_cases),
                "dipole_type": dipole_type,
                "dipole_direction": dipole_dir,
                "dipole_normalization_scale": primary_metrics["dipole_normalization_scale"],
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
                "strategy": "sheet_freetri_conformal_tet",
                "global_hmax_divisor": args.global_mesh_divisor,
                "interior_hmax_divisor": interior_hmax_divisor,
                "sheet_tangential_divisor": args.sheet_tangential_divisor,
                "far_field_surface_divisor": far_field_surface_divisor,
                "source_ball_radius_m": source_ball_radius_m,
                "source_ball_hmax_divisor": source_ball_hmax_divisor,
                "inner_band_width_m": inner_band_width_m,
                "inner_band_hmax_divisor": inner_band_hmax_divisor,
                "outer_band_width_m": outer_band_width_m,
                "outer_band_hmax_divisor": outer_band_hmax_divisor,
                "total_elements": mesh_stats.total_elements,
                "prisms": mesh_stats.prisms,
                "tetrahedra": mesh_stats.tetrahedra,
                "sheet_triangles": mesh_stats.source_triangles,
            },
            "solver": {
                "linear_solver": args.linear_solver,
                "case_count": len(excitation_cases),
                "lu_factorization_reused": is_multi_case,
                "solve_duration_seconds": solve_duration,
                "cores": args.cores,
            },
            "far_field_metrics": primary_metrics["far_field_metrics"],
            "files": {
                "mph_model": mph_path.name if mph_path else None,
                "radial_csv": primary_metrics["files"]["radial_csv"],
                "zx_csv": primary_metrics["files"]["zx_csv"],
                "far_field_cuts_csv": primary_metrics["files"]["far_field_cuts_csv"],
            },
            "cases": per_case_metrics,
            "total_execution_seconds": time.time() - total_start_time,
        }
        metrics_path = out_dir / "comsol_metrics.json"
        if metrics_path.is_file():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    old_metrics = json.load(f)
                old_cases = old_metrics.get("cases", {})
                old_cases.update(per_case_metrics)
                metrics["cases"] = old_cases
            except Exception:
                pass
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"    Saved: {metrics_path.name}")

        # Export standard schema_version 2 run.json for plot_fields.py integration
        raw_cfg = cfg_defaults.get("raw_config", {})
        case_id = raw_cfg.get("case_id", "comsol_case_a_validation")

        run_manifest = {
            "schema_version": 2,
            "application": "anisotropic_gstc_sphere",
            "case_id": case_id,
            "backend": "comsol",
            "comsol_formulation": "case_a_zero_thickness_sheet",
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
                },
                "physical_construction": "interior for r<a; exterior for r>=a (zero-thickness sheet)",
            },
            "simulation": {
                "cases": manifest_cases,
            },
            "files": {
                "configuration": "configuration.json",
            },
        }
        run_manifest_path = out_dir / "run.json"
        if run_manifest_path.is_file():
            try:
                with run_manifest_path.open("r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
                old_cases = {
                    c["id"]: c
                    for c in old_manifest.get("simulation", {}).get("cases", [])
                    if isinstance(c, dict) and "id" in c
                }
                for mc in manifest_cases:
                    old_cases[mc["id"]] = mc
                manifest_cases = list(old_cases.values())
            except Exception:
                pass
        run_manifest["simulation"]["cases"] = manifest_cases
        with run_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)
        print(f"    Saved: {run_manifest_path.name}")

        if args.config and args.config.is_file():
            shutil.copyfile(args.config, out_dir / "configuration.json")
            print("    Saved: configuration.json (snapshot)")

        print(f"  Phase 7 completed in {time.time() - t0:.2f} s")

    finally:
        if server is not None and client is not None:
            print("\n[Shutdown] Stopping COMSOL Server...")
            stop_comsol(server, client)
        shutil.rmtree(runtime_dir, ignore_errors=True)

    total_time = time.time() - total_start_time
    print("=" * 80)
    print(f"COMSOL CASE A VALIDATION COMPLETE IN {total_time:.2f} s ({total_time / 60:.2f} min)")
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

