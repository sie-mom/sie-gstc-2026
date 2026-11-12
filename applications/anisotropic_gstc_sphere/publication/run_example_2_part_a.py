#!/usr/bin/env python3
"""End-to-end reproduction runner for Example 2 Part A (Electric to Magnetic).

Pipeline:
  1. Solve dielectric sphere with synthesized anisotropic metasurface using C++ SIE-GSTC solver.
  2. Compute independent SVWF collocation benchmark fields.
  3. Generate publication-ready figures (2D field maps and 1D radial cut) in PDF and PNG.
  4. Generate metasurface constitutive parameter postprocessing plots (Y, Z, chi, condition number).

Usage:
  .venv/bin/python applications/anisotropic_gstc_sphere/publication/run_example_2_part_a.py [options]

Options:
  --plots-only        Skip solver runs and regenerate figures from existing results.
  --force             Delete existing simulation results and re-run C++ SIE solve from scratch.
  --vmin-db           Colorbar minimum in normalized dB (default: -112.0).
  --vmax-db           Colorbar maximum in normalized dB (default: -58.0).
  --skip-parameters   Skip metasurface constitutive parameter postprocessing plots.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PUBLICATION_DIR = Path(__file__).resolve().parent
APP_DIR = PUBLICATION_DIR.parent
REPO_ROOT = APP_DIR.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = (
    APP_DIR
    / "configurations"
    / "dipole_pol_rot"
    / "e2m_eps7.json"
)
_RESULTS_CANDIDATE = (
    APP_DIR
    / "results"
    / "dipole_pol_rot"
    / "e2m_eps7"
)
RESULTS_DIR = (
    _RESULTS_CANDIDATE
    if _RESULTS_CANDIDATE.exists()
    else (
        APP_DIR
        / "results"
        / "dipole_pol_rot"
        / "electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
    )
)
EXECUTABLE_PATH = REPO_ROOT / "build" / "applications" / "anisotropic_gstc_sphere" / "anisotropic_sphere_app"
PYTHON_EXE = sys.executable


def run_command(command: list[str], description: str) -> None:
    """Execute a subprocess command with clean logging."""
    print(f"\n---> [RUNNING] {description}")
    print(f"     Command: {' '.join(str(c) for c in command)}")
    start_time = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}: {description}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"---> [COMPLETED] {description} in {elapsed:.1f}s")


def run_svwf_benchmark() -> None:
    """Generate independent SVWF collocation benchmark data."""
    svwf_csv = RESULTS_DIR / "cases" / "electric_incidence" / "fields" / "svwf_radial.csv"
    if svwf_csv.is_file():
        print(f"\n---> [SKIPPED] SVWF Benchmark already exists at: {svwf_csv}")
        return

    print("\n---> [RUNNING] Solving SVWF Collocation Benchmark")
    start = time.time()
    import numpy as np
    import pandas as pd
    from applications.anisotropic_gstc_sphere.workflow import load_workflow_configuration
    from sie_mom.gstc import extract_reciprocal_lossless_least_squares, form_boundary_quantities
    from sie_mom.excitation import evaluate_dipole_fields
    from spherical_expansion.solver import solve_collocation_general
    from spherical_expansion.svwf_sphere import compute_fields
    from spherical_expansion.coefficients import save_coefficients
    from sie_mom.postprocessing import load_field_csv

    cfg = load_workflow_configuration(CONFIG_PATH)
    case = cfg.case
    a = case.radius_m

    th = np.array([np.deg2rad(45.0)])
    ph = np.array([0.0])

    E_int, H_int = evaluate_dipole_fields(case.k_interior, a, th, ph, case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type="electric")
    E_ext, H_ext = evaluate_dipole_fields(case.k_exterior, a, th, ph, case.eta_exterior, effective_moment=(0.0, 0.0, 11.386035931884505), dipole_type="magnetic")

    bq = form_boundary_quantities(E_int[1:].conj(), H_int[1:].conj(), E_ext[1:].conj(), H_ext[1:].conj())
    res_ext = extract_reciprocal_lossless_least_squares([bq], reference_impedance=case.eta_exterior)
    params_svwf = res_ext.parameters

    tensors = {
        "Y": params_svwf.Y_ee[:, :, 0],
        "Chi": params_svwf.Y_em[:, :, 0],
        "Gamma": params_svwf.Y_me[:, :, 0],
        "Z": params_svwf.Y_mm[:, :, 0],
    }

    def incident_field(theta, phi):
        E_inc, H_inc = evaluate_dipole_fields(case.k_interior, a, theta, phi, case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type="electric")
        return E_inc[1].conj(), E_inc[2].conj(), H_inc[1].conj(), H_inc[2].conj()

    res = solve_collocation_general(
        n_max=8, k0=case.k_exterior, eta0=case.eta_exterior,
        k1=case.k_interior, eta1=case.eta_interior, radius=a,
        gstc_params=tensors, n_theta=36, n_phi=72, return_metrics=True,
        incident_field_func=incident_field, is_interior_source=True,
    )
    ext_m, ext_n, int_m, int_n, _ = res

    case_dir = RESULTS_DIR / "cases" / "electric_incidence"
    fields_dir = case_dir / "fields"
    save_coefficients(case_dir / "svwf_coefficients.npz", ext_m, ext_n, int_m, int_n)

    data_rad = load_field_csv(fields_dir / "sie_physical_radial.csv")
    x_r, y_r, z_r = data_rad["x"], data_rad["y"], data_rad["z"]
    r_pts = np.sqrt(x_r**2 + y_r**2 + z_r**2)
    th_r = np.arccos(np.clip(z_r / r_pts, -1.0, 1.0))
    ph_r = np.arctan2(y_r, x_r)
    r_norm = r_pts / a

    int_mask = r_norm < 1.0
    ext_mask = r_norm >= 1.0

    E_svwf_sph = np.zeros((3, len(r_pts)), dtype=complex)
    H_svwf_sph = np.zeros((3, len(r_pts)), dtype=complex)

    E_ext_svwf, H_ext_svwf = compute_fields(case.k_exterior, r_pts[ext_mask], th_r[ext_mask], ph_r[ext_mask], ext_m, ext_n, kind=3, eta=case.eta_exterior, grid=False)
    E_svwf_sph[:, ext_mask] = np.conjugate(np.asarray(E_ext_svwf))
    H_svwf_sph[:, ext_mask] = np.conjugate(np.asarray(H_ext_svwf))

    E_inc_int, H_inc_int = evaluate_dipole_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], case.eta_interior, effective_moment=(0.0, 0.0, 1.0), dipole_type="electric")
    E_refl_int, H_refl_int = compute_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], int_m, int_n, kind=1, eta=case.eta_interior, grid=False)
    E_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(E_inc_int) + np.asarray(E_refl_int))
    H_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(H_inc_int) + np.asarray(H_refl_int))

    def sph_to_cart(v_sph, th, ph):
        vr, vth, vph = v_sph[0], v_sph[1], v_sph[2]
        sin_th, cos_th = np.sin(th), np.cos(th)
        sin_ph, cos_ph = np.sin(ph), np.cos(ph)
        vx = vr * sin_th * cos_ph + vth * cos_th * cos_ph - vph * sin_ph
        vy = vr * sin_th * sin_ph + vth * cos_th * sin_ph + vph * cos_ph
        vz = vr * cos_th - vth * sin_th
        return vx, vy, vz

    Ex, Ey, Ez = sph_to_cart(E_svwf_sph, th_r, ph_r)
    Hx, Hy, Hz = sph_to_cart(H_svwf_sph, th_r, ph_r)

    df_out = pd.DataFrame({
        "point_index": np.arange(len(r_pts)),
        "x": x_r, "y": y_r, "z": z_r,
        "region": data_rad["region"].astype(int),
        "Ex_real": Ex.real, "Ex_imag": Ex.imag,
        "Ey_real": Ey.real, "Ey_imag": Ey.imag,
        "Ez_real": Ez.real, "Ez_imag": Ez.imag,
        "Hx_real": Hx.real, "Hx_imag": Hx.imag,
        "Hy_real": Hy.real, "Hy_imag": Hy.imag,
        "Hz_real": Hz.real, "Hz_imag": Hz.imag,
    })
    df_out.to_csv(fields_dir / "svwf_radial.csv", index=False)
    print(f"---> [COMPLETED] SVWF Benchmark solved and saved in {time.time() - start:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip C++ SIE solve and regenerate plots from existing data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing results and re-run C++ SIE solve from scratch",
    )
    parser.add_argument(
        "--vmin-db",
        type=float,
        default=-112.0,
        help="Colorbar minimum in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--vmax-db",
        type=float,
        default=-58.0,
        help="Colorbar maximum in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-parameters",
        action="store_true",
        help="Skip metasurface constitutive parameter postprocessing plots",
    )
    args = parser.parse_args()

    # Resolve configured mesh path
    import json
    with open(CONFIG_PATH) as f:
        cfg_raw = json.load(f)
    mesh_rel = cfg_raw.get("sie", {}).get("mesh", {}).get("path", "")
    mesh_full = Path(mesh_rel) if Path(mesh_rel).is_absolute() else APP_DIR / mesh_rel

    print("==================================================================")
    print(" Full End-to-End Runner: Example 2 Part A (Electric to Magnetic)")
    print(f" Configuration: {CONFIG_PATH}")
    print(f" Mesh:          {mesh_full}")
    print(f" Results Dir:   {RESULTS_DIR}")
    print("==================================================================")

    # 1. Step 1: Solve C++ Spherical GSTC SIE
    has_results = (RESULTS_DIR / "batch_metrics.json").is_file()
    if args.plots_only:
        print("\n---> [SKIPPED] Step 1: C++ SIE solve (--plots-only requested)")
    elif has_results and not args.force:
        print("\n---> [SKIPPED] Step 1: C++ SIE solve (results already exist; pass --force to re-solve)")
    else:
        if args.force and RESULTS_DIR.is_dir():
            print(f"\n---> [CLEAN] Removing existing results directory: {RESULTS_DIR}")
            shutil.rmtree(RESULTS_DIR)

        cmd = [
            PYTHON_EXE,
            str(APP_DIR / "sie_gstc_sphere.py"),
            "--config", str(CONFIG_PATH),
            "--run-dir", str(RESULTS_DIR),
            "--executable", str(EXECUTABLE_PATH),
        ]
        if mesh_full.is_file():
            cmd.append("--use-existing-mesh")
        run_command(cmd, "Solve Spherical Anisotropic GSTC SIE (C++)")

    # 2. Step 2: Ensure independent SVWF benchmark is solved
    run_svwf_benchmark()

    # 3. Step 3: Generate publication figures
    cmd_plot = [
        PYTHON_EXE,
        str(PUBLICATION_DIR / "plot_example_2_part_a.py"),
        "--config", str(CONFIG_PATH),
        "--results-dir", str(RESULTS_DIR),
        "--output-dir", str(PUBLICATION_DIR / "figures"),
        "--vmin-db", str(args.vmin_db),
        "--vmax-db", str(args.vmax_db),
    ]
    run_command(cmd_plot, "Generate Publication Figures (Figure 1 and Figure 2)")

    # 4. Step 4: Generate metasurface parameter plots (Y, Z, chi, condition number)
    if not args.skip_parameters:
        param_output_dir = PUBLICATION_DIR / "figures" / "surface_parameters"
        param_output_dir.mkdir(parents=True, exist_ok=True)
        tensors_bin = RESULTS_DIR / "face_tensors.bin"
        if tensors_bin.is_file() and mesh_full.is_file():
            cmd_params = [
                PYTHON_EXE,
                str(APP_DIR / "plot_surface_parameters.py"),
                "--tensors", str(tensors_bin),
                "--mesh", str(mesh_full),
                "--output-dir", str(param_output_dir),
                "--title-suffix", "Part A: Electric to Magnetic (10 GHz, eps=7)",
                "--csv",
            ]
            run_command(cmd_params, "Plot Metasurface Constitutive Parameters (Y, Z, chi, cond)")
        else:
            print(f"\n---> [WARNING] Step 4 skipped: missing {tensors_bin} or {mesh_full}")

    print("\n==================================================================")
    print(" End-to-End Pipeline Finished Successfully!")
    print(f" Figures available in: {PUBLICATION_DIR / 'figures'}")
    if not args.skip_parameters:
        print(f" Surface parameters in: {PUBLICATION_DIR / 'figures' / 'surface_parameters'}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

