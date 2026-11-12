#!/usr/bin/env python3
"""End-to-end reproduction runner for Example 2 Part B (Dual Polarization Rotator).

Pipeline:
  1. Solve dielectric sphere with synthesized dual-polarization anisotropic metasurface
     using the C++ SIE-GSTC solver (batch solve of both electric_incidence and magnetic_incidence).
  2. Compute independent SVWF collocation benchmark fields for both transformations.
  3. Generate publication-ready figures (2D field maps and 1D radial cuts) in PDF and PNG.
  4. Generate metasurface constitutive parameter postprocessing plots (Y, Z, chi, condition number).

Usage:
  .venv/bin/python applications/anisotropic_gstc_sphere/publication/run_example_2_part_b.py [options]

Options:
  --plots-only        Skip solver runs and regenerate figures from existing results.
  --force             Delete existing simulation results and re-run C++ SIE solve from scratch.
  --vmin-db           Colorbar minimum in normalized dB (default: -112.0).
  --vmax-db           Colorbar maximum in normalized dB (default: -58.0).
  --skip-parameters   Skip metasurface constitutive parameter postprocessing plots.
"""

from __future__ import annotations

import argparse
import json
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
    / "dual_pol_eps7.json"
)
_RESULTS_CANDIDATE = (
    APP_DIR
    / "results"
    / "dipole_pol_rot"
    / "dual_pol_eps7"
)
RESULTS_DIR = (
    _RESULTS_CANDIDATE
    if _RESULTS_CANDIDATE.exists()
    else (
        APP_DIR
        / "results"
        / "dipole_pol_rot"
        / "dual_pol_rot_10ghz_eps7_reciprocal_lossless"
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


def run_svwf_benchmark(force: bool = False) -> None:
    """Generate independent SVWF collocation benchmark data for both transformations."""
    csv_case1 = RESULTS_DIR / "cases" / "electric_incidence" / "fields" / "svwf_radial.csv"
    csv_case2 = RESULTS_DIR / "cases" / "magnetic_incidence" / "fields" / "svwf_radial.csv"
    if csv_case1.is_file() and csv_case2.is_file() and not force:
        print("\n---> [SKIPPED] SVWF Benchmark already exists for both cases.")
        return

    print("\n---> [RUNNING] Solving SVWF Collocation Benchmark for Dual Transformations")
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

    # Form boundary quantities for both transformations
    ref_el_int = cfg.fields["electric_z_interior"]
    ref_mg_ext = cfg.fields["magnetic_z_exterior"]
    m_el_int = tuple(x.real for x in ref_el_int.effective_moment)
    m_mg_ext = tuple(x.real for x in ref_mg_ext.effective_moment)
    E_int_1, H_int_1 = evaluate_dipole_fields(case.k_interior, a, th, ph, case.eta_interior, effective_moment=m_el_int, dipole_type="electric")
    E_ext_1, H_ext_1 = evaluate_dipole_fields(case.k_exterior, a, th, ph, case.eta_exterior, effective_moment=m_mg_ext, dipole_type="magnetic")
    bq1 = form_boundary_quantities(E_int_1[1:].conj(), H_int_1[1:].conj(), E_ext_1[1:].conj(), H_ext_1[1:].conj())

    ref_mg_int = cfg.fields["magnetic_z_interior"]
    ref_el_ext = cfg.fields["electric_z_exterior"]
    m_mg_int = tuple(x.real for x in ref_mg_int.effective_moment)
    m_el_ext = tuple(x.real for x in ref_el_ext.effective_moment)
    E_int_2, H_int_2 = evaluate_dipole_fields(case.k_interior, a, th, ph, case.eta_interior, effective_moment=m_mg_int, dipole_type="magnetic")
    E_ext_2, H_ext_2 = evaluate_dipole_fields(case.k_exterior, a, th, ph, case.eta_exterior, effective_moment=m_el_ext, dipole_type="electric")
    bq2 = form_boundary_quantities(E_int_2[1:].conj(), H_int_2[1:].conj(), E_ext_2[1:].conj(), H_ext_2[1:].conj())

    # Extract single reciprocal lossless metasurface satisfying both simultaneously
    res_ext = extract_reciprocal_lossless_least_squares([bq1, bq2], reference_impedance=case.eta_exterior)
    params_svwf = res_ext.parameters

    tensors = {
        "Y": params_svwf.Y_ee[:, :, 0],
        "Chi": params_svwf.Y_em[:, :, 0],
        "Gamma": params_svwf.Y_me[:, :, 0],
        "Z": params_svwf.Y_mm[:, :, 0],
    }

    def sph_to_cart(v_sph, th_pts, ph_pts):
        vr, vth, vph = v_sph[0], v_sph[1], v_sph[2]
        sin_th, cos_th = np.sin(th_pts), np.cos(th_pts)
        sin_ph, cos_ph = np.sin(ph_pts), np.cos(ph_pts)
        vx = vr * sin_th * cos_ph + vth * cos_th * cos_ph - vph * sin_ph
        vy = vr * sin_th * sin_ph + vth * cos_th * sin_ph + vph * cos_ph
        vz = vr * cos_th - vth * sin_th
        return vx, vy, vz

    def solve_and_save_radial(case_id: str, dipole_type: str, dipole_moment: tuple[float, float, float]):
        case_dir = RESULTS_DIR / "cases" / case_id
        fields_dir = case_dir / "fields"
        fields_dir.mkdir(parents=True, exist_ok=True)

        def inc_func(theta, phi):
            E_inc, H_inc = evaluate_dipole_fields(case.k_interior, a, theta, phi, case.eta_interior, effective_moment=dipole_moment, dipole_type=dipole_type)
            return E_inc[1].conj(), E_inc[2].conj(), H_inc[1].conj(), H_inc[2].conj()

        res = solve_collocation_general(
            n_max=8, k0=case.k_exterior, eta0=case.eta_exterior,
            k1=case.k_interior, eta1=case.eta_interior, radius=a,
            gstc_params=tensors, n_theta=36, n_phi=72, return_metrics=True,
            incident_field_func=inc_func, is_interior_source=True,
        )
        ext_m, ext_n, int_m, int_n, _ = res
        save_coefficients(case_dir / "svwf_coefficients.npz", ext_m, ext_n, int_m, int_n)

        sie_csv = fields_dir / "sie_physical_radial.csv"
        if not sie_csv.is_file():
            print(f"[WARNING] {sie_csv} not found; cannot compute radial SVWF curve.")
            return

        data_rad = load_field_csv(sie_csv)
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

        E_inc_int, H_inc_int = evaluate_dipole_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], case.eta_interior, effective_moment=dipole_moment, dipole_type=dipole_type)
        E_refl_int, H_refl_int = compute_fields(case.k_interior, r_pts[int_mask], th_r[int_mask], ph_r[int_mask], int_m, int_n, kind=1, eta=case.eta_interior, grid=False)
        E_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(E_inc_int) + np.asarray(E_refl_int))
        H_svwf_sph[:, int_mask] = np.conjugate(np.conjugate(H_inc_int) + np.asarray(H_refl_int))

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
        out_csv = fields_dir / "svwf_radial.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"     Saved SVWF radial data: {out_csv}")

    solve_and_save_radial("electric_incidence", "electric", m_el_int)
    solve_and_save_radial("magnetic_incidence", "magnetic", m_mg_int)
    print(f"---> [COMPLETED] SVWF Benchmarks solved in {time.time() - start:.1f}s")


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
        help="Colorbar minimum for electric incidence in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--vmax-db",
        type=float,
        default=-58.0,
        help="Colorbar maximum for electric incidence in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--mag-vmin-db",
        type=float,
        default=-75.0,
        help="Colorbar minimum for magnetic incidence in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--mag-vmax-db",
        type=float,
        default=-45.0,
        help="Colorbar maximum for magnetic incidence in normalized dB (default: %(default)s)",
    )
    parser.add_argument(
        "--norm",
        choices=["origin", "radiated"],
        default="origin",
        help="Normalization reference: 'origin' (dipole core peak) or 'radiated' (exterior radiated peak = 0 dB) (default: %(default)s)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for 2D field maps (default: %(default)s, e.g. viridis, turbo, inferno, plasma)",
    )
    parser.add_argument(
        "--skip-parameters",
        action="store_true",
        help="Skip metasurface constitutive parameter postprocessing plots",
    )
    args = parser.parse_args()

    # Resolve configured mesh path
    with open(CONFIG_PATH) as f:
        cfg_raw = json.load(f)
    mesh_rel = cfg_raw.get("sie", {}).get("mesh", {}).get("path", "")
    mesh_full = Path(mesh_rel) if Path(mesh_rel).is_absolute() else APP_DIR / mesh_rel

    print("==================================================================")
    print(" Full End-to-End Runner: Example 2 Part B (Dual Polarization Rotator)")
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
        run_command(cmd, "Solve Spherical Anisotropic GSTC SIE Batch (C++)")

    # 2. Step 2: Ensure independent SVWF benchmarks are solved
    run_svwf_benchmark(force=args.force)

    # 3. Step 3: Generate publication figures
    cmd_plot = [
        PYTHON_EXE,
        str(PUBLICATION_DIR / "plot_example_2_part_b.py"),
        "--config", str(CONFIG_PATH),
        "--results-dir", str(RESULTS_DIR),
        "--output-dir", str(PUBLICATION_DIR / "figures"),
        "--vmin-db", str(args.vmin_db),
        "--vmax-db", str(args.vmax_db),
        "--mag-vmin-db", str(args.mag_vmin_db),
        "--mag-vmax-db", str(args.mag_vmax_db),
        "--norm", str(args.norm),
        "--cmap", str(args.cmap),
    ]
    run_command(cmd_plot, "Generate Publication Figures (Figure 1 and Figure 2)")

    # 4. Step 4: Generate metasurface parameter plots (Y, Z, chi, condition number)
    if not args.skip_parameters:
        param_output_dir = PUBLICATION_DIR / "figures" / "surface_parameters_part_b"
        param_output_dir.mkdir(parents=True, exist_ok=True)
        tensors_bin = RESULTS_DIR / "face_tensors.bin"
        if tensors_bin.is_file() and mesh_full.is_file():
            cmd_params = [
                PYTHON_EXE,
                str(APP_DIR / "plot_surface_parameters.py"),
                "--tensors", str(tensors_bin),
                "--mesh", str(mesh_full),
                "--output-dir", str(param_output_dir),
                "--title-suffix", "Part B: Dual Pol. Rotator (10 GHz, eps=7)",
                "--csv",
            ]
            run_command(cmd_params, "Plot Metasurface Constitutive Parameters (Y, Z, chi, cond)")
        else:
            print(f"\n---> [WARNING] Step 4 skipped: missing {tensors_bin} or {mesh_full}")

    print("\n==================================================================")
    print(" End-to-End Pipeline Finished Successfully!")
    print(f" Figures available in: {PUBLICATION_DIR / 'figures'}")
    if not args.skip_parameters:
        print(f" Surface parameters in: {PUBLICATION_DIR / 'figures' / 'surface_parameters_part_b'}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
