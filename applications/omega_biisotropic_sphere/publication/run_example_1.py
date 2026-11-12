#!/usr/bin/env python3
"""End-to-end reproduction runner for Example 1 (Budhu 10 GHz, eps_r=7).

Pipeline:
  1. Solve bare dielectric sphere baseline using PMCHWT MoM solver.
  2. Solve dielectric sphere with synthesized omega metasurface using SIE-GSTC MoM solver.
  3. Generate publication-ready figures (2D maps and 1D radial cut) in PDF and PNG.

Usage:
  python run_example_1.py [options]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PUBLICATION_DIR = Path(__file__).resolve().parent
APP_DIR = PUBLICATION_DIR.parent
REPO_ROOT = APP_DIR.parents[1]

# Executable defaults
PMCHWT_APP = REPO_ROOT / "build" / "applications" / "pmchwt_dielectric_sphere" / "pmchwt_dielectric_sphere_app"
OMEGA_APP = REPO_ROOT / "build" / "applications" / "omega_biisotropic_sphere" / "omega_sphere_sie"
PYTHON_EXE = sys.executable

# Case settings
MESH_PATH = APP_DIR / "meshes" / "budhu_10ghz_eps7" / "sphere.obj"
CONFIG_PATH = APP_DIR / "configurations" / "budhu_10ghz_eps7.json"
RESULTS_DIR = APP_DIR / "results" / "budhu_10ghz_eps7"
BARE_SPHERE_DIR = RESULTS_DIR / "pmchwt" / "bare_sphere"
FIGURES_DIR = PUBLICATION_DIR / "figures"


def run_command(command: list[str], description: str) -> None:
    """Execute a subprocess command with clear logging."""
    print(f"\n---> [RUNNING] {description}")
    print(f"     Command: {' '.join(str(c) for c in command)}")
    start_time = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}: {description}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"---> [COMPLETED] {description} in {elapsed:.1f}s")


def check_prerequisites(skip_simulations: bool = False) -> None:
    """Verify that required executables and configuration files exist."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    if not MESH_PATH.is_file():
        raise FileNotFoundError(f"Mesh file not found: {MESH_PATH}")

    if not skip_simulations:
        missing = []
        if not PMCHWT_APP.is_file():
            missing.append(f"PMCHWT executable ({PMCHWT_APP})")
        if not OMEGA_APP.is_file():
            missing.append(f"Omega SIE executable ({OMEGA_APP})")
        if missing:
            msg = (
                "Missing required C++ build artifacts:\n  - "
                + "\n  - ".join(missing)
                + "\n\nPlease build them first using:\n"
                "  cmake -B build -S .\n"
                "  cmake --build build --target pmchwt_dielectric_sphere_app omega_sphere_sie -j\n"
            )
            print(msg, file=sys.stderr)
            sys.exit(1)


def run_pmchwt_bare_sphere(no_residual: bool = False) -> None:
    """Run PMCHWT solver for the bare dielectric sphere (no metasurface)."""
    BARE_SPHERE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PMCHWT_APP),
        "--mesh", str(MESH_PATH.resolve()),
        "--output-dir", str(BARE_SPHERE_DIR.resolve()),
        "--frequency-hz", "10000000000.0",
        "--epsilon-r-interior", "7.0",
        "--mu-r-interior", "1.0",
        "--epsilon-r-exterior", "1.0",
        "--mu-r-exterior", "1.0",
        "--effective-moment-interior", "1.0",
        "--formulation", "t",
        "--quadrature-rule", "7",
        "--zx-grid-size", "200",
        "--zx-span-radii", "2.0",
        "--radial-points", "300",
        "--radial-theta-degrees", "45.0",
        "--radial-phi-degrees", "0.0",
        "--radial-min-radii", "0.05",
        "--radial-max-radii", "2.0",
    ]
    if no_residual:
        cmd.append("--no-residual")
    run_command(cmd, "PMCHWT Bare Dielectric Sphere MoM Simulation")


def run_sie_omega_sphere(no_residual: bool = False) -> None:
    """Run SIE-GSTC solver for the sphere with synthesized omega metasurface."""
    sie_script = APP_DIR / "sie_dipole_impedance_matching.py"
    cmd = [
        PYTHON_EXE,
        str(sie_script),
        "--config", str(CONFIG_PATH),
        "--use-existing-mesh",
    ]
    if no_residual:
        cmd.append("--no-residual")
    run_command(cmd, "SIE-GSTC Omega Metasurface MoM Simulation")


def run_publication_plots() -> None:
    """Generate Figure 1 (side-by-side right half x >= 0) and Figure 2 (radial cut)."""
    plot_script = PUBLICATION_DIR / "plot_example_1.py"
    bare_csv = BARE_SPHERE_DIR / "zx_fields.csv"

    # Primary publication layout: right half (x >= 0) side-by-side (single-column width 3.5 in)
    cmd = [
        PYTHON_EXE,
        str(plot_script),
        "--results-dir", str(RESULTS_DIR),
        "--output-dir", str(FIGURES_DIR),
        "--bare-csv", str(bare_csv),
        "--crop", "right",
    ]
    run_command(cmd, "Generate Primary Publication Figures (Figure 1 Side-by-Side & Figure 2 Radial Cut)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-pmchwt",
        action="store_true",
        help="Skip PMCHWT bare sphere simulation and reuse existing results",
    )
    parser.add_argument(
        "--skip-sie",
        action="store_true",
        help="Skip SIE-GSTC omega metasurface simulation and reuse existing results",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip both simulations and only re-generate publication figures",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Skip post-solve residual reassembly to speed up simulations",
    )
    args = parser.parse_args()

    skip_pmchwt = args.skip_pmchwt or args.plots_only
    skip_sie = args.skip_sie or args.plots_only

    print("==================================================================")
    print("Example 1 Pipeline: Impedance Matching of Dielectric Sphere")
    print("==================================================================")
    print(f"Repository Root:   {REPO_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Figures Directory: {FIGURES_DIR}")

    check_prerequisites(skip_simulations=skip_pmchwt and skip_sie)

    # 1. PMCHWT
    if not skip_pmchwt:
        run_pmchwt_bare_sphere(no_residual=args.no_residual)
    else:
        print("\n---> [SKIPPED] PMCHWT Bare Sphere Simulation (reusing existing data)")

    # 2. SIE-GSTC Omega
    if not skip_sie:
        run_sie_omega_sphere(no_residual=args.no_residual)
    else:
        print("\n---> [SKIPPED] SIE-GSTC Omega Metasurface Simulation (reusing existing data)")

    # 3. Plots
    run_publication_plots()

    print("\n==================================================================")
    print("Pipeline Execution Completed Successfully!")
    print(f"Publication figures saved in: {FIGURES_DIR}")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

