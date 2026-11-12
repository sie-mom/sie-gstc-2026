#!/usr/bin/env python3
"""End-to-end reproduction runner for Example 3 (Circular-Arc Spindle GSTC Surface).

Executes the full pipeline from mesh generation and tensor synthesis to CUDA SIE solve,
observation sampling, field evaluation, optional NGSolve FEM cross-validation,
and publication-ready figure generation.

Pipeline Stages:
  Stage 1: Generate tip-graded mesh and synthesize 10-parameter physical GSTC tensors.
  Stage 2: Single-LU batch CUDA solve of dual-polarization states (z- and y-dipoles).
  Stage 3: Generate and classify 2D zx-plane grid and 1D radial line-cut observations.
  Stage 4: Integrate surface currents to evaluate near fields.
  Stage 4B (Optional): Independent FEM cross-validation via NGSolve (Order 2).
  Stage 5: Generate publication 2D field maps and 1D radial comparison cuts.
  Stage 6: Generate publication 3D surface parameter maps (Im(Y), Im(Z), Re(chi), fit residual).

Usage:
  python applications/gstc_surface/publication/run_example_3.py [options]

Key Options:
  --plots-only        Skip simulation stages 1-4 and generate figures from existing data.
  --force             Ignore cached stage results and re-run stages from scratch.
  --with-ngsolve      Run independent NGSolve FEM validation stage (Order 2).
  --ngsolve-python    Explicit path to NGSolve Python interpreter.
  --threads           Number of CPU threads for observations and NGSolve (default: auto).
  --config            Configuration JSON path (default: configs/ogive/e50_ppw5_tip025.json).
  --run-dir           Working results directory (default: results/ogive/e50_ppw5_tip025/run01).
  --figures-dir       Output directory for publication figures (default: publication/figures).
  --vmin-db           Colorbar lower limit for 2D field maps in dB (default: -30.0).
  --vmax-db           Colorbar upper limit for 2D field maps in dB (default: 10.0).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PUBLICATION_DIR = Path(__file__).resolve().parent
APP_DIR = PUBLICATION_DIR.parent
REPO_ROOT = APP_DIR.parents[1]

for path_entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from applications.gstc_surface.workflow import load_workflow_configuration

DEFAULT_CONFIG = APP_DIR / "configs" / "ogive" / "e50_ppw5_tip025.json"
DEFAULT_RUN_DIR = APP_DIR / "results" / "ogive" / "e50_ppw5_tip025" / "run01"
DEFAULT_FIGURES_DIR = PUBLICATION_DIR / "figures"

# Default NGSolve interpreter: prioritize root .venv, fall back to experiments venv or current Python
_NG_BIN = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
_EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
_ROOT_VENV_PYTHON = REPO_ROOT / ".venv" / _NG_BIN
_LEGACY_NG_PYTHON = (
    REPO_ROOT
    / "applications"
    / "anisotropic_gstc_sphere"
    / "experiments"
    / "ngsolve"
    / ".venv"
    / _NG_BIN
)
DEFAULT_NGSOLVE_PYTHON = (
    _ROOT_VENV_PYTHON
    if _ROOT_VENV_PYTHON.is_file()
    else (_LEGACY_NG_PYTHON if _LEGACY_NG_PYTHON.is_file() else Path(sys.executable))
)

PMCHWT_APP = (
    REPO_ROOT
    / "build"
    / "applications"
    / "pmchwt_dielectric_sphere"
    / f"pmchwt_dielectric_sphere_app{_EXE_SUFFIX}"
)

EVAL_FIELDS_APP = (
    REPO_ROOT
    / "build"
    / "applications"
    / "gstc_surface"
    / f"evaluate_surface_fields{_EXE_SUFFIX}"
)


def _log_header(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n {title}\n{sep}")


def _run_step(cmd: list[str], description: str, *, env: dict[str, str] | None = None) -> None:
    print(f"\n---> [RUNNING] {description}")
    print(f"     Command: {' '.join(str(c) for c in cmd)}")
    start_time = time.time()
    call_env = os.environ.copy()
    if env:
        call_env.update(env)
    # Ensure REPO_ROOT and src are on PYTHONPATH
    existing_pythonpath = call_env.get("PYTHONPATH", "")
    src_path = str(REPO_ROOT / "src")
    repo_path = str(REPO_ROOT)
    call_env["PYTHONPATH"] = (
        f"{repo_path}{os.pathsep}{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else f"{repo_path}{os.pathsep}{src_path}"
    )

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=call_env)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}: {description}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"---> [COMPLETED] {description} in {elapsed:.1f}s")


def run_pmchwt_baseline(
    mesh_path: Path,
    obs_dir: Path,
    pmchwt_dir: Path,
    *,
    force: bool = False,
    no_residual: bool = False,
) -> None:
    """Run PMCHWT solver for the bare dielectric spindle baseline (no metasurface)."""
    if not PMCHWT_APP.is_file():
        raise FileNotFoundError(
            f"PMCHWT executable not found at {PMCHWT_APP}. Build it with:\n"
            "  cmake --build build --target pmchwt_dielectric_sphere_app -j"
        )
    if not EVAL_FIELDS_APP.is_file():
        raise FileNotFoundError(
            f"Field evaluator not found at {EVAL_FIELDS_APP}. Build it with:\n"
            "  cmake --build build --target evaluate_surface_fields -j"
        )

    obs_zx = obs_dir / "observations_zx.csv"
    obs_rad = obs_dir / "observations_radial.csv"
    if not obs_zx.is_file():
        raise FileNotFoundError(f"observations_zx.csv missing at {obs_zx}")

    if force and pmchwt_dir.exists():
        shutil.rmtree(pmchwt_dir)
    pmchwt_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("case1_electric_z", ["0.0", "0.0", "1.0"]),
        ("case2_electric_y", ["0.0", "1.0", "0.0"]),
    ]

    for case_folder, moment in cases:
        case_dir = pmchwt_dir / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)
        zx_out = case_dir / "zx_fields.csv"
        rad_out = case_dir / "radial_fields.csv"
        cur_out = case_dir / "currents.bin"

        if zx_out.is_file() and rad_out.is_file() and not force:
            print(f"---> [SKIPPED] PMCHWT baseline already computed at {case_dir}")
            continue

        for existing_file in (zx_out, rad_out, cur_out):
            if existing_file.is_file():
                existing_file.unlink()

        cmd = [
            str(PMCHWT_APP),
            "--mesh",
            str(mesh_path),
            "--output-dir",
            str(case_dir),
            "--output-fields",
            str(zx_out),
            "--output-currents",
            str(cur_out),
            "--observations",
            str(obs_zx),
            "--frequency-hz",
            "2000000000.0",
            "--epsilon-r-interior",
            "5.0",
            "--mu-r-interior",
            "1.0",
            "--epsilon-r-exterior",
            "1.0",
            "--mu-r-exterior",
            "1.0",
            "--dipole",
            "interior",
            "electric",
            *moment,
            "0.0",
            "0.0",
            "0.0",
            "--quadrature-rule",
            "7",
            "--singularity-threshold-lambda0",
            "0.1",
        ]
        if no_residual:
            cmd.append("--no-residual")
        _run_step(cmd, f"PMCHWT Bare Spindle Solve & 2D Grid ({case_folder})")

        if obs_rad.is_file():
            cmd_rad = [
                str(EVAL_FIELDS_APP),
                "--mesh",
                str(mesh_path),
                "--currents",
                str(cur_out),
                "--observations",
                str(obs_rad),
                "--output",
                str(rad_out),
                "--frequency-hz",
                "2000000000.0",
                "--epsilon-r-interior",
                "5.0",
                "--mu-r-interior",
                "1.0",
                "--epsilon-r-exterior",
                "1.0",
                "--mu-r-exterior",
                "1.0",
                "--dipole",
                "interior",
                "electric",
                *moment,
                "0.0",
                "0.0",
                "0.0",
                "--quadrature-rule",
                "7",
            ]
            _run_step(cmd_rad, f"PMCHWT 1D Radial Cut Field Evaluation ({case_folder})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Example 3 (Circular-Arc Spindle GSTC Metasurface) and generate publication figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the case configuration JSON.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Root directory for simulation inputs, solves, and fields.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Destination directory for output publication figures (PDF & PNG).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip simulation stages 1-4 and generate publication figures from existing data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running simulation stages even if output files already exist.",
    )
    parser.add_argument(
        "--with-ngsolve",
        action="store_true",
        help="Run independent NGSolve FEM cross-validation (Order 2).",
    )
    parser.add_argument(
        "--ngsolve-python",
        type=Path,
        default=DEFAULT_NGSOLVE_PYTHON,
        help="Path to the NGSolve Python virtual environment interpreter.",
    )
    parser.add_argument(
        "--use-existing-mesh",
        action="store_true",
        default=True,
        help="Reuse existing surface mesh if present instead of regenerating via Gmsh (default: True).",
    )
    parser.add_argument(
        "--regenerate-mesh",
        action="store_true",
        help="Force regeneration of the surface mesh via Gmsh, overwriting any existing OBJ file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="Number of CPU threads to use for multi-threaded stages.",
    )
    parser.add_argument(
        "--skip-pmchwt",
        action="store_true",
        help="Skip bare dielectric PMCHWT baseline simulation.",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Skip residual reassembly during PMCHWT solve to save GPU memory and time.",
    )
    parser.add_argument(
        "--vmin-db",
        type=float,
        default=-30.0,
        help="Colorbar lower bound in normalized dB for 2D field maps.",
    )
    parser.add_argument(
        "--vmax-db",
        type=float,
        default=10.0,
        help="Colorbar upper bound in normalized dB for 2D field maps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_exe = sys.executable

    config_path = args.config.resolve()
    run_dir = args.run_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = run_dir / "inputs"
    sie_dir = run_dir / "sie"
    obs_dir = sie_dir / "observations" / "run01"
    fields_dir = sie_dir / "fields" / "run01"
    pmchwt_dir = run_dir / "pmchwt"
    ng_dir = run_dir / "ngsolve" / "order2_run01"
    mesh_path = sie_dir / "surface.obj"

    _log_header("Publication Workflow: Example 3 (Circular-Arc Spindle GSTC Metasurface)")
    print(f"Configuration:   {config_path}")
    print(f"Run Directory:   {run_dir}")
    print(f"PMCHWT Baseline: {pmchwt_dir}")
    print(f"Figures Output:  {figures_dir}")
    print(f"Plots-Only Mode: {args.plots_only}")
    print(f"Force Re-run:    {args.force}")
    print(f"Skip PMCHWT:     {args.skip_pmchwt}")
    print(f"With NGSolve:    {args.with_ngsolve}")
    print(f"CPU Threads:     {args.threads}")

    # =========================================================================
    # Stage 1: Inputs & Mesh Preparation
    # =========================================================================
    stage1_done = (
        (inputs_dir / "face_tensors.bin").is_file()
        and (inputs_dir / "sie_inputs.json").is_file()
        and (inputs_dir / "configuration.json").is_file()
    )
    if args.plots_only:
        print("\n---> [SKIPPED] Stage 1: Inputs & Mesh (--plots-only requested)")
    elif stage1_done and not args.force:
        print(f"\n---> [SKIPPED] Stage 1: Inputs already prepared at {inputs_dir}")
    else:
        if args.force and inputs_dir.exists():
            shutil.rmtree(inputs_dir)
        wf_cfg = load_workflow_configuration(config_path)
        configured_mesh = wf_cfg.mesh_path

        cmd_stage1 = [
            python_exe,
            "-m",
            "applications.gstc_surface.prepare_sie_inputs",
            "--config",
            str(config_path),
            "--run-dir",
            str(inputs_dir),
        ]
        if args.regenerate_mesh:
            if configured_mesh.is_file():
                print(f"     [REGENERATE] Removing existing mesh at {configured_mesh}")
                configured_mesh.unlink()
        elif args.use_existing_mesh and configured_mesh.is_file():
            cmd_stage1.append("--use-existing-mesh")
        _run_step(cmd_stage1, "Stage 1: Generate Mesh and Synthesize Face Tensors")

    # =========================================================================
    # Stage 2: Single-LU Batch CUDA SIE Solve
    # =========================================================================
    stage2_done = (sie_dir / "batch_metrics.json").is_file()
    if args.plots_only:
        print("\n---> [SKIPPED] Stage 2: CUDA SIE Solve (--plots-only requested)")
    elif stage2_done and not args.force:
        print(f"\n---> [SKIPPED] Stage 2: SIE solve already completed at {sie_dir}")
    else:
        if args.force and sie_dir.exists():
            shutil.rmtree(sie_dir)
        cmd_stage2 = [
            python_exe,
            "-m",
            "applications.gstc_surface.sie_gstc_surface",
            "--inputs-dir",
            str(inputs_dir),
            "--run-dir",
            str(sie_dir),
        ]
        _run_step(cmd_stage2, "Stage 2: Solve Both Transformations via Batch GPU LU")

    # =========================================================================
    # Stage 3: Observation Points Generation & Classification
    # =========================================================================
    stage3_done = (obs_dir / "observations.json").is_file()
    if args.plots_only:
        print("\n---> [SKIPPED] Stage 3: Observation Grid (--plots-only requested)")
    elif stage3_done and not args.force:
        print(f"\n---> [SKIPPED] Stage 3: Observations already exist at {obs_dir}")
    else:
        if args.force and obs_dir.exists():
            shutil.rmtree(obs_dir)
        cmd_stage3 = [
            python_exe,
            "-m",
            "applications.gstc_surface.observations",
            "--solve-dir",
            str(sie_dir),
            "--output-dir",
            str(obs_dir),
        ]
        _run_step(cmd_stage3, "Stage 3: Generate and Classify Observation Grids")

    # =========================================================================
    # Stage 4: Surface Current Field Evaluation
    # =========================================================================
    stage4_done = (fields_dir / "fields.json").is_file()
    if args.plots_only:
        print("\n---> [SKIPPED] Stage 4: Field Evaluation (--plots-only requested)")
    elif stage4_done and not args.force:
        print(f"\n---> [SKIPPED] Stage 4: Fields already evaluated at {fields_dir}")
    else:
        if args.force and fields_dir.exists():
            shutil.rmtree(fields_dir)
        cmd_stage4 = [
            python_exe,
            "-m",
            "applications.gstc_surface.evaluate_fields",
            "--solve-dir",
            str(sie_dir),
            "--observations-dir",
            str(obs_dir),
            "--output-dir",
            str(fields_dir),
        ]
        _run_step(cmd_stage4, "Stage 4: Integrate Surface Currents to Evaluate Near Fields")

    # =========================================================================
    # Stage 4B (Optional): Independent NGSolve FEM Validation
    # =========================================================================
    if args.with_ngsolve:
        stage4b_done = (ng_dir / "fields" / "run01" / "fields.json").is_file()
        ng_python = args.ngsolve_python.absolute()
        if not ng_python.is_file():
            print(
                f"\n[WARNING] NGSolve Python interpreter not found at {ng_python}."
                "\n          Skipping Stage 4B. Specify path via --ngsolve-python.",
                file=sys.stderr,
            )
        elif stage4b_done and not args.force:
            print(f"\n---> [SKIPPED] Stage 4B: NGSolve validation already completed at {ng_dir}")
        else:
            if args.force and ng_dir.exists():
                shutil.rmtree(ng_dir)
            ng_env = {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
            }
            cmd_stage4b = [
                str(ng_python),
                "-u",
                "-m",
                "applications.gstc_surface.validate_ngsolve",
                "--inputs-dir",
                str(inputs_dir),
                "--observations-dir",
                str(obs_dir),
                "--sie-fields-dir",
                str(fields_dir),
                "--output-dir",
                str(ng_dir),
                "--order",
                "2",
                "--ppw",
                "4",
                "--air-lambda",
                "1.8",
                "--pml-lambda",
                "0.25",
                "--pml-strength",
                "3",
                "--threads",
                str(args.threads),
                "--inverse",
                "auto",
            ]
            _run_step(cmd_stage4b, "Stage 4B: Independent NGSolve FEM Validation (Order 2)", env=ng_env)

    # =========================================================================
    # Stage 4C: Bare Dielectric PMCHWT Baseline Simulation
    # =========================================================================
    if not args.skip_pmchwt and not args.plots_only:
        _log_header("Stage 4C: PMCHWT Bare Dielectric Baseline Simulation")
        actual_mesh = mesh_path if mesh_path.is_file() else (inputs_dir / "surface.obj")
        run_pmchwt_baseline(
            actual_mesh,
            obs_dir,
            pmchwt_dir,
            force=args.force,
            no_residual=args.no_residual,
        )

    # =========================================================================
    # Stage 5: Publication Field Plots (Figure 1 & Figure 2)
    # =========================================================================
    _log_header("Generating Publication Field Figures")
    cmd_stage5 = [
        python_exe,
        "-m",
        "applications.gstc_surface.publication.plot_fields_publication",
        "--fields-dir",
        str(fields_dir),
        "--output-dir",
        str(figures_dir),
        "--vmin-db",
        str(args.vmin_db),
        "--vmax-db",
        str(args.vmax_db),
    ]
    if pmchwt_dir.is_dir():
        cmd_stage5.extend(["--pmchwt-dir", str(pmchwt_dir)])
    # If NGSolve results are available, pass them to overlay on the 1D cut
    if (ng_dir / "fields" / "run01" / "fields.json").is_file():
        cmd_stage5.extend(["--ngsolve-fields-dir", str(ng_dir / "fields" / "run01")])
    _run_step(cmd_stage5, "Stage 5: Generate Publication 2D Maps and 1D Line Cuts")

    # =========================================================================
    # Stage 6: Publication Surface Parameter Maps (Figure 3)
    # =========================================================================
    _log_header("Generating Publication Surface Parameter Figures")
    param_figures_dir = figures_dir / "surface_parameters"
    cmd_stage6 = [
        python_exe,
        "-m",
        "applications.gstc_surface.publication.plot_parameters_publication",
        "--inputs-dir",
        str(inputs_dir),
        "--output-dir",
        str(param_figures_dir),
        "--percentile",
        "98",
        "--show-edges",
    ]
    _run_step(cmd_stage6, "Stage 6: Generate Publication 3D Surface Parameter Maps")

    _log_header("Example 3 Publication Figures Successfully Generated")
    print(f"Output files saved in: {figures_dir}")
    if figures_dir.is_dir():
        for p in sorted(figures_dir.glob("*.png")):
            print(f"  - {p.name}")
        for p in sorted(figures_dir.glob("*.pdf")):
            print(f"  - {p.name}")
    if param_figures_dir.is_dir():
        print(f"\nSurface parameter figures saved in: {param_figures_dir}")
        for p in sorted(param_figures_dir.glob("*.png")):
            print(f"  - {p.name}")
        for p in sorted(param_figures_dir.glob("*.pdf")):
            print(f"  - {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

