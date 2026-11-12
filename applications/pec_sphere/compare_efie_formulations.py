#!/usr/bin/env python3
"""Compare PEC T-EFIE with N-EFIE or N-MFIE currents and fields.

Results are kept under <output-dir>/data/{t,n_surface|n_line|n_mfie}/ and /plots/.
Use --reuse-currents to plot a completed solve without rebuilding its matrices.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parents[1]
VENV_PACKAGES = (
    REPO / ".venv" / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
if VENV_PACKAGES.is_dir() and str(VENV_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_PACKAGES))

BINARY = REPO / "build/applications/pec_sphere/pec_sphere_app"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frequency-hz", type=float, default=1e9)
    parser.add_argument("--epsilon-r-exterior", type=float, default=1.0)
    parser.add_argument("--mu-r-exterior", type=float, default=1.0)
    parser.add_argument("--quadrature-rule", type=int, default=7)
    parser.add_argument("--singularity-threshold-lambda0", type=float, default=0.1)
    parser.add_argument("--compare-with", choices=("n-efie", "n-mfie"), default="n-efie")
    parser.add_argument("--ntl-form", choices=("surface", "line"), default="surface")
    parser.add_argument("--line-quadrature-points", type=int, default=20)
    parser.add_argument("--zx-grid-size", type=int, default=80)
    parser.add_argument("--zx-span-radii", type=float, default=2.0)
    parser.add_argument("--radial-points", type=int, default=100)
    parser.add_argument("--radial-theta-degrees", type=float, default=45.0)
    parser.add_argument("--radial-phi-degrees", type=float, default=0.0)
    parser.add_argument("--radial-min-radii", type=float, default=0.05)
    parser.add_argument("--radial-max-radii", type=float, default=2.0)
    parser.add_argument("--no-2d", action="store_true")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument(
        "--reuse-currents", action="store_true",
        help="Use currents.csv from an existing output directory; do not solve again",
    )
    args = parser.parse_args()
    if not args.mesh.is_file():
        parser.error(f"mesh not found: {args.mesh}")
    if args.frequency_hz <= 0 or not math.isfinite(args.frequency_hz):
        parser.error("--frequency-hz must be positive and finite")
    if args.radial_points not in (0,) and args.radial_points < 2:
        parser.error("--radial-points must be 0 or at least 2")
    if args.zx_grid_size not in (0,) and args.zx_grid_size < 2:
        parser.error("--zx-grid-size must be 0 or at least 2")
    if args.compare_with == "n-mfie" and args.ntl_form == "line":
        parser.error("--ntl-form line applies only to N-EFIE")
    if args.compare_with == "n-efie" and args.ntl_form == "line" and not 1 <= args.line_quadrature_points <= 20:
        parser.error("--line-quadrature-points must be in [1, 20]")
    if args.reuse_currents and args.no_fields:
        parser.error("--reuse-currents requires field evaluation")
    if args.reuse_currents:
        for name in ("t", comparison_directory(args)):
            path = args.output_dir / "data" / name / "currents.csv"
            if not path.is_file():
                parser.error(f"saved currents not found: {path}")
    return args


def comparison_directory(args: argparse.Namespace) -> str:
    return "n_mfie" if args.compare_with == "n-mfie" else f"n_{args.ntl_form}"


def infer_sphere_radius(mesh_path: Path) -> float:
    radii = []
    with mesh_path.open() as stream:
        for line in stream:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                radii.append(math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2))
    if not radii:
        raise RuntimeError(f"mesh has no vertices: {mesh_path}")
    return sum(radii) / len(radii)


def common_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        str(BINARY), "--mesh", str(args.mesh), "--output-dir", str(run_dir),
        "--frequency-hz", str(args.frequency_hz),
        "--epsilon-r-exterior", str(args.epsilon_r_exterior),
        "--mu-r-exterior", str(args.mu_r_exterior),
        "--quadrature-rule", str(args.quadrature_rule),
        "--zx-grid-size", str(0 if args.no_2d else args.zx_grid_size),
        "--zx-span-radii", str(args.zx_span_radii),
        "--radial-points", str(args.radial_points),
        "--radial-theta-degrees", str(args.radial_theta_degrees),
        "--radial-phi-degrees", str(args.radial_phi_degrees),
        "--radial-min-radii", str(args.radial_min_radii),
        "--radial-max-radii", str(args.radial_max_radii),
    ]
    return command


def run(args: argparse.Namespace, formulation: str, run_dir: Path) -> list[complex]:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = common_command(args, run_dir)
    if args.reuse_currents:
        command += ["--currents-file", str(run_dir / "currents.csv")]
        log_name = "fields.log"
    else:
        command += [
            "--singularity-threshold-lambda0", str(args.singularity_threshold_lambda0),
            "--formulation", formulation,
        ]
        if formulation == "n":
            command += ["--ntl-form", args.ntl_form]
            if args.ntl_form == "line":
                command += ["--line-quadrature-points", str(args.line_quadrature_points)]
        if args.no_fields:
            command.append("--no-fields")
        if args.no_residual:
            command.append("--no-residual")
        log_name = "run.log"
    completed = subprocess.run(command, text=True, capture_output=True)
    (run_dir / log_name).write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        label = "N-MFIE" if formulation == "n-mfie" else f"{formulation.upper()}-EFIE"
        raise RuntimeError(f"{label} failed; see {run_dir / log_name}")
    print(completed.stdout, end="", flush=True)
    with (run_dir / "currents.csv").open(newline="") as stream:
        return [complex(float(row["J2_real"]), float(row["J2_imag"]))
                for row in csv.DictReader(stream)]


def load_fields(path: Path, np):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty field CSV: {path}")
    data = {key: np.array([float(row[key]) for row in rows])
            for key in rows[0]}
    for field in ("E", "H"):
        data[f"{field}_mag"] = np.sqrt(sum(
            data[f"{field}_{component}_{part}"] ** 2
            for component in "xyz" for part in ("re", "im")
        ))
    return data


def relative_l2(np, reference, comparison, key: str) -> float:
    denominator = np.sum(reference[key] ** 2)
    return float(np.sqrt(np.sum((reference[key] - comparison[key]) ** 2)
                         / denominator)) if denominator > 0 else float("nan")


def plot_2d(plt, np, t_data, n_data, key: str, title: str, unit: str,
            radius: float, path: Path, comparison_label: str,
            signed: bool = False) -> None:
    count = len(t_data["x"])
    size = math.isqrt(count)
    if size * size != count or len(n_data["x"]) != count:
        raise RuntimeError("ZX field grid has an unexpected shape")
    x = t_data["x"].reshape(size, size)
    z = t_data["z"].reshape(size, size)
    t = t_data[key].reshape(size, size)
    n = n_data[key].reshape(size, size)
    difference = t - n if signed else np.abs(t - n)
    if signed:
        common_limit = max(float(np.percentile(np.abs(np.r_[t.flat, n.flat]), 98)), 1e-12)
        difference_limit = max(float(np.percentile(np.abs(difference), 98)), 1e-12)
        panels = [
            (t, "T-EFIE", "RdBu_r", -common_limit, common_limit),
            (n, comparison_label, "RdBu_r", -common_limit, common_limit),
            (difference, f"T-EFIE - {comparison_label}", "coolwarm", -difference_limit, difference_limit),
        ]
    else:
        common_limit = max(float(np.percentile(np.r_[t.flat, n.flat], 99)), 1e-12)
        difference_limit = max(float(np.percentile(difference, 99)), 1e-12)
        panels = [
            (t, "T-EFIE", "viridis", 0, common_limit),
            (n, comparison_label, "viridis", 0, common_limit),
            (difference, f"|T-EFIE - {comparison_label}|", "magma", 0, difference_limit),
        ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, (values, label, colors, minimum, maximum) in zip(axes, panels):
        image = axis.pcolormesh(x, z, values, shading="auto", cmap=colors,
                                vmin=minimum, vmax=maximum)
        axis.add_patch(plt.Circle((0, 0), radius, fill=False, color="black",
                                  linestyle="--", linewidth=1.5))
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")
        axis.set_title(label)
        fig.colorbar(image, ax=axis, shrink=0.85).set_label(unit)
    fig.suptitle(f"PEC sphere, y=0: {title}")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_radial(plt, np, t_data, n_data, radius: float, path: Path,
                comparison_label: str) -> None:
    r = np.sqrt(t_data["x"] ** 2 + t_data["y"] ** 2 + t_data["z"] ** 2)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for axis, key, unit in (
        (axes[0], "E_mag", "V/m"), (axes[1], "H_mag", "A/m"),
    ):
        axis.plot(r / radius, t_data[key], label="T-EFIE")
        axis.plot(r / radius, n_data[key], label=comparison_label, linestyle="--")
        axis.axvline(1, color="tab:red", linestyle=":", label="PEC boundary")
        axis.set_ylabel(f"|{key[0]}| ({unit})")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
    axes[1].set_xlabel("r/a")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    if not (REPO / "build/CMakeCache.txt").exists():
        subprocess.run(["cmake", "-S", str(REPO), "-B", str(REPO / "build")], check=True)
    subprocess.run(["cmake", "--build", str(REPO / "build"), "--target",
                    "pec_sphere_app", "-j", "2"], check=True)
    data = args.output_dir / "data"
    t_dir, n_dir = data / "t", data / comparison_directory(args)
    comparison_label = "N-MFIE" if args.compare_with == "n-mfie" else "N-EFIE"
    print("Evaluating T-EFIE", flush=True)
    t_current = run(args, "t", t_dir)
    if args.compare_with == "n-mfie":
        print("Evaluating N-MFIE", flush=True)
        n_current = run(args, "n-mfie", n_dir)
    else:
        print(f"Evaluating N-EFIE Form {'A' if args.ntl_form == 'line' else 'B'}", flush=True)
        n_current = run(args, "n", n_dir)
    if len(t_current) != len(n_current):
        raise RuntimeError(f"T-EFIE and {comparison_label} returned different edge counts")
    denominator = sum(abs(value) ** 2 for value in t_current)
    if denominator == 0:
        raise RuntimeError("T-EFIE current has zero norm")
    current_difference = math.sqrt(
        sum(abs(t - n) ** 2 for t, n in zip(t_current, n_current)) / denominator
    )
    print(f"Relative L2 difference in J2 ({comparison_label} versus T-EFIE): "
          f"{current_difference:.9e}")
    if args.no_fields:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = args.output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    radius = infer_sphere_radius(args.mesh)
    if args.radial_points > 0:
        t_radial = load_fields(t_dir / "radial_fields.csv", np)
        n_radial = load_fields(n_dir / "radial_fields.csv", np)
        for key in ("E_mag", "H_mag"):
            print(f"Relative L2 difference in radial {key}: "
                  f"{relative_l2(np, t_radial, n_radial, key):.9e}")
        plot_radial(plt, np, t_radial, n_radial, radius, plots / "radial_cut.png",
                    comparison_label)
    if not args.no_2d and args.zx_grid_size > 0:
        t_zx = load_fields(t_dir / "zx_fields.csv", np)
        n_zx = load_fields(n_dir / "zx_fields.csv", np)
        for key in ("E_mag", "H_mag"):
            print(f"Relative L2 difference in ZX {key}: "
                  f"{relative_l2(np, t_zx, n_zx, key):.9e}")
        plot_2d(plt, np, t_zx, n_zx, "E_mag", "|E|", "V/m", radius,
                plots / "2d_electric_field_magnitude.png", comparison_label)
        plot_2d(plt, np, t_zx, n_zx, "H_mag", "|H|", "A/m", radius,
                plots / "2d_magnetic_field_magnitude.png", comparison_label)
        plot_2d(plt, np, t_zx, n_zx, "E_x_re", "Re(Ex)", "V/m", radius,
                plots / "2d_real_ex.png", comparison_label, signed=True)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
