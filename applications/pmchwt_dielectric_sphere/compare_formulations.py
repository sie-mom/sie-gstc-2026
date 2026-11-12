#!/usr/bin/env python3
"""Run both PMCHWT formulations (T and N) and generate 2D & 1D validation plots.

Saves structured outputs:
  <output-dir>/data/t/   (T-PMCHWT CSVs and run.log)
  <output-dir>/data/n/   (N-PMCHWT CSVs and run.log)
  <output-dir>/plots/    (2D heatmaps and radial field comparisons)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

# Ensure matplotlib can find a writable config cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

# Fallback: if running with system python without virtualenv activated, add .venv to sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_PACKAGES = (
    _REPO_ROOT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
)
if _VENV_PACKAGES.is_dir() and str(_VENV_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_VENV_PACKAGES))

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from sie_mom.postprocessing import complex_vector, load_field_csv

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parents[1]
DEFAULT_MESH = APP_DIR / "meshes" / "icosahedron_20_faces_r0p03m.obj"
DEFAULT_OUTPUT = APP_DIR / "results"
BINARY = REPO / "build" / "applications" / "pmchwt_dielectric_sphere" / "pmchwt_dielectric_sphere_app"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH,
                        help="Path to triangular OBJ mesh")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Root output directory (contains data/ and plots/)")
    parser.add_argument("--frequency-hz", type=float, default=1e9,
                        help="Frequency in Hz")
    parser.add_argument("--epsilon-r-interior", type=float, default=2.5,
                        help="Interior relative permittivity")
    parser.add_argument("--mu-r-interior", type=float, default=1.0,
                        help="Interior relative permeability")
    parser.add_argument("--epsilon-r-exterior", type=float, default=1.0,
                        help="Exterior relative permittivity")
    parser.add_argument("--mu-r-exterior", type=float, default=1.0,
                        help="Exterior relative permeability")
    parser.add_argument("--formulations", nargs="+", choices=("t", "n", "muller"),
                        default=["t", "n", "muller"],
                        help="Formulations to run and compare (default: t n muller)")
    parser.add_argument("--quadrature-rule", type=int, default=5,
                        help="Dunavant rule order")
    parser.add_argument("--ntl-form", choices=("line", "surface"), default="line",
                        help="nTL integral for N-formulations: Form A line or Form B surface (default: surface)")
    parser.add_argument("--line-quadrature-points", type=int, default=20,
                        help="Gauss-Legendre points for Form A line (default: 20)")
    parser.add_argument("--zx-grid-size", type=int, default=80,
                        help="Grid points per axis in the 2D y=0 (zx) plane")
    parser.add_argument("--zx-span-radii", type=float, default=2.0,
                        help="Half-span of 2D grid in units of sphere radius")
    parser.add_argument("--radial-points", type=int, default=100,
                        help="Number of radial sampling points")
    parser.add_argument("--radial-min-radii", type=float, default=0.05,
                        help="Minimum radial sampling distance (normalized by radius r/a)")
    parser.add_argument("--radial-max-radii", type=float, default=2.0,
                        help="Maximum radial sampling distance (normalized by radius r/a)")
    parser.add_argument("--radial-theta-degrees", type=float, default=45.0,
                        help="Polar angle theta for radial cut (default: 45)")
    parser.add_argument("--radial-phi-degrees", type=float, default=0.0,
                        help="Azimuthal angle phi for radial cut (default: 0)")
    parser.add_argument("--singularity-threshold-lambda0", type=float, default=0.1,
                        help="Singularity extraction threshold in lambda0 units (default: 0.1)")
    parser.add_argument("--no-2d", action="store_true",
                        help="Skip 2D grid evaluation and plots")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip building and running the solvers; just reload CSVs and plot")
    parser.add_argument("--include-mie", action="store_true",
                        help="Calculate and plot Mie series using scattnlay (requires pip install scattnlay)")
    args = parser.parse_args()
    if args.radial_points < 2:
        parser.error("--radial-points must be at least 2")
    if args.ntl_form == "line" and not 1 <= args.line_quadrature_points <= 20:
        parser.error("--line-quadrature-points must be in [1, 20] for line form")
    if not args.mesh.is_file():
        parser.error(f"mesh not found: {args.mesh}")
    return args



def calculate_mie_series(args, x, y, z, radius):
    try:
        from scattnlay import fieldnlay
    except ImportError:
        print("Error: --include-mie requires scattnlay. Run: pip install scattnlay")
        sys.exit(1)
        
    c0 = 299792458.0
    lambda0 = c0 / args.frequency_hz
    n_med = np.sqrt(args.epsilon_r_exterior * args.mu_r_exterior)
    n_sph = np.sqrt(args.epsilon_r_interior * args.mu_r_interior)
    
    size_param = np.array([2 * np.pi * n_med * radius / lambda0])
    m_rel = np.array([n_sph / n_med])
    
    scaling_factor = (2 * np.pi * n_med) / lambda0
    xp = x * scaling_factor
    yp = y * scaling_factor
    zp = z * scaling_factor
    
    terms, E, H = fieldnlay(size_param, m_rel, xp, yp, zp)
    
    # scattnlay returns complex arrays of shape (N, 3)
    # The solver outputs E_mag and H_mag. Let's compute them.
    # Note: scattnlay assumes incident plane wave E0 = 1 V/m.
    E_mag = np.linalg.norm(E, axis=1)
    H_mag = np.linalg.norm(H, axis=1)
    
    return {
        "x": x, "y": y, "z": z,
        "E_mag": E_mag,
        "H_mag": H_mag,
        "E_x_re": E[:, 0].real,
    }

def build_app():
    build_dir = REPO / "build"
    if not (build_dir / "CMakeCache.txt").exists():
        subprocess.run(["cmake", "-S", str(REPO), "-B", str(build_dir)], check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target",
         "pmchwt_dielectric_sphere_app", "-j", "2"],
        check=True,
    )


def infer_sphere_radius(mesh_path: Path) -> float:
    radii = []
    with mesh_path.open() as stream:
        for line in stream:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    radii.append(math.sqrt(x * x + y * y + z * z))
    return sum(radii) / len(radii) if radii else 1.0


def run_formulation(args, formulation: str, data_dir: Path) -> Path:
    form_dir = data_dir / formulation
    form_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(BINARY),
        "--mesh", str(args.mesh),
        "--output-dir", str(form_dir),
        "--frequency-hz", str(args.frequency_hz),
        "--epsilon-r-interior", str(args.epsilon_r_interior),
        "--mu-r-interior", str(args.mu_r_interior),
        "--epsilon-r-exterior", str(args.epsilon_r_exterior),
        "--mu-r-exterior", str(args.mu_r_exterior),
        "--quadrature-rule", str(args.quadrature_rule),
        "--radial-points", str(args.radial_points),
        "--radial-theta-degrees", str(args.radial_theta_degrees),
        "--radial-phi-degrees", str(args.radial_phi_degrees),
        "--radial-min-radii", str(args.radial_min_radii),
        "--radial-max-radii", str(args.radial_max_radii),
        "--formulation", formulation,
        "--singularity-threshold-lambda0", str(args.singularity_threshold_lambda0),
    ]
    if formulation in ("n", "muller"):
        command.extend(["--ntl-form", args.ntl_form])
        if args.ntl_form == "line":
            command.extend(["--line-quadrature-points", str(args.line_quadrature_points)])
    if args.no_2d:
        command.append("--no-2d-grid")
    else:
        command.extend([
            "--zx-grid-size", str(args.zx_grid_size),
            "--zx-span-radii", str(args.zx_span_radii),
        ])

    label_map = {
        "t": "T-PMCHWT",
        "n": f"N-PMCHWT ({args.ntl_form} nTL)",
        "muller": f"N-Müller ({args.ntl_form} nTL)",
    }
    label = label_map.get(formulation, formulation.upper())
    print(f"\n--- Running {label} on {args.mesh.name} ---", flush=True)
    print(f"  command: {shlex.join(command)}", flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    (form_dir / "run.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stdout + result.stderr)
        raise RuntimeError(f"{label} failed; see {form_dir / 'run.log'}")
    for line in result.stdout.splitlines():
        if line.strip():
            print(f"  {line}")
    return form_dir


def load_field_data(csv_path: Path):
    if not csv_path.is_file():
        return None
    data = load_field_csv(csv_path)
    data["E_x_re"] = data["Ex_real"]
    data["E_x_im"] = data["Ex_imag"]
    data["E_mag"] = np.linalg.norm(complex_vector(data, "E"), axis=1)
    data["H_mag"] = np.linalg.norm(complex_vector(data, "H"), axis=1)
    return data


def relative_l2(t_data, n_data, field_key: str) -> float:
    diff = np.sum((t_data[field_key] - n_data[field_key]) ** 2)
    ref = np.sum(t_data[field_key] ** 2)
    return float(np.sqrt(diff / ref)) if ref > 0 else float("nan")


def plot_2d_slice(
    data_a, data_b,
    label_a: str, label_b: str,
    field_key: str,
    title: str,
    unit: str,
    radius: float,
    output_path: Path,
    is_diverging: bool = False,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Skipping plot {output_path.name}: matplotlib is not available.")
        return

    n_points = len(data_a["x"])
    grid_size = int(round(math.sqrt(n_points)))
    if grid_size * grid_size != n_points:
        print(f"Skipping 2D plot: point count {n_points} is not a perfect square.")
        return

    x_grid = data_a["x"].reshape((grid_size, grid_size))
    z_grid = data_a["z"].reshape((grid_size, grid_size))
    a_vals = data_a[field_key].reshape((grid_size, grid_size))
    b_vals = data_b[field_key].reshape((grid_size, grid_size))
    diff_vals = (a_vals - b_vals) if is_diverging else np.abs(a_vals - b_vals)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    if is_diverging:
        limit_a = max(float(np.percentile(np.abs(a_vals), 98)), 1e-12)
        limit_b = max(float(np.percentile(np.abs(b_vals), 98)), 1e-12)
        diff_limit = max(float(np.percentile(np.abs(diff_vals), 98)), 1e-12)

        panels = [
            (axes[0], a_vals, f"{label_a} (peak: {np.max(np.abs(a_vals)):.2f} {unit})",
             "RdBu_r", -limit_a, limit_a, f"{title} ({unit})"),
            (axes[1], b_vals, f"{label_b} (peak: {np.max(np.abs(b_vals)):.2f} {unit})",
             "RdBu_r", -limit_b, limit_b, f"{title} ({unit})"),
            (axes[2], diff_vals, f"Difference {label_a} - {label_b} (peak: {np.max(np.abs(diff_vals)):.2f} {unit})",
             "coolwarm", -diff_limit, diff_limit, f"Difference ({unit})"),
        ]
    else:
        vmax_a = max(float(np.percentile(a_vals, 99)), 1e-12)
        vmax_b = max(float(np.percentile(b_vals, 99)), 1e-12)
        diff_vmax = max(float(np.percentile(diff_vals, 99)), 1e-12)

        panels = [
            (axes[0], a_vals, f"{label_a} (max: {np.max(a_vals):.2f} {unit})",
             "viridis", 0.0, vmax_a, f"{title} ({unit})"),
            (axes[1], b_vals, f"{label_b} (max: {np.max(b_vals):.2f} {unit})",
             "viridis", 0.0, vmax_b, f"{title} ({unit})"),
            (axes[2], diff_vals, f"Difference |{label_a} - {label_b}| (max: {np.max(diff_vals):.2f} {unit})",
             "magma", 0.0, diff_vmax, f"Difference ({unit})"),
        ]

    for ax, data_grid, panel_title, colormap, c_min, c_max, label in panels:
        pcm = ax.pcolormesh(
            x_grid, z_grid, data_grid,
            shading="auto", cmap=colormap, vmin=c_min, vmax=c_max
        )
        circle = plt.Circle((0, 0), radius, fill=False, color="black", linestyle="--", linewidth=1.5)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.85)
        cbar.set_label(label)

    fig.suptitle(f"2D Cut (y=0 Plane): {title}", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved 2D plot: {output_path}")


def plot_radial_cuts(radial_results: dict, radius: float, output_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping radial plot: matplotlib is not available.")
        return

    first_data = next(iter(radial_results.values()))
    r = np.sqrt(first_data["x"] ** 2 + first_data["y"] ** 2 + first_data["z"] ** 2)

    styles = {
        "T-PMCHWT": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.0},
        "N-PMCHWT": {"color": "tab:orange", "linestyle": "--", "linewidth": 1.8},
        "N-Müller": {"color": "tab:green", "linestyle": ":", "linewidth": 2.0},
        "Mie Series": {"color": "black", "linestyle": "-.", "linewidth": 1.8},
    }
    fallback_colors = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)

    for idx, (label, data) in enumerate(radial_results.items()):
        style = styles.get(label, {"color": fallback_colors[idx % len(fallback_colors)], "linestyle": "-", "linewidth": 1.8})
        axes[0].plot(r, data["E_mag"], label=label, **style)
        axes[1].plot(r, data["H_mag"], label=label, **style)

    axes[0].axvline(radius, color="tab:red", linestyle=":", label=f"Boundary (r={radius:.3f} m)")
    axes[0].set_ylabel("|E| (V/m)")
    axes[0].set_title("Electric Field Magnitude along Radial Cut")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].axvline(radius, color="tab:red", linestyle=":", label=f"Boundary (r={radius:.3f} m)")
    axes[1].set_xlabel("Radius r (m)")
    axes[1].set_ylabel("|H| (A/m)")
    axes[1].set_title("Magnetic Field Magnitude along Radial Cut")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved radial plot: {output_path}")


def main():
    args = arguments()
    if not args.plot_only:
        build_app()

    data_dir = args.output_dir / "data"
    plots_dir = args.output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    radius = infer_sphere_radius(args.mesh)
    print(f"Inferred sphere radius: {radius:.6f} m")

    form_labels = {
        "t": "T-PMCHWT",
        "n": "N-PMCHWT",
        "muller": "N-Müller",
    }

    results_radial = {}
    results_zx = {}

    for form in args.formulations:
        form_dir = data_dir / form
        if not args.plot_only:
            run_formulation(args, form, data_dir)
        radial_data = load_field_data(form_dir / "radial_fields.csv")
        if radial_data is not None:
            results_radial[form_labels.get(form, form)] = radial_data
        if not args.no_2d:
            zx_data = load_field_data(form_dir / "zx_fields.csv")
            if zx_data is not None:
                results_zx[form] = zx_data

    if args.include_mie and np is not None and results_radial:
        print("\n--- Calculating Mie Series (scattnlay) ---")
        ref_data = next(iter(results_radial.values()))
        mie_radial = calculate_mie_series(args, ref_data["x"], ref_data["y"], ref_data["z"], radius)
        results_radial["Mie Series"] = mie_radial
        
        if not args.no_2d and results_zx:
            ref_zx = next(iter(results_zx.values()))
            mie_zx = calculate_mie_series(args, ref_zx["x"], ref_zx["y"], ref_zx["z"], radius)
            results_zx["Mie Series"] = mie_zx

    if len(results_radial) >= 2 and np is not None:
        print("\n--- Validation Metrics (Radial Cut) ---")
        labels = list(results_radial.keys())
        baseline_label = labels[0]
        baseline_data = results_radial[baseline_label]
        for other_label in labels[1:]:
            rel_e = relative_l2(baseline_data, results_radial[other_label], "E_mag")
            rel_h = relative_l2(baseline_data, results_radial[other_label], "H_mag")
            print(f"  {baseline_label} vs {other_label}:")
            print(f"    Relative L2 difference in |E|: {rel_e:.4e}")
            print(f"    Relative L2 difference in |H|: {rel_h:.4e}")

        plot_radial_cuts(results_radial, radius, plots_dir / "radial_cut.png")

    if not args.no_2d and len(results_zx) >= 2 and np is not None:
        forms = [f for f in args.formulations if f in results_zx]
        base_form = forms[0]
        base_label = form_labels.get(base_form, base_form)

        for comp_form in forms[1:]:
            comp_label = form_labels.get(comp_form, comp_form)
            suffix = f"_{base_form}_vs_{comp_form}" if len(forms) > 2 else ""

            print(f"\n--- Generating 2D Field Plots ({base_label} vs {comp_label}) ---")
            plot_2d_slice(
                results_zx[base_form], results_zx[comp_form],
                base_label, comp_label,
                field_key="E_mag",
                title=f"Electric Field Magnitude |E|",
                unit="V/m",
                radius=radius,
                output_path=plots_dir / f"2d_electric_field_magnitude{suffix}.png",
                is_diverging=False,
            )
            plot_2d_slice(
                results_zx[base_form], results_zx[comp_form],
                base_label, comp_label,
                field_key="H_mag",
                title=f"Magnetic Field Magnitude |H|",
                unit="A/m",
                radius=radius,
                output_path=plots_dir / f"2d_magnetic_field_magnitude{suffix}.png",
                is_diverging=False,
            )
            plot_2d_slice(
                results_zx[base_form], results_zx[comp_form],
                base_label, comp_label,
                field_key="E_x_re",
                title=f"Re(Ex) Field Component",
                unit="V/m",
                radius=radius,
                output_path=plots_dir / f"2d_real_ex{suffix}.png",
                is_diverging=True,
            )

        if args.include_mie and "Mie Series" in results_zx:
            print(f"\n--- Generating 2D Field Plots ({base_label} vs Mie Series) ---")
            suffix_mie = f"_{base_form}_vs_mie"
            plot_2d_slice(
                results_zx[base_form], results_zx["Mie Series"],
                base_label, "Mie Series",
                field_key="E_mag",
                title=f"Electric Field Magnitude |E|",
                unit="V/m",
                radius=radius,
                output_path=plots_dir / f"2d_electric_field_magnitude{suffix_mie}.png",
                is_diverging=False,
            )
            plot_2d_slice(
                results_zx[base_form], results_zx["Mie Series"],
                base_label, "Mie Series",
                field_key="H_mag",
                title=f"Magnetic Field Magnitude |H|",
                unit="A/m",
                radius=radius,
                output_path=plots_dir / f"2d_magnetic_field_magnitude{suffix_mie}.png",
                is_diverging=False,
            )
            plot_2d_slice(
                results_zx[base_form], results_zx["Mie Series"],
                base_label, "Mie Series",
                field_key="E_x_re",
                title=f"Re(Ex) Field Component",
                unit="V/m",
                radius=radius,
                output_path=plots_dir / f"2d_real_ex{suffix_mie}.png",
                is_diverging=True,
            )

    print(f"\nCompleted! Results organized in:\n  Data:  {data_dir}\n  Plots: {plots_dir}")


if __name__ == "__main__":
    main()
