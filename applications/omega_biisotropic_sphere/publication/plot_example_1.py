#!/usr/bin/env python3
"""Generate publication-quality figures for Example 1 (Budhu 10 GHz, eps_r=7).

Generates:
  1. Figure 1: 2D maps of |E_theta| in dB (right half-plane x >= 0)
     - Panel (a): Without metasurface (PMCHWT MoM)
     - Panel (b): With metasurface (SIE-GSTC MoM)
     - Side-by-side single-column layout (3.5 in) with shared z-axis and colorbar.
  2. Figure 2: 1D radial cut of Re(E_theta) along theta = 45 deg
     - Comparison across Target (Prescribed), SIE (MoM), SVWF, and COMSOL.

Designed for IEEE journal two-column formatting (3.5 in column width).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Circle
import numpy as np

# Use shared postprocessing module from src/sie_mom
from sie_mom.postprocessing import (
    budhu_moment_scale,
    dominant_tangential_components,
    load_field_csv,
    to_decibels,
    zx_grid,
)

# IEEE Journal typography / styling defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "budhu_10ghz_eps7"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
SPHERE_RADIUS_M = 0.02098547206  # 0.7 lambda_0 at 10 GHz


def _crop_zx_grid(
    data: dict[str, np.ndarray],
    grid_size: int,
    field_values: np.ndarray,
    crop: str = "right",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape 2D ZX grid and crop to right half (x >= 0) or upper half (z >= 0)."""
    x, z, order = zx_grid(data, grid_size)
    values_2d = field_values[order].reshape((grid_size, grid_size))

    if crop == "right":
        # Keep x >= 0 (columns where x >= -1e-12)
        x_mask = x >= -1e-12
        x_crop = x[x_mask]
        values_crop = values_2d[:, x_mask]
        xx, zz = np.meshgrid(x_crop, z)
    else:
        # Keep z >= 0 (rows where z >= -1e-12)
        z_mask = z >= -1e-12
        z_crop = z[z_mask]
        values_crop = values_2d[z_mask, :]
        xx, zz = np.meshgrid(x, z_crop)

    return xx, zz, values_crop


def plot_figure_1_2d_maps(
    bare_csv: Path,
    sie_csv: Path,
    output_dir: Path,
    *,
    grid_size: int = 200,
    radius_m: float = SPHERE_RADIUS_M,
    crop: str = "right",
    db_dynamic_range: float = 45.0,
) -> tuple[Path, Path]:
    """Plot |E_theta| in dB for bare sphere vs omega metasurface."""
    print(f"Loading bare sphere data: {bare_csv}")
    bare_data = load_field_csv(bare_csv)
    print(f"Loading SIE omega data:  {sie_csv}")
    sie_data = load_field_csv(sie_csv)

    eth_bare, _ = dominant_tangential_components(bare_data)
    eth_sie, _ = dominant_tangential_components(sie_data)

    xx_bare, zz_bare, e2d_bare = _crop_zx_grid(bare_data, grid_size, np.abs(eth_bare), crop=crop)
    xx_sie, zz_sie, e2d_sie = _crop_zx_grid(sie_data, grid_size, np.abs(eth_sie), crop=crop)

    # Normalize coordinates by sphere radius a
    x_norm = xx_bare / radius_m
    z_norm = zz_bare / radius_m

    # Convert to unfloored dB matching the diagnostic fields_zx.png convention
    db_bare = to_decibels(e2d_bare, floor_relative_db=280.0)
    db_sie = to_decibels(e2d_sie, floor_relative_db=280.0)

    # Focus colormap on propagating lobes by clipping the extreme 2% singularity core
    combined = np.concatenate([db_bare[np.isfinite(db_bare)], db_sie[np.isfinite(db_sie)]])
    vmin, vmax = np.percentile(combined, (2.0, 98.0))
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    cmap = "viridis"

    titles = (
        r"(a) Without metasurface",
        r"(b) With metasurface",
    )
    datasets = (db_bare, db_sie)

    if crop == "right":
        # Right half-plane (x >= 0, z in [-2, 2]):
        # Two panels side-by-side perfectly fit a single column (3.5 in) with a 1:1 aspect ratio.
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.2), sharex=True, sharey=True, constrained_layout=True)

        mesh_artists = []
        for ax, data_db, title in zip(axes, datasets, titles):
            mesh = ax.pcolormesh(
                x_norm,
                z_norm,
                data_db,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            mesh_artists.append(mesh)

            # Draw half-circle outline of the sphere on right (radius = 1.0, theta in [-pi/2, pi/2])
            theta_arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
            ax.plot(np.cos(theta_arc), np.sin(theta_arc), color="white", linestyle="--", linewidth=0.9, alpha=0.9)

            # Draw centered z-directed dipole indicator arrow
            ax.annotate(
                "",
                xy=(0.0, 0.22),
                xytext=(0.0, -0.22),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
            )

            ax.set_aspect("equal", "box")
            ax.set_title(title, pad=3)
            ax.set_xlabel(r"$x/a$")
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(-2.0, 2.0)

        axes[0].set_ylabel(r"$z/a$")
        cbar = fig.colorbar(mesh_artists[1], ax=axes, orientation="horizontal", fraction=0.06, pad=0.07, shrink=0.85)
    else:
        # Upper half-plane (z >= 0, x in [-2, 2]):
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True, sharey=True, constrained_layout=True)

        mesh_artists = []
        for ax, data_db, title in zip(axes, datasets, titles):
            mesh = ax.pcolormesh(
                x_norm,
                z_norm,
                data_db,
                shading="auto",
                cmap=cmap,
                norm=norm,
            )
            mesh_artists.append(mesh)

            # Draw half-circle outline of the sphere (radius = 1.0, theta in [0, pi])
            theta_arc = np.linspace(0, np.pi, 200)
            ax.plot(np.cos(theta_arc), np.sin(theta_arc), color="white", linestyle="--", linewidth=0.9, alpha=0.9)

            ax.set_aspect("equal", "box")
            ax.set_title(title, pad=3)
            ax.set_ylabel(r"$z/a$")
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(0.0, 2.0)

        axes[1].set_xlabel(r"$x/a$")
        cbar = fig.colorbar(mesh_artists[1], ax=axes, orientation="horizontal", fraction=0.06, pad=0.08, shrink=0.85)

    cbar.set_label(r"$|E_\theta|$ [dB]")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{crop}_half" if crop != "right" else ""
    pdf_path = output_dir / f"figure_1_2d_field_maps{suffix}.pdf"
    png_path = output_dir / f"figure_1_2d_field_maps{suffix}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1: {pdf_path}")
    print(f"Saved Figure 1: {png_path}")
    return pdf_path, png_path


def plot_figure_2_1d_radial_cut(
    results_dir: Path,
    output_dir: Path,
    *,
    radius_m: float = SPHERE_RADIUS_M,
) -> tuple[Path, Path]:
    """Plot 1D radial cut of Re(E_theta) comparing Target, SIE, SVWF, and COMSOL."""
    prescribed_csv = results_dir / "sie" / "least_squares" / "fields" / "prescribed_radial.csv"
    sie_csv = results_dir / "sie" / "least_squares" / "fields" / "sie_physical_radial.csv"
    svwf_csv = results_dir / "svwf" / "least_squares" / "fields" / "svwf_radial.csv"
    comsol_csv = results_dir / "comsol" / "omega_case19_projector_shell90_glob20_int35_6_1" / "fields" / "comsol_radial.csv"

    print("Loading radial cut datasets...")
    pre_data = load_field_csv(prescribed_csv)
    sie_data = load_field_csv(sie_csv)
    svwf_data = load_field_csv(svwf_csv)
    has_comsol = comsol_csv.is_file()
    if has_comsol:
        com_data = load_field_csv(comsol_csv)

    eth_pre, _ = dominant_tangential_components(pre_data)
    eth_sie, _ = dominant_tangential_components(sie_data)
    eth_svwf, _ = dominant_tangential_components(svwf_data)
    if has_comsol:
        eth_com, _ = dominant_tangential_components(com_data)

    # Budhu physical scale I Delta z = 1 A m
    manifest = {
        "frequency_hz": 1e10,
        "interior": {"epsilon_r": 7.0, "mu_r": 1.0},
        "source": {"effective_moment_interior": 1.0},
    }
    scale = budhu_moment_scale(manifest)

    r_pre = np.sqrt(pre_data["x"] ** 2 + pre_data["y"] ** 2 + pre_data["z"] ** 2) / radius_m
    r_sie = np.sqrt(sie_data["x"] ** 2 + sie_data["y"] ** 2 + sie_data["z"] ** 2) / radius_m
    r_svwf = np.sqrt(svwf_data["x"] ** 2 + svwf_data["y"] ** 2 + svwf_data["z"] ** 2) / radius_m
    if has_comsol:
        r_com = np.sqrt(com_data["x"] ** 2 + com_data["y"] ** 2 + com_data["z"] ** 2) / radius_m

    # Re(E_theta) scaled by 1e-5 (units: 10^5 V/m) matching Budhu's Fig. 3 convention
    e_pre_re = np.real(scale * eth_pre) / 1e5
    e_sie_re = np.real(scale * eth_sie) / 1e5
    e_svwf_re = np.real(scale * eth_svwf) / 1e5
    if has_comsol:
        # COMSOL source is already physical 1 A m, so only divide by 1e5
        e_com_re = np.real(eth_com) / 1e5

    # Single-column width figure (3.4 in x 2.4 in)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)

    # Sparse SIE circles show agreement without adding another overlapping line.
    ax.plot(r_pre, e_pre_re, label="Prescribed Target", color="black", linestyle="--", linewidth=1.0)
    ax.plot(
        r_sie, e_sie_re, label="SIE-GSTC (MoM)", color="#d95f02", linestyle="none",
        marker="o", markevery=(0, 10), markersize=3.0, markerfacecolor="none", markeredgewidth=0.8,
    )
    # Start at index 5 and step by 24: these odd indices never coincide with SIE's multiples of 10.
    ax.plot(r_svwf, e_svwf_re, label="SVWF Collocation", color="#1b9e77", linestyle="none",
        marker="^", markevery=(5, 24), markersize=3.0, markerfacecolor="none", markeredgewidth=0.8)
    if has_comsol:
        ax.plot(r_com, e_com_re, label="COMSOL (FEM)", color="#0072B2", linestyle="-.", linewidth=0.8)

    # Metasurface interface indicator
    ax.axvline(1.0, color="#e41a1c", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel(r"Normalized Radial Distance $r/a$")
    ax.set_ylabel(r"$\mathrm{Re}(E_\theta) \times 10^5$ [V/m]")
    ax.set_xlim(0.1, 2.0)
    ax.set_title(r"Radial Field Comparison ($\theta = 45^\circ, \phi = 0^\circ$)", pad=3)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, edgecolor="none", loc="upper right")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_2_radial_cut_comparison.pdf"
    png_path = output_dir / "figure_2_radial_cut_comparison.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2: {pdf_path}")
    print(f"Saved Figure 2: {png_path}")
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to results root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated figures (default: %(default)s)",
    )
    parser.add_argument(
        "--crop",
        choices=("right", "upper"),
        default="right",
        help="Region to plot: 'right' (x >= 0, side-by-side single-column) or 'upper' (z >= 0, vertical stack)",
    )
    parser.add_argument(
        "--bare-csv",
        type=Path,
        default=None,
        help="Optional path to bare sphere ZX CSV (default: results_dir/pmchwt/bare_sphere/zx_fields.csv)",
    )
    args = parser.parse_args()

    bare_csv = (
        args.bare_csv
        if args.bare_csv is not None
        else args.results_dir / "pmchwt" / "bare_sphere" / "zx_fields.csv"
    )
    sie_csv = args.results_dir / "sie" / "least_squares" / "fields" / "sie_physical_zx.csv"

    print("=== Generating Paper 1 Figures ===")
    plot_figure_1_2d_maps(bare_csv, sie_csv, args.output_dir, crop=args.crop)
    plot_figure_2_1d_radial_cut(args.results_dir, args.output_dir)
    print("=== Figure Generation Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

