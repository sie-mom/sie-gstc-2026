#!/usr/bin/env python3
"""Generate publication-quality figures for Example 2 Part A (Dipole Polarization Rotator).

Case:
  Electric Dipole -> Magnetic-Type Radiation Metasurface
  Frequency: 10 GHz, Interior: eps_r=7, Exterior: eps_r=1, Radius: 0.7 lambda_0.

Generates:
  1. Figure 1: 2D field maps of |E_theta| and |E_phi| in dB (right half-plane x >= 0)
     - Panel (a): |E_theta| in dB [TM mode: active interior dipole, extinguished exterior].
     - Panel (b): |E_phi| in dB [TE mode: extinguished interior, active exterior magnetic donut].
     - Side-by-side single-column layout (3.5 in) with shared colorbar [105, 150] dB.
  2. Figure 2: Stacked radial cuts along theta = 45 deg, phi = 0 deg
     - Top: Re(E_theta) across the full radial range.
     - Bottom: Re(E_phi) across the full radial range.
     - Comparison across Prescribed Target, SIE-GSTC (MoM), and SVWF Collocation.

Designed for IEEE journal two-column formatting (3.5 in column width).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path so 'applications' can always be imported
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from applications.anisotropic_gstc_sphere.prescribed_fields import evaluate_prescribed_fields
from applications.anisotropic_gstc_sphere.workflow import load_workflow_configuration
from sie_mom.postprocessing import (
    budhu_moment_scale,
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

DEFAULT_CONFIG_FILE = (
    Path(__file__).resolve().parents[1]
    / "configurations"
    / "dipole_pol_rot"
    / "e2m_eps7.json"
)
_RESULTS_CANDIDATE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "dipole_pol_rot"
    / "e2m_eps7"
)
DEFAULT_RESULTS_DIR = (
    _RESULTS_CANDIDATE
    if _RESULTS_CANDIDATE.exists()
    else (
        Path(__file__).resolve().parents[1]
        / "results"
        / "dipole_pol_rot"
        / "electric_to_magnetic_10ghz_eps7_reciprocal_lossless"
    )
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


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
        x_mask = x >= -1e-12
        x_crop = x[x_mask]
        values_crop = values_2d[:, x_mask]
        xx, zz = np.meshgrid(x_crop, z)
    else:
        z_mask = z >= -1e-12
        z_crop = z[z_mask]
        values_crop = values_2d[z_mask, :]
        xx, zz = np.meshgrid(x, z_crop)

    return xx, zz, values_crop


def plot_figure_1_2d_maps(
    sie_zx_csv: Path,
    output_dir: Path,
    radius_m: float,
    *,
    grid_size: int = 200,
    crop: str = "right",
    vmin_db: float = -112.0,
    vmax_db: float = -58.0,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot 2D field maps of |E_theta| and |E_phi| in normalized dB."""
    print(f"Loading 2D ZX field data: {sie_zx_csv}")
    data_zx = load_field_csv(sie_zx_csv)

    x, z, order = zx_grid(data_zx, grid_size)
    Ex = (
        data_zx["Ex_real"][order].reshape((grid_size, grid_size))
        + 1j * data_zx["Ex_imag"][order].reshape((grid_size, grid_size))
    )
    Ey = (
        data_zx["Ey_real"][order].reshape((grid_size, grid_size))
        + 1j * data_zx["Ey_imag"][order].reshape((grid_size, grid_size))
    )
    Ez = (
        data_zx["Ez_real"][order].reshape((grid_size, grid_size))
        + 1j * data_zx["Ez_imag"][order].reshape((grid_size, grid_size))
    )

    xx, zz = np.meshgrid(x, z)
    rr = np.sqrt(xx**2 + zz**2)
    phi = np.where(xx >= 0, 0.0, np.pi)
    th = np.arccos(np.clip(zz / rr, -1.0, 1.0))

    E_th_mag = np.abs(Ex * np.cos(th) * np.cos(phi) - Ez * np.sin(th))
    E_ph_mag = np.abs(-Ex * np.sin(phi) + Ey * np.cos(phi))

    # Crop to right half (x >= 0)
    x_mask = x >= -1e-12
    x_crop = x[x_mask] / radius_m
    z_crop = z / radius_m
    E_th_crop = E_th_mag[:, x_mask]
    E_ph_crop = E_ph_mag[:, x_mask]

    # Shared peak reference (matching Example 1 dipole singularity normalization)
    ref_peak = np.max(E_th_crop)
    db_th = 20.0 * np.log10(np.maximum(E_th_crop / ref_peak, 1e-15))
    db_ph = 20.0 * np.log10(np.maximum(E_ph_crop / ref_peak, 1e-15))

    norm = Normalize(vmin=vmin_db, vmax=vmax_db)
    cmap = "viridis"

    titles = (
        r"(a) $|E_\theta|$ [TM mode]",
        r"(b) $|E_\phi|$ [TE mode]",
    )
    datasets = (db_th, db_ph)

    # Side-by-side single-column layout (3.5 in width)
    fig, axes = plt.subplots(
        1, 2, figsize=(3.5, 3.2), sharex=True, sharey=True, constrained_layout=True
    )

    mesh_artists = []
    for ax, data_db, title in zip(axes, datasets, titles):
        mesh = ax.pcolormesh(
            x_crop,
            z_crop,
            data_db,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        mesh_artists.append(mesh)

        # Draw half-circle outline of the sphere (radius = 1.0, theta in [-pi/2, pi/2])
        theta_arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.plot(
            np.cos(theta_arc),
            np.sin(theta_arc),
            color="white",
            linestyle="--",
            linewidth=0.9,
            alpha=0.9,
        )

        # Centered z-directed dipole indicator arrow
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
    cbar = fig.colorbar(
        mesh_artists[1],
        ax=axes,
        orientation="horizontal",
        fraction=0.06,
        pad=0.07,
        shrink=0.85,
    )
    cbar.set_label(r"Normalized Field Magnitude [dB]")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_1_part_a_2d_field_maps.pdf"
    png_path = output_dir / "figure_1_part_a_2d_field_maps.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 1: {pdf_path}")
    print(f"Saved Figure 1: {png_path}")
    return pdf_path, png_path


def plot_figure_2_1d_radial_cut(
    results_dir: Path,
    config: object,
    output_dir: Path,
    case_name: str = "electric_incidence",
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot full-range E_theta and E_phi radial cuts in two stacked panels."""
    radius_m = config.case.radius_m
    fields_dir = results_dir / "cases" / case_name / "fields"
    sie_csv = fields_dir / "sie_physical_radial.csv"
    svwf_csv = fields_dir / "svwf_radial.csv"
    comsol_csv = fields_dir / "comsol_radial.csv"

    print(f"Loading SIE radial data: {sie_csv}")
    sie_data = load_field_csv(sie_csv)

    x_r, y_r, z_r = sie_data["x"], sie_data["y"], sie_data["z"]
    r_pts = np.sqrt(x_r**2 + y_r**2 + z_r**2)
    r_norm = r_pts / radius_m
    th_r = np.arccos(np.clip(z_r / r_pts, -1.0, 1.0))
    ph_r = np.arctan2(y_r, x_r)

    # 1. SIE spherical components
    Ex_s = sie_data["Ex_real"] + 1j * sie_data["Ex_imag"]
    Ey_s = sie_data["Ey_real"] + 1j * sie_data["Ey_imag"]
    Ez_s = sie_data["Ez_real"] + 1j * sie_data["Ez_imag"]
    E_th_sie = (
        Ex_s * np.cos(th_r) * np.cos(ph_r)
        + Ey_s * np.cos(th_r) * np.sin(ph_r)
        - Ez_s * np.sin(th_r)
    )
    E_ph_sie = -Ex_s * np.sin(ph_r) + Ey_s * np.cos(ph_r)

    # 2. Prescribed target fields
    transformation = config.transformations[0]
    ref_int = transformation.gstc.interior
    ref_ext = transformation.gstc.exterior
    E_int_pre, _ = evaluate_prescribed_fields(
        ref_int, "interior", config, radius=r_pts, theta=th_r, phi=ph_r
    )
    E_ext_pre, _ = evaluate_prescribed_fields(
        ref_ext, "exterior", config, radius=r_pts, theta=th_r, phi=ph_r
    )
    # Each component uses the prescribed field of its own region.
    interior = r_norm <= 1.0
    E_th_pre = np.where(interior, E_int_pre[1], E_ext_pre[1])
    E_ph_pre = np.where(interior, E_int_pre[2], E_ext_pre[2])

    # 3. SVWF field if available
    has_svwf = svwf_csv.is_file()
    if has_svwf:
        print(f"Loading SVWF radial data: {svwf_csv}")
        svwf_data = load_field_csv(svwf_csv)
        Ex_sv = svwf_data["Ex_real"] + 1j * svwf_data["Ex_imag"]
        Ey_sv = svwf_data["Ey_real"] + 1j * svwf_data["Ey_imag"]
        Ez_sv = svwf_data["Ez_real"] + 1j * svwf_data["Ez_imag"]
        E_th_svwf = (
            Ex_sv * np.cos(th_r) * np.cos(ph_r)
            + Ey_sv * np.cos(th_r) * np.sin(ph_r)
            - Ez_sv * np.sin(th_r)
        )
        E_ph_svwf = -Ex_sv * np.sin(ph_r) + Ey_sv * np.cos(ph_r)

    # 4. COMSOL field if available in the future
    has_comsol = comsol_csv.is_file()
    if has_comsol:
        print(f"Loading COMSOL radial data: {comsol_csv}")
        com_data = load_field_csv(comsol_csv)
        Ex_c = com_data["Ex_real"] + 1j * com_data["Ex_imag"]
        Ey_c = com_data["Ey_real"] + 1j * com_data["Ey_imag"]
        Ez_c = com_data["Ez_real"] + 1j * com_data["Ez_imag"]
        E_th_com = (
            Ex_c * np.cos(th_r) * np.cos(ph_r)
            + Ey_c * np.cos(th_r) * np.sin(ph_r)
            - Ez_c * np.sin(th_r)
        )
        E_ph_com = -Ex_c * np.sin(ph_r) + Ey_c * np.cos(ph_r)

    # Physical scaling: I Delta z = 1 A m
    manifest = {
        "frequency_hz": config.case.frequency_hz,
        "interior": {
            "epsilon_r": config.case.relative_permittivity_interior,
            "mu_r": config.case.relative_permeability_interior,
        },
        "source": {"effective_moment_interior": 1.0},
    }
    scale = budhu_moment_scale(manifest)

    # Limit the data as well as the axis so hidden near-source samples do not set the y range.
    visible = (r_norm >= 0.1) & (r_norm <= 2.0)
    r_plot = r_norm[visible]

    # One component per panel, each spanning the interior and exterior.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True, constrained_layout=True)
    components = (
        (E_th_pre, E_th_sie, E_th_svwf if has_svwf else None,
         E_th_com if has_comsol else None, r"$\mathrm{Re}(E_\theta)$ [$10^5$ V/m]", "(a) Theta component"),
        (E_ph_pre, E_ph_sie, E_ph_svwf if has_svwf else None,
         E_ph_com if has_comsol else None, r"$\mathrm{Re}(E_\phi)$ [$10^5$ V/m]", "(b) Phi component"),
    )
    for ax, (target, sie, svwf, comsol, ylabel, title) in zip(axes, components):
        ax.plot(r_plot, np.real(scale * target[visible]) / 1e5,
                color="black", linestyle="--", linewidth=1.0, label="Prescribed Target")
        ax.plot(r_plot, np.real(scale * sie[visible]) / 1e5,
                color="#d95f02", linestyle="none", marker="o", markevery=(0, 10),
                markersize=3.0, markerfacecolor="none", markeredgewidth=0.8, label="SIE-GSTC (MoM)")
        if has_svwf:
            ax.plot(r_plot, np.real(scale * svwf[visible]) / 1e5,
                    color="#1b9e77", linestyle="none", marker="^", markevery=(5, 24),
                    markersize=3.0, markerfacecolor="none", markeredgewidth=0.8, label="SVWF Collocation")
        if has_comsol:
            ax.plot(r_plot, np.real(comsol[visible]) / 1e5,
                    color="#0072B2", linestyle="-.", linewidth=0.8, label="COMSOL (FEM)")
        ax.axvline(1.0, color="#e41a1c", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=3)
        ax.set_xlim(0.1, 2.0)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel(r"Normalized Radial Distance $r/a$")
    fig.suptitle(r"Radial fields ($\theta = 45^\circ,\ \phi = 0^\circ$)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=6.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_2_part_a_radial_cut_transformation.pdf"
    png_path = output_dir / "figure_2_part_a_radial_cut_transformation.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Figure 2: {pdf_path}")
    print(f"Saved Figure 2: {png_path}")
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="Path to workflow configuration JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to run results root (default: %(default)s)",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="electric_incidence",
        help="Simulation case name (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated figures (default: %(default)s)",
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
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs (default: %(default)s)",
    )
    args = parser.parse_args()

    cfg = load_workflow_configuration(args.config)
    sie_zx = args.results_dir / "cases" / args.case / "fields" / "sie_physical_zx.csv"

    print("==================================================================")
    print(" Generating Publication Figures for Example 2, Part A")
    print(f" Configuration: {args.config}")
    print(f" Results Dir:   {args.results_dir}")
    print(f" Output Dir:    {args.output_dir}")
    print("==================================================================")

    # 1. Figure 1: 2D Field Maps (|E_theta| vs |E_phi|)
    plot_figure_1_2d_maps(
        sie_zx,
        args.output_dir,
        radius_m=cfg.case.radius_m,
        vmin_db=args.vmin_db,
        vmax_db=args.vmax_db,
        dpi=args.dpi,
    )

    # 2. Figure 2: 1D Radial Cut Transformation
    plot_figure_2_1d_radial_cut(
        args.results_dir,
        cfg,
        args.output_dir,
        case_name=args.case,
        dpi=args.dpi,
    )

    print("==================================================================")
    print(" All Example 2 Part A Publication Figures Successfully Generated!")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
