#!/usr/bin/env python3
"""Generate publication-quality figures for Example 2 Part B (Dual Polarization Rotator).

Case:
  Dual Complementary Polarization Rotation Metasurface
  Transformations:
    1. electric_to_magnetic: Electric Dipole (interior) -> Magnetic Radiation (exterior)
    2. magnetic_to_electric: Magnetic Dipole (interior) -> Electric Radiation (exterior)
  Frequency: 10 GHz, Interior: eps_r=7, Exterior: eps_r=1, Radius: 0.7 lambda_0.

Generates:
  1. 2D Field Maps:
     - figure_1_part_b_case1_electric_incidence_2d_field_maps.pdf / .png
     - figure_1_part_b_case2_magnetic_incidence_2d_field_maps.pdf / .png
     - figure_1_part_b_combined_2d_field_maps.pdf / .png (4-panel dual comparison)
  2. 1D Radial Cuts:
     - figure_2_part_b_case1_electric_incidence_radial_cut.pdf / .png
     - figure_2_part_b_case2_magnetic_incidence_radial_cut.pdf / .png
     - figure_2_part_b_radial_cut_transformation.pdf / .png (2-panel dual comparison)

Designed for IEEE journal formatting (3.5 in single column, 7.1 in double column).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
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
    / "dual_pol_eps7.json"
)
_RESULTS_CANDIDATE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "dipole_pol_rot"
    / "dual_pol_eps7"
)
DEFAULT_RESULTS_DIR = (
    _RESULTS_CANDIDATE
    if _RESULTS_CANDIDATE.exists()
    else (
        Path(__file__).resolve().parents[1]
        / "results"
        / "dipole_pol_rot"
        / "dual_pol_rot_10ghz_eps7_reciprocal_lossless"
    )
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def _process_2d_zx(sie_zx_csv: Path, grid_size: int = 200, radius_m: float = 0.02098547):
    """Load and process planar field map into spherical components on right half-plane."""
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

    x_mask = x >= -1e-12
    x_crop = x[x_mask] / radius_m
    z_crop = z / radius_m
    E_th_crop = E_th_mag[:, x_mask]
    E_ph_crop = E_ph_mag[:, x_mask]
    return x_crop, z_crop, E_th_crop, E_ph_crop


def plot_single_case_2d_maps(
    sie_zx_csv: Path,
    output_dir: Path,
    radius_m: float,
    case_name: str,
    *,
    grid_size: int = 200,
    vmin_db: float = -112.0,
    vmax_db: float = -58.0,
    norm_mode: str = "origin",
    cmap: str = "viridis",
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot 2D field maps for a single simulation case (side-by-side single column)."""
    print(f"Loading 2D ZX field data for {case_name}: {sie_zx_csv}")
    x_crop, z_crop, E_th_crop, E_ph_crop = _process_2d_zx(sie_zx_csv, grid_size, radius_m)

    if norm_mode == "radiated":
        if case_name == "electric_incidence":
            ref_rad = np.max(E_ph_crop)
            data_left = 20.0 * np.log10(np.maximum(E_th_crop / np.max(E_th_crop), 1e-15))
            data_right = 20.0 * np.log10(np.maximum(E_ph_crop / ref_rad, 1e-15))
            title_left = r"(a) $|E_\theta|$ [TM mode]"
            title_right = r"(b) $|E_\phi|$ [TE mode]"
            arrow_color = "red"
        else:  # magnetic_incidence
            ref_rad = np.max(E_th_crop)
            data_left = 20.0 * np.log10(np.maximum(E_ph_crop / np.max(E_ph_crop), 1e-15))
            data_right = 20.0 * np.log10(np.maximum(E_th_crop / ref_rad, 1e-15))
            title_left = r"(a) $|E_\phi|$ [TE mode]"
            title_right = r"(b) $|E_\theta|$ [TM mode]"
            arrow_color = "blue"
        cbar_label = r"Normalized Field [dB] (Peak = $0\,\mathrm{dB}$)"
    else:
        if case_name == "electric_incidence":
            ref_peak = np.max(E_th_crop)
            data_left = 20.0 * np.log10(np.maximum(E_th_crop / ref_peak, 1e-15))
            data_right = 20.0 * np.log10(np.maximum(E_ph_crop / ref_peak, 1e-15))
            title_left = r"(a) $|E_\theta|$ [TM mode]"
            title_right = r"(b) $|E_\phi|$ [TE mode]"
            arrow_color = "red"
        else:  # magnetic_incidence
            ref_peak = np.max(E_ph_crop)
            data_left = 20.0 * np.log10(np.maximum(E_ph_crop / ref_peak, 1e-15))
            data_right = 20.0 * np.log10(np.maximum(E_th_crop / ref_peak, 1e-15))
            title_left = r"(a) $|E_\phi|$ [TE mode]"
            title_right = r"(b) $|E_\theta|$ [TM mode]"
            arrow_color = "blue"
        cbar_label = r"Normalized Field Magnitude [dB]"

    norm = Normalize(vmin=vmin_db, vmax=vmax_db)

    fig, axes = plt.subplots(
        1, 2, figsize=(3.5, 3.2), sharex=True, sharey=True, constrained_layout=True
    )

    mesh_artists = []
    for ax, data_db, title in zip(axes, (data_left, data_right), (title_left, title_right)):
        mesh = ax.pcolormesh(x_crop, z_crop, data_db, shading="auto", cmap=cmap, norm=norm)
        mesh_artists.append(mesh)

        theta_arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.plot(np.cos(theta_arc), np.sin(theta_arc), color="white", linestyle="--", linewidth=0.9, alpha=0.9)
        ax.annotate("", xy=(0.0, 0.22), xytext=(0.0, -0.22), arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.3))

        ax.set_aspect("equal", "box")
        ax.set_title(title, pad=3)
        ax.set_xlabel(r"$x/a$")
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-2.0, 2.0)

    axes[0].set_ylabel(r"$z/a$")
    cbar = fig.colorbar(mesh_artists[1], ax=axes, orientation="horizontal", fraction=0.06, pad=0.07, shrink=0.85)
    cbar.set_label(cbar_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"figure_1_part_b_{case_name}"
    pdf_path = output_dir / f"{prefix}_2d_field_maps.pdf"
    png_path = output_dir / f"{prefix}_2d_field_maps.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {prefix}: {pdf_path}")
    print(f"Saved {prefix}: {png_path}")
    return pdf_path, png_path


def plot_combined_4panel_2d_maps(
    csv_case1: Path,
    csv_case2: Path,
    output_dir: Path,
    radius_m: float,
    *,
    grid_size: int = 200,
    vmin_db: float = -112.0,
    vmax_db: float = -58.0,
    norm_mode: str = "origin",
    cmap: str = "viridis",
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot comprehensive 4-panel (2x2) 2D field maps proving full electromagnetic duality."""
    x1, z1, Eth1, Eph1 = _process_2d_zx(csv_case1, grid_size, radius_m)
    x2, z2, Eth2, Eph2 = _process_2d_zx(csv_case2, grid_size, radius_m)

    if norm_mode == "radiated":
        db_1_th = 20.0 * np.log10(np.maximum(Eth1 / np.max(Eth1), 1e-15))
        db_1_ph = 20.0 * np.log10(np.maximum(Eph1 / np.max(Eph1), 1e-15))
        db_2_ph = 20.0 * np.log10(np.maximum(Eph2 / np.max(Eph2), 1e-15))
        db_2_th = 20.0 * np.log10(np.maximum(Eth2 / np.max(Eth2), 1e-15))
        cbar_label = r"Normalized Field [dB] (Peak = $0\,\mathrm{dB}$)"
    else:
        ref1 = np.max(Eth1)
        ref2 = np.max(Eph2)

        db_1_th = 20.0 * np.log10(np.maximum(Eth1 / ref1, 1e-15))
        db_1_ph = 20.0 * np.log10(np.maximum(Eph1 / ref1, 1e-15))
        db_2_ph_raw = 20.0 * np.log10(np.maximum(Eph2 / ref2, 1e-15))
        db_2_th_raw = 20.0 * np.log10(np.maximum(Eth2 / ref2, 1e-15))

        # Align Case 2 exterior radiated peak to Case 1 to avoid artificial yellow saturation
        # and provide a clean, shared colorbar spanning both cases simultaneously.
        shift = np.max(db_2_th_raw) - np.max(db_1_ph)
        db_2_ph = db_2_ph_raw - shift
        db_2_th = db_2_th_raw - shift
        cbar_label = r"Normalized Field Magnitude [dB]"

    norm = Normalize(vmin=vmin_db, vmax=vmax_db)

    fig, axes = plt.subplots(
        2, 2, figsize=(4.8, 5.2), sharex=True, sharey=True, constrained_layout=True
    )

    panels = [
        (axes[0, 0], db_1_th, r"(a) Electric Inc.: $|E_\theta|$ [TM]", "red"),
        (axes[0, 1], db_1_ph, r"(b) Electric Inc.: $|E_\phi|$ [TE]", "red"),
        (axes[1, 0], db_2_ph, r"(c) Magnetic Inc.: $|E_\phi|$ [TE]", "blue"),
        (axes[1, 1], db_2_th, r"(d) Magnetic Inc.: $|E_\theta|$ [TM]", "blue"),
    ]

    theta_arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
    for ax, data_db, title, arr_color in panels:
        im = ax.pcolormesh(x1, z1, data_db, shading="auto", cmap=cmap, norm=norm)
        ax.plot(np.cos(theta_arc), np.sin(theta_arc), color="white", linestyle="--", linewidth=0.9, alpha=0.9)
        ax.annotate("", xy=(0.0, 0.22), xytext=(0.0, -0.22), arrowprops=dict(arrowstyle="->", color=arr_color, lw=1.3))
        ax.set_aspect("equal", "box")
        ax.set_title(title, pad=3)
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(-2.0, 2.0)

    axes[0, 0].set_ylabel(r"$z/a$")
    axes[1, 0].set_ylabel(r"$z/a$")
    axes[1, 0].set_xlabel(r"$x/a$")
    axes[1, 1].set_xlabel(r"$x/a$")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.7)
    cbar.set_label(cbar_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_1_part_b_combined_2d_field_maps.pdf"
    png_path = output_dir / "figure_1_part_b_combined_2d_field_maps.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Combined Figure 1: {pdf_path}")
    print(f"Saved Combined Figure 1: {png_path}")
    return pdf_path, png_path


def _extract_1d_radial_curves(results_dir: Path, config: object, case_name: str):
    """Extract Target, SIE, and SVWF 1D radial curves for a given case."""
    radius_m = config.case.radius_m
    fields_dir = results_dir / "cases" / case_name / "fields"
    sie_csv = fields_dir / "sie_physical_radial.csv"
    svwf_csv = fields_dir / "svwf_radial.csv"
    comsol_csv = fields_dir / "comsol_radial.csv"

    sie_data = load_field_csv(sie_csv)
    x_sie = sie_data["x"]
    y_sie = sie_data["y"]
    z_sie = sie_data["z"]
    r_sie = np.sqrt(x_sie**2 + y_sie**2 + z_sie**2)
    th_sie = np.arccos(np.clip(z_sie / r_sie, -1.0, 1.0))
    ph_sie = np.arctan2(y_sie, x_sie)
    r_norm = r_sie / radius_m

    manifest = {
        "frequency_hz": config.case.frequency_hz,
        "interior": {
            "epsilon_r": config.case.relative_permittivity_interior,
            "mu_r": config.case.relative_permeability_interior,
        },
        "source": {"effective_moment_interior": 1.0},
    }
    scale = budhu_moment_scale(manifest)

    Ex_sie = (sie_data["Ex_real"] + 1j * sie_data["Ex_imag"]) * scale
    Ey_sie = (sie_data["Ey_real"] + 1j * sie_data["Ey_imag"]) * scale
    Ez_sie = (sie_data["Ez_real"] + 1j * sie_data["Ez_imag"]) * scale

    # Spherical components
    Eth_sie = Ex_sie * np.cos(th_sie) * np.cos(ph_sie) + Ey_sie * np.cos(th_sie) * np.sin(ph_sie) - Ez_sie * np.sin(th_sie)
    Eph_sie = -Ex_sie * np.sin(ph_sie) + Ey_sie * np.cos(ph_sie)

    # SVWF
    has_svwf = svwf_csv.is_file()
    Eth_svwf = None
    Eph_svwf = None
    if has_svwf:
        svwf_data = load_field_csv(svwf_csv)
        Ex_sv = (svwf_data["Ex_real"] + 1j * svwf_data["Ex_imag"]) * scale
        Ey_sv = (svwf_data["Ey_real"] + 1j * svwf_data["Ey_imag"]) * scale
        Ez_sv = (svwf_data["Ez_real"] + 1j * svwf_data["Ez_imag"]) * scale
        Eth_svwf = Ex_sv * np.cos(th_sie) * np.cos(ph_sie) + Ey_sv * np.cos(th_sie) * np.sin(ph_sie) - Ez_sv * np.sin(th_sie)
        Eph_svwf = -Ex_sv * np.sin(ph_sie) + Ey_sv * np.cos(ph_sie)

    # COMSOL
    has_comsol = comsol_csv.is_file()
    Eth_comsol = None
    Eph_comsol = None
    r_norm_comsol = None
    if has_comsol:
        comsol_data = load_field_csv(comsol_csv)
        r_c = np.sqrt(comsol_data["x"]**2 + comsol_data["y"]**2 + comsol_data["z"]**2)
        th_c = np.arccos(np.clip(comsol_data["z"] / r_c, -1.0, 1.0))
        ph_c = np.arctan2(comsol_data["y"], comsol_data["x"])
        r_norm_comsol = r_c / radius_m
        Ex_c = (comsol_data["Ex_real"] + 1j * comsol_data["Ex_imag"]) * scale
        Ey_c = (comsol_data["Ey_real"] + 1j * comsol_data["Ey_imag"]) * scale
        Ez_c = (comsol_data["Ez_real"] + 1j * comsol_data["Ez_imag"]) * scale
        Eth_comsol = Ex_c * np.cos(th_c) * np.cos(ph_c) + Ey_c * np.cos(th_c) * np.sin(ph_c) - Ez_c * np.sin(th_c)
        Eph_comsol = -Ex_c * np.sin(ph_c) + Ey_c * np.cos(ph_c)

    # Target (Prescribed)
    int_mask = r_norm <= 1.0
    ext_mask = r_norm > 1.0

    if case_name == "electric_incidence":
        # Target: int is electric, ext is magnetic
        ref_int = config.fields["electric_z_interior"]
        ref_ext = config.fields["magnetic_z_exterior"]
    else:
        # Target: int is magnetic, ext is electric
        ref_int = config.fields["magnetic_z_interior"]
        ref_ext = config.fields["electric_z_exterior"]

    from sie_mom.excitation import evaluate_dipole_fields
    scale_tgt = scale

    # Interior target
    m_int = tuple(x.real for x in ref_int.effective_moment)
    m_ext = tuple(x.real for x in ref_ext.effective_moment)
    E_tgt_int, _ = evaluate_dipole_fields(
        config.case.k_interior, r_sie[int_mask], th_sie[int_mask], ph_sie[int_mask],
        config.case.eta_interior, effective_moment=m_int,
        dipole_type=ref_int.dipole_type,
    )
    # Exterior target
    E_tgt_ext, _ = evaluate_dipole_fields(
        config.case.k_exterior, r_sie[ext_mask], th_sie[ext_mask], ph_sie[ext_mask],
        config.case.eta_exterior, effective_moment=m_ext,
        dipole_type=ref_ext.dipole_type,
    )

    # Piecewise curve construction:
    # Case 1 (electric): int is Eth, ext is Eph
    # Case 2 (magnetic): int is Eph, ext is Eth
    target_curve = np.zeros(len(r_norm))
    sie_curve = np.zeros(len(r_norm))
    svwf_curve = np.zeros(len(r_norm)) if has_svwf else None
    comsol_curve = np.zeros(len(r_norm_comsol)) if has_comsol else None

    if case_name == "electric_incidence":
        target_curve[int_mask] = (E_tgt_int[1] * scale_tgt).real
        target_curve[ext_mask] = (E_tgt_ext[2] * scale_tgt).real
        sie_curve[int_mask] = Eth_sie[int_mask].real
        sie_curve[ext_mask] = Eph_sie[ext_mask].real
        if has_svwf:
            svwf_curve[int_mask] = Eth_svwf[int_mask].real
            svwf_curve[ext_mask] = Eph_svwf[ext_mask].real
        if has_comsol:
            c_int = r_norm_comsol <= 1.0
            c_ext = r_norm_comsol > 1.0
            comsol_curve[c_int] = Eth_comsol[c_int].real
            comsol_curve[c_ext] = Eph_comsol[c_ext].real
    else:
        target_curve[int_mask] = (E_tgt_int[2] * scale_tgt).real
        target_curve[ext_mask] = (E_tgt_ext[1] * scale_tgt).real
        sie_curve[int_mask] = Eph_sie[int_mask].real
        sie_curve[ext_mask] = Eth_sie[ext_mask].real
        if has_svwf:
            svwf_curve[int_mask] = Eph_svwf[int_mask].real
            svwf_curve[ext_mask] = Eth_svwf[ext_mask].real
        if has_comsol:
            c_int = r_norm_comsol <= 1.0
            c_ext = r_norm_comsol > 1.0
            comsol_curve[c_int] = Eph_comsol[c_int].real
            comsol_curve[c_ext] = Eth_comsol[c_ext].real

    return {
        "r_norm": r_norm,
        "target": target_curve,
        "sie": sie_curve,
        "svwf": svwf_curve,
        "comsol": comsol_curve,
        "r_norm_comsol": r_norm_comsol,
        "has_svwf": has_svwf,
        "has_comsol": has_comsol,
    }


def plot_single_case_1d_radial_cut(
    results_dir: Path,
    config: object,
    output_dir: Path,
    case_name: str,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot 1D radial cut piecewise transformation for one case."""
    curves = _extract_1d_radial_curves(results_dir, config, case_name)
    r_norm = curves["r_norm"]
    target = curves["target"]
    sie = curves["sie"]
    svwf = curves["svwf"]
    comsol = curves["comsol"]
    has_svwf = curves["has_svwf"]
    has_comsol = curves["has_comsol"]

    mask = r_norm >= 0.1
    r_v = r_norm[mask]
    target_v = target[mask]
    sie_v = sie[mask]
    svwf_v = svwf[mask] if has_svwf else None

    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)

    ax.plot(r_v, target_v / 1e5, color="black", linestyle="-", linewidth=1.2, label="Target (Prescribed)")
    ax.plot(r_v, sie_v / 1e5, color="#ff7f0e", linestyle="--", linewidth=1.1, label="SIE-GSTC (MoM)")
    if has_svwf:
        ax.plot(r_v, svwf_v / 1e5, color="#2ca02c", linestyle=":", linewidth=1.1, label="SVWF Collocation")
    if has_comsol:
        mask_c = curves["r_norm_comsol"] >= 0.1
        ax.plot(curves["r_norm_comsol"][mask_c], comsol[mask_c] / 1e5, color="#1f77b4", linestyle="-.", linewidth=1.0, label="COMSOL (FEM)")

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.8)

    if case_name == "electric_incidence":
        y_label = r"$\mathrm{Re}(E) \times 10^5$ [V/m]"
        badge_int = r"Interior: $\mathrm{Re}(E_\theta)$ [TM]"
        badge_ext = r"Exterior: $\mathrm{Re}(E_\phi)$ [TE]"
        ax.set_ylim(-8.0, 18.0)
    else:
        y_label = r"$\mathrm{Re}(E) \times 10^5$ [V/m]"
        badge_int = r"Interior: $\mathrm{Re}(E_\phi)$ [TE]"
        badge_ext = r"Exterior: $\mathrm{Re}(E_\theta)$ [TM]"
        ax.set_ylim(-30.0, 10.0)

    ax.text(0.55, 0.90, badge_int, transform=ax.transAxes, fontsize=6.5, color="#333333", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="none", alpha=0.8))
    ax.text(0.98, 0.90, badge_ext, transform=ax.transAxes, fontsize=6.5, color="#333333", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="none", alpha=0.8))

    ax.set_xlabel(r"Normalized Radius $r/a$")
    ax.set_ylabel(y_label)
    ax.set_xlim(0.1, 2.0)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"figure_2_part_b_{case_name}"
    pdf_path = output_dir / f"{prefix}_radial_cut.pdf"
    png_path = output_dir / f"{prefix}_radial_cut.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {prefix}: {pdf_path}")
    print(f"Saved {prefix}: {png_path}")
    return pdf_path, png_path


def plot_combined_2panel_1d_radial_cut(
    results_dir: Path,
    config: object,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot side-by-side 2-panel 1D radial cut comparison for both cases (3.5 in single column or 7.1 in double)."""
    curves1 = _extract_1d_radial_curves(results_dir, config, "electric_incidence")
    curves2 = _extract_1d_radial_curves(results_dir, config, "magnetic_incidence")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)

    cases_info = [
        (axes[0], curves1, r"(a) Case 1: Electric Inc. ($\mathbf{p} \to \mathbf{m}$)",
         r"Int: $\mathrm{Re}(E_\theta)$ [TM]", r"Ext: $\mathrm{Re}(E_\phi)$ [TE]", (-8.0, 18.0)),
        (axes[1], curves2, r"(b) Case 2: Magnetic Inc. ($\mathbf{m} \to \mathbf{p}$)",
         r"Int: $\mathrm{Re}(E_\phi)$ [TE]", r"Ext: $\mathrm{Re}(E_\theta)$ [TM]", (-30.0, 10.0)),
    ]

    for ax, c, title, badge_int, badge_ext, ylim in cases_info:
        mask = c["r_norm"] >= 0.1
        r_v = c["r_norm"][mask]
        ax.plot(r_v, c["target"][mask] / 1e5, color="black", linestyle="-", linewidth=1.2, label="Target (Prescribed)")
        ax.plot(r_v, c["sie"][mask] / 1e5, color="#ff7f0e", linestyle="--", linewidth=1.1, label="SIE-GSTC (MoM)")
        if c["has_svwf"]:
            ax.plot(r_v, c["svwf"][mask] / 1e5, color="#2ca02c", linestyle=":", linewidth=1.1, label="SVWF Collocation")
        if c["has_comsol"]:
            mask_c = c["r_norm_comsol"] >= 0.1
            ax.plot(c["r_norm_comsol"][mask_c], c["comsol"][mask_c] / 1e5, color="#1f77b4", linestyle="-.", linewidth=1.0, label="COMSOL (FEM)")

        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_title(title, pad=3)
        ax.set_xlabel(r"Normalized Radius $r/a$")
        ax.set_ylabel(r"$\mathrm{Re}(E) \times 10^5$ [V/m]")
        ax.set_xlim(0.1, 2.0)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle=":", alpha=0.6)

        ax.text(0.55, 0.90, badge_int, transform=ax.transAxes, fontsize=6.5, color="#333333", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="none", alpha=0.8))
        ax.text(0.98, 0.90, badge_ext, transform=ax.transAxes, fontsize=6.5, color="#333333", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="none", alpha=0.8))

    axes[1].legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=6.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure_2_part_b_radial_cut_transformation.pdf"
    png_path = output_dir / "figure_2_part_b_radial_cut_transformation.png"
    fig.savefig(pdf_path, dpi=dpi)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved Combined Figure 2: {pdf_path}")
    print(f"Saved Combined Figure 2: {png_path}")
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
        help="Path to results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to output figures directory (default: %(default)s)",
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
        help="Normalization reference: 'origin' (dipole core peak, legacy Part A style) or 'radiated' (exterior radiated peak = 0 dB) (default: %(default)s)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for 2D field maps (default: %(default)s, e.g. viridis, turbo, inferno, plasma)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution (default: %(default)s)",
    )
    args = parser.parse_args()

    # Automatically adjust defaults if radiated mode is selected
    if args.norm == "radiated":
        if args.vmin_db == -112.0 and args.vmax_db == -58.0:
            args.vmin_db = -30.0
            args.vmax_db = 0.0
        if args.mag_vmin_db == -75.0 and args.mag_vmax_db == -45.0:
            args.mag_vmin_db = -30.0
            args.mag_vmax_db = 0.0

    cfg = load_workflow_configuration(args.config)
    radius_m = cfg.case.radius_m

    print("==================================================================")
    print(" Generating Publication Figures for Example 2, Part B")
    print(f" Configuration: {args.config}")
    print(f" Results Dir:   {args.results_dir}")
    print(f" Output Dir:    {args.output_dir}")
    print(f" Norm Mode:     {args.norm}")
    print(f" Colormap:      {args.cmap}")
    print("==================================================================")

    csv_case1 = args.results_dir / "cases" / "electric_incidence" / "fields" / "sie_physical_zx.csv"
    csv_case2 = args.results_dir / "cases" / "magnetic_incidence" / "fields" / "sie_physical_zx.csv"

    # 1. 2D Field Maps
    if csv_case1.is_file():
        plot_single_case_2d_maps(csv_case1, args.output_dir, radius_m, "electric_incidence", vmin_db=args.vmin_db, vmax_db=args.vmax_db, norm_mode=args.norm, cmap=args.cmap, dpi=args.dpi)
    if csv_case2.is_file():
        plot_single_case_2d_maps(csv_case2, args.output_dir, radius_m, "magnetic_incidence", vmin_db=args.mag_vmin_db, vmax_db=args.mag_vmax_db, norm_mode=args.norm, cmap=args.cmap, dpi=args.dpi)
    if csv_case1.is_file() and csv_case2.is_file():
        plot_combined_4panel_2d_maps(csv_case1, csv_case2, args.output_dir, radius_m, vmin_db=args.vmin_db, vmax_db=args.vmax_db, norm_mode=args.norm, cmap=args.cmap, dpi=args.dpi)

    # 2. 1D Radial Cuts
    if (args.results_dir / "cases" / "electric_incidence" / "fields" / "sie_physical_radial.csv").is_file():
        plot_single_case_1d_radial_cut(args.results_dir, cfg, args.output_dir, "electric_incidence", dpi=args.dpi)
    if (args.results_dir / "cases" / "magnetic_incidence" / "fields" / "sie_physical_radial.csv").is_file():
        plot_single_case_1d_radial_cut(args.results_dir, cfg, args.output_dir, "magnetic_incidence", dpi=args.dpi)
    if (args.results_dir / "cases" / "electric_incidence" / "fields" / "sie_physical_radial.csv").is_file() and \
       (args.results_dir / "cases" / "magnetic_incidence" / "fields" / "sie_physical_radial.csv").is_file():
        plot_combined_2panel_1d_radial_cut(args.results_dir, cfg, args.output_dir, dpi=args.dpi)

    print("\n==================================================================")
    print(" All Example 2 Part B Publication Figures Successfully Generated!")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

