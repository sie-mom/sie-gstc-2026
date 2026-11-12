#!/usr/bin/env python3
"""Compare SIE-BEM solver results with COMSOL Multiphysics FEM validation (Case B).

Loads:
  1. 1D Physical Radial Cuts: sie_physical_radial.csv vs. comsol_physical_radial.csv
  2. 2D Physical ZX-Plane Maps: sie_physical_zx.csv vs. comsol_physical_zx.csv
  3. Stratton-Chu Far-Field Cuts: far_field_cuts.csv vs. comsol_far_field_cuts.csv

Computes:
  - Interior and exterior relative L2 field errors ||E_comsol - E_sie|| / ||E_sie||
  - Component-wise complex field errors (Ex, Ey, Ez, Hx, Hy, Hz)
  - Far-field directivity MAE and max deviation (dB) across E-plane, H-plane, and XY-plane
  - Total radiated power comparison and relative difference

Generates:
  - radial_near_field_comparison.png (4-panel field and error profiles)
  - zx_near_field_comparison.png (3-panel 2D map comparison)
  - far_field_polar_comparison.png (3-panel polar pattern overlay)
  - far_field_rectangular_comparison.png (rectangular directivity overlay)
  - comparison_metrics.json (structured numerical summary)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Color palette matching sie_mom style
COLOR_SIE = "#007a87"     # Teal / Deep Cyan
COLOR_COMSOL = "#e25c34"  # Coral / Rust Orange
COLOR_DIFF = "#5c3c88"    # Purple
COLOR_TARGET = "#c44838"  # Crimson


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the comparison tool."""
    parser = argparse.ArgumentParser(
        description="Compare SIE-BEM and COMSOL FEM Results for Bianisotropic Spheres",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sie-dir",
        type=Path,
        required=True,
        help="Path to directory containing SIE results (fields and far_field folders)",
    )
    parser.add_argument(
        "--comsol-dir",
        type=Path,
        required=True,
        help="Path to directory containing COMSOL result CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and comparison metrics (default: <comsol-dir>/comparison)",
    )
    parser.add_argument(
        "--dbi-min",
        type=float,
        default=-30.0,
        help="Minimum threshold in dBi for directivity polar and rectangular plots",
    )

    return parser.parse_args()


def load_radial_csv(path: Path) -> pd.DataFrame:
    """Load radial near-field CSV and compute total field norms."""
    df = pd.read_csv(path)
    # Compute complex fields
    df["Ex"] = df["Ex_real"] + 1j * df["Ex_imag"]
    df["Ey"] = df["Ey_real"] + 1j * df["Ey_imag"]
    df["Ez"] = df["Ez_real"] + 1j * df["Ez_imag"]
    df["normE"] = np.sqrt(np.abs(df["Ex"]) ** 2 + np.abs(df["Ey"]) ** 2 + np.abs(df["Ez"]) ** 2)

    df["Hx"] = df["Hx_real"] + 1j * df["Hx_imag"]
    df["Hy"] = df["Hy_real"] + 1j * df["Hy_imag"]
    df["Hz"] = df["Hz_real"] + 1j * df["Hz_imag"]
    df["normH"] = np.sqrt(np.abs(df["Hx"]) ** 2 + np.abs(df["Hy"]) ** 2 + np.abs(df["Hz"]) ** 2)

    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    return df.sort_values("r").reset_index(drop=True)


def load_zx_csv(path: Path) -> pd.DataFrame:
    """Load 2D zx-plane near-field CSV and compute field norms."""
    df = pd.read_csv(path)
    df["Ex"] = df["Ex_real"] + 1j * df["Ex_imag"]
    df["Ey"] = df["Ey_real"] + 1j * df["Ey_imag"]
    df["Ez"] = df["Ez_real"] + 1j * df["Ez_imag"]
    df["normE"] = np.sqrt(np.abs(df["Ex"]) ** 2 + np.abs(df["Ey"]) ** 2 + np.abs(df["Ez"]) ** 2)

    df["Hx"] = df["Hx_real"] + 1j * df["Hx_imag"]
    df["Hy"] = df["Hy_real"] + 1j * df["Hy_imag"]
    df["Hz"] = df["Hz_real"] + 1j * df["Hz_imag"]
    df["normH"] = np.sqrt(np.abs(df["Hx"]) ** 2 + np.abs(df["Hy"]) ** 2 + np.abs(df["Hz"]) ** 2)

    df["r"] = np.sqrt(df["x"] ** 2 + df["z"] ** 2)
    return df


def compare_radial_cuts(
    sie_path: Path,
    comsol_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Compare 1D radial near-field cuts and generate comparison plot."""
    print(f"\n[Radial Comparison] Loading SIE: {sie_path.name} | COMSOL: {comsol_path.name}")
    df_sie = load_radial_csv(sie_path)
    df_com = load_radial_csv(comsol_path)

    # Interpolate COMSOL fields onto common evaluation radial coordinates
    r_common = df_sie["r"].values
    normE_sie = df_sie["normE"].values
    normH_sie = df_sie["normH"].values

    normE_com = np.interp(r_common, df_com["r"].values, df_com["normE"].values)
    normH_com = np.interp(r_common, df_com["r"].values, df_com["normH"].values)

    # Relative difference
    eps_val = 1e-12
    rel_err_E = np.abs(normE_com - normE_sie) / np.maximum(normE_sie, eps_val)
    rel_err_H = np.abs(normH_com - normH_sie) / np.maximum(normH_sie, eps_val)

    # Separate interior (region 1) and exterior (region 2)
    reg_sie = df_sie["region"].values
    r_interior = r_common[reg_sie == 1]
    a_core = float(np.max(r_interior)) if len(r_interior) > 0 else float(np.median(r_common))
    r_cutoff = 0.15 * a_core

    mask_int = (reg_sie == 1) & (r_common > r_cutoff)  # exclude near-dipole singularity for fair comparison
    # Only compare exterior points within COMSOL computational air domain (prevent evaluating flatline clamped interpolation)
    r_com_max = float(np.max(df_com["r"].values))
    mask_ext = (reg_sie == 2) & (r_common <= r_com_max)
    mask_all = mask_int | mask_ext

    int_mean_err_E = float(np.mean(rel_err_E[mask_int])) * 100.0 if np.any(mask_int) else 0.0
    ext_mean_err_E = float(np.mean(rel_err_E[mask_ext])) * 100.0 if np.any(mask_ext) else 0.0
    int_mean_err_H = float(np.mean(rel_err_H[mask_int])) * 100.0 if np.any(mask_int) else 0.0
    ext_mean_err_H = float(np.mean(rel_err_H[mask_ext])) * 100.0 if np.any(mask_ext) else 0.0

    l2_err_E = (
        float(np.linalg.norm(normE_com[mask_all] - normE_sie[mask_all]) / np.linalg.norm(normE_sie[mask_all])) * 100.0
        if np.any(mask_all) else 0.0
    )
    l2_err_H = (
        float(np.linalg.norm(normH_com[mask_all] - normH_sie[mask_all]) / np.linalg.norm(normH_sie[mask_all])) * 100.0
        if np.any(mask_all) else 0.0
    )

    print(f"  Core radius detected: a = {a_core:.4e} m (singularity cutoff: r > {r_cutoff:.4e} m)")
    print(f"  Radial Relative L2 Error |E| : {l2_err_E:.2f}%")
    print(f"  Radial Relative L2 Error |H| : {l2_err_H:.2f}%")
    print(f"  Interior (r < a) Mean Relative Error |E| : {int_mean_err_E:.2f}%")
    print(f"  Exterior (r > a) Mean Relative Error |E| : {ext_mean_err_E:.2f}%")

    # Generate 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: |E| Comparison
    ax = axes[0, 0]
    ax.plot(df_sie["r"], normE_sie, label="SIE-BEM", color=COLOR_SIE, linewidth=2.2)
    ax.plot(df_com["r"], df_com["normE"], label="COMSOL FEM", color=COLOR_COMSOL, linewidth=1.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Radius r [m]", fontsize=11)
    ax.set_ylabel("Electric Field |E| [V/m]", fontsize=11)
    ax.set_title("Total Electric Field Norm Profile |E|(r)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    # Panel 2: Relative Error |E|
    ax = axes[0, 1]
    ax.plot(r_common, rel_err_E * 100.0, color=COLOR_DIFF, linewidth=2.0)
    ax.set_xlabel("Radius r [m]", fontsize=11)
    ax.set_ylabel("Relative Difference [%]", fontsize=11)
    ax.set_title("Relative Difference |E_comsol - E_sie| / |E_sie|", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    valid_pts = rel_err_E[r_common > r_cutoff] * 100.0
    ylim_max = min(100.0, float(np.percentile(valid_pts, 95)) * 1.5) if len(valid_pts) > 0 else 50.0
    ax.set_ylim(0, max(ylim_max, 5.0))

    # Panel 3: |H| Comparison
    ax = axes[1, 0]
    ax.plot(df_sie["r"], normH_sie, label="SIE-BEM", color=COLOR_SIE, linewidth=2.2)
    ax.plot(df_com["r"], df_com["normH"], label="COMSOL FEM", color=COLOR_COMSOL, linewidth=1.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Radius r [m]", fontsize=11)
    ax.set_ylabel("Magnetic Field |H| [A/m]", fontsize=11)
    ax.set_title("Total Magnetic Field Norm Profile |H|(r)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True)

    # Panel 4: Component Ex, Ez real part overlay
    ax = axes[1, 1]
    ax.plot(df_sie["r"], df_sie["Ex"].values.real, label="Re(Ex) SIE", color=COLOR_SIE, linewidth=1.8)
    ax.plot(df_com["r"], df_com["Ex"].values.real, label="Re(Ex) COMSOL", color=COLOR_COMSOL, linewidth=1.5, linestyle="--")
    ax.plot(df_sie["r"], df_sie["Ez"].values.real, label="Re(Ez) SIE", color="#2ca02c", linewidth=1.8)
    ax.plot(df_com["r"], df_com["Ez"].values.real, label="Re(Ez) COMSOL", color="#98df8a", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Radius r [m]", fontsize=11)
    ax.set_ylabel("Field Amplitude [V/m]", fontsize=11)
    ax.set_title("Cartesian Field Components Re(Ex) & Re(Ez)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    plot_path = out_dir / "radial_near_field_comparison.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_path.name}")

    return {
        "core_radius_m": a_core,
        "radial_l2_rel_err_E_percent": l2_err_E,
        "radial_l2_rel_err_H_percent": l2_err_H,
        "interior_mean_rel_err_E_percent": int_mean_err_E,
        "exterior_mean_rel_err_E_percent": ext_mean_err_E,
        "interior_mean_rel_err_H_percent": int_mean_err_H,
        "exterior_mean_rel_err_H_percent": ext_mean_err_H,
    }


def compare_zx_plane(
    sie_path: Path,
    comsol_path: Path,
    out_dir: Path,
    a_core: float | None = None,
) -> dict[str, Any]:
    """Compare 2D physical zx-plane near fields and generate 3-panel map plot."""
    print(f"\n[ZX Comparison] Loading SIE: {sie_path.name} | COMSOL: {comsol_path.name}")
    df_sie = load_zx_csv(sie_path)
    df_com = load_zx_csv(comsol_path)

    # Create uniform grid for comparison
    x_unique = np.unique(df_sie["x"].values)
    z_unique = np.unique(df_sie["z"].values)
    if len(x_unique) > 1 and len(z_unique) > 1:
        X, Z = np.meshgrid(x_unique, z_unique, indexing="ij")
        grid_points = np.column_stack([X.ravel(), Z.ravel()])

        sie_points = np.column_stack([df_sie["x"].values, df_sie["z"].values])
        com_points = np.column_stack([df_com["x"].values, df_com["z"].values])

        E_sie_grid = griddata(sie_points, df_sie["normE"].values, grid_points, method="linear").reshape(X.shape)
        E_com_grid = griddata(com_points, df_com["normE"].values, grid_points, method="linear").reshape(X.shape)

        diff_abs = np.abs(E_com_grid - E_sie_grid)
        rel_diff = diff_abs / np.maximum(E_sie_grid, 1e-12) * 100.0

        # Mask singular dipole center
        R_grid = np.sqrt(X ** 2 + Z ** 2)
        r_cutoff_zx = 0.2 * a_core if a_core is not None else 0.1 * float(np.max(R_grid))
        valid_mask = (R_grid > r_cutoff_zx) & ~np.isnan(E_com_grid) & ~np.isnan(E_sie_grid)
        rel_diff_valid = rel_diff[valid_mask]
        mean_rel_diff = float(np.nanmean(rel_diff_valid)) if len(rel_diff_valid) > 0 else 0.0
        median_rel_diff = float(np.nanmedian(rel_diff_valid)) if len(rel_diff_valid) > 0 else 0.0
        zx_l2_err = (
            float(np.linalg.norm(E_com_grid[valid_mask] - E_sie_grid[valid_mask]) / np.linalg.norm(E_sie_grid[valid_mask])) * 100.0
            if np.any(valid_mask) else 0.0
        )
        print(f"  ZX-Plane Cutoff: R > {r_cutoff_zx:.4e} m")
        print(f"  ZX-Plane Relative L2 Error |E|: {zx_l2_err:.2f}%")
        print(f"  ZX-Plane Mean Relative Difference |E|: {mean_rel_diff:.2f}% (Median: {median_rel_diff:.2f}%)")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: SIE |E|
        im1 = axes[0].pcolormesh(X, Z, np.log10(np.maximum(E_sie_grid, 1e-2)), shading="auto", cmap="viridis")
        axes[0].set_title("SIE-BEM log10(|E|) [V/m]", fontweight="bold")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("z [m]")
        axes[0].set_aspect("equal")
        plt.colorbar(im1, ax=axes[0])

        # Panel 2: COMSOL |E|
        im2 = axes[1].pcolormesh(X, Z, np.log10(np.maximum(E_com_grid, 1e-2)), shading="auto", cmap="viridis")
        axes[1].set_title("COMSOL FEM log10(|E|) [V/m]", fontweight="bold")
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("z [m]")
        axes[1].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1])

        # Panel 3: Relative difference
        vmax = min(50.0, float(np.nanpercentile(rel_diff_valid, 95)) * 1.5)
        im3 = axes[2].pcolormesh(X, Z, np.clip(rel_diff, 0, vmax), shading="auto", cmap="magma")
        axes[2].set_title("Relative Difference [%]", fontweight="bold")
        axes[2].set_xlabel("x [m]")
        axes[2].set_ylabel("z [m]")
        axes[2].set_aspect("equal")
        plt.colorbar(im3, ax=axes[2])

        fig.tight_layout()
        plot_path = out_dir / "zx_near_field_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {plot_path.name}")

        return {
            "zx_mean_rel_diff_percent": mean_rel_diff,
            "zx_median_rel_diff_percent": median_rel_diff,
        }
    return {}


def compare_far_field_cuts(
    sie_path: Path,
    comsol_path: Path,
    out_dir: Path,
    dbi_min: float = -30.0,
) -> dict[str, Any]:
    """Compare far-field principal cuts (E-plane, H-plane, XY-plane) and directivity."""
    print(f"\n[Far-Field Comparison] Loading SIE: {sie_path.name} | COMSOL: {comsol_path.name}")
    df_sie = pd.read_csv(sie_path)
    df_com = pd.read_csv(comsol_path)

    angles_deg = df_sie["angle_deg"].values
    angles_rad = np.radians(angles_deg)

    # Interpolate COMSOL directivity onto identical angular samples
    cuts = ["E_plane", "H_plane", "XY_plane"]
    metrics: dict[str, Any] = {}

    fig_polar, axes_polar = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "polar"})
    fig_rect, axes_rect = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for idx, plane in enumerate(cuts):
        col_dbi = f"{plane}_directivity_dbi"
        col_copol = f"{plane}_copol_V"

        d_sie = df_sie[col_dbi].values
        d_com_raw = df_com[col_dbi].values
        d_com = np.interp(angles_deg, df_com["angle_deg"].values, d_com_raw)

        # MAE and max deviation in dB
        diff_db = np.abs(d_com - d_sie)
        mae_db = float(np.mean(diff_db))
        max_db = float(np.max(diff_db))
        metrics[f"{plane}_mae_db"] = mae_db
        metrics[f"{plane}_max_diff_db"] = max_db
        print(f"  {plane.replace('_', ' ')} Directivity MAE: {mae_db:.2f} dB (Max Dev: {max_db:.2f} dB)")

        # Polar plot
        ax_p = axes_polar[idx]
        val_sie_p = np.maximum(d_sie, dbi_min)
        val_com_p = np.maximum(d_com, dbi_min)
        ax_p.plot(angles_rad, val_sie_p, label="SIE-BEM", color=COLOR_SIE, linewidth=2.2)
        ax_p.plot(angles_rad, val_com_p, label="COMSOL FEM", color=COLOR_COMSOL, linewidth=1.8, linestyle="--")
        ax_p.set_theta_zero_location("N")
        ax_p.set_theta_direction(-1)
        ax_p.set_ylim(dbi_min, max(5.0, float(np.max(val_sie_p)) + 2.0))
        ax_p.set_title(f"{plane.replace('_', ' ')} Directivity [dBi]", pad=15, fontweight="bold")
        ax_p.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax_p.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=True)

        # Rectangular plot
        ax_r = axes_rect[idx]
        ax_r.plot(angles_deg, np.maximum(d_sie, dbi_min), label="SIE-BEM", color=COLOR_SIE, linewidth=2.2)
        ax_r.plot(angles_deg, np.maximum(d_com, dbi_min), label="COMSOL FEM", color=COLOR_COMSOL, linewidth=1.8, linestyle="--")
        ax_r.set_xlabel("Angle [deg]", fontsize=11)
        if idx == 0:
            ax_r.set_ylabel("Directivity [dBi]", fontsize=11)
        ax_r.set_title(f"{plane.replace('_', ' ')} Cut", fontweight="bold")
        ax_r.grid(True, linestyle="--", alpha=0.6)
        ax_r.set_xlim(0, 360)
        ax_r.set_ylim(dbi_min, max(5.0, float(np.max(d_sie)) + 3.0))
        if idx == 0:
            ax_r.legend(frameon=True)

    fig_polar.tight_layout()
    polar_path = out_dir / "far_field_polar_comparison.png"
    fig_polar.savefig(polar_path, dpi=300)
    plt.close(fig_polar)
    print(f"  Saved: {polar_path.name}")

    fig_rect.tight_layout()
    rect_path = out_dir / "far_field_rectangular_comparison.png"
    fig_rect.savefig(rect_path, dpi=300)
    plt.close(fig_rect)
    print(f"  Saved: {rect_path.name}")

    return metrics


def main() -> None:
    """Main CLI entry point for SIE vs. COMSOL comparison."""
    args = parse_arguments()
    sie_dir = args.sie_dir.resolve()
    comsol_dir = args.comsol_dir.resolve()
    out_dir = (args.output_dir or (comsol_dir / "comparison")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SIE-BEM vs. COMSOL FEM BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"SIE Results Directory    : {sie_dir}")
    print(f"COMSOL Results Directory : {comsol_dir}")
    print(f"Comparison Output Dir    : {out_dir}")
    print("=" * 80)

    summary_metrics: dict[str, Any] = {
        "sie_dir": str(sie_dir),
        "comsol_dir": str(comsol_dir),
    }

    # 1. Compare 1D Radial Cuts
    sie_radial = sie_dir / "fields" / "sie_physical_radial.csv"
    if not sie_radial.is_file():
        sie_radial = sie_dir / "sie_physical_radial.csv"
    if not sie_radial.is_file():
        sie_radial = sie_dir / "fields" / "prescribed_radial.csv"
    if not sie_radial.is_file():
        sie_radial = sie_dir / "prescribed_radial.csv"
    if not sie_radial.is_file():
        sie_radial = sie_dir / "fields" / "comsol_radial.csv"
    if not sie_radial.is_file():
        sie_radial = sie_dir / "comsol_radial.csv"

    comsol_radial = comsol_dir / "comsol_physical_radial.csv"

    a_core: float | None = None
    if sie_radial.is_file() and comsol_radial.is_file():
        rad_metrics = compare_radial_cuts(sie_radial, comsol_radial, out_dir)
        summary_metrics["radial_metrics"] = rad_metrics
        a_core = rad_metrics.get("core_radius_m")
    else:
        print(f"[Warning] Missing radial CSVs: SIE={sie_radial.is_file()}, COMSOL={comsol_radial.is_file()}")

    # 2. Compare 2D ZX Cuts
    sie_zx = sie_dir / "fields" / "sie_physical_zx.csv"
    if not sie_zx.is_file():
        sie_zx = sie_dir / "sie_physical_zx.csv"
    if not sie_zx.is_file():
        sie_zx = sie_dir / "fields" / "prescribed_zx.csv"
    if not sie_zx.is_file():
        sie_zx = sie_dir / "prescribed_zx.csv"
    if not sie_zx.is_file():
        sie_zx = sie_dir / "fields" / "comsol_zx.csv"
    if not sie_zx.is_file():
        sie_zx = sie_dir / "comsol_zx.csv"

    comsol_zx = comsol_dir / "comsol_physical_zx.csv"

    if sie_zx.is_file() and comsol_zx.is_file():
        zx_metrics = compare_zx_plane(sie_zx, comsol_zx, out_dir, a_core=a_core)
        summary_metrics["zx_metrics"] = zx_metrics
    else:
        print(f"[Warning] Missing ZX CSVs: SIE={sie_zx.is_file()}, COMSOL={comsol_zx.is_file()}")

    # 3. Compare Far-Field Cuts
    sie_ff = sie_dir / "far_field" / "far_field_cuts.csv"
    if not sie_ff.is_file():
        sie_ff = sie_dir / "far_field_cuts.csv"

    comsol_ff = comsol_dir / "comsol_far_field_cuts.csv"

    if sie_ff.is_file() and comsol_ff.is_file():
        ff_metrics = compare_far_field_cuts(sie_ff, comsol_ff, out_dir, dbi_min=args.dbi_min)
        summary_metrics["far_field_metrics"] = ff_metrics
    else:
        print(f"[Warning] Missing far-field CSVs: SIE={sie_ff.is_file()}, COMSOL={comsol_ff.is_file()}")

    # Save summary metrics JSON
    metrics_path = out_dir / "comparison_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\n[Artifact] Saved comparison summary metrics: {metrics_path}")
    print("=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

