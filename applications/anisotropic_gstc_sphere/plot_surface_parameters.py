#!/usr/bin/env python3
"""Plot spherical GSTC metasurface constitutive parameters on an angular (theta, phi) grid.

Visualizes the physical surface parameter blocks across the spherical surface:
  1. Im(Y_ee) in S [Siemens] (electric sheet admittance)
  2. Im(Z_mm) in Ohm [Ohms] (magnetic sheet impedance)
  3. Re(chi) dimensionless (magneto-electric coupling tensor)
  4. Balanced map condition number kappa(M)

Can read directly from physical 32-complex tensor binaries (face_tensors.bin)
or 10-parameter normalized design vectors (thetas.bin).

Usage Examples:
  # Plot from physical 32-complex tensors (face_tensors.bin):
  .venv/bin/python applications/anisotropic_gstc_sphere/plot_surface_parameters.py \\
      --tensors applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7/face_tensors.bin \\
      --mesh applications/anisotropic_gstc_sphere/meshes/example_1_omega_matching_10ghz_eps7/sphere.obj \\
      --output-dir applications/anisotropic_gstc_sphere/results/dipole_pol_rot/e2m_eps7/parameter_plots \\
      --title-suffix "Electric to Magnetic (10 GHz, eps=7)" \\
      --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import griddata

from sie_mom.mesh.obj_mesh import load_triangle_obj

ETA_0 = 376.73031366857


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spherical GSTC metasurface constitutive parameters on a (theta, phi) map.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tensors",
        type=Path,
        help="Path to physical 32-complex tensors binary file (e.g. face_tensors.bin).",
    )
    group.add_argument(
        "--thetas",
        type=Path,
        help="Path to 10-parameter vector binary file (e.g. thetas.bin).",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to sphere OBJ mesh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated PNG plots and summary metrics.",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="Optional title suffix.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI (default: 160).",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const=True,
        default=False,
        help="Export per-face physical parameters and coordinates to CSV.",
    )
    return parser.parse_args()


def extract_thetas_from_tensors(records: np.ndarray, eta0: float = ETA_0) -> np.ndarray:
    """Extract normalized parameters (N_faces, 10) from physical tensors (N_faces, 32)."""
    num_faces = records.shape[0]
    thetas = np.zeros((num_faces, 10), dtype=np.float64)
    for m in range(num_faces):
        L = records[m, :16].reshape(4, 4)
        thetas[m, 0] = L[0, 0].imag * eta0
        thetas[m, 1] = L[1, 1].imag * eta0
        thetas[m, 2] = L[0, 1].imag * eta0
        thetas[m, 3] = L[0, 2].real
        thetas[m, 4] = L[0, 3].real
        thetas[m, 5] = L[1, 2].real
        thetas[m, 6] = L[1, 3].real
        thetas[m, 7] = L[2, 2].imag / eta0
        thetas[m, 8] = L[3, 3].imag / eta0
        thetas[m, 9] = L[2, 3].imag / eta0
    return thetas


def interpolate_to_grid(
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    n_theta: int = 181,
    n_phi: int = 361,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate unstructured face values to a regular (theta, phi) grid with periodic wrapping."""
    grid_phi_1d = np.linspace(0.0, 360.0, n_phi)
    grid_theta_1d = np.linspace(0.0, 180.0, n_theta)
    grid_phi, grid_theta = np.meshgrid(grid_phi_1d, grid_theta_1d)

    theta_deg = np.rad2deg(theta)
    phi_deg = np.rad2deg(phi)

    # Periodic padding across phi = 0 / 360 degrees
    phi_pad = np.concatenate([phi_deg - 360.0, phi_deg, phi_deg + 360.0])
    theta_pad = np.concatenate([theta_deg, theta_deg, theta_deg])
    val_pad = np.concatenate([values, values, values])

    grid_val = griddata((phi_pad, theta_pad), val_pad, (grid_phi, grid_theta), method="linear")
    grid_val_nearest = griddata((phi_pad, theta_pad), val_pad, (grid_phi, grid_theta), method="nearest")
    nan_mask = np.isnan(grid_val)
    grid_val[nan_mask] = grid_val_nearest[nan_mask]

    return grid_phi, grid_theta, grid_val


def compute_condition_numbers(thetas: np.ndarray) -> np.ndarray:
    """Compute 2-norm condition number of the dimensionless impedance-balanced 4x4 matrix."""
    num_faces = thetas.shape[0]
    conds = np.zeros(num_faces, dtype=np.float64)

    y1, y2, y3 = thetas[:, 0], thetas[:, 1], thetas[:, 2]
    c1, c2, c3, c4 = thetas[:, 3], thetas[:, 4], thetas[:, 5], thetas[:, 6]
    z1, z2, z3 = thetas[:, 7], thetas[:, 8], thetas[:, 9]

    for m in range(num_faces):
        M = np.array([
            [1j * y1[m], 1j * y3[m], c1[m], c2[m]],
            [1j * y3[m], 1j * y2[m], c3[m], c4[m]],
            [-c1[m], -c3[m], 1j * z1[m], 1j * z3[m]],
            [-c2[m], -c4[m], 1j * z3[m], 1j * z2[m]],
        ], dtype=np.complex128)
        s = np.linalg.svd(M, compute_uv=False)
        conds[m] = s[0] / max(s[-1], 1e-15)

    return conds


def plot_parameters(
    thetas_path: Path | None,
    tensors_path: Path | None,
    mesh_path: Path,
    output_dir: Path,
    title_suffix: str = "",
    dpi: int = 160,
    csv_export: bool | str | Path = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load mesh centroids and compute spherical coordinates
    mesh = load_triangle_obj(mesh_path)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    r = np.linalg.norm(centroids, axis=1)
    theta = np.arccos(np.clip(centroids[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(centroids[:, 1], centroids[:, 0]) % (2 * np.pi)

    # 2. Load parameters
    if thetas_path is not None:
        thetas = np.fromfile(thetas_path, dtype=np.float64).reshape(-1, 10)
    else:
        records = np.fromfile(tensors_path, dtype=np.complex128).reshape(-1, 32)
        thetas = extract_thetas_from_tensors(records)

    assert len(thetas) == len(theta), f"Mesh faces ({len(theta)}) != parameter rows ({len(thetas)})"

    # Compute physical components
    # Yee (imaginary, S)
    Y_tt = thetas[:, 0] / ETA_0
    Y_pp = thetas[:, 1] / ETA_0
    Y_tp = thetas[:, 2] / ETA_0

    # Chi (real, dimensionless)
    chi_tt = thetas[:, 3]
    chi_tp = thetas[:, 4]
    chi_pt = thetas[:, 5]
    chi_pp = thetas[:, 6]

    # Zmm (imaginary, Ohm)
    Z_tt = thetas[:, 7] * ETA_0
    Z_pp = thetas[:, 8] * ETA_0
    Z_tp = thetas[:, 9] * ETA_0

    # Condition number
    cond = compute_condition_numbers(thetas)

    title_tag = f" | {title_suffix}" if title_suffix else ""

    # =========================================================================
    # Figure 1: Im(Y_ee)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    y_components = [
        (Y_tt, r"$\operatorname{Im} Y_{\theta\theta}$"),
        (Y_tp, r"$\operatorname{Im} Y_{\theta\phi}$"),
        (Y_pp, r"$\operatorname{Im} Y_{\phi\phi}$"),
    ]
    y_limit = max(np.percentile(np.abs(comp), 98) for comp, _ in y_components)
    if y_limit == 0 or not np.isfinite(y_limit):
        y_limit = 1.0
    for ax, (comp, name) in zip(axes, y_components):
        gphi, gtheta, gval = interpolate_to_grid(theta, phi, comp)
        im = ax.pcolormesh(
            gphi, gtheta, gval, cmap="RdBu_r", shading="auto",
            vmin=-y_limit, vmax=y_limit,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_xlim(0, 360)
        ax.set_ylim(180, 0)
    fig.colorbar(im, ax=list(axes), shrink=0.85, label=r"$\operatorname{Im} Y$ [S]")
    fig.suptitle(f"Electric Surface Admittance Block $Y_{{ee}}${title_tag}", fontsize=14)
    fig.savefig(output_dir / "Y_imaginary.pdf")
    fig.savefig(output_dir / "Y_imaginary.png", dpi=dpi)
    plt.close(fig)
    print(f"[Plot] Saved: Y_imaginary.png and Y_imaginary.pdf")

    # =========================================================================
    # Figure 2: Im(Z_mm)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    z_components = [
        (Z_tt, r"$\operatorname{Im} Z_{\theta\theta}$"),
        (Z_tp, r"$\operatorname{Im} Z_{\theta\phi}$"),
        (Z_pp, r"$\operatorname{Im} Z_{\phi\phi}$"),
    ]
    z_limit = max(np.percentile(np.abs(comp), 98) for comp, _ in z_components)
    if z_limit == 0 or not np.isfinite(z_limit):
        z_limit = 1.0
    for ax, (comp, name) in zip(axes, z_components):
        gphi, gtheta, gval = interpolate_to_grid(theta, phi, comp)
        im = ax.pcolormesh(
            gphi, gtheta, gval, cmap="RdBu_r", shading="auto",
            vmin=-z_limit, vmax=z_limit,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_xlim(0, 360)
        ax.set_ylim(180, 0)
    fig.colorbar(im, ax=list(axes), shrink=0.85, label=r"$\operatorname{Im} Z$ [$\Omega$]")
    fig.suptitle(f"Magnetic Surface Impedance Block $Z_{{mm}}${title_tag}", fontsize=14)
    fig.savefig(output_dir / "Z_imaginary.pdf")
    fig.savefig(output_dir / "Z_imaginary.png", dpi=dpi)
    plt.close(fig)
    print(f"[Plot] Saved: Z_imaginary.png and Z_imaginary.pdf")

    # =========================================================================
    # Figure 3: Re(chi)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    chi_components = [
        (chi_tt, r"$\operatorname{Re} \chi_{\theta\theta}$"),
        (chi_tp, r"$\operatorname{Re} \chi_{\theta\phi}$"),
        (chi_pt, r"$\operatorname{Re} \chi_{\phi\theta}$"),
        (chi_pp, r"$\operatorname{Re} \chi_{\phi\phi}$"),
    ]
    chi_limit = max(np.percentile(np.abs(comp), 98) for comp, _ in chi_components)
    if chi_limit == 0 or not np.isfinite(chi_limit):
        chi_limit = 1.0
    for ax, (comp, name) in zip(axes.ravel(), chi_components):
        gphi, gtheta, gval = interpolate_to_grid(theta, phi, comp)
        im = ax.pcolormesh(
            gphi, gtheta, gval, cmap="RdBu_r", shading="auto",
            vmin=-chi_limit, vmax=chi_limit,
        )
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(r"$\phi$ [deg]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_xlim(0, 360)
        ax.set_ylim(180, 0)
    fig.colorbar(im, ax=list(axes.ravel()), shrink=0.85, label=r"$\operatorname{Re} \chi$ [dimensionless]")
    fig.suptitle(rf"Magneto-Electric Coupling Block $\chi${title_tag}", fontsize=14)
    fig.savefig(output_dir / "chi_real.pdf")
    fig.savefig(output_dir / "chi_real.png", dpi=dpi)
    plt.close(fig)
    print(f"[Plot] Saved: chi_real.png and chi_real.pdf")

    # =========================================================================
    # Figure 4: Condition Number
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    gphi, gtheta, gcond = interpolate_to_grid(theta, phi, cond)
    cond_min = max(1.0, float(np.percentile(cond, 1)))
    cond_max = float(np.percentile(cond, 99))
    if cond_max <= cond_min:
        cond_max = 10.0 * cond_min

    cmap = plt.get_cmap("viridis").copy()
    im = ax.pcolormesh(
        gphi, gtheta, gcond, cmap=cmap, shading="auto",
        norm=LogNorm(vmin=cond_min, vmax=cond_max),
    )
    ax.set_title(rf"Impedance-Balanced Map Condition Number $\kappa(M)${title_tag}", fontsize=13)
    ax.set_xlabel(r"$\phi$ [deg]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_xlim(0, 360)
    ax.set_ylim(180, 0)
    fig.colorbar(im, ax=ax, label=r"Condition Number $\kappa(M)$ [log scale]")
    fig.savefig(output_dir / "map_condition_number.pdf")
    fig.savefig(output_dir / "map_condition_number.png", dpi=dpi)
    plt.close(fig)
    print(f"[Plot] Saved: map_condition_number.png and map_condition_number.pdf")

    # =========================================================================
    # Summary JSON Metrics
    # =========================================================================
    stats = {
        "num_faces": len(thetas),
        "thetas_norm": float(np.linalg.norm(thetas)),
        "condition_number": {
            "min": float(np.min(cond)),
            "p25": float(np.percentile(cond, 25)),
            "median": float(np.median(cond)),
            "p75": float(np.percentile(cond, 75)),
            "p95": float(np.percentile(cond, 95)),
            "max": float(np.max(cond)),
        },
        "Y_ee_imag_S": {
            "min": float(min(np.min(Y_tt), np.min(Y_pp), np.min(Y_tp))),
            "max": float(max(np.max(Y_tt), np.max(Y_pp), np.max(Y_tp))),
        },
        "chi_real": {
            "min": float(min(np.min(chi_tt), np.min(chi_pp), np.min(chi_tp), np.min(chi_pt))),
            "max": float(max(np.max(chi_tt), np.max(chi_pp), np.max(chi_tp), np.max(chi_pt))),
        },
        "Z_mm_imag_Ohm": {
            "min": float(min(np.min(Z_tt), np.min(Z_pp), np.min(Z_tp))),
            "max": float(max(np.max(Z_tt), np.max(Z_pp), np.max(Z_tp))),
        },
    }
    stats_path = output_dir / "parameter_checks.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[Export] Saved parameter checks: {stats_path.name}")

    # =========================================================================
    # Per-Face Physical Parameters CSV Export
    # =========================================================================
    if csv_export:
        if isinstance(csv_export, (str, Path)) and not isinstance(csv_export, bool):
            csv_path = Path(csv_export)
        else:
            csv_path = output_dir / "face_parameters.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "face_index",
                "x_m",
                "y_m",
                "z_m",
                "theta_deg",
                "phi_deg",
                "Y_ee_theta_theta_S",
                "Y_ee_theta_phi_S",
                "Y_ee_phi_phi_S",
                "chi_theta_theta",
                "chi_theta_phi",
                "chi_phi_theta",
                "chi_phi_phi",
                "Z_mm_theta_theta_Ohm",
                "Z_mm_theta_phi_Ohm",
                "Z_mm_phi_phi_Ohm",
                "condition_number",
                "theta_0_y1",
                "theta_1_y2",
                "theta_2_y3",
                "theta_3_c1",
                "theta_4_c2",
                "theta_5_c3",
                "theta_6_c4",
                "theta_7_z1",
                "theta_8_z2",
                "theta_9_z3",
            ])
            for m in range(len(thetas)):
                writer.writerow([
                    m,
                    f"{centroids[m, 0]:.6f}",
                    f"{centroids[m, 1]:.6f}",
                    f"{centroids[m, 2]:.6f}",
                    f"{np.rad2deg(theta[m]):.4f}",
                    f"{np.rad2deg(phi[m]):.4f}",
                    f"{Y_tt[m]:.8e}",
                    f"{Y_tp[m]:.8e}",
                    f"{Y_pp[m]:.8e}",
                    f"{chi_tt[m]:.8e}",
                    f"{chi_tp[m]:.8e}",
                    f"{chi_pt[m]:.8e}",
                    f"{chi_pp[m]:.8e}",
                    f"{Z_tt[m]:.8e}",
                    f"{Z_tp[m]:.8e}",
                    f"{Z_pp[m]:.8e}",
                    f"{cond[m]:.6e}",
                    f"{thetas[m, 0]:.8e}",
                    f"{thetas[m, 1]:.8e}",
                    f"{thetas[m, 2]:.8e}",
                    f"{thetas[m, 3]:.8e}",
                    f"{thetas[m, 4]:.8e}",
                    f"{thetas[m, 5]:.8e}",
                    f"{thetas[m, 6]:.8e}",
                    f"{thetas[m, 7]:.8e}",
                    f"{thetas[m, 8]:.8e}",
                    f"{thetas[m, 9]:.8e}",
                ])
        print(f"[Export] Saved per-face parameters CSV: {csv_path.name}")


def main() -> int:
    args = parse_args()
    plot_parameters(
        thetas_path=args.thetas,
        tensors_path=args.tensors,
        mesh_path=args.mesh,
        output_dir=args.output_dir,
        title_suffix=args.title_suffix,
        dpi=args.dpi,
        csv_export=args.csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

