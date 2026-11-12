#!/usr/bin/env python3
"""Generate publication-quality 2D field maps and 1D line cuts for Example 3.

Case:
  Circular-Arc Spindle (Ovil / Ogive) GSTC Metasurface (50% elongation, L/D = 1.5).
  Dual Prescribed Dipole Transformations:
    1. electric_z_matching: z-directed electric dipole (interior) -> exterior radiation.
       Active transverse electric component in zx-plane (phi=0): E_theta. (E_phi = 0 by symmetry).
    2. electric_y_matching: y-directed electric dipole (interior) -> exterior radiation.
       Active transverse electric component in zx-plane (phi=0): E_phi. (E_theta = 0 by symmetry).

Generates:
  1. Figure 1A: 2D zx-plane map of |E_theta| [dB] for Transformation 1 (z-dipole).
  2. Figure 1B: 2D zx-plane map of |E_phi| [dB] for Transformation 2 (y-dipole).
  3. Figure 1 Combined: 2-panel side-by-side figure (a) |E_theta| and (b) |E_phi|.
  4. Figure 2: 2-panel 1D radial line cuts (theta=45 deg, phi=0 deg) comparing
     Prescribed Target vs. SIE-GSTC (MoM) vs. NGSolve (Order 2 FEM).

Designed for IEEE journal formatting with high-resolution vector PDF and 300 DPI PNG outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
import numpy as np
from matplotlib.colors import Normalize

from applications.gstc_surface.artifacts import artifact_path, load_workflow_configuration
from applications.gstc_surface.plot_fields import (
    _mesh_section,
    display_reference,
    prescribed_samples,
    reference_transformation,
)
from sie_mom.postprocessing import (
    load_field_csv,
    spherical_components,
    to_decibels,
    zx_grid,
)

# IEEE Journal typography defaults
plt.rcParams.update(
    {
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
    }
)


def _draw_dipole_marker(ax, orientation: str) -> None:
    """Draw physical dipole orientation: small vertical arrow for z-dipole, simple dot for y-dipole."""
    if orientation == "z":
        # Clean, small vertical arrow along +z
        arrow_len = 0.018
        ax.annotate(
            "",
            xy=(0, arrow_len),
            xytext=(0, -arrow_len),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, mutation_scale=10),
        )
    elif orientation == "y":
        # Clean solid dot for out-of-plane y-dipole
        ax.plot(0, 0, "ko", markersize=4.5)
    else:
        ax.plot(0, 0, "k+", markersize=6, markeredgewidth=1.2)


def _load_case_data(
    fields_dir: Path, case_id: str, config: object, pmchwt_dir: Path | None = None
) -> dict:
    """Load field manifest, configuration, mesh boundary, and field CSVs for a case."""
    manifest_path = fields_dir / "fields.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling = manifest.get("sampling", {})

    case_entry = next((c for c in manifest["cases"] if c["id"] == case_id), None)
    if case_entry is None:
        raise ValueError(f"case {case_id} not found in {manifest_path}")

    zx_path = artifact_path(fields_dir, case_entry["grids"]["zx"]["path"])
    radial_path = artifact_path(fields_dir, case_entry["grids"]["radial"]["path"])

    sie_zx = load_field_csv(zx_path)
    sie_radial = load_field_csv(radial_path)

    trans = reference_transformation(config, case_id)
    target_zx = prescribed_samples(sie_zx, config, trans)
    target_radial = prescribed_samples(sie_radial, config, trans)

    bare_zx = None
    bare_radial = None
    if pmchwt_dir and pmchwt_dir.is_dir():
        subfolder = "case1_electric_z" if "z" in case_id else "case2_electric_y"
        b_zx = pmchwt_dir / subfolder / "zx_fields.csv"
        b_rad = pmchwt_dir / subfolder / "radial_fields.csv"
        if b_zx.is_file():
            bare_zx = load_field_csv(b_zx)
        if b_rad.is_file():
            bare_radial = load_field_csv(b_rad)

    _, _, section = _mesh_section(fields_dir, manifest, None)

    return {
        "sie_zx": sie_zx,
        "target_zx": target_zx,
        "sie_radial": sie_radial,
        "target_radial": target_radial,
        "bare_zx": bare_zx,
        "bare_radial": bare_radial,
        "sampling": sampling,
        "section": section,
        "transformation": trans,
    }


def plot_figure_1a(
    data: dict,
    output_dir: Path,
    *,
    vmin_db: float = -30.0,
    vmax_db: float = 10.0,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Figure 1A: 2D zx-plane map of |E_theta| [dB] for Transformation 1 (z-dipole).

    When bare PMCHWT baseline data is present, produces side-by-side comparison:
      (a) Without metasurface (PMCHWT) vs (b) With metasurface (SIE-GSTC).
    """
    sie_zx = data["sie_zx"]
    target_zx = data["target_zx"]
    bare_zx = data.get("bare_zx")
    grid_size = data["sampling"].get("zx_grid_size", 200)
    section = data["section"]

    x, z, order = zx_grid(sie_zx, grid_size)
    ref = display_reference(target_zx, "E", percentile=100, region="exterior")

    eth_sie = spherical_components(sie_zx, "E")[:, 1]
    eth_grid_sie = to_decibels(eth_sie, reference=ref)[order].reshape(len(z), len(x))
    norm = Normalize(vmin=vmin_db, vmax=vmax_db)

    output_dir.mkdir(parents=True, exist_ok=True)

    if bare_zx is not None:
        eth_bare = spherical_components(bare_zx, "E")[:, 1]
        eth_grid_bare = to_decibels(eth_bare, reference=ref)[order].reshape(len(z), len(x))

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True, constrained_layout=True)
        panels = [
            (axes[0], eth_grid_bare, r"(a) Without metasurface (PMCHWT)"),
            (axes[1], eth_grid_sie,  r"(b) With metasurface (SIE-GSTC)"),
        ]
        im = None
        for ax, grid_vals, title in panels:
            im = ax.pcolormesh(x, z, grid_vals, shading="auto", cmap="viridis", norm=norm)
            for curve in section:
                ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
                ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
            _draw_dipole_marker(ax, "z")
            ax.set_aspect("equal")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(z[0], z[-1])
            ax.set_xlabel(r"$x$ [m]")
            ax.set_title(title, pad=4)

        axes[0].set_ylabel(r"$z$ [m]")
        cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, extend="both")
        cb.set_label(r"Normalized $|E_\theta|$ [dB]")
        fig.suptitle(r"Transformation 1 ($z$-dipole, $\mathbf{p} \parallel \hat{\mathbf{z}}$): $|E_\theta|$", y=1.02, fontsize=9)

        pdf_path = output_dir / "figure_1a_case1_electric_z_incidence_2d_field_map.pdf"
        png_path = output_dir / "figure_1a_case1_electric_z_incidence_2d_field_map.png"
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Duplicate to explicit side_by_side naming
        shutil.copy2(pdf_path, output_dir / "figure_1a_case1_side_by_side.pdf")
        shutil.copy2(png_path, output_dir / "figure_1a_case1_side_by_side.png")
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
        im = ax.pcolormesh(x, z, eth_grid_sie, shading="auto", cmap="viridis", norm=norm)
        for curve in section:
            ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
            ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
        _draw_dipole_marker(ax, "z")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(z[0], z[-1])
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$z$ [m]")
        ax.set_title(r"Transformation 1 ($z$-dipole, $\mathbf{p} \parallel \hat{\mathbf{z}}$): $|E_\theta|$", pad=4)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04, extend="both")
        cb.set_label(r"Normalized $|E_\theta|$ [dB]")
        pdf_path = output_dir / "figure_1a_case1_electric_z_incidence_2d_field_map.pdf"
        png_path = output_dir / "figure_1a_case1_electric_z_incidence_2d_field_map.png"
        fig.savefig(pdf_path, dpi=dpi)
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    print(f"Saved Figure 1A: {pdf_path}")
    print(f"Saved Figure 1A: {png_path}")
    return pdf_path, png_path


def plot_figure_1b(
    data: dict,
    output_dir: Path,
    *,
    vmin_db: float = -30.0,
    vmax_db: float = 10.0,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Figure 1B: 2D zx-plane map of |E_phi| [dB] for Transformation 2 (y-dipole).

    When bare PMCHWT baseline data is present, produces side-by-side comparison:
      (a) Without metasurface (PMCHWT) vs (b) With metasurface (SIE-GSTC).
    """
    sie_zx = data["sie_zx"]
    target_zx = data["target_zx"]
    bare_zx = data.get("bare_zx")
    grid_size = data["sampling"].get("zx_grid_size", 200)
    section = data["section"]

    x, z, order = zx_grid(sie_zx, grid_size)
    ref = display_reference(target_zx, "E", percentile=100, region="exterior")

    eph_sie = spherical_components(sie_zx, "E")[:, 2]
    eph_grid_sie = to_decibels(eph_sie, reference=ref)[order].reshape(len(z), len(x))
    norm = Normalize(vmin=vmin_db, vmax=vmax_db)

    output_dir.mkdir(parents=True, exist_ok=True)

    if bare_zx is not None:
        eph_bare = spherical_components(bare_zx, "E")[:, 2]
        eph_grid_bare = to_decibels(eph_bare, reference=ref)[order].reshape(len(z), len(x))

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True, constrained_layout=True)
        panels = [
            (axes[0], eph_grid_bare, r"(a) Without metasurface (PMCHWT)"),
            (axes[1], eph_grid_sie,  r"(b) With metasurface (SIE-GSTC)"),
        ]
        im = None
        for ax, grid_vals, title in panels:
            im = ax.pcolormesh(x, z, grid_vals, shading="auto", cmap="viridis", norm=norm)
            for curve in section:
                ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
                ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
            _draw_dipole_marker(ax, "y")
            ax.set_aspect("equal")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(z[0], z[-1])
            ax.set_xlabel(r"$x$ [m]")
            ax.set_title(title, pad=4)

        axes[0].set_ylabel(r"$z$ [m]")
        cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, extend="both")
        cb.set_label(r"Normalized $|E_\phi|$ [dB]")
        fig.suptitle(r"Transformation 2 ($y$-dipole, $\mathbf{p} \parallel \hat{\mathbf{y}}$): $|E_\phi|$", y=1.02, fontsize=9)

        pdf_path = output_dir / "figure_1b_case2_electric_y_incidence_2d_field_map.pdf"
        png_path = output_dir / "figure_1b_case2_electric_y_incidence_2d_field_map.png"
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Duplicate to explicit side_by_side naming
        shutil.copy2(pdf_path, output_dir / "figure_1b_case2_side_by_side.pdf")
        shutil.copy2(png_path, output_dir / "figure_1b_case2_side_by_side.png")
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
        im = ax.pcolormesh(x, z, eph_grid_sie, shading="auto", cmap="viridis", norm=norm)
        for curve in section:
            ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
            ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
        _draw_dipole_marker(ax, "y")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(z[0], z[-1])
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$z$ [m]")
        ax.set_title(r"Transformation 2 ($y$-dipole, $\mathbf{p} \parallel \hat{\mathbf{y}}$): $|E_\phi|$", pad=4)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04, extend="both")
        cb.set_label(r"Normalized $|E_\phi|$ [dB]")
        pdf_path = output_dir / "figure_1b_case2_electric_y_incidence_2d_field_map.pdf"
        png_path = output_dir / "figure_1b_case2_electric_y_incidence_2d_field_map.png"
        fig.savefig(pdf_path, dpi=dpi)
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    print(f"Saved Figure 1B: {pdf_path}")
    print(f"Saved Figure 1B: {png_path}")
    return pdf_path, png_path


def plot_figure_1_combined(
    data_z: dict,
    data_y: dict,
    output_dir: Path,
    *,
    vmin_db: float = -30.0,
    vmax_db: float = 10.0,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Figure 1 (Combined): 2x2 grid (or 1x2 if PMCHWT missing) comparing without vs with metasurface."""
    sie_zx_z = data_z["sie_zx"]
    target_zx_z = data_z["target_zx"]
    bare_zx_z = data_z.get("bare_zx")
    grid_size = data_z["sampling"].get("zx_grid_size", 200)
    section = data_z["section"]

    x, z, order = zx_grid(sie_zx_z, grid_size)

    # Case 1 (z-dipole)
    ref_z = display_reference(target_zx_z, "E", percentile=100, region="exterior")
    eth_sie = spherical_components(sie_zx_z, "E")[:, 1]
    eth_grid_sie = to_decibels(eth_sie, reference=ref_z)[order].reshape(len(z), len(x))

    # Case 2 (y-dipole)
    sie_zx_y = data_y["sie_zx"]
    target_zx_y = data_y["target_zx"]
    bare_zx_y = data_y.get("bare_zx")
    ref_y = display_reference(target_zx_y, "E", percentile=100, region="exterior")
    eph_sie = spherical_components(sie_zx_y, "E")[:, 2]
    eph_grid_sie = to_decibels(eph_sie, reference=ref_y)[order].reshape(len(z), len(x))

    norm = Normalize(vmin=vmin_db, vmax=vmax_db)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bare_zx_z is not None and bare_zx_y is not None:
        eth_bare = spherical_components(bare_zx_z, "E")[:, 1]
        eth_grid_bare = to_decibels(eth_bare, reference=ref_z)[order].reshape(len(z), len(x))

        eph_bare = spherical_components(bare_zx_y, "E")[:, 2]
        eph_grid_bare = to_decibels(eph_bare, reference=ref_y)[order].reshape(len(z), len(x))

        fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.8), sharex=True, sharey=True, constrained_layout=True)
        grid_panels = [
            (axes[0, 0], eth_grid_bare, r"(a) Trans. 1: Without metasurface", "z"),
            (axes[0, 1], eth_grid_sie,  r"(b) Trans. 1: With metasurface",    "z"),
            (axes[1, 0], eph_grid_bare, r"(c) Trans. 2: Without metasurface", "y"),
            (axes[1, 1], eph_grid_sie,  r"(d) Trans. 2: With metasurface",    "y"),
        ]
        im = None
        for ax, data_grid, title, orientation in grid_panels:
            im = ax.pcolormesh(x, z, data_grid, shading="auto", cmap="viridis", norm=norm)
            for curve in section:
                ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
                ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
            _draw_dipole_marker(ax, orientation)
            ax.set_aspect("equal")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(z[0], z[-1])
            ax.set_title(title, pad=4)

        axes[0, 0].set_ylabel(r"$z$ [m]")
        axes[1, 0].set_ylabel(r"$z$ [m]")
        axes[1, 0].set_xlabel(r"$x$ [m]")
        axes[1, 1].set_xlabel(r"$x$ [m]")

        cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06, shrink=0.75, extend="both")
        cb.set_label(r"Normalized Field Magnitude [dB] ($|E_\theta|$ for Trans. 1, $|E_\phi|$ for Trans. 2)")

        pdf_path = output_dir / "figure_1_combined_2d_field_maps.pdf"
        png_path = output_dir / "figure_1_combined_2d_field_maps.png"
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # Duplicate to explicit 2x2 naming
        shutil.copy2(pdf_path, output_dir / "figure_1_combined_2x2_field_maps.pdf")
        shutil.copy2(png_path, output_dir / "figure_1_combined_2x2_field_maps.png")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2), constrained_layout=True)
        panels = [
            (
                axes[0],
                eth_grid_sie,
                r"(a) Transformation 1 ($z$-dipole, $\mathbf{p} \parallel \hat{\mathbf{z}}$): $|E_\theta|$",
                "z",
            ),
            (
                axes[1],
                eph_grid_sie,
                r"(b) Transformation 2 ($y$-dipole, $\mathbf{p} \parallel \hat{\mathbf{y}}$): $|E_\phi|$",
                "y",
            ),
        ]
        im = None
        for ax, grid_vals, title, orientation in panels:
            im = ax.pcolormesh(x, z, grid_vals, shading="auto", cmap="viridis", norm=norm)
            for curve in section:
                ax.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.5)
                ax.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.7)
            _draw_dipole_marker(ax, orientation)
            ax.set_aspect("equal")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(z[0], z[-1])
            ax.set_xlabel(r"$x$ [m]")
            ax.set_title(title, pad=4)

        axes[0].set_ylabel(r"$z$ [m]")
        axes[1].tick_params(labelleft=False)

        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, pad=0.03, extend="both")
        cb.set_label(r"Normalized Field Magnitude [dB]")

        pdf_path = output_dir / "figure_1_combined_2d_field_maps.pdf"
        png_path = output_dir / "figure_1_combined_2d_field_maps.png"
        fig.savefig(pdf_path, dpi=dpi)
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)

    print(f"Saved Figure 1 (Combined): {pdf_path}")
    print(f"Saved Figure 1 (Combined): {png_path}")
    return pdf_path, png_path


def plot_figure_2_radial_cuts(
    data_z: dict,
    data_y: dict,
    output_dir: Path,
    ngsolve_fields_dir: Path | None = None,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Figure 2: 2-panel 1D radial line cuts (theta=45 deg, phi=0 deg).

    Primary plot: Real parts Re(E_theta) and Re(E_phi) proving phase, wavelength, and continuity.
    Secondary plot: Magnitude |E_theta| and |E_phi| saved alongside as companion figure.
    """
    sie_rad_z = data_z["sie_radial"]
    tgt_rad_z = data_z["target_radial"]
    sie_rad_y = data_y["sie_radial"]
    tgt_rad_y = data_y["target_radial"]

    r = np.sqrt(sie_rad_z["x"] ** 2 + sie_rad_z["y"] ** 2 + sie_rad_z["z"] ** 2)
    regions = sie_rad_z["region"]

    # Boundary crossing
    crossings = np.flatnonzero(regions[1:] != regions[:-1])
    r_boundary = 0.5 * (r[crossings[0]] + r[crossings[0] + 1]) if len(crossings) > 0 else 0.1177

    # Case 1 (z-dipole): E_theta (complex)
    tgt_eth_c = spherical_components(tgt_rad_z, "E")[:, 1]
    sie_eth_c = spherical_components(sie_rad_z, "E")[:, 1]

    # Case 2 (y-dipole): E_phi (complex)
    tgt_eph_c = spherical_components(tgt_rad_y, "E")[:, 2]
    sie_eph_c = spherical_components(sie_rad_y, "E")[:, 2]

    # Check NGSolve availability
    ng_eth_c = None
    ng_eph_c = None
    if ngsolve_fields_dir and (ngsolve_fields_dir / "fields.json").is_file():
        try:
            ng_manifest = json.loads((ngsolve_fields_dir / "fields.json").read_text(encoding="utf-8"))
            for case_entry in ng_manifest["cases"]:
                c_id = case_entry["id"]
                csv_p = artifact_path(ngsolve_fields_dir, case_entry["grids"]["radial"]["path"])
                if csv_p.is_file():
                    ng_csv = load_field_csv(csv_p)
                    if c_id == "electric_z_incidence":
                        ng_eth_c = spherical_components(ng_csv, "E")[:, 1]
                    elif c_id == "electric_y_incidence":
                        ng_eph_c = spherical_components(ng_csv, "E")[:, 2]
        except Exception as err:
            print(f"[WARNING] Could not load NGSolve comparison fields: {err}", file=sys.stderr)

    # Filter out extreme dipole singularity near center (r < 0.02 m) for clear wave visualization
    mask_r = r >= 0.02
    r_plot = r[mask_r]
    scale = 1e-5

    # -------------------------------------------------------------------------
    # Helper to generate polished two-tier (field + difference subpanel) figure
    # -------------------------------------------------------------------------
    def _render_2tier_cut(
        vals_z: tuple[np.ndarray, np.ndarray, np.ndarray | None],
        vals_y: tuple[np.ndarray, np.ndarray, np.ndarray | None],
        ylabels: tuple[str, str],
        diff_labels: tuple[str, str],
        pdf_name: str,
        png_name: str,
    ) -> tuple[Path, Path]:
        fig = plt.figure(figsize=(7.1, 3.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.06)

        axes_top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        axes_bot = [
            fig.add_subplot(gs[1, 0], sharex=axes_top[0]),
            fig.add_subplot(gs[1, 1], sharex=axes_top[1]),
        ]

        panels = [
            (
                axes_top[0],
                axes_bot[0],
                vals_z[0],
                vals_z[1],
                vals_z[2],
                r"(a) Transformation 1 ($z$-dipole)",
                ylabels[0],
                diff_labels[0],
            ),
            (
                axes_top[1],
                axes_bot[1],
                vals_y[0],
                vals_y[1],
                vals_y[2],
                r"(b) Transformation 2 ($y$-dipole)",
                ylabels[1],
                diff_labels[1],
            ),
        ]

        for ax, ax_err, tgt_v, sie_v, ng_v, title, ylabel, diff_ylabel in panels:
            tgt_sub = tgt_v[mask_r] * scale
            sie_sub = sie_v[mask_r] * scale
            ng_sub = ng_v[mask_r] * scale if ng_v is not None else None

            diff_sie = np.abs(sie_sub - tgt_sub)
            diff_ng = np.abs(ng_sub - tgt_sub) if ng_sub is not None else None

            # --- Top Panel: Field Curves ---
            ax.axvspan(r_plot[0], r_boundary, color="#e6f2ff", alpha=0.5, label="_Interior")
            ax.axvspan(r_boundary, r_plot[-1], color="#fff5eb", alpha=0.5, label="_Exterior")
            ax.axvline(x=r_boundary, color="#777777", linestyle="--", linewidth=0.9)

            ax.plot(r_plot, tgt_sub, "k-", linewidth=1.3, label="Target (Prescribed)")
            ax.plot(
                r_plot,
                sie_sub,
                color="#0066cc",
                linestyle="--",
                linewidth=1.2,
                label="SIE-GSTC (MoM)",
            )
            if ng_sub is not None:
                ax.plot(
                    r_plot,
                    ng_sub,
                    color="#d95f02",
                    linestyle=":",
                    linewidth=1.2,
                    marker="o",
                    markersize=3.6,
                    markeredgewidth=0.9,
                    markerfacecolor="none",
                    markevery=10,
                    label="NGSolve (Order 2 FEM)",
                )

            ax.set_title(title, pad=4, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=7.5)
            ax.set_xlim(r_plot[0], r_plot[-1])
            ax.tick_params(labelbottom=False, labelsize=7)
            ax.grid(True, linestyle=":", alpha=0.5)

            # Ensure ample headroom for permittivity badges
            all_vals = [tgt_sub, sie_sub]
            if ng_sub is not None:
                all_vals.append(ng_sub)
            y_min = min(float(np.nanmin(v)) for v in all_vals)
            y_max = max(float(np.nanmax(v)) for v in all_vals)
            y_span = max(y_max - y_min, 1e-6)
            ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.24 * y_span)

            # Prominent permittivity badges with background
            ax.text(
                0.28,
                0.88,
                "Interior\n" + r"($\varepsilon_r=5$)",
                transform=ax.transAxes,
                fontsize=6.5,
                ha="center",
                va="center",
                color="#003366",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#b0c4de",
                    alpha=0.92,
                ),
            )
            ax.text(
                0.80,
                0.88,
                "Exterior\n" + r"($\varepsilon_r=1$)",
                transform=ax.transAxes,
                fontsize=6.5,
                ha="center",
                va="center",
                color="#8b3a00",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="#f4a460",
                    alpha=0.92,
                ),
            )

            # --- Bottom Panel: Absolute Difference ---
            ax_err.axvspan(r_plot[0], r_boundary, color="#e6f2ff", alpha=0.5)
            ax_err.axvspan(r_boundary, r_plot[-1], color="#fff5eb", alpha=0.5)
            ax_err.axvline(x=r_boundary, color="#777777", linestyle="--", linewidth=0.9)

            ax_err.plot(
                r_plot,
                diff_sie,
                color="#0066cc",
                linestyle="--",
                linewidth=1.1,
                label=r"$|\text{SIE} - \text{Target}|$",
            )
            if diff_ng is not None:
                ax_err.plot(
                    r_plot,
                    diff_ng,
                    color="#d95f02",
                    linestyle=":",
                    linewidth=1.1,
                    marker="o",
                    markersize=3.2,
                    markeredgewidth=0.8,
                    markerfacecolor="none",
                    markevery=10,
                    label=r"$|\text{FEM} - \text{Target}|$",
                )

            ax_err.set_xlabel(r"Radial Distance $r$ [m]", fontsize=8)
            ax_err.set_ylabel(diff_ylabel, fontsize=7)
            ax_err.set_xlim(r_plot[0], r_plot[-1])
            ax_err.set_ylim(bottom=0.0)

            # Headroom for error legend
            max_d = float(np.nanmax(diff_sie))
            if diff_ng is not None:
                max_d = max(max_d, float(np.nanmax(diff_ng)))
            ax_err.set_ylim(0.0, max_d * 1.38)

            ax_err.tick_params(labelsize=7)
            ax_err.grid(True, linestyle=":", alpha=0.5)

        axes_top[1].legend(
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fontsize=6.5,
        )
        axes_bot[1].legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fontsize=6.0,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_p = output_dir / pdf_name
        png_p = output_dir / png_name
        fig.savefig(pdf_p, dpi=dpi)
        fig.savefig(png_p, dpi=dpi)
        plt.close(fig)
        print(f"Saved: {pdf_p}")
        print(f"Saved: {png_p}")
        return pdf_p, png_p

    # 1. Primary Figure 2: Real Part Re(E) + Difference Subpanel
    pdf_real, png_real = _render_2tier_cut(
        (tgt_eth_c.real, sie_eth_c.real, ng_eth_c.real if ng_eth_c is not None else None),
        (tgt_eph_c.real, sie_eph_c.real, ng_eph_c.real if ng_eph_c is not None else None),
        (
            r"$\operatorname{Re}(E_\theta) \times 10^5$ [V/m]",
            r"$\operatorname{Re}(E_\phi) \times 10^5$ [V/m]",
        ),
        (
            r"Diff. [$\times 10^5$ V/m]",
            r"Diff. [$\times 10^5$ V/m]",
        ),
        "figure_2_radial_cut_comparison.pdf",
        "figure_2_radial_cut_comparison.png",
    )

    # 2. Companion Figure: Magnitude |E| + Difference Subpanel
    _render_2tier_cut(
        (np.abs(tgt_eth_c), np.abs(sie_eth_c), np.abs(ng_eth_c) if ng_eth_c is not None else None),
        (np.abs(tgt_eph_c), np.abs(sie_eph_c), np.abs(ng_eph_c) if ng_eph_c is not None else None),
        (
            r"$|E_\theta| \times 10^5$ [V/m]",
            r"$|E_\phi| \times 10^5$ [V/m]",
        ),
        (
            r"Diff. [$\times 10^5$ V/m]",
            r"Diff. [$\times 10^5$ V/m]",
        ),
        "figure_2_radial_cut_magnitude.pdf",
        "figure_2_radial_cut_magnitude.png",
    )

    return pdf_real, png_real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures for Example 3 fields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fields-dir",
        type=Path,
        required=True,
        help="Path to evaluated fields directory containing fields.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated publication figures.",
    )
    parser.add_argument(
        "--pmchwt-dir",
        type=Path,
        default=None,
        help="Optional path to bare dielectric PMCHWT baseline directory.",
    )
    parser.add_argument(
        "--ngsolve-fields-dir",
        type=Path,
        default=None,
        help="Optional path to NGSolve fields directory for overlaying FEM validation on Figure 2.",
    )
    parser.add_argument(
        "--vmin-db",
        type=float,
        default=-30.0,
        help="Colorbar lower bound in dB for 2D field maps.",
    )
    parser.add_argument(
        "--vmax-db",
        type=float,
        default=10.0,
        help="Colorbar upper bound in dB for 2D field maps.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution for PNG figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields_dir = args.fields_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = fields_dir / "configuration.json"
    if not config_path.is_file():
        config_path = fields_dir.parent.parent / "configuration.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration.json not found in {fields_dir} or parent")

    config = load_workflow_configuration(config_path)

    # Resolve PMCHWT directory if not explicitly provided
    pmchwt_dir = args.pmchwt_dir.resolve() if args.pmchwt_dir else None
    if pmchwt_dir is None:
        candidates = [
            fields_dir.parents[2] / "pmchwt",
            fields_dir.parent.parent.parent / "pmchwt",
            fields_dir.parents[1] / "pmchwt",
        ]
        pmchwt_dir = next((c for c in candidates if c.is_dir()), None)

    if pmchwt_dir:
        print(f"PMCHWT baseline located at: {pmchwt_dir}")

    print("Loading data for Transformation 1 (electric_z_incidence)...")
    data_z = _load_case_data(fields_dir, "electric_z_incidence", config, pmchwt_dir)

    print("Loading data for Transformation 2 (electric_y_incidence)...")
    data_y = _load_case_data(fields_dir, "electric_y_incidence", config, pmchwt_dir)

    print("\nGenerating Figure 1A (Transformation 1 2D field map)...")
    plot_figure_1a(data_z, output_dir, vmin_db=args.vmin_db, vmax_db=args.vmax_db, dpi=args.dpi)

    print("\nGenerating Figure 1B (Transformation 2 2D field map)...")
    plot_figure_1b(data_y, output_dir, vmin_db=args.vmin_db, vmax_db=args.vmax_db, dpi=args.dpi)

    print("\nGenerating Figure 1 Combined (Dual 2D field maps)...")
    plot_figure_1_combined(
        data_z, data_y, output_dir, vmin_db=args.vmin_db, vmax_db=args.vmax_db, dpi=args.dpi
    )

    print("\nGenerating Figure 2 (1D radial line cuts comparison)...")
    plot_figure_2_radial_cuts(
        data_z, data_y, output_dir, ngsolve_fields_dir=args.ngsolve_fields_dir, dpi=args.dpi
    )

    print(f"\nAll publication field figures successfully written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
