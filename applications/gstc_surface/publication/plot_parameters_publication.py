#!/usr/bin/env python3
"""Generate publication-quality 3D surface parameter maps for Example 3.

Case:
  Circular-Arc Spindle (Ovil / Ogive) GSTC Metasurface (50% elongation, L/D = 1.5).
  Displays synthesized physical 10-parameter GSTC surface tensors:
    - Electric admittance block: Im(Y_11), Im(Y_12), Im(Y_22) [S]
    - Magnetic impedance block: Im(Z_11), Im(Z_12), Im(Z_22) [Ohm]
    - Magnetoelectric coupling block: Re(chi_11), Re(chi_12), Re(chi_21), Re(chi_22) [dimensionless]
    - Local fitting residual: relative residual per triangle face.

To save space in journal articles, renders a single broadside orthographic view (+y view)
showing the full spindle length along x and height along z in single-row layouts.
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
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from applications.gstc_surface.plot_surface_parameters import (
    load_surface_parameters,
)
from sie_mom.postprocessing.mesh_plotting import face_value_limits, plot_mesh_face_values

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

# Broadside view from +y (elev=0, azim=90) displays the zx profile clearly
BROADSIDE_VIEW = (0, 90)


def _render_single_row(
    mesh: object,
    arrays: list[np.ndarray],
    titles: list[str],
    *,
    limits: tuple[float, float],
    cmap: str,
    unit: str,
    output_pdf: Path,
    output_png: Path,
    dpi: int = 300,
    show_edges: bool = True,
    fig_width: float = 7.1,
    fig_height: float = 2.4,
) -> None:
    """Render a single-row multi-panel figure for a block of surface parameters."""
    n_cols = len(arrays)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [axes]

    norm = Normalize(*limits)
    first_im = None

    for col, (vals, title) in enumerate(zip(arrays, titles)):
        ax = axes[col]
        im = plot_mesh_face_values(
            mesh,
            vals,
            ax=ax,
            norm=norm,
            cmap=cmap,
            view=BROADSIDE_VIEW,
            show_edges=show_edges,
        )
        if first_im is None:
            first_im = im

        ax.set_title(title, pad=-1, fontsize=8)
        ax.set_xlabel(r"$x$ [m]", labelpad=-3)
        ax.set_ylabel("")
        ax.set_yticks([])
        # Suppress 3D z-label to avoid gutter collision between subplots; place 2D label on left margin
        ax.set_zlabel("")
        if col == 0:
            ax.text2D(0.08, 0.5, r"$z$ [m]", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=8)
        else:
            ax.set_zticklabels([])

        ax.tick_params(labelsize=6.5, pad=-1)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(4))

    below = any(np.any(vals < limits[0]) for vals in arrays)
    above = any(np.any(vals > limits[1]) for vals in arrays)
    extend = "both" if below and above else "min" if below else "max" if above else "neither"

    cb = fig.colorbar(
        first_im,
        ax=axes,
        shrink=0.82,
        pad=0.03,
        label=unit,
        extend=extend,
    )
    cb.ax.tick_params(labelsize=7)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=dpi)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


def _render_grid_2x2(
    mesh: object,
    arrays: list[np.ndarray],
    titles: list[str],
    *,
    limits: tuple[float, float],
    cmap: str,
    unit: str,
    output_pdf: Path,
    output_png: Path,
    dpi: int = 300,
    show_edges: bool = True,
    fig_width: float = 7.1,
    fig_height: float = 4.2,
) -> None:
    """Render a 2x2 grid of 3D broadside surface parameter maps with shared colorbar."""
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.88,
        bottom=0.08,
        top=0.94,
        wspace=-0.08,
        hspace=0.04,
    )
    norm = Normalize(*limits)
    first_im = None

    for idx, (vals, title) in enumerate(zip(arrays, titles)):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c], projection="3d")
        im = plot_mesh_face_values(
            mesh,
            vals,
            ax=ax,
            norm=norm,
            cmap=cmap,
            view=BROADSIDE_VIEW,
            show_edges=show_edges,
        )
        if first_im is None:
            first_im = im

        # Zoom in on broadside projection to eliminate dead margin in 3D box
        try:
            ax.set_box_aspect(ax.get_box_aspect(), zoom=1.22)
        except Exception:
            pass

        ax.set_title(title, pad=-1, fontsize=8)
        if r == 1:
            ax.set_xlabel(r"$x$ [m]", labelpad=-3, fontsize=7.5)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        ax.set_ylabel("")
        ax.set_yticks([])

        if c == 0:
            ax.set_zlabel(r"$z$ [m]", labelpad=-3, fontsize=7.5)
            # Note: Alternatively, lines 111-115 can be used here for 2D margin placement:
            # ax.set_zlabel("")
            # ax.text2D(0.08, 0.5, r"$z$ [m]", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=7.5)
        else:
            ax.set_zlabel("")
            ax.set_zticklabels([])

        ax.tick_params(labelsize=6.5, pad=-1)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_locator(MaxNLocator(4))

    below = any(np.any(vals < limits[0]) for vals in arrays)
    above = any(np.any(vals > limits[1]) for vals in arrays)
    extend = "both" if below and above else "min" if below else "max" if above else "neither"

    cax = fig.add_axes([0.90, 0.12, 0.022, 0.76])
    cb = fig.colorbar(first_im, cax=cax, label=unit, extend=extend)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(unit, fontsize=8)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=dpi)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


def generate_surface_parameter_figures(
    inputs_dir: Path,
    output_dir: Path,
    *,
    percentile: float = 98.0,
    show_edges: bool = True,
    dpi: int = 300,
) -> None:
    """Generate all publication surface parameter figures from prepared inputs."""
    mesh, maps, residual, prov = load_surface_parameters(inputs_dir)
    print(f"Loaded mesh with {len(mesh.faces)} faces from {inputs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Figure 3A: Electric Admittance Im(Y) [S] (3 subplots)
    # -------------------------------------------------------------------------
    y_arrays = [
        maps[:, 0, 0].imag,
        maps[:, 0, 1].imag,
        maps[:, 1, 1].imag,
    ]
    y_titles = [
        r"(a) $\operatorname{Im}(Y_{11})$",
        r"(b) $\operatorname{Im}(Y_{12})$",
        r"(c) $\operatorname{Im}(Y_{22})$",
    ]
    # Compute robust symmetric limits across all Y components
    y_all = np.concatenate(y_arrays)
    y_lim = face_value_limits(y_all, percentile=percentile)

    _render_single_row(
        mesh,
        y_arrays,
        y_titles,
        limits=y_lim,
        cmap="RdBu_r",
        unit=r"Admittance [S]",
        output_pdf=output_dir / "figure_3a_surface_parameters_admittance.pdf",
        output_png=output_dir / "figure_3a_surface_parameters_admittance.png",
        dpi=dpi,
        show_edges=show_edges,
        fig_width=7.1,
        fig_height=2.3,
    )

    # -------------------------------------------------------------------------
    # 2. Figure 3B: Magnetic Impedance Im(Z) [Ohm] (3 subplots)
    # -------------------------------------------------------------------------
    z_arrays = [
        maps[:, 2, 2].imag,
        maps[:, 2, 3].imag,
        maps[:, 3, 3].imag,
    ]
    z_titles = [
        r"(a) $\operatorname{Im}(Z_{11})$",
        r"(b) $\operatorname{Im}(Z_{12})$",
        r"(c) $\operatorname{Im}(Z_{22})$",
    ]
    z_all = np.concatenate(z_arrays)
    z_lim = face_value_limits(z_all, percentile=percentile)

    _render_single_row(
        mesh,
        z_arrays,
        z_titles,
        limits=z_lim,
        cmap="RdBu_r",
        unit=r"Impedance [$\Omega$]",
        output_pdf=output_dir / "figure_3b_surface_parameters_impedance.pdf",
        output_png=output_dir / "figure_3b_surface_parameters_impedance.png",
        dpi=dpi,
        show_edges=show_edges,
        fig_width=7.1,
        fig_height=2.3,
    )

    # -------------------------------------------------------------------------
    # 3. Figure 3C: Magnetoelectric Coupling Re(chi) (4 subplots)
    # -------------------------------------------------------------------------
    chi_arrays = [
        maps[:, 0, 2].real,
        maps[:, 0, 3].real,
        maps[:, 1, 2].real,
        maps[:, 1, 3].real,
    ]
    chi_titles = [
        r"(a) $\operatorname{Re}(\chi_{11})$",
        r"(b) $\operatorname{Re}(\chi_{12})$",
        r"(c) $\operatorname{Re}(\chi_{21})$",
        r"(d) $\operatorname{Re}(\chi_{22})$",
    ]
    chi_all = np.concatenate(chi_arrays)
    chi_lim = face_value_limits(chi_all, percentile=percentile)

    _render_grid_2x2(
        mesh,
        chi_arrays,
        chi_titles,
        limits=chi_lim,
        cmap="RdBu_r",
        unit=r"Coupling $\chi$",
        output_pdf=output_dir / "figure_3c_surface_parameters_coupling.pdf",
        output_png=output_dir / "figure_3c_surface_parameters_coupling.png",
        dpi=dpi,
        show_edges=show_edges,
        fig_width=7.1,
        fig_height=4.2,
    )

    # -------------------------------------------------------------------------
    # 4. Figure 3D: Relative Fitting Residual (1 subplot)
    # -------------------------------------------------------------------------
    res_lim = (0.0, float(np.percentile(residual, percentile)))
    _render_single_row(
        mesh,
        [residual],
        [r"Synthesis Relative Residual"],
        limits=res_lim,
        cmap="viridis",
        unit=r"Relative Error",
        output_pdf=output_dir / "figure_3d_surface_parameters_residual.pdf",
        output_png=output_dir / "figure_3d_surface_parameters_residual.png",
        dpi=dpi,
        show_edges=show_edges,
        fig_width=3.5,
        fig_height=2.6,
    )

    # -------------------------------------------------------------------------
    # 5. Figure 3: Combined 4-Panel Overview Figure (IEEE double-column 1-row)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.1, 2.2),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    combined_items = [
        (
            axes[0],
            maps[:, 0, 0].imag,
            r"(a) $\operatorname{Im}(Y_{11})$",
            face_value_limits(maps[:, 0, 0].imag, percentile=percentile),
            "RdBu_r",
            "[S]",
        ),
        (
            axes[1],
            maps[:, 2, 2].imag,
            r"(b) $\operatorname{Im}(Z_{11})$",
            face_value_limits(maps[:, 2, 2].imag, percentile=percentile),
            "RdBu_r",
            r"[$\Omega$]",
        ),
        (
            axes[2],
            maps[:, 0, 3].real,
            r"(c) $\operatorname{Re}(\chi_{12})$",
            face_value_limits(maps[:, 0, 3].real, percentile=percentile),
            "RdBu_r",
            "",
        ),
        (
            axes[3],
            residual,
            r"(d) Fit Residual",
            res_lim,
            "viridis",
            "",
        ),
    ]

    for ax, vals, title, lims, cmap, unit in combined_items:
        norm = Normalize(*lims)
        im = plot_mesh_face_values(
            mesh,
            vals,
            ax=ax,
            norm=norm,
            cmap=cmap,
            view=BROADSIDE_VIEW,
            show_edges=show_edges,
        )
        ax.set_title(title, pad=-1, fontsize=7.5)
        ax.set_xlabel(r"$x$ [m]", labelpad=-3)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_zlabel(r"$z$ [m]" if ax == axes[0] else "", labelpad=-3)
        # Note: Alternatively, lines 111-115 can be used here for 2D margin placement:
        # ax.set_zlabel("")
        # if ax == axes[0]:
        #     ax.text2D(0.08, 0.5, r"$z$ [m]", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=8)
        if ax != axes[0]:
            ax.set_zticklabels([])

        ax.tick_params(labelsize=6, pad=-1)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(4))

        cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.04)
        cb.ax.tick_params(labelsize=6)
        if unit:
            cb.set_label(unit, fontsize=6.5)

    comb_pdf = output_dir / "figure_3_combined_surface_parameters.pdf"
    comb_png = output_dir / "figure_3_combined_surface_parameters.png"
    fig.savefig(comb_pdf, dpi=dpi)
    fig.savefig(comb_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved Combined Figure 3: {comb_pdf}")
    print(f"Saved Combined Figure 3: {comb_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication 3D surface parameter figures for Example 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        required=True,
        help="Path to inputs directory containing face_tensors.bin and sie_inputs.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated publication surface parameter figures.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="Percentile limit for robust colormap scaling.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        default=True,
        help="Render mesh triangle edges on 3D surface.",
    )
    parser.add_argument(
        "--no-edges",
        action="store_false",
        dest="show_edges",
        help="Do not render triangle edges.",
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
    generate_surface_parameter_figures(
        args.inputs_dir.resolve(),
        args.output_dir.resolve(),
        percentile=args.percentile,
        show_edges=args.show_edges,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
