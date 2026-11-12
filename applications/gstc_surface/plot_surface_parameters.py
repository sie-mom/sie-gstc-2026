"""Plot saved local GSTC coefficients on the actual mesh, without a solve.

Run as ``python -m applications.gstc_surface.plot_surface_parameters``.
Uses reciprocal-lossless prepared maps in their original face-edge basis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from sie_mom.mesh.face_frames import compute_face_frames
from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh
from sie_mom.postprocessing.mesh_plotting import face_value_limits, plot_mesh_face_values

APPLICATION_DIR = Path(__file__).resolve().parent
VIEWS = ((0, -90), (0, 90))
BLOCKS = (
    ("electric", "Electric admittance", "S", "Y", "imag", ((0, 0), (0, 1), (1, 1))),
    ("magnetic", "Magnetic impedance", "Ohm", "Z", "imag", ((2, 2), (2, 3), (3, 3))),
    (
        "coupling",
        "Electric-magnetic coupling",
        "dimensionless",
        r"\chi",
        "real",
        ((0, 2), (0, 3), (1, 2), (1, 3)),
    ),
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("prepared artifact must remain inside its input directory")
    return path


def load_surface_parameters(inputs_dir, *, mesh_path=None):
    """Check mesh identity and the association of saved maps/diagnostics to faces.

    The configuration is provenance, not a geometry factory. A local solve mesh
    or explicit --mesh can replace an unavailable original path, but only when
    its bytes match the prepared mesh SHA-256. No analytic surface is required.
    """
    root = Path(inputs_dir).resolve()
    manifest_path = root / "sie_inputs.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != "gstc_surface_inputs"
        or manifest.get("status") not in ("prepared", "prepared_with_warnings")
        or manifest.get("basis") != "face"
    ):
        raise ValueError("expected prepared gstc_surface inputs in the face basis")
    files = manifest["files"]
    config_path = _artifact(root, files["configuration"])
    config = _read_json(config_path)
    if config != manifest["configuration"]:
        raise ValueError("prepared manifest and configuration snapshot disagree")
    if config["extraction_method"] != "reciprocal_lossless":
        raise ValueError("ten-real-coefficient plots require reciprocal_lossless extraction")
    if mesh_path is None:
        candidates = (
            root / "surface.obj",
            APPLICATION_DIR / config["sie"]["mesh"]["path"],
            Path(manifest["mesh"]["path"]),
        )
        mesh_path = next(
            (p for p in candidates if p.is_file() and _sha256(p) == manifest["mesh"]["sha256"]),
            None,
        )
        if mesh_path is None:
            raise ValueError("prepared mesh SHA-256 match not found; supply --mesh PATH")
    mesh_path = Path(mesh_path).resolve()
    if _sha256(mesh_path) != manifest["mesh"]["sha256"]:
        raise ValueError("plot mesh SHA-256 differs from the prepared mesh")
    mesh = load_triangle_obj(mesh_path)
    report = validate_triangle_mesh(mesh)
    report.require_valid()
    count = report.face_count
    if any(
        getattr(report, key) != manifest["mesh"][key]
        for key in ("vertex_count", "face_count", "edge_count")
    ):
        raise ValueError("prepared mesh counts disagree with actual mesh")
    tensor_path = _artifact(root, files["tensors"])
    layout = manifest["tensors"]
    if (
        layout["face_count"] != count
        or layout["precision"] != "complex128"
        or layout["byte_order"] != "little-endian"
        or layout["ordering"] != "per face: row-major fields-to-currents map, then its inverse"
        or tensor_path.stat().st_size != count * 32 * 16
    ):
        raise ValueError("unsupported tensor binary layout or face count")
    records = np.fromfile(tensor_path, dtype="<c16").reshape(count, 32)
    if not np.isfinite(records).all():
        raise ValueError("tensor binary contains non-finite coefficients")
    maps = records[:, :16].reshape(count, 4, 4)
    samples_path = _artifact(root, files["samples"])
    frames = compute_face_frames(mesh)
    with np.load(samples_path, allow_pickle=False) as samples:
        if not np.array_equal(samples["face_index"], np.arange(count)) or not np.array_equal(
            samples["surface_maps"], maps
        ):
            raise ValueError("saved samples and tensor maps disagree in face order or values")
        for name in ("centroids", "t1", "t2", "normals"):
            expected = getattr(frames, name)
            if samples[name].shape != expected.shape or not np.allclose(
                samples[name], expected, rtol=1e-12, atol=1e-14
            ):
                raise ValueError("saved sample geometry differs from the mesh face frame")
        residual = samples["relative_residual"].copy()
    if residual.shape != (count,) or not np.isfinite(residual).all() or np.any(residual < 0):
        raise ValueError("expected finite nonnegative per-face fitting residuals")
    provenance = {
        "input_preparation": {"directory": str(root), "sha256": _sha256(manifest_path)},
        "mesh": {"path": str(mesh_path), "sha256": _sha256(mesh_path), "face_count": count},
        "tensor_sha256": _sha256(tensor_path),
        "samples_sha256": _sha256(samples_path),
        "warnings": manifest.get("warnings", []),
    }
    return mesh, maps, residual, provenance


def _statistics(values, limits):
    minimum, maximum = int(np.argmin(values)), int(np.argmax(values))
    return {
        "minimum": float(values[minimum]),
        "minimum_face_index": minimum,
        "maximum": float(values[maximum]),
        "maximum_face_index": maximum,
        "clipped_face_count": int(np.count_nonzero((values < limits[0]) | (values > limits[1]))),
    }


def _figure(mesh, arrays, labels, *, limits, cmap, title, unit, path, dpi, show_edges):
    single = len(arrays) == 1
    fig, axes = plt.subplots(
        1 if single else 2,
        2 if single else len(arrays),
        figsize=(10, 4.8) if single else (4 * len(arrays), 6.8),
        squeeze=False,
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    try:
        norm = Normalize(*limits)
        for row, view in enumerate(VIEWS):
            for column, (values, label) in enumerate(zip(arrays, labels)):
                ax = axes[0, row] if single else axes[row, column]
                image = plot_mesh_face_values(
                    mesh, values, ax=ax, norm=norm, cmap=cmap, view=view, show_edges=show_edges
                )
                ax.set_title(f"{label}\nView from {'-y' if row == 0 else '+y'}", fontsize=11)
                ax.set_xlabel("x [m]", labelpad=0)
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.set_zlabel("z [m]", labelpad=0)
                ax.tick_params(labelsize=8, pad=0)
                # Fewer ticks remain legible in a multi-panel mesh figure.
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.zaxis.set_major_locator(MaxNLocator(5))
        below = any(np.any(values < limits[0]) for values in arrays)
        above = any(np.any(values > limits[1]) for values in arrays)
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.8,
            pad=0.03,
            label=unit,
            extend=extend,
        )
        fig.suptitle(
            title + "\nLocal face frame (1,2); flat face colors; orthographic views", fontsize=12
        )
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def plot_surface_parameters(
    inputs_dir, *, output_dir=None, mesh_path=None, percentile=98, dpi=150, show_edges=False
):
    """Save three physical tensor blocks and a separate joint-fit residual map."""
    # Validate inputs and display options before creating an output directory.
    face_value_limits([0.0], percentile=percentile)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    mesh, maps, residual, provenance = load_surface_parameters(inputs_dir, mesh_path=mesh_path)
    root = Path(inputs_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else root / "plots/surface/run01"
    summary = {
        "schema_version": 1,
        "application": "gstc_surface",
        "stage": "surface_parameter_plots",
        **provenance,
        "basis": "face_edge: t1, t2; outward triangle normal",
        "basis_rotation": False,
        "interpolation": "none",
        "lighting": False,
        "display_units": "physical tensor coefficients from face_tensors.bin",
        "notation": {"Y": "Y_ee", "chi": "Y_em", "gamma": "Y_me = -Y_em.T", "Z": "Y_mm"},
        "color_percentile": percentile,
        "percentile_weighting": "one sample per face coefficient",
        "views_degrees": VIEWS,
        "show_edges": show_edges,
        "blocks": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, title, unit, symbol, part, entries in BLOCKS:
        arrays = [getattr(maps[:, i, j], part) for i, j in entries]
        limits = face_value_limits(np.stack(arrays), percentile=percentile)
        components = [f"{i % 2 + 1}{j % 2 + 1}" for i, j in entries]
        operation = "Im" if part == "imag" else "Re"
        labels = [rf"$\operatorname{{{operation}}} {symbol}_{{{ij}}}$" for ij in components]
        filename = f"{name}.png"
        _figure(
            mesh,
            arrays,
            labels,
            limits=limits,
            cmap="RdBu_r",
            title=f"{title} | +/- {percentile:g}th percentile of absolute block values",
            unit=unit,
            path=output / filename,
            dpi=dpi,
            show_edges=show_edges,
        )
        summary["blocks"][name] = {
            "file": filename,
            "unit": unit,
            "part": part,
            "literature_symbol": symbol,
            "limits": limits,
            "components": {
                ij: _statistics(values, limits) for ij, values in zip(components, arrays)
            },
        }
    percent_residual = 100 * residual
    limits = face_value_limits(percent_residual, percentile=percentile, symmetric=False)
    _figure(
        mesh,
        [percent_residual],
        ["Joint relative fitting residual"],
        limits=limits,
        cmap="viridis",
        title=f"Joint extraction residual | {percentile:g}th percentile limit",
        unit="%",
        path=output / "fit_residual.png",
        dpi=dpi,
        show_edges=show_edges,
    )
    summary["fit_residual"] = {
        "file": "fit_residual.png",
        "unit": "%",
        "limits": limits,
        **_statistics(percent_residual, limits),
    }
    summary_path = output / "plots.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved 4 surface figures and plots.json in {output}", flush=True)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="Fresh directory; existing output is preserved"
    )
    parser.add_argument(
        "--mesh", dest="mesh_path", type=Path, help="Exact prepared mesh if relocated"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=98,
        help="Absolute block color percentile; 100 shows all extrema (default: 98)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show-edges", action="store_true")
    args = parser.parse_args()
    plot_surface_parameters(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
