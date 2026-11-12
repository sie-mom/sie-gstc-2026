"""Plot saved general-surface E/H and their prescribed targets, without solving.

Shared postprocessing handles CSV input, complex vectors, spherical projection,
ZX ordering and dB conversion. Spherical components use the global origin and
the standard z polar axis. Real/imaginary curves retain physical field units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from applications.anisotropic_gstc_sphere.prescribed_fields import (
    evaluate_prescribed_fields_cartesian,
)
from sie_mom.mesh.obj_mesh import load_triangle_obj
from sie_mom.postprocessing import (
    complex_vector,
    load_field_csv,
    spherical_components,
    spherical_coordinates_from_cartesian,
    to_decibels,
    zx_grid,
)

from .artifacts import artifact_path as _artifact
from .artifacts import read_json as _read_json
from .artifacts import sha256 as _sha256
from .artifacts import write_json as _write_json
from .workflow import load_workflow_configuration

BASES = {"spherical": ("r", "theta", "phi"), "cartesian": ("x", "y", "z")}
UNITS = {"E": "V/m", "H": "A/m"}
REGIONS = ((1, "Interior", "#007a87"), (2, "Exterior", "#c44838"))


def reference_transformation(configuration, case_id, explicit=None):
    """Match physical excitation references, never infer a target from case names."""
    case = next(case for case in configuration.simulation.cases if case.id == case_id)
    matches = []
    for transformation in configuration.transformations:
        if explicit is not None:
            matched = transformation.id == explicit
        else:
            matched = all(
                getattr(transformation.gstc, source.region) == source.field
                for source in case.excitations
            )
        if matched:
            matches.append(transformation)
    if len(matches) != 1:
        raise ValueError(
            f"expected one reference transformation for {case_id}; "
            "select --case and --reference-transformation explicitly"
        )
    return matches[0]


def prescribed_samples(data, configuration, transformation):
    """Evaluate existing Cartesian target routines only in each saved region."""
    points = np.column_stack([data[axis] for axis in "xyz"])
    result = {name: data[name] for name in ("point_index", "x", "y", "z", "region")}
    vectors = {field: np.empty((len(points), 3), dtype=complex) for field in "EH"}
    if not np.isin(data["region"], (1, 2)).all():
        raise ValueError("expected interior/exterior region labels 1 and 2")
    for region_id, region, _ in REGIONS:
        mask = data["region"] == region_id
        if not np.any(mask):
            continue
        fields = evaluate_prescribed_fields_cartesian(
            getattr(transformation.gstc, region.lower()),
            region.lower(),
            configuration,
            points_xyz=points[mask],
        )
        for field, values in zip("EH", fields):
            if not np.isfinite(values).all():
                raise ValueError("prescribed field is non-finite at an observation point")
            vectors[field][mask] = values.T
    for field in "EH":
        for index, axis in enumerate("xyz"):
            result[f"{field}{axis}_real"] = vectors[field][:, index].real
            result[f"{field}{axis}_imag"] = vectors[field][:, index].imag
    return result


def field_components(data, field, basis):
    if basis == "spherical":
        return spherical_components(data, field, mask_origin=True)
    return complex_vector(data, field)


def display_reference(target, field, *, percentile=100, region="all"):
    """One vector reference for all components; zero components keep that scale."""
    magnitudes = np.linalg.norm(complex_vector(target, field), axis=1)
    if region == "exterior":
        magnitudes = magnitudes[target["region"] == 2]
    elif region != "all":
        raise ValueError("reference region must be all or exterior")
    if not magnitudes.size:
        raise ValueError("no samples in the selected reference region")
    reference = float(np.percentile(magnitudes, percentile))
    if reference <= 0:
        reference = float(np.max(magnitudes))
    return reference if reference > 0 else 1.0


def display_magnitude(values, *, scale, reference, floor_db):
    shown = (
        to_decibels(values, reference=reference, floor_relative_db=floor_db)
        if scale == "db"
        else np.abs(values)
    )
    return np.where(np.isfinite(values), shown, np.nan)


def comparison_errors(data, target):
    """Unweighted complex-vector L2 comparisons, separately in each medium."""
    result = {}
    for region_id, region, _ in REGIONS:
        mask = data["region"] == region_id
        if not np.any(mask):
            continue
        result[region.lower()] = {}
        for field in "EH":
            desired = complex_vector(target, field)[mask]
            error = complex_vector(data, field)[mask] - desired
            denominator = float(np.linalg.norm(desired))
            result[region.lower()][field] = {
                "point_count": int(np.count_nonzero(mask)),
                "relative_complex_l2": float(np.linalg.norm(error) / denominator)
                if denominator > 0
                else None,
                "absolute_vector_rms": float(np.linalg.norm(error) / np.sqrt(len(error))),
            }
    return result


def _mesh_section(fields_dir, manifest, mesh_path):
    """Draw the actual numerical boundary; no analytic-shape approximation."""
    if mesh_path is None:
        candidates = [
            fields_dir.parent.parent / "surface.obj",
            Path(manifest["solve_directory"]) / "surface.obj",
        ]
        mesh_path = next((path for path in candidates if path.is_file()), None)
    if mesh_path is None:
        raise FileNotFoundError("saved solve mesh not found; supply --mesh PATH")
    mesh_path = Path(mesh_path).resolve()
    if _sha256(mesh_path) != manifest["mesh_sha256"]:
        raise ValueError("plot mesh SHA-256 differs from the mesh used for these fields")
    mesh = load_triangle_obj(mesh_path)
    import trimesh

    surface = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, process=False, validate=False
    )
    section = surface.section(plane_origin=(0, 0, 0), plane_normal=(0, 1, 0))
    return mesh_path, len(mesh.faces), [] if section is None else section.discrete


def _symbol(field, component):
    subscript = "\\" + component if component in ("theta", "phi") else component
    return f"{field}_{{{subscript}}}"


def _save_figure(figure, path):
    try:
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _plot_zx(
    data,
    target,
    *,
    field,
    basis,
    scale,
    reference,
    floor_db,
    sampling,
    section,
    sources,
    title,
    path,
    reference_percentile,
    reference_region,
    limits_db,
    solver_label="SIE",
):
    x, z, order = zx_grid(data, sampling["zx_grid_size"])
    arrays = [
        display_magnitude(
            field_components(table, field, basis),
            scale=scale,
            reference=reference,
            floor_db=floor_db,
        )
        for table in (data, target)
    ]
    norm = Normalize(*limits_db) if scale == "db" else Normalize(0, reference)
    figure, axes = plt.subplots(3, 2, figsize=(12, 7.8), layout="constrained", squeeze=False)
    figure.suptitle(title + "\nOutline: saved mesh; black cross: physical dipole")
    for row, component in enumerate(BASES[basis]):
        for column, (name, values) in enumerate(zip((solver_label, "Prescribed"), arrays)):
            axis = axes[row, column]
            image = axis.pcolormesh(
                x,
                z,
                values[order, row].reshape(len(z), len(x)),
                shading="auto",
                cmap="viridis",
                norm=norm,
            )
            for curve in section:
                axis.plot(curve[:, 0], curve[:, 2], color="black", linewidth=1.7)
                axis.plot(curve[:, 0], curve[:, 2], color="white", linewidth=0.8)
            for source in sources:
                if np.isclose(source[1], 0, atol=1e-12, rtol=0):
                    axis.plot(source[0], source[2], "kx", markersize=5)
            axis.set(title=f"{name}: $|{_symbol(field, component)}|$", ylabel="z [m]")
            axis.set_aspect("equal")
            axis.set_xlim(x[0], x[-1])
            axis.set_ylim(z[0], z[-1])
            if row == 2:
                axis.set_xlabel("x [m]")
    if scale == "db":
        label = (
            f"dB re {reference:.4g} {UNITS[field]} "
            f"(prescribed {reference_region} vector p{reference_percentile:g}); "
            f"display range [{limits_db[0]:g}, {limits_db[1]:g}] dB"
        )
    else:
        label = (
            f"Field magnitude [{UNITS[field]}]; "
            f"upper color limit: prescribed {reference_region} vector p{reference_percentile:g}"
        )
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label=label,
        shrink=0.94,
        extend="both" if scale == "db" else "max",
    )
    _save_figure(figure, path)


def _regional_curves(axis, radii, regions, computed, prescribed, solver_label="SIE"):
    for region_id, name, color in REGIONS:
        mask = regions == region_id
        if np.any(mask):
            # NaNs break curves across intervening regions of a concave object.
            axis.plot(
                radii, np.where(mask, computed, np.nan), color=color, label=f"{name} {solver_label}"
            )
            axis.plot(
                radii,
                np.where(mask, prescribed, np.nan),
                color=color,
                linestyle="--",
                label=f"{name} prescribed",
            )
    crossings = np.flatnonzero(regions[1:] != regions[:-1])
    for index in crossings:
        axis.axvline((radii[index] + radii[index + 1]) / 2, color="0.5", linestyle=":")
    axis.grid(alpha=0.25)


def _plot_radial(
    data,
    target,
    *,
    field,
    basis,
    scale,
    reference,
    floor_db,
    title,
    path,
    complex_path,
    solver_label="SIE",
):
    radii, _, _ = spherical_coordinates_from_cartesian(*[data[axis] for axis in "xyz"])
    order = np.argsort(radii)
    radii, regions = radii[order], data["region"][order]
    computed = field_components(data, field, basis)[order]
    prescribed = field_components(target, field, basis)[order]
    display = [
        display_magnitude(values, scale=scale, reference=reference, floor_db=floor_db)
        for values in (computed, prescribed)
    ]
    figure, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, layout="constrained")
    figure.suptitle(title)
    for index, (axis, component) in enumerate(zip(axes, BASES[basis])):
        _regional_curves(
            axis, radii, regions, display[0][:, index], display[1][:, index], solver_label
        )
        axis.set_ylabel(
            f"$|{_symbol(field, component)}|$ " + ("[dB]" if scale == "db" else f"[{UNITS[field]}]")
        )
        if scale == "db":
            axis.set_ylim(bottom=-floor_db - 2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("r from global origin [m]")
    if scale == "db":
        figure.supxlabel(
            f"Common reference: {reference:.4g} {UNITS[field]}; floor: -{floor_db:g} dB"
        )
    _save_figure(figure, path)

    figure, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True, layout="constrained")
    figure.suptitle(title)
    for row, component in enumerate(BASES[basis]):
        for column, (name, part) in enumerate((("Re", np.real), ("Im", np.imag))):
            axis = axes[row, column]
            _regional_curves(
                axis, radii, regions, part(computed[:, row]), part(prescribed[:, row]), solver_label
            )
            axis.set_ylabel(rf"${name}({_symbol(field, component)})$ [{UNITS[field]}]")
    axes[0, 0].legend(fontsize=8, ncol=2)
    for axis in axes[-1]:
        axis.set_xlabel("r from global origin [m]")
    _save_figure(figure, complex_path)


def plot_fields(
    fields_dir,
    *,
    output_dir=None,
    case_id=None,
    reference_id=None,
    basis="spherical",
    scale="db",
    floor_db=80,
    map_reference_percentile=98,
    map_reference_region="all",
    vmin_db=None,
    vmax_db=None,
    mesh_path=None,
):
    """Generate paired component plots and raw complex error summaries."""
    if basis not in BASES or scale not in ("db", "linear"):
        raise ValueError("unsupported plotting basis or scale")
    if not np.isfinite(floor_db) or floor_db <= 0:
        raise ValueError("floor_db must be finite and positive")
    if not np.isfinite(map_reference_percentile) or not 0 < map_reference_percentile <= 100:
        raise ValueError("map_reference_percentile must be in (0, 100]")
    if map_reference_region not in ("all", "exterior"):
        raise ValueError("map_reference_region must be all or exterior")
    if scale != "db" and (vmin_db is not None or vmax_db is not None):
        raise ValueError("dB color limits require --scale db")
    limits_db = (
        float(-floor_db if vmin_db is None else vmin_db),
        float(0 if vmax_db is None else vmax_db),
    )
    if not np.isfinite(limits_db).all() or limits_db[0] >= limits_db[1]:
        raise ValueError("dB color limits must be finite with vmin_db < vmax_db")
    # A zero-value floor must not truncate a deeper requested color range.
    map_floor_db = max(floor_db, -limits_db[0])
    if reference_id is not None and case_id is None:
        raise ValueError("an explicit reference transformation requires --case")
    fields_dir = Path(fields_dir).resolve()
    manifest = _read_json(fields_dir / "fields.json")
    if manifest.get("application") != "gstc_surface" or manifest.get("status") != "evaluated":
        raise ValueError("expected an evaluated gstc_surface field run")
    configuration = load_workflow_configuration(fields_dir / "configuration.json")
    mesh_path, face_count, section = _mesh_section(fields_dir, manifest, mesh_path)
    cases = [case for case in manifest["cases"] if case_id in (None, case["id"])]
    if not cases:
        raise ValueError(f"unknown saved field case: {case_id}")
    prepared = []
    for case in cases:
        transformation = reference_transformation(configuration, case["id"], reference_id)
        grids = {}
        for name in ("zx", "radial"):
            saved = case["grids"][name]
            path = _artifact(fields_dir, saved["path"])
            if _sha256(path) != saved["sha256"]:
                raise ValueError("field CSV SHA-256 mismatch")
            data = load_field_csv(path)
            if name == "zx":
                zx_grid(data, manifest["sampling"]["zx_grid_size"])
            target = prescribed_samples(data, configuration, transformation)
            references = {
                field: display_reference(
                    target,
                    field,
                    percentile=map_reference_percentile if name == "zx" else 100,
                    region=map_reference_region if name == "zx" else "all",
                )
                for field in "EH"
            }
            grids[name] = (data, target, references)
        prepared.append((case["id"], transformation, grids))
    output_dir = Path(output_dir or fields_dir / "plots/run01").resolve()
    if output_dir.exists():
        raise FileExistsError(f"plot directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "application": "gstc_surface",
        "stage": "plots",
        "fields_directory": str(fields_dir),
        "fields_manifest_sha256": _sha256(fields_dir / "fields.json"),
        "mesh": {
            "path": str(mesh_path),
            "sha256": manifest["mesh_sha256"],
            "face_count": face_count,
        },
        "basis": basis,
        "origin_m": [0, 0, 0],
        "polar_axis": "z" if basis == "spherical" else None,
        "scale": scale,
        "floor_db": floor_db,
        "warnings": manifest.get("warnings", []),
        "solver_label": manifest.get("solver_label", "SIE"),
        "map_reference_percentile": map_reference_percentile,
        "map_reference_region": map_reference_region,
        "map_limits_db": list(limits_db) if scale == "db" else None,
        "map_floor_db": map_floor_db if scale == "db" else None,
        "cases": [],
    }
    for case_id, transformation, grids in prepared:
        directory = output_dir / case_id
        directory.mkdir()
        case = next(case for case in configuration.simulation.cases if case.id == case_id)
        sources = [
            configuration.fields[source.field.field_id].source_position
            for source in case.excitations
            if hasattr(configuration.fields[source.field.field_id], "source_position")
        ]
        report = {
            "id": case_id,
            "reference_transformation": transformation.id,
            "grids": {},
            "files": [],
        }
        for name, (data, target, references) in grids.items():
            report["grids"][name] = {
                "errors": comparison_errors(data, target),
                "display_references": {},
            }
            frame = (
                "spherical components; global origin (0,0,0), z polar axis"
                if basis == "spherical"
                else "Cartesian components; radial distances from global origin (0,0,0)"
            )
            title = f"{case_id} vs {transformation.id}; {face_count} triangles\n{frame}"
            for field in "EH":
                percentile = map_reference_percentile if name == "zx" else 100
                reference = references[field]
                report["grids"][name]["display_references"][field] = reference
                path = directory / f"{name}_{field}_{basis}_{scale}.png"
                options = dict(
                    field=field,
                    basis=basis,
                    scale=scale,
                    reference=reference,
                    floor_db=map_floor_db if name == "zx" else floor_db,
                    title=title,
                    path=path,
                    solver_label=summary["solver_label"],
                )
                if name == "zx":
                    _plot_zx(
                        data,
                        target,
                        sampling=manifest["sampling"],
                        section=section,
                        sources=sources,
                        reference_percentile=percentile,
                        reference_region=map_reference_region,
                        limits_db=limits_db,
                        **options,
                    )
                else:
                    complex_path = directory / f"radial_{field}_{basis}_complex.png"
                    _plot_radial(data, target, complex_path=complex_path, **options)
                    report["files"].append(str(complex_path.relative_to(output_dir)))
                report["files"].append(str(path.relative_to(output_dir)))
        summary["cases"].append(report)
        print(f"{case_id}: {len(report['files'])} figures saved in {directory}", flush=True)
    _write_json(output_dir / "plots.json", summary)
    return output_dir / "plots.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--reference-transformation", dest="reference_id")
    parser.add_argument("--basis", choices=tuple(BASES), default="spherical")
    parser.add_argument("--scale", choices=("db", "linear"), default="db")
    parser.add_argument(
        "--floor-db",
        type=float,
        default=80,
        help="Positive dB depth for numerical zeros and radial curves (default: 80)",
    )
    parser.add_argument(
        "--map-reference-percentile",
        type=float,
        default=98,
        help="Percentile of prescribed vector magnitude used as map reference",
    )
    parser.add_argument(
        "--map-reference-region",
        choices=("all", "exterior"),
        default="all",
        help="Samples used for the common map reference (default: all)",
    )
    parser.add_argument("--vmin-db", type=float, help="Map color minimum (default: -floor-db)")
    parser.add_argument("--vmax-db", type=float, help="Map color maximum (default: 0)")
    parser.add_argument("--mesh", dest="mesh_path", type=Path)
    args = vars(parser.parse_args())
    fields_dir = args.pop("fields_dir")
    print(plot_fields(fields_dir, **args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
