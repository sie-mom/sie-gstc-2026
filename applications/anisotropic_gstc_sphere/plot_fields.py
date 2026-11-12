"""Plot saved physical E/H fields for independent spherical GSTC simulation cases.

Read run metadata and its configuration snapshot without solving. An explicitly
selected transformation supplies optional prescribed total-field comparisons.
Figures use the saved sample coordinates and are written to each case's plots
directory. Spherical components are referred to the sphere center, not to a
dipole's source position. Field amplitudes are never normalized or rescaled.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from sie_mom.postprocessing import (
    complex_vector,
    load_field_csv,
    spherical_components,
    spherical_coordinates_from_cartesian,
    zx_grid,
)

if __package__:
    from .prescribed_fields import evaluate_prescribed_fields
    from .workflow import DipoleFieldDefinition, load_workflow_configuration
else:
    from prescribed_fields import evaluate_prescribed_fields
    from workflow import DipoleFieldDefinition, load_workflow_configuration


BASES = {"spherical": ("r", "theta", "phi"), "cartesian": ("x", "y", "z")}
FIELD_PARTS = ("magnitude", "real", "imag", "phase")
REGION_STYLES = ((1, "Interior", "#007a87"), (3, "Shell", "#777777"), (2, "Exterior", "#c44838"))
REGION_MARKERS = {1: "o", 2: "^", 3: "s"}
FIELD_UNITS = {"E": "V/m", "H": "A/m"}
MAGNITUDE_SCALES = ("linear", "log", "db")
MAP_COLOR_LIMITS = ("percentile", "full")
DEFAULT_MAP_PERCENTILES = (2, 98)


def _artifact_path(run_dir: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"artifact path must be relative to the run directory: {relative}")
    resolved = (run_dir / path).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError(f"artifact path escapes the run directory: {relative}")
    return resolved


def _coordinates(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([data[axis] for axis in "xyz"])


def _spherical_coordinates(data):
    return spherical_coordinates_from_cartesian(data["x"], data["y"], data["z"])



def _field_components(data, field, basis):
    if basis == "cartesian":
        return complex_vector(data, field)
    return spherical_components(data, field, mask_origin=True)


def _quantity(data, selection, basis):
    field = selection[0]
    if selection in FIELD_UNITS:
        return np.linalg.norm(complex_vector(data, field), axis=1)
    return _field_components(data, field, basis)[:, BASES[basis].index(selection[2:])]


def _quantity_symbol(selection):
    if selection in FIELD_UNITS:
        return selection
    component = selection[2:]
    if component in ("theta", "phi"):
        component = "\\" + component
    return f"{selection[0]}_{{{component}}}"


def prescribed_samples(data, configuration, transformation):
    """Evaluate the selected regional targets on saved points, without scaling.

    Only the reference belonging to each point's saved region is evaluated.
    Exact dipole-source samples are undefined and are masked in the figures.
    """

    radii, theta, phi = _spherical_coordinates(data)
    coordinates = _coordinates(data)
    vectors = {field: np.full((len(radii), 3), np.nan, dtype=complex) for field in "EH"}
    for region_id, region in ((1, "interior"), (2, "exterior")):
        reference = getattr(transformation.gstc, region)
        indices = np.flatnonzero(data["region"] == region_id)
        if reference is not None:
            definition = configuration.fields[reference.field_id]
            if isinstance(definition, DipoleFieldDefinition):
                distance = np.linalg.norm(coordinates[indices] - definition.source_position, axis=1)
                tolerance = (
                    32
                    * np.finfo(float).eps
                    * max(
                        1, configuration.case.radius_m, np.linalg.norm(definition.source_position)
                    )
                )
                indices = indices[distance > tolerance]
        if not len(indices):
            continue
        electric, magnetic = evaluate_prescribed_fields(
            reference,
            region,
            configuration,
            radius=radii[indices],
            theta=theta[indices],
            phi=phi[indices],
        )
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ValueError("non-finite prescribed fields away from source singularities")
        st, ct = np.sin(theta[indices]), np.cos(theta[indices])
        sp, cp = np.sin(phi[indices]), np.cos(phi[indices])
        for field, spherical in (("E", electric), ("H", magnetic)):
            r, t, p = spherical
            vectors[field][indices] = np.column_stack(
                (
                    r * st * cp + t * ct * cp - p * sp,
                    r * st * sp + t * ct * sp + p * cp,
                    r * ct - t * st,
                )
            )
    result = {name: data[name] for name in ("point_index", "x", "y", "z", "region")}
    for field in "EH":
        for column, axis in enumerate("xyz"):
            result[f"{field}{axis}_real"] = vectors[field][:, column].real
            result[f"{field}{axis}_imag"] = vectors[field][:, column].imag
    return result


def _save_figure(figure, path: Path) -> None:
    try:
        figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)


def _radial_curve(axis, radii, values, regions, reference_values=None) -> None:
    for region, label, color in REGION_STYLES:
        mask = regions == region
        if np.any(mask):
            axis.plot(
                radii[mask],
                values[mask],
                color=color,
                label=label if reference_values is None else f"{label} computed",
                linestyle="-" if reference_values is None else ":",
                linewidth=1.4 if reference_values is None else 0.8,
                marker=None if reference_values is None else REGION_MARKERS[region],
                markerfacecolor="none",
                markersize=4.5,
                markevery=max(1, np.count_nonzero(mask) // 12),
                zorder=3,
            )
            if reference_values is not None:
                axis.plot(
                    radii[mask],
                    reference_values[mask],
                    color="black",
                    label=f"{label} prescribed",
                    linestyle="-",
                    linewidth=1.4,
                    zorder=2,
                )
    axis.axvline(1, color="black", linestyle="--", linewidth=0.9)
    axis.grid(True, alpha=0.25)


def _display_magnitude(values, scale):
    """Convert magnitudes for display only; dB references are one SI field unit."""

    if scale != "db":
        return np.ma.masked_invalid(values)
    positive = np.ma.masked_less_equal(np.ma.masked_invalid(values), 0)
    return 20 * np.ma.log10(positive)


def _magnitude_label(field, scale):
    symbol = _quantity_symbol(field)
    if scale == "db":
        return f"$|{symbol}|$ [dB re 1 {FIELD_UNITS[field[0]]}]"
    return f"$|{symbol}|$ [{FIELD_UNITS[field[0]]}]"


def _display_values(values, part, scale):
    values = np.ma.masked_invalid(values)
    if part == "magnitude":
        return _display_magnitude(np.abs(values), scale)
    if part == "phase":
        return np.ma.masked_where(
            np.abs(values) == 0, np.ma.masked_invalid(np.angle(values, deg=True))
        )
    return values.real if part == "real" else values.imag


def _value_label(selection, part, scale):
    if part == "magnitude":
        return _magnitude_label(selection, scale)
    symbol = _quantity_symbol(selection)
    if part == "phase":
        return rf"$\arg({symbol})$ [deg]"
    operator = "Re" if part == "real" else "Im"
    return rf"$\mathrm{{{operator}}}({symbol})$ [{FIELD_UNITS[selection[0]]}]"


def _comparison_suffix(reference_id):
    return "" if reference_id is None else f"_vs_{reference_id}"


def _selected_filename(cut, basis, selections, part, reference_id):
    return f"{cut}_{basis}_{'_'.join(selections)}_{part}{_comparison_suffix(reference_id)}.png"


def _plot_selected_radial(
    data,
    *,
    radius_m,
    title,
    directory,
    scale,
    basis,
    selections,
    part,
    reference,
    reference_id,
):
    radii = np.linalg.norm(_coordinates(data), axis=1) / radius_m
    order = np.argsort(radii)
    radii, regions = radii[order], data["region"][order]
    figure, axes = plt.subplots(
        len(selections),
        1,
        figsize=(9, max(4, 3 * len(selections))),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    figure.suptitle(title)
    for axis, selection in zip(axes[:, 0], selections):
        values = _display_values(_quantity(data, selection, basis), part, scale)[order]
        reference_values = (
            None if reference is None else _quantity(reference, selection, basis)[order]
        )
        prescribed = (
            None if reference_values is None else _display_values(reference_values, part, scale)
        )
        _radial_curve(axis, radii, values, regions, prescribed)
        finite = np.ma.concatenate(
            [values] if prescribed is None else [values, prescribed]
        ).compressed()
        if part == "magnitude" and scale == "log" and np.any(finite > 0):
            axis.set_yscale("log", nonpositive="mask")
        if not finite.size:
            axis.text(0.5, 0.5, "No finite displayed values", transform=axis.transAxes, ha="center")
        if reference_values is not None and part == "magnitude" and scale in ("log", "db"):
            zero_regions = []
            for region, label, _ in REGION_STYLES:
                regional = np.ma.masked_invalid(reference_values[regions == region]).compressed()
                if regional.size and np.all(regional == 0):
                    zero_regions.append(label)
            if zero_regions:
                axis.text(
                    0.02,
                    0.95,
                    f"Prescribed = 0: {', '.join(zero_regions)}",
                    transform=axis.transAxes,
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
        axis.set_ylabel(_value_label(selection, part, scale))
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("r/a")
    path = directory / _selected_filename("radial", basis, selections, part, reference_id)
    _save_figure(figure, path)
    return [path]


def _plot_radial(
    data,
    *,
    radius_m,
    title,
    directory,
    magnitude_scale,
    basis="spherical",
    selections=None,
    part="magnitude",
    reference=None,
    reference_id=None,
) -> list[Path]:
    coordinates = _coordinates(data)
    first_radius = np.linalg.norm(coordinates[0])
    theta = np.degrees(np.arccos(np.clip(coordinates[0, 2] / first_radius, -1, 1)))
    phi = np.degrees(np.arctan2(coordinates[0, 1], coordinates[0, 0]))
    title += "\n" + rf"Radial cut: $\theta = {theta:g}^\circ$, $\phi = {phi:g}^\circ$"
    if selections is not None:
        return _plot_selected_radial(
            data,
            radius_m=radius_m,
            title=title,
            directory=directory,
            scale=magnitude_scale,
            basis=basis,
            selections=selections,
            part=part,
            reference=reference,
            reference_id=reference_id,
        )
    radii = np.linalg.norm(_coordinates(data), axis=1) / radius_m
    order = np.argsort(radii)
    radii, regions = radii[order], data["region"][order]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
    figure.suptitle(title)
    for axis, field in zip(axes, "EH"):
        magnitude = np.linalg.norm(complex_vector(data, field), axis=1)[order]
        prescribed = (
            None
            if reference is None
            else np.linalg.norm(complex_vector(reference, field), axis=1)[order]
        )
        _radial_curve(
            axis,
            radii,
            _display_magnitude(magnitude, magnitude_scale),
            regions,
            None if prescribed is None else _display_magnitude(prescribed, magnitude_scale),
        )
        if magnitude_scale == "log" and (
            np.any(magnitude > 0) or (prescribed is not None and np.any(prescribed > 0))
        ):
            axis.set_yscale("log", nonpositive="mask")
        if magnitude_scale == "db" and not np.any(magnitude > 0):
            axis.text(0.5, 0.5, r"Zero field ($-\infty$ dB)", transform=axis.transAxes, ha="center")
        axis.set_ylabel(_magnitude_label(field, magnitude_scale))
    axes[0].legend()
    axes[-1].set_xlabel("r/a")
    paths = [directory / f"radial_magnitudes{_comparison_suffix(reference_id)}.png"]
    _save_figure(figure, paths[0])
    for component in BASES[basis]:
        paths.extend(
            _plot_selected_radial(
                data,
                radius_m=radius_m,
                title=title,
                directory=directory,
                scale=magnitude_scale,
                basis=basis,
                selections=(f"E_{component}", f"H_{component}"),
                part="magnitude",
                reference=reference,
                reference_id=reference_id,
            )
        )
    for field in "EH":
        field_name = "electric" if field == "E" else "magnetic"
        components = _field_components(data, field, basis)[order]
        prescribed = (
            None if reference is None else _field_components(reference, field, basis)[order]
        )
        figure, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, layout="constrained")
        figure.suptitle(title)
        for row, component in enumerate(BASES[basis]):
            for column, (part, values) in enumerate(
                (
                    ("Re", _display_values(components, "real", "linear")),
                    ("Im", _display_values(components, "imag", "linear")),
                )
            ):
                axis = axes[row, column]
                target = (
                    None
                    if prescribed is None
                    else _display_values(prescribed, "real" if column == 0 else "imag", "linear")[
                        :, row
                    ]
                )
                _radial_curve(axis, radii, values[:, row], regions, target)
                symbol = _quantity_symbol(f"{field}_{component}")
                axis.set_ylabel(rf"$\mathrm{{{part}}}({symbol})$ [{FIELD_UNITS[field]}]")
        axes[0, 0].legend()
        for axis in axes[-1]:
            axis.set_xlabel("r/a")
        basis_suffix = "" if basis == "spherical" else "_cartesian"
        path = (
            directory
            / f"radial_{field_name}_components{basis_suffix}{_comparison_suffix(reference_id)}.png"
        )
        _save_figure(figure, path)
        paths.append(path)
    return paths


def _color_bounds(values, color_limits, percentiles=DEFAULT_MAP_PERCENTILES):
    if color_limits == "percentile":
        lower, upper = np.percentile(values, percentiles)
        return float(lower), float(upper)
    return float(values.min()), float(values.max())


def _magnitude_norm(values, scale, color_limits="full", percentiles=DEFAULT_MAP_PERCENTILES):
    values = np.ma.masked_invalid(values).compressed()
    if not values.size:
        return Normalize(vmin=-1 if scale == "db" else 0, vmax=1)
    if scale == "db":
        lower, upper = _color_bounds(values, color_limits, percentiles)
        if np.isclose(lower, upper, rtol=1e-12, atol=0):
            lower, upper = lower - 1, upper + 1
        return Normalize(vmin=lower, vmax=upper)
    positive = values[values > 0]
    if scale == "log" and positive.size:
        lower, upper = _color_bounds(positive, color_limits, percentiles)
        if np.isclose(lower, upper, rtol=1e-12, atol=0):
            lower, upper = lower / 10, upper * 10
        return LogNorm(vmin=lower, vmax=upper)
    maximum = float(values.max())
    if color_limits == "percentile":
        lower, upper = _color_bounds(values, color_limits, percentiles)
        if not np.isclose(lower, upper, rtol=1e-12, atol=0):
            return Normalize(vmin=lower, vmax=upper)
        return Normalize(vmin=0, vmax=upper * 1.01 if upper > 0 else 1)
    return Normalize(vmin=0, vmax=maximum if maximum > 0 else 1)


def _color_norm(values, part, scale, color_limits="full", percentiles=DEFAULT_MAP_PERCENTILES):
    if part == "magnitude":
        return _magnitude_norm(values, scale, color_limits, percentiles)
    if part == "phase":
        return Normalize(vmin=-180, vmax=180)
    finite = np.ma.masked_invalid(values).compressed()
    lower, upper = _color_bounds(finite, color_limits, percentiles) if finite.size else (0, 0)
    limit = max(abs(lower), abs(upper))
    return Normalize(vmin=-limit if limit else -1, vmax=limit if limit else 1)


def _plot_zx(
    data,
    *,
    radius_m,
    grid,
    title,
    directory,
    magnitude_scale,
    sources,
    basis="spherical",
    selections=None,
    part="magnitude",
    reference=None,
    reference_id=None,
    reference_sources=(),
    color_limits="percentile",
    percentiles=DEFAULT_MAP_PERCENTILES,
) -> Path:
    x, z, order = grid
    selected = ("E", "H") if selections is None else selections
    rows = len(selected) if reference is not None else (len(selected) + 1) // 2
    columns = 2 if reference is not None or len(selected) > 1 else 1
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, max(5, 5 * rows)), squeeze=False, layout="constrained"
    )
    figure.suptitle(title)
    cmap = "viridis" if part == "magnitude" else "twilight" if part == "phase" else "RdBu_r"
    for index, selection in enumerate(selected):
        displayed = [
            _display_values(_quantity(data, selection, basis), part, magnitude_scale)[
                order
            ].reshape(len(z), len(x))
        ]
        if reference is not None:
            displayed.append(
                _display_values(_quantity(reference, selection, basis), part, magnitude_scale)[
                    order
                ].reshape(len(z), len(x))
            )
        shared_values = np.ma.concatenate([array.ravel() for array in displayed])
        norm = _color_norm(shared_values, part, magnitude_scale, color_limits, percentiles)
        finite = shared_values.compressed()
        below = finite.size and finite.min() < norm.vmin
        above = finite.size and finite.max() > norm.vmax
        extend = "both" if below and above else "min" if below else "max" if above else "neither"
        for column, display in enumerate(displayed):
            axis = axes[index, column] if reference is not None else axes.ravel()[index]
            artist = axis.pcolormesh(
                x / radius_m, z / radius_m, display, shading="nearest", cmap=cmap, norm=norm
            )
            if display.count() == 0:
                message = "Zero or undefined field" if part == "magnitude" else "Undefined values"
                axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center")
            if finite.size:
                figure.colorbar(
                    artist,
                    ax=axis,
                    label=_value_label(selection, part, magnitude_scale),
                    extend=extend,
                )
            axis.add_patch(
                plt.Circle((0, 0), 1, edgecolor="white", facecolor="none", linestyle="--")
            )
            panel_sources = reference_sources if column == 1 else sources
            for label, position in panel_sources:
                if abs(position[1]) <= 1e-10 * radius_m:
                    axis.scatter(
                        position[0] / radius_m,
                        position[2] / radius_m,
                        marker="x",
                        color="red",
                        s=40,
                        label=label,
                    )
            symbol = _quantity_symbol(selection)
            quantity_title = (
                f"$|{symbol}|$"
                if part == "magnitude"
                else _value_label(selection, part, magnitude_scale).split(" [")[0]
            )
            panel_label = "Prescribed" if column == 1 else "Computed"
            axis.set(
                xlabel="x/a",
                ylabel="z/a",
                title=f"{panel_label} {quantity_title}"
                if reference is not None
                else quantity_title,
                xlim=(x[0] / radius_m, x[-1] / radius_m),
                ylim=(z[0] / radius_m, z[-1] / radius_m),
            )
            axis.set_aspect("equal")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8, loc="upper right")
    if reference is None:
        for axis in axes.ravel()[len(selected) :]:
            axis.set_visible(False)
    filename = (
        f"zx_magnitudes{_comparison_suffix(reference_id)}.png"
        if selections is None
        else _selected_filename("zx", basis, selected, part, reference_id)
    )
    path = directory / filename
    _save_figure(figure, path)
    return path


def _validate_selection(basis, selections, part, scale, name):
    if basis not in BASES:
        raise ValueError(f"{name} basis must be spherical or cartesian")
    if part not in FIELD_PARTS:
        raise ValueError(f"{name} part must be magnitude, real, imag or phase")
    if selections is None:
        if part != "magnitude":
            raise ValueError(f"{name} components must be selected for {part} plots")
        return
    allowed = ("E", "H") + tuple(
        f"{field}_{component}" for field in "EH" for component in BASES[basis]
    )
    if (
        isinstance(selections, str)
        or not selections
        or len(set(selections)) != len(selections)
        or any(selection not in allowed for selection in selections)
    ):
        raise ValueError(f"{name} components must be unique selections from {', '.join(allowed)}")
    if part != "magnitude":
        if scale != "linear":
            raise ValueError(f"{name} {part} plots require a linear scale")
        if any(selection in FIELD_UNITS for selection in selections):
            raise ValueError(f"{name} {part} plots require vector components, not E/H norms")


def plot_fields(
    run_dir: Path,
    *,
    case_id: str | None = None,
    magnitude_scale: str = "log",
    line_scale: str | None = None,
    map_scale: str | None = None,
    map_color_limits: str = "percentile",
    map_percentiles: tuple[float, float] = DEFAULT_MAP_PERCENTILES,
    line_basis: str = "spherical",
    map_basis: str = "spherical",
    line_components: tuple[str, ...] | None = None,
    map_components: tuple[str, ...] | None = None,
    line_part: str = "magnitude",
    map_part: str = "magnitude",
    reference_transformation: str | None = None,
) -> list[Path]:
    """Plot one completed case or all cases, without changing any solver artifacts."""

    for name, scale in (("magnitude", magnitude_scale), ("line", line_scale), ("map", map_scale)):
        if scale not in MAGNITUDE_SCALES and (name == "magnitude" or scale is not None):
            raise ValueError(f"{name} scale must be linear, log or db")
    line_scale = magnitude_scale if line_scale is None else line_scale
    map_scale = magnitude_scale if map_scale is None else map_scale
    if map_color_limits not in MAP_COLOR_LIMITS:
        raise ValueError("map color limits must be percentile or full")
    try:
        percentiles = np.asarray(map_percentiles, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("map percentiles must satisfy 0 <= LOW < HIGH <= 100") from None
    if (
        percentiles.shape != (2,)
        or not np.isfinite(percentiles).all()
        or not 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError("map percentiles must satisfy 0 <= LOW < HIGH <= 100")
    map_percentiles = tuple(percentiles)
    _validate_selection(line_basis, line_components, line_part, line_scale, "line")
    _validate_selection(map_basis, map_components, map_part, map_scale, "map")
    if reference_transformation is not None and case_id is None:
        raise ValueError("a reference transformation requires an explicit simulation --case")
    run_dir = run_dir.resolve()
    with (run_dir / "run.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("application") != "anisotropic_gstc_sphere"
        or manifest.get("backend") not in ("sie", "svwf", "comsol")
    ):
        raise ValueError(
            "expected an anisotropic GSTC SIE, SVWF, or COMSOL run manifest with schema_version 2"
        )
    configuration = load_workflow_configuration(
        _artifact_path(run_dir, manifest["files"]["configuration"])
    )
    reference = None
    reference_sources = []
    if reference_transformation is not None:
        reference = next(
            (
                state
                for state in configuration.transformations
                if state.id == reference_transformation
            ),
            None,
        )
        if reference is None:
            raise ValueError(f"unknown reference transformation: {reference_transformation}")
        for region in ("interior", "exterior"):
            field_reference = getattr(reference.gstc, region)
            if field_reference is not None:
                definition = configuration.fields[field_reference.field_id]
                if isinstance(definition, DipoleFieldDefinition):
                    reference_sources.append(
                        (f"{region}: {field_reference.field_id}", definition.source_position)
                    )
    radius = configuration.case.radius_m
    if not (
        np.isclose(manifest["radius_m"], radius, rtol=1e-12, atol=0)
        and np.isclose(
            manifest["frequency_hz"], configuration.case.frequency_hz, rtol=1e-12, atol=0
        )
    ):
        raise ValueError("run radius or frequency disagrees with its configuration snapshot")
    field_data = manifest["field_data"]
    if any(
        field_data.get(key) != value
        for key, value in {
            "coordinate_system": "Cartesian",
            "coordinate_unit": "m",
            "electric_field_unit": "V/m",
            "magnetic_field_unit": "A/m",
        }.items()
    ):
        raise ValueError("unsupported saved field coordinates or units")
    cases = manifest["simulation"]["cases"]
    if case_id is not None:
        cases = [case for case in cases if case["id"] == case_id]
        if not cases:
            raise ValueError(f"unknown simulation case: {case_id}")
    if not cases:
        raise ValueError("run has no simulation cases to plot")
    definitions = {case.id: case for case in configuration.simulation.cases}
    saved_cases = {case["id"]: case for case in configuration.to_dict()["simulation"]["cases"]}
    prepared = []
    for case in cases:
        if case.get("status") != "completed":
            raise ValueError(f"simulation case {case['id']} is not completed")
        if case["id"] not in definitions:
            raise ValueError(
                f"simulation case {case['id']} is missing from the configuration snapshot"
            )
        if case["excitations"] != saved_cases[case["id"]]["excitations"]:
            raise ValueError(f"simulation case {case['id']} excitations disagree with its snapshot")
        files = case["files"]
        if "computed_radial" not in files or "computed_zx" not in files:
            raise ValueError(f"simulation case {case['id']} has no saved physical fields")
        radial = load_field_csv(_artifact_path(run_dir, files["computed_radial"]), radius_m=radius)
        zx = load_field_csv(_artifact_path(run_dir, files["computed_zx"]), radius_m=radius)
        expected_radial_points = manifest.get("sampling", {}).get(
            "radial_points", configuration.sampling.radial_points
        )
        if len(radial["x"]) != expected_radial_points:
            raise ValueError("radial field count does not match the saved sampling")
        radii = np.linalg.norm(_coordinates(radial), axis=1)
        if np.any(radii == 0) or len(np.unique(radii)) != len(radii):
            raise ValueError("radial field coordinates must have distinct, nonzero radii")
        directions = _coordinates(radial) / radii[:, None]
        if not np.allclose(directions, directions[0], rtol=0, atol=1e-6):
            raise ValueError("radial field samples do not lie on one radial ray")
        expected_zx_grid = manifest.get("sampling", {}).get(
            "zx_grid_size", configuration.sampling.zx_grid_size
        )
        grid = zx_grid(zx, expected_zx_grid)
        directory = _artifact_path(run_dir, files["plots"])
        sources = []
        for excitation in definitions[case["id"]].excitations:
            field = configuration.fields[excitation.field.field_id]
            if isinstance(field, DipoleFieldDefinition):
                sources.append(
                    (f"{excitation.region}: {excitation.field.field_id}", field.source_position)
                )
        title = fill(f"{manifest['case_id']} / {case['id']}", width=75)
        title += (
            f"\nPhysical fields | {configuration.case.frequency_hz / 1e6:g} MHz | a = {radius:g} m"
        )
        if manifest["backend"] == "svwf":
            label = {"forward": "forward validation", "inverse": "inverse candidate"}.get(
                manifest.get("solution"), "fields"
            )
            title += f"\nSVWF {label}"
        elif manifest["backend"] == "comsol":
            title += "\nCOMSOL FEM validation"
        if reference is not None:
            title += "\n" + fill(f"Prescribed transformation: {reference.id}", width=75)
            if manifest.get("post_processing", {}).get("interior_reference_enforced") is False:
                title += "\nInterior reference: initial prescription (not enforced)"
        target_radial = (
            None if reference is None else prescribed_samples(radial, configuration, reference)
        )
        target_zx = None if reference is None else prescribed_samples(zx, configuration, reference)
        prepared.append((radial, zx, grid, directory, sources, title, target_radial, target_zx))
    paths = []
    with plt.rc_context({"font.size": 10}):
        for radial, zx, grid, directory, sources, title, target_radial, target_zx in prepared:
            directory.mkdir(parents=True, exist_ok=True)
            paths.extend(
                _plot_radial(
                    radial,
                    radius_m=radius,
                    title=title,
                    directory=directory,
                    magnitude_scale=line_scale,
                    basis=line_basis,
                    selections=line_components,
                    part=line_part,
                    reference=target_radial,
                    reference_id=reference_transformation,
                )
            )
            map_selections = (
                [None] + [(f"E_{component}", f"H_{component}") for component in BASES[map_basis]]
                if map_components is None
                else [map_components]
            )
            for selections in map_selections:
                paths.append(
                    _plot_zx(
                        zx,
                        radius_m=radius,
                        grid=grid,
                        title=title,
                        directory=directory,
                        magnitude_scale=map_scale,
                        sources=sources,
                        basis=map_basis,
                        selections=selections,
                        part=map_part,
                        reference=target_zx,
                        reference_id=reference_transformation,
                        reference_sources=reference_sources,
                        color_limits=map_color_limits,
                        percentiles=map_percentiles,
                    )
                )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--results-dir", dest="run_dir", type=Path, required=True)
    parser.add_argument(
        "--case", dest="case_id", help="Simulation case ID; omitted means all cases"
    )
    parser.add_argument(
        "--magnitude-scale",
        choices=MAGNITUDE_SCALES,
        default="log",
        help="Shared magnitude display scale; individual line/map settings override it",
    )
    parser.add_argument(
        "--line-scale", choices=MAGNITUDE_SCALES, help="Radial magnitude display scale"
    )
    parser.add_argument("--map-scale", choices=MAGNITUDE_SCALES, help="2D magnitude display scale")
    parser.add_argument(
        "--map-color-limits",
        choices=MAP_COLOR_LIMITS,
        default="percentile",
        help="Map limits: configurable percentiles (default) or full range; no field rescaling",
    )
    parser.add_argument(
        "--map-percentiles",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_MAP_PERCENTILES,
        help="Bounds: 0 <= LOW < HIGH <= 100 (default: 2 98); ignored by full/phase maps",
    )
    parser.add_argument("--line-basis", choices=tuple(BASES), default="spherical")
    parser.add_argument("--map-basis", choices=tuple(BASES), default="spherical")
    components = ("E", "H") + tuple(
        f"{field}_{component}" for field in "EH" for basis in BASES.values() for component in basis
    )
    parser.add_argument(
        "--line-components",
        nargs="+",
        choices=components,
        help="Optional filter; omitted means vector norms and all components of --line-basis",
    )
    parser.add_argument(
        "--map-components",
        nargs="+",
        choices=components,
        help="Optional filter; omitted means vector norms and all components of --map-basis",
    )
    parser.add_argument("--line-part", choices=FIELD_PARTS, default="magnitude")
    parser.add_argument("--map-part", choices=FIELD_PARTS, default="magnitude")
    parser.add_argument(
        "--reference-transformation", help="Explicit prescribed transformation ID; requires --case"
    )
    args = parser.parse_args()
    try:
        paths = plot_fields(
            args.run_dir,
            case_id=args.case_id,
            magnitude_scale=args.magnitude_scale,
            line_scale=args.line_scale,
            map_scale=args.map_scale,
            map_color_limits=args.map_color_limits,
            map_percentiles=tuple(args.map_percentiles),
            line_basis=args.line_basis,
            map_basis=args.map_basis,
            line_components=None if args.line_components is None else tuple(args.line_components),
            map_components=None if args.map_components is None else tuple(args.map_components),
            line_part=args.line_part,
            map_part=args.map_part,
            reference_transformation=args.reference_transformation,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
