#!/usr/bin/env python3
"""Plot standardized omega-sphere output from either validation backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.constants import epsilon_0, mu_0

from sie_mom.postprocessing import (
    budhu_moment_scale as _paper_current_moment_scale,
    complex_vector,
    dominant_tangential_components as _dominant_tangential_components,
    to_decibels,
)

try:
    from .workflow import DEFAULT_CASE_ID, default_run_directory, load_field_csv
except ImportError:  # Support direct execution from the repository root.
    from workflow import (  # type: ignore
        DEFAULT_CASE_ID,
        default_run_directory,
        load_field_csv,
    )


def _complex_vector(data: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    return complex_vector(data, prefix).T


def _decibels(values: np.ndarray) -> np.ndarray:
    return to_decibels(values, floor_relative_db=280.0)



def _shared_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(tuple(array[np.isfinite(array)] for array in arrays))
    lower, upper = np.percentile(values, (2.0, 98.0))
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def _assert_matching_coordinates(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
) -> None:
    for name in ("point_index", "region"):
        np.testing.assert_array_equal(first[name], second[name])
    # NumPy and C++ can round the same sampling formula differently. Allow
    # only floating-point roundoff relative to the spatial extent of the data.
    length_scale = max(
        float(np.max(np.abs(data[coordinate])))
        for data in (first, second)
        for coordinate in ("x", "y", "z")
    )
    tolerance = 32 * np.finfo(float).eps * length_scale
    for coordinate in ("x", "y", "z"):
        np.testing.assert_allclose(
            first[coordinate], second[coordinate], rtol=0.0, atol=tolerance
        )


def _sum_representations(
    interior: dict[str, np.ndarray],
    exterior: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return the algebraic representation sum used only as a diagnostic."""

    _assert_matching_coordinates(interior, exterior)
    result = {name: values.copy() for name, values in interior.items()}
    for field in ("E", "H"):
        for axis in ("x", "y", "z"):
            for part in ("real", "imag"):
                name = f"{field}{axis}_{part}"
                result[name] += exterior[name]
    return result


def create_plots(
    run_dir: Path,
    *,
    comparison_run: Path | None = None,
) -> tuple[Path, ...]:
    """Create physical-comparison and optional SIE-representation figures."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required; install the project with the 'plot' extra"
        ) from error

    with (run_dir / "run.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    files = manifest["files"]
    backend = str(manifest.get("backend", "svwf")).upper()
    reference_label = manifest.get("reference_label", "Prescribed")
    prescribed_zx = load_field_csv(
        run_dir / (files.get("reference_zx") or files["prescribed_zx"])
    )
    computed_zx = load_field_csv(
        run_dir / (files.get("computed_zx") or files["svwf_zx"])
    )
    prescribed_radial = load_field_csv(
        run_dir
        / (files.get("reference_radial") or files["prescribed_radial"])
    )
    computed_radial = load_field_csv(
        run_dir / (files.get("computed_radial") or files["svwf_radial"])
    )
    _assert_matching_coordinates(prescribed_zx, computed_zx)
    _assert_matching_coordinates(prescribed_radial, computed_radial)

    computed_sets = [(backend, computed_zx, computed_radial)]
    if comparison_run is not None:
        with (comparison_run / "run.json").open(encoding="utf-8") as stream:
            comparison_manifest = json.load(stream)
        comparison_files = comparison_manifest["files"]
        if manifest.get("cap_perturbation") != comparison_manifest.get(
            "cap_perturbation"
        ):
            raise ValueError("comparison runs use different cap perturbations")
        comparison_zx = load_field_csv(
            comparison_run
            / (
                comparison_files.get("computed_zx")
                or comparison_files["svwf_zx"]
            )
        )
        comparison_radial = load_field_csv(
            comparison_run
            / (
                comparison_files.get("computed_radial")
                or comparison_files["svwf_radial"]
            )
        )
        _assert_matching_coordinates(prescribed_zx, comparison_zx)
        _assert_matching_coordinates(prescribed_radial, comparison_radial)
        if not np.isclose(
            float(manifest["frequency_hz"]),
            float(comparison_manifest["frequency_hz"]),
        ) or not np.isclose(
            float(manifest["radius_m"]), float(comparison_manifest["radius_m"])
        ):
            raise ValueError("comparison run uses a different frequency or radius")
        comparison_backend = str(
            comparison_manifest.get("backend", "comparison")
        ).upper()
        computed_sets.append(
            (comparison_backend, comparison_zx, comparison_radial)
        )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    radius = float(manifest["radius_m"])
    grid_shape = tuple(manifest["sampling"]["zx"]["shape"])

    prescribed_e_theta, prescribed_h_phi = _dominant_tangential_components(
        prescribed_zx
    )
    tangential_zx = [
        _dominant_tangential_components(data)
        for _, data, _ in computed_sets
    ]
    e_db = tuple(
        [_decibels(prescribed_e_theta)]
        + [_decibels(electric) for electric, _ in tangential_zx]
    )
    h_db = tuple(
        [_decibels(prescribed_h_phi)]
        + [_decibels(magnetic) for _, magnetic in tangential_zx]
    )
    e_limits = _shared_limits(*e_db)
    h_limits = _shared_limits(*h_db)
    x_grid = prescribed_zx["x"].reshape(grid_shape) / radius
    z_grid = prescribed_zx["z"].reshape(grid_shape) / radius

    figure, axes = plt.subplots(
        len(computed_sets) + 1,
        2,
        figsize=(12, 5 * (len(computed_sets) + 1)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    row_labels = (reference_label, *(label for label, _, _ in computed_sets))
    column_data = (e_db, h_db)
    column_titles = (r"$|E_\theta|$ (dB)", r"$|H_\phi|$ (dB)")
    column_limits = (e_limits, h_limits)
    for row in range(len(row_labels)):
        for column in range(2):
            axis = axes[row, column]
            lower, upper = column_limits[column]
            image = axis.pcolormesh(
                x_grid,
                z_grid,
                column_data[column][row].reshape(grid_shape),
                shading="auto",
                cmap="viridis",
                vmin=lower,
                vmax=upper,
            )
            axis.add_patch(
                plt.Circle((0.0, 0.0), 1.0, fill=False, color="white", linestyle="--")
            )
            axis.set_aspect("equal")
            axis.set_title(f"{row_labels[row]} {column_titles[column]}")
            axis.set_xlabel(r"$x/a$")
            axis.set_ylabel(r"$z/a$")
            figure.colorbar(image, ax=axis, shrink=0.85)
    fields_path = plots_dir / "fields_zx.png"
    figure.savefig(fields_path, dpi=200)
    plt.close(figure)

    prescribed_e_theta, prescribed_h_phi = _dominant_tangential_components(
        prescribed_radial
    )
    tangential_radial = [
        _dominant_tangential_components(data)
        for _, _, data in computed_sets
    ]
    radial_radius = np.sqrt(
        prescribed_radial["x"] ** 2
        + prescribed_radial["y"] ** 2
        + prescribed_radial["z"] ** 2
    )
    normalized_radius = radial_radius / radius

    figure, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)
    radial_data = (
        (
            np.abs(prescribed_e_theta),
            [np.abs(electric) for electric, _ in tangential_radial],
            r"$|E_\theta|$ (V/m)",
        ),
        (
            np.abs(prescribed_h_phi),
            [np.abs(magnetic) for _, magnetic in tangential_radial],
            r"$|H_\phi|$ (A/m)",
        ),
    )
    line_styles = ("--", "-.", ":")
    line_colors = ("tab:blue", "tab:orange", "tab:green")
    for axis, (prescribed, computed_values, label) in zip(
        axes, radial_data, strict=True
    ):
        axis.plot(normalized_radius, prescribed, color="black", label=reference_label)
        for index, ((backend_label, _, _), values) in enumerate(
            zip(computed_sets, computed_values, strict=True)
        ):
            axis.plot(
                normalized_radius,
                values,
                color=line_colors[index % len(line_colors)],
                linestyle=line_styles[index % len(line_styles)],
                label=backend_label,
            )
        axis.axvline(1.0, color="tab:red", linestyle=":", label="Sheet")
        axis.set_yscale("log")
        axis.set_xlabel(r"$r/a$")
        axis.set_ylabel(label)
        axis.grid(True, which="both", linestyle=":", alpha=0.5)
        axis.legend()
    radial_path = plots_dir / "radial_cut.png"
    figure.savefig(radial_path, dpi=200)
    plt.close(figure)

    # Budhu's Fig. 3 uses Re(E_x) and the physical current-moment
    # normalization I Delta z = 1 A m in the dielectric interior.  The saved
    # fields instead use an arbitrary real effective moment, so apply the one
    # global complex scale that converts both regions to the paper convention.
    paper_scale = _paper_current_moment_scale(manifest)
    prescribed_ex = paper_scale * _complex_vector(prescribed_zx, "E")[0]
    prescribed_ex_scaled = np.real(prescribed_ex) / 1.0e5
    computed_ex_scaled = [
        np.real(paper_scale * _complex_vector(data, "E")[0]) / 1.0e5
        for _, data, _ in computed_sets
    ]

    figure, axes = plt.subplots(
        1,
        len(computed_sets) + 1,
        figsize=(6 * (len(computed_sets) + 1), 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x_grid_m = prescribed_zx["x"].reshape(grid_shape)
    z_grid_m = prescribed_zx["z"].reshape(grid_shape)
    for axis, values, title in zip(
        axes[0],
        (prescribed_ex_scaled, *computed_ex_scaled),
        (reference_label, *(f"{label} with omega sheet" for label, _, _ in computed_sets)),
        strict=True,
    ):
        image = axis.pcolormesh(
            x_grid_m,
            z_grid_m,
            values.reshape(grid_shape),
            shading="auto",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.add_patch(
            plt.Circle(
                (0.0, 0.0), radius, fill=False, color="black", linestyle="--"
            )
        )
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel(r"$x$ (m)")
        axis.set_ylabel(r"$z$ (m)")
        colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
        colorbar.set_label(r"$\operatorname{Re}(E_x)$ ($10^5$ V/m)")
    real_ex_zx_path = plots_dir / "real_ex_zx.png"
    figure.savefig(real_ex_zx_path, dpi=300)
    plt.close(figure)

    prescribed_radial_ex = (
        paper_scale * _complex_vector(prescribed_radial, "E")[0]
    )
    computed_radial_ex = [
        paper_scale * _complex_vector(data, "E")[0]
        for _, _, data in computed_sets
    ]
    radial_radius_m = np.sqrt(
        prescribed_radial["x"] ** 2
        + prescribed_radial["y"] ** 2
        + prescribed_radial["z"] ** 2
    )
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(
        radial_radius_m,
        np.real(prescribed_radial_ex) / 1.0e3,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=reference_label,
    )
    for index, ((backend_label, _, _), values) in enumerate(
        zip(computed_sets, computed_radial_ex, strict=True)
    ):
        axis.plot(
            radial_radius_m,
            np.real(values) / 1.0e3,
            color=line_colors[index % len(line_colors)],
            linestyle=line_styles[index % len(line_styles)],
            linewidth=1.5,
            label=f"{backend_label} with omega sheet",
        )
    axis.axvline(radius, color="tab:red", linestyle="--", label="Sheet")
    axis.set_xlabel(r"$r$ (m)")
    axis.set_ylabel(r"$\operatorname{Re}(E_x)$ (kV/m)")
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend()
    real_ex_radial_path = plots_dir / "real_ex_radial_cut.png"
    figure.savefig(real_ex_radial_path, dpi=300)
    plt.close(figure)

    output_paths = [fields_path, radial_path, real_ex_zx_path, real_ex_radial_path]
    representation_keys = {
        "interior_zx": "interior_representation_zx",
        "exterior_zx": "exterior_representation_zx",
        "interior_radial": "interior_representation_radial",
        "exterior_radial": "exterior_representation_radial",
    }
    if all(key in files for key in representation_keys.values()):
        interior_zx = load_field_csv(run_dir / files[representation_keys["interior_zx"]])
        exterior_zx = load_field_csv(run_dir / files[representation_keys["exterior_zx"]])
        interior_radial = load_field_csv(
            run_dir / files[representation_keys["interior_radial"]]
        )
        exterior_radial = load_field_csv(
            run_dir / files[representation_keys["exterior_radial"]]
        )
        for representation in (interior_zx, exterior_zx):
            _assert_matching_coordinates(computed_zx, representation)
        for representation in (interior_radial, exterior_radial):
            _assert_matching_coordinates(computed_radial, representation)
        sum_zx = _sum_representations(interior_zx, exterior_zx)
        sum_radial = _sum_representations(interior_radial, exterior_radial)

        representation_zx = (interior_zx, exterior_zx, sum_zx)
        representation_titles = (
            "Interior representation\n(should vanish for $r>a$)",
            "Exterior representation\n(should vanish for $r<a$)",
            "Representation sum\n(diagnostic only)",
        )
        representation_ex = [
            np.real(paper_scale * _complex_vector(data, "E")[0]) / 1.0e5
            for data in representation_zx
        ]
        figure, axes = plt.subplots(
            1, 3, figsize=(18, 5), constrained_layout=True, sharex=True, sharey=True
        )
        for axis, values, title in zip(
            axes, representation_ex, representation_titles, strict=True
        ):
            image = axis.pcolormesh(
                x_grid_m,
                z_grid_m,
                values.reshape(grid_shape),
                shading="auto",
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
            )
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0), radius, fill=False, color="black", linestyle="--"
                )
            )
            axis.set_aspect("equal")
            axis.set_title(title)
            axis.set_xlabel(r"$x$ (m)")
            axis.set_ylabel(r"$z$ (m)")
            colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
            colorbar.set_label(r"$\operatorname{Re}(E_x)$ ($10^5$ V/m)")
        representations_zx_path = plots_dir / "sie_representations_zx.png"
        figure.savefig(representations_zx_path, dpi=300)
        plt.close(figure)

        int_e_theta, int_h_phi = _dominant_tangential_components(interior_radial)
        ext_e_theta, ext_h_phi = _dominant_tangential_components(exterior_radial)
        sum_e_theta, sum_h_phi = _dominant_tangential_components(sum_radial)
        physical_e_theta, physical_h_phi = _dominant_tangential_components(
            computed_radial
        )
        figure, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
        diagnostic_radial_data = (
            (
                (int_e_theta, ext_e_theta, sum_e_theta, physical_e_theta),
                r"$|E_\theta|$ (V/m)",
            ),
            (
                (int_h_phi, ext_h_phi, sum_h_phi, physical_h_phi),
                r"$|H_\phi|$ (A/m)",
            ),
        )
        diagnostic_labels = (
            "Interior representation",
            "Exterior representation",
            "Representation sum (diagnostic)",
            "Piecewise physical field",
        )
        diagnostic_styles = ("--", "-.", ":", "-")
        for axis, (values, ylabel) in zip(
            axes, diagnostic_radial_data, strict=True
        ):
            for field, label, linestyle in zip(
                values, diagnostic_labels, diagnostic_styles, strict=True
            ):
                axis.plot(
                    normalized_radius,
                    np.abs(field),
                    linestyle=linestyle,
                    label=label,
                )
            axis.axvline(1.0, color="tab:red", linestyle=":", label="Sheet")
            axis.set_yscale("log")
            axis.set_xlabel(r"$r/a$")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", linestyle=":", alpha=0.5)
            axis.legend()
        representations_radial_path = plots_dir / "sie_representations_radial.png"
        figure.savefig(representations_radial_path, dpi=300)
        plt.close(figure)
        output_paths.extend((representations_zx_path, representations_radial_path))

    return tuple(output_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_run_dir = default_run_directory(
        DEFAULT_CASE_ID, "svwf", "least_squares"
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument(
        "--comparison-run",
        type=Path,
        help="Optional second backend run sampled at identical coordinates",
    )
    arguments = parser.parse_args()
    plot_paths = create_plots(
        arguments.run_dir, comparison_run=arguments.comparison_run
    )
    print("Omega-biisotropic dipole-matching plots")
    for path in plot_paths:
        print(f"  {path.stem}: {path}")


if __name__ == "__main__":
    main()
