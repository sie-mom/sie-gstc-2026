#!/usr/bin/env python3
"""Plot NGSolve ZX fields with the application's existing SIE/COMSOL renderer."""

import argparse
import json
from pathlib import Path

import model as m
import numpy as np
from plot_fields import BASES, _plot_zx, prescribed_samples
from workflow import DipoleFieldDefinition, load_workflow_configuration

from sie_mom.postprocessing import zx_grid


def field_table(samples):
    xyz = samples["xyz_m"]
    data = {"point_index": np.arange(len(xyz)), "region": samples["region"]}
    data.update({axis: xyz[:, i] for i, axis in enumerate("xyz")})
    for field in "EH":
        for column, axis in enumerate("xyz"):
            values = samples[f"{field}_cartesian"][:, column]
            data[f"{field}{axis}_real"] = values.real
            data[f"{field}{axis}_imag"] = values.imag
    return data


def plot(
    run_dir,
    output_dir=None,
    data_dir=None,
    basis="spherical",
    part="magnitude",
    scale="db",
    color_limits="percentile",
    percentiles=(2, 98),
    compare_target=False,
):
    if part != "magnitude" and scale != "linear":
        raise ValueError("real/imag/phase maps require --scale linear")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    report = json.loads((run_dir / "metrics.json").read_text())
    config = load_workflow_configuration(run_dir / "configuration.json")
    output_dir = output_dir or run_dir / "plots_zx"
    data_dir = data_dir or run_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"plot directory is not empty: {output_dir}")
    # Validate all inputs before creating figures.
    prepared = []
    cases = {case.id: case for case in config.simulation.cases}
    transformations = {item.id: item for item in config.transformations}
    for case_id, metrics in report["cases"].items():
        with np.load(data_dir / case_id / "zx_fields.npz") as samples:
            data = field_table(samples)
            grid = zx_grid(data, int(samples["grid_size"]))
        sources = []
        for excitation in cases[case_id].excitations:
            definition = config.fields[excitation.field.field_id]
            if isinstance(definition, DipoleFieldDefinition):
                sources.append((excitation.field.field_id, definition.source_position))
        reference, reference_id, reference_sources = None, None, []
        if compare_target:
            reference_id = metrics.get("target_transformation")
            if reference_id not in transformations:
                raise ValueError(f"{case_id} has no unique matching target transformation")
            reference = prescribed_samples(data, config, transformations[reference_id])
            for region in ("interior", "exterior"):
                field = getattr(transformations[reference_id].gstc, region)
                if field is not None:
                    definition = config.fields[field.field_id]
                    if isinstance(definition, DipoleFieldDefinition):
                        reference_sources.append(
                            (f"{region}: {field.field_id}", definition.source_position)
                        )
        prepared.append((case_id, data, grid, sources, reference, reference_id, reference_sources))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_id, data, grid, sources, reference, reference_id, reference_sources in prepared:
        directory = output_dir / case_id
        directory.mkdir()
        title = (
            f"NGSolve FEM / {case_id}\n"
            f"{config.case.frequency_hz / 1e9:g} GHz | ZX plane (y=0)\n"
            "Masked: PML/outside air, source singularity, immediate sheet vicinity"
        )
        selections = [None] if part == "magnitude" else []
        selections += [(f"E_{c}", f"H_{c}") for c in BASES[basis]]
        for selected in selections:
            paths.append(
                _plot_zx(
                    data,
                    radius_m=config.case.radius_m,
                    grid=grid,
                    title=title,
                    directory=directory,
                    magnitude_scale=scale,
                    sources=sources,
                    basis=basis,
                    selections=selected,
                    part=part,
                    reference=reference,
                    reference_id=reference_id,
                    reference_sources=reference_sources,
                    color_limits=color_limits,
                    percentiles=percentiles,
                )
            )
    (output_dir / "plot_settings.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir.resolve()),
                "data_dir": str(data_dir.resolve()),
                "basis": basis,
                "part": part,
                "scale": scale,
                "color_limits": color_limits,
                "percentiles": list(percentiles),
                "compare_target": compare_target,
                "renderer": str(m.APPLICATION / "plot_fields.py"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(paths)} ZX maps to {output_dir}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, help="optional directory from sample_zx.py")
    parser.add_argument("--basis", choices=tuple(BASES), default="spherical")
    parser.add_argument(
        "--part", choices=("magnitude", "real", "imag", "phase"), default="magnitude"
    )
    parser.add_argument("--scale", choices=("linear", "log", "db"), default="db")
    parser.add_argument("--color-limits", choices=("percentile", "full"), default="percentile")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(2, 98))
    parser.add_argument("--compare-target", action="store_true")
    args = parser.parse_args()
    plot(
        args.run_dir,
        args.output_dir,
        args.data_dir,
        args.basis,
        args.part,
        args.scale,
        args.color_limits,
        args.percentiles,
        args.compare_target,
    )
