#!/usr/bin/env python3
"""Plot saved NGSolve radial fields with the shared SIE/COMSOL renderer."""

import argparse
import json
from pathlib import Path

import model as m
import numpy as np
from plot_fields import _plot_radial
from plot_zx import field_table


def plot(run_dir, output_dir=None, scale="db", compare_target=False):
    report = json.loads((run_dir / "metrics.json").read_text())
    config = m.load_workflow_configuration(run_dir / "configuration.json")
    output_dir = output_dir or run_dir / "plots_radial_spherical"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"plot directory is not empty: {output_dir}")
    prepared = []
    for case_id, metrics in report["cases"].items():
        with np.load(run_dir / case_id / "radial_fields.npz") as samples:
            values = {
                "xyz_m": samples["xyz_m"],
                "region": np.where(samples["interior"], 1, 2),
                "E_cartesian": samples["E_cartesian"],
                "H_cartesian": samples["H_cartesian"],
            }
            data = field_table(values)
            reference, reference_id = None, None
            if compare_target:
                reference_id = metrics.get("target_transformation")
                if not reference_id:
                    raise ValueError(f"{case_id} has no unique matching target transformation")
                reference = field_table(
                    {
                        **values,
                        "E_cartesian": samples["target_E_cartesian"],
                        "H_cartesian": samples["target_H_cartesian"],
                    }
                )
        prepared.append((case_id, data, reference, reference_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_id, data, reference, reference_id in prepared:
        directory = output_dir / case_id
        directory.mkdir()
        paths.extend(
            _plot_radial(
                data,
                radius_m=config.case.radius_m,
                title=(
                    f"NGSolve FEM / {case_id}\n"
                    f"{config.case.frequency_hz / 1e9:g} GHz | {run_dir.name}"
                ),
                directory=directory,
                magnitude_scale=scale,
                basis="spherical",
                reference=reference,
                reference_id=reference_id,
            )
        )
    (output_dir / "plot_settings.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir.resolve()),
                "basis": "spherical",
                "magnitude_scale": scale,
                "real_imag_scale": "linear",
                "compare_target": compare_target,
                "reference": "saved prescribed fields; no amplitude rescaling",
                "renderer": str(m.APPLICATION / "plot_fields.py"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(paths)} radial figures to {output_dir}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scale", choices=("linear", "log", "db"), default="db")
    parser.add_argument("--compare-target", action="store_true")
    args = parser.parse_args()
    plot(args.run_dir, args.output_dir, args.scale, args.compare_target)
