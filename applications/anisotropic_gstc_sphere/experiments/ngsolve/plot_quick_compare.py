#!/usr/bin/env python3
"""Quick computed/target ZX comparisons with common exterior field references."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize
from model import load_workflow_configuration
from plot_fields import _quantity, _quantity_symbol, prescribed_samples
from plot_zx import field_table

from sie_mom.postprocessing import zx_grid


def exterior_references(reference):
    """One prescribed exterior vector-magnitude peak for E and one for H."""
    exterior = reference["region"] == 2
    values = {}
    for field in "EH":
        magnitudes = _quantity(reference, field, "spherical")[exterior]
        finite = magnitudes[np.isfinite(magnitudes)]
        peak = float(finite.max()) if len(finite) else 0
        if peak <= 0:
            raise ValueError(f"no nonzero prescribed exterior {field} reference")
        values[field] = peak
    return values


def plot(run_dir, output_dir, dynamic_range_db=60):
    from matplotlib import pyplot as plt

    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0:
        raise ValueError("dynamic range must be finite and positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    report = json.loads((run_dir / "metrics.json").read_text())
    config = load_workflow_configuration(run_dir / "configuration.json")
    transformations = {item.id: item for item in config.transformations}
    prepared = []
    for case_id, metrics in report["cases"].items():
        with np.load(run_dir / case_id / "zx_fields.npz") as samples:
            data = field_table(samples)
            x, z, order = zx_grid(data, int(samples["grid_size"]))
        reference = prescribed_samples(
            data, config, transformations[metrics["target_transformation"]]
        )
        prepared.append((case_id, data, reference, x, z, order, exterior_references(reference)))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_run": str(run_dir.resolve()),
        "limits_db": [-dynamic_range_db, 0],
        "reference_definition": "prescribed exterior vector-magnitude peak, separately for E/H",
        "reference_shared_across": "computed/target panels and all components within E or H",
        "reference_values": {},
        "clipping": "display only; saved complex data unchanged",
    }
    for case_id, data, reference, x, z, order, references in prepared:
        metadata["reference_values"][case_id] = references
        directory = output_dir / case_id
        directory.mkdir()
        for component in (None, "r", "theta", "phi"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
            fig.suptitle(
                f"NGSolve / {case_id} — quick ZX comparison\n"
                "Common prescribed exterior E/H references; above 0 dB saturated\n"
                "Blank regions: excluded source/interface/PML samples"
            )
            for row, field in enumerate("EH"):
                selection = field if component is None else f"{field}_{component}"
                for column, table in enumerate((data, reference)):
                    magnitude = np.abs(_quantity(table, selection, "spherical"))
                    db = 20 * np.log10(np.maximum(magnitude / references[field], 1e-30))
                    image = np.ma.masked_invalid(db[order].reshape(len(z), len(x)))
                    ax = axes[row, column]
                    artist = ax.pcolormesh(
                        x / config.case.radius_m,
                        z / config.case.radius_m,
                        image,
                        shading="nearest",
                        cmap="viridis",
                        norm=Normalize(-dynamic_range_db, 0),
                    )
                    ax.add_patch(
                        plt.Circle((0, 0), 1, fill=False, edgecolor="white", linestyle="--")
                    )
                    ax.set(
                        xlabel="x/a",
                        ylabel="z/a",
                        aspect="equal",
                        title=f"{'NGSolve' if column == 0 else 'Prescribed'} "
                        f"$|{_quantity_symbol(selection)}|$",
                    )
                    fig.colorbar(
                        artist,
                        ax=ax,
                        extend="both",
                        label=f"dB re prescribed exterior peak |{field}|",
                    )
            name = "norms" if component is None else component
            fig.savefig(directory / f"zx_{name}_quick_comparison.png", dpi=180)
            plt.close(fig)
    (output_dir / "plot_settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved quick comparisons to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dynamic-range-db", type=float, default=60)
    args = parser.parse_args()
    plot(args.run_dir, args.output_dir, args.dynamic_range_db)
