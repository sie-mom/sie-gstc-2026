#!/usr/bin/env python3
"""Overlay SIE, checkpointed NGSolve and prescribed radial fields and leakage."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import model as m
import ngsolve as ng
import numpy as np
from plot_fields import (
    BASES,
    FIELD_UNITS,
    _field_components,
    _quantity_symbol,
    _save_figure,
    prescribed_samples,
)
from plot_zx import field_table

from sie_mom.postprocessing import complex_vector, load_field_csv

STYLES = (
    ("Prescribed", "black", "-", None),
    ("SIE", "#d27400", "--", "s"),
    ("NGSolve", "#007a87", ":", "o"),
)
ZERO_TOLERANCE = 1e-12


def db(values, floor_db):
    """Display floor only: all statistics use the original linear ratios."""
    return 20 * np.log10(np.maximum(np.abs(values), 10 ** (floor_db / 20)))


def sample_checkpoint(run_dir, config, sie, mesh, space, case_id):
    xyz = np.column_stack([sie[axis] for axis in "xyz"])
    solution = ng.GridFunction(space)
    solution.Load(str(run_dir / case_id / "solution.sol"))
    system = SimpleNamespace(mesh=mesh, config=config)
    fields = m.total_fields(system, solution, *m.incident_cf(config, case_id))
    sampled = {"xyz_m": xyz, "region": sie["region"]}
    k0 = m.k0_of(config)
    for field, column, scale in (("E", 0, k0**3), ("H", 1, k0**3 / m.ETA0)):
        values = np.empty((len(xyz), 3), complex)
        for index, region in ((0, 1), (1, 2)):
            mask = sie["region"] == region
            values[mask] = m.evaluate(fields[index][column], mesh, xyz[mask] * k0) * scale
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite checkpoint {field} fields")
        sampled[f"{field}_cartesian"] = values
    return field_table(sampled)


def leakage_statistics(ratios):
    """Unweighted radial sample statistics; zero is represented as null dB."""
    rms = float(np.sqrt(np.mean(np.abs(ratios) ** 2)))
    peak = float(np.max(np.abs(ratios)))
    return {
        "rms_amplitude_ratio": rms,
        "peak_amplitude_ratio": peak,
        "rms_db": float(20 * np.log10(rms)) if rms else None,
        "peak_db": float(20 * np.log10(peak)) if peak else None,
    }


def draw_curves(ax, radius, regions, arrays, transform):
    for name, color, linestyle, marker in STYLES:
        for region in (1, 2):
            mask = regions == region
            ax.plot(
                radius[mask],
                transform(arrays[name][mask]),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor="none",
                markersize=4,
                markevery=max(1, np.count_nonzero(mask) // 14),
                linewidth=1.3,
                label=name if region == 1 else None,
                zorder=3 if name != "Prescribed" else 2,
            )
    ax.axvline(1, color="black", linestyle="--", linewidth=0.8)
    ax.grid(alpha=0.25)


def render_case(directory, title, radius, regions, tables, floor_db, metrics_range):
    from matplotlib import pyplot as plt

    vectors = {name: {f: complex_vector(t, f) for f in "EH"} for name, t in tables.items()}
    spherical = {
        name: {f: _field_components(t, f, "spherical") for f in "EH"} for name, t in tables.items()
    }
    norms = {f: np.linalg.norm(vectors["Prescribed"][f], axis=1) for f in "EH"}
    exterior = regions == 2
    references = {f: float(norms[f][exterior].max()) for f in "EH"}
    if any(not np.isfinite(norms[f]).all() or np.any(norms[f] <= 0) for f in "EH"):
        raise ValueError("suppression comparison requires nonzero prescribed vector fields")
    paths = []
    # Same amplitude reference for both solvers, the target and every component.
    for component in (None, 0, 1, 2):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
        fig.suptitle(title + "\nShared prescribed exterior peak references; display floor only")
        for ax, field in zip(axes, "EH"):
            arrays = {
                name: np.linalg.norm(vectors[name][field], axis=1)
                if component is None
                else np.abs(spherical[name][field][:, component])
                for name in tables
            }
            draw_curves(ax, radius, regions, arrays, lambda v: db(v / references[field], floor_db))
            peak = max(float(np.max(v)) for v in arrays.values()) / references[field]
            ax.set_ylim(floor_db, max(5, 5 * np.ceil(20 * np.log10(peak) / 5) + 5))
            symbol = field if component is None else f"{field}_{BASES['spherical'][component]}"
            ax.set_ylabel(rf"$|{_quantity_symbol(symbol)}|$ [dB re exterior peak $|{field}|$]")
            ax.text(0.02, 0.03, f"At floor: ≤ {floor_db:g} dB", transform=ax.transAxes, fontsize=8)
        axes[0].legend(ncol=3)
        axes[-1].set_xlabel("r/a")
        stem = "magnitudes" if component is None else f"spherical_{BASES['spherical'][component]}"
        path = directory / f"radial_{stem}_sie_ngsolve_prescribed.png"
        _save_figure(fig, path)
        paths.append(path)
    for field in "EH":
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, layout="constrained")
        fig.suptitle(title)
        for row, component in enumerate(BASES["spherical"]):
            arrays = {name: spherical[name][field][:, row] for name in tables}
            for column, part in enumerate(("real", "imag")):
                ax = axes[row, column]
                draw_curves(ax, radius, regions, arrays, lambda v, p=part: getattr(v, p))
                symbol = _quantity_symbol(f"{field}_{component}")
                ax.set_ylabel(
                    rf"$\mathrm{{{'Re' if column == 0 else 'Im'}}}({symbol})$ "
                    f"[{FIELD_UNITS[field]}]"
                )
        axes[0, 0].legend(ncol=3, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("r/a")
        path = directory / f"radial_{field}_real_imag_sie_ngsolve_prescribed.png"
        _save_figure(fig, path)
        paths.append(path)
    # Dedicated leakage panels normalize by the local prescribed vector norm.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col", layout="constrained")
    fig.suptitle(
        title + "\nUnwanted components / local prescribed vector magnitude; lower is better"
    )
    metrics = {"exterior_peak_reference_SI": references, "regions": {}}
    for column, (region, label) in enumerate(((1, "Interior"), (2, "Exterior"))):
        region_mask = regions == region
        score_mask = (
            region_mask
            & (radius >= metrics_range[0])
            & (radius <= metrics_range[1])
            & (np.abs(radius - 1) >= 0.05)
        )
        if not np.any(score_mask):
            raise ValueError(f"no common scoring samples in {label}")
        for row, field in enumerate("EH"):
            ax = axes[row, column]
            ideal_ratio = np.abs(spherical["Prescribed"][field]) / norms[field][:, None]
            zero = np.max(ideal_ratio[region_mask], axis=0) <= ZERO_TOLERANCE
            zero_indices = np.flatnonzero(zero)
            entry = {
                "zero_components": [BASES["spherical"][j] for j in zero_indices],
                "maximum_target_zero_ratio": float(ideal_ratio[region_mask][:, zero].max())
                if zero.any()
                else None,
                "scoring_sample_count": int(score_mask.sum()),
                "scoring_r_over_a_min_max": [
                    float(radius[score_mask].min()),
                    float(radius[score_mask].max()),
                ],
                "solvers": {},
            }
            for name, color, _, marker in STYLES[1:]:
                parts = {}
                for j, linestyle in zip(zero_indices, ("-", "--", ":")):
                    ratio = np.abs(spherical[name][field][:, j]) / norms[field]
                    ax.plot(
                        radius[region_mask],
                        db(ratio[region_mask], floor_db),
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markerfacecolor="none",
                        markersize=3,
                        markevery=max(1, int(region_mask.sum()) // 14),
                        label=f"{name} {field}_{BASES['spherical'][j]}",
                    )
                    parts[BASES["spherical"][j]] = leakage_statistics(ratio[score_mask])
                combined = np.linalg.norm(spherical[name][field][:, zero], axis=1) / norms[field]
                error = vectors[name][field][score_mask] - vectors["Prescribed"][field][score_mask]
                entry["solvers"][name] = {
                    "combined_unwanted": leakage_statistics(combined[score_mask]),
                    "components": parts,
                    "complex_vector_relative_l2_error": float(
                        np.linalg.norm(error)
                        / np.linalg.norm(vectors["Prescribed"][field][score_mask])
                    ),
                }
            ax.set(
                title=f"{label}: unwanted {field} components",
                ylim=(floor_db, 5),
                ylabel=rf"$20\log_{{10}}(|{field}_j|/\|{field}_{{target}}(r)\|)$ [dB]",
            )
            ax.axvline(1, color="black", linestyle="--", linewidth=0.8)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            ax.text(
                0.02,
                0.03,
                f"Ideal: −∞ dB; display floor {floor_db:g} dB",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.set_xlabel("r/a")
            metrics["regions"][f"{label.lower()}_{field}"] = entry
    path = directory / "radial_suppression_sie_ngsolve.png"
    _save_figure(fig, path)
    paths.append(path)
    return paths, metrics


def plot(
    run_dir, sie_dir=m.DEFAULT_INPUT, output_dir=None, floor_db=-100, metrics_range=(0.2, 1.3)
):
    if not np.isfinite(floor_db) or floor_db >= 0:
        raise ValueError("floor must be a finite negative number")
    if not 0 < metrics_range[0] < 0.95 < 1.05 < metrics_range[1]:
        raise ValueError("metrics range must include interior and exterior away from the sheet")
    output_dir = output_dir or run_dir / "plots_radial_sie"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"plot directory is not empty: {output_dir}")
    report = json.loads((run_dir / "metrics.json").read_text())
    if report["ngsolve"] != ng.__version__:
        raise ValueError("checkpoint requires matching NGSolve version")
    if report["inputs"]["configuration_sha256"] != m.sha256(sie_dir / "configuration.json"):
        raise ValueError("SIE and NGSolve configuration snapshots differ")
    if report["inputs"]["tensors_sha256"] != m.sha256(sie_dir / "face_tensors.bin"):
        raise ValueError("SIE and NGSolve tensor inputs differ")
    manifest = json.loads((sie_dir / "run.json").read_text())
    if manifest["backend"] != "sie" or manifest["status"] != "completed":
        raise ValueError("SIE run is not completed")
    if manifest["phasor_convention"] != "exp(+j omega t)":
        raise ValueError("unexpected SIE phasor convention")
    config = m.load_workflow_configuration(run_dir / "configuration.json")
    transformations = {t.id: t for t in config.transformations}
    mesh = ng.Mesh(str(run_dir / "mesh.vol.gz"))
    space = m.make_space(mesh, report["settings"]["order"])
    if space.ndof != report["dofs"]:
        raise ValueError("checkpoint space DOFs disagree")
    max_radius = (report["settings"]["air_lambda"] - 0.02) * 2 * np.pi / m.k0_of(config)
    if metrics_range[1] * config.case.radius_m > max_radius:
        raise ValueError("common scoring range extends beyond the physical sampling domain")
    prepared = []
    sources = {}
    for case_id, case_metrics in report["cases"].items():
        source = sie_dir / "cases" / case_id / "fields/sie_physical_radial.csv"
        data = load_field_csv(source, radius_m=config.case.radius_m)
        xyz = np.column_stack([data[axis] for axis in "xyz"])
        radii = np.linalg.norm(xyz, axis=1)
        if np.any(radii <= 0) or not np.allclose(
            xyz / radii[:, None], xyz[0] / radii[0], atol=1e-8, rtol=0
        ):
            raise ValueError("SIE points must lie on a nonzero radial ray")
        valid = (radii <= max_radius) & (
            np.abs(radii - config.case.radius_m) > 1e-3 * config.case.radius_m
        )
        data = {k: v[valid] for k, v in data.items()}
        order = np.argsort(radii[valid])
        data = {k: v[order] for k, v in data.items()}
        radius = radii[valid][order] / config.case.radius_m
        target = prescribed_samples(
            data, config, transformations[case_metrics["target_transformation"]]
        )
        fem = sample_checkpoint(run_dir, config, data, mesh, space, case_id)
        prepared.append(
            (case_id, radius, data["region"], {"Prescribed": target, "SIE": data, "NGSolve": fem})
        )
        sources[case_id] = {"csv": str(source.resolve()), "sha256": m.sha256(source)}
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ngsolve_run": str(run_dir.resolve()),
        "sie_run": str(sie_dir.resolve()),
        "sie_sources": sources,
        "display_floor_db": floor_db,
        "sampling": "NGSolve checkpoint evaluated at SIE coordinates; no interpolation or solve",
        "magnitude_reference": (
            "prescribed exterior vector peak on plotted ray; one shared reference per E/H"
        ),
        "suppression_reference": "local prescribed vector norm, separately for E/H and region",
        "zero_component_relative_tolerance": ZERO_TOLERANCE,
        "scoring": {
            "r_over_a_range": list(metrics_range),
            "exclude_abs_r_over_a_minus_1_below": 0.05,
            "aggregation": "unweighted RMS of local amplitude ratios; no dB clipping in statistics",
        },
        "cases": {},
    }
    paths = []
    for case_id, radius, regions, tables in prepared:
        directory = output_dir / case_id
        directory.mkdir()
        # Use the actual CSV ray, which may differ from saved sampling settings.
        xyz = np.array([tables["SIE"][axis][0] for axis in "xyz"])
        theta = np.degrees(np.arccos(xyz[2] / np.linalg.norm(xyz)))
        phi = np.degrees(np.arctan2(xyz[1], xyz[0]))
        title = f"{case_id} | {run_dir.name}\nSIE / NGSolve / prescribed | θ={theta:g}°, φ={phi:g}°"
        files, metrics = render_case(
            directory, title, radius, regions, tables, floor_db, metrics_range
        )
        paths.extend(files)
        result["cases"][case_id] = metrics
        np.savez_compressed(
            directory / "common_radial_fields.npz",
            radius_over_a=radius,
            region=regions,
            xyz_m=np.column_stack([tables["SIE"][a] for a in "xyz"]),
            **{
                f"{name}_{field}_cartesian": complex_vector(table, field)
                for name, table in tables.items()
                for field in "EH"
            },
        )
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {len(paths)} overlay/suppression figures to {output_dir}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sie-dir", type=Path, default=m.DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--floor-db", type=float, default=-100)
    parser.add_argument("--metrics-range", type=float, nargs=2, default=(0.2, 1.3))
    args = parser.parse_args()
    plot(args.run_dir, args.sie_dir, args.output_dir, args.floor_db, args.metrics_range)
