"""Prepare radial/ZX total-field CSVs from an inverse experiment's saved SVWFs.

The default selects independently solved forward coefficients. Select inverse
to inspect the fixed exterior target expansion and optimized interior response.
Only field reconstruction is performed; synthesis and forward solves are not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from spherical_expansion.coefficients import load_coefficients
from spherical_expansion.svwf_sphere import compute_fields

if __package__:
    from ...prescribed_fields import evaluate_prescribed_fields
    from ...workflow import load_workflow_configuration, write_json
    from ..prepare_plot_fields import observation_points, spherical_to_cartesian, write_field_csv
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from applications.anisotropic_gstc_sphere.experiments.prepare_plot_fields import (
        observation_points,
        spherical_to_cartesian,
        write_field_csv,
    )
    from applications.anisotropic_gstc_sphere.prescribed_fields import evaluate_prescribed_fields
    from applications.anisotropic_gstc_sphere.workflow import (
        load_workflow_configuration,
        write_json,
    )


SOLUTIONS = ("forward", "inverse")
ARTIFACT = "svwf_inverse_plot_data"


def total_fields(configuration, simulation_case, points, coefficients):
    """Reconstruct total Cartesian E/H in exp(+j omega t), without amplitude scaling."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("observation points must have nonempty shape (point, 3)")
    if not np.isfinite(points).all():
        raise ValueError("observation points must be finite")
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii == 0):
        raise ValueError("plot sampling must exclude the sphere center")
    theta = np.arccos(np.clip(points[:, 2] / radii, -1, 1))
    phi = np.arctan2(points[:, 1], points[:, 0])
    regions = np.where(radii < configuration.case.radius_m, 1, 2)
    totals = [np.empty((3, len(points)), complex) for _ in "EH"]
    for region_id, region, families, kind in (
        (1, "interior", coefficients[2:], 1),
        (2, "exterior", coefficients[:2], 3),
    ):
        indices = np.flatnonzero(regions == region_id)
        if not len(indices):
            continue
        expanded = compute_fields(
            getattr(configuration.case, f"k_{region}"), radii[indices], theta[indices],
            phi[indices], *families, kind=kind,
            eta=getattr(configuration.case, f"eta_{region}"), grid=False,
        )
        # Archives use exp(-j omega t); analytic incident fields already use +j.
        fields = [np.asarray(field).conj() for field in expanded]
        for excitation in simulation_case.excitations:
            if excitation.region != region:
                continue
            incident = evaluate_prescribed_fields(
                excitation.field, region, configuration, radius=radii[indices],
                theta=theta[indices], phi=phi[indices],
            )
            for total, contribution in zip(fields, incident):
                total += contribution
        for total, field in zip(totals, fields):
            total[:, indices] = spherical_to_cartesian(field, theta[indices], phi[indices])
    if not all(np.isfinite(field).all() for field in totals):
        raise ValueError("nonfinite reconstructed fields; check for samples at incident sources")
    return regions, tuple(totals)


def load_saved_inputs(experiment_dir, solution):
    """Validate the experiment snapshot and all selected archives before writing output."""
    if solution not in SOLUTIONS:
        raise ValueError("solution must be forward or inverse")
    with (experiment_dir / "experiment.json").open(encoding="utf-8") as stream:
        experiment = json.load(stream)
    method = experiment.get("method")
    if method not in ("joint_svwf_reciprocal_lossless", "forward_svwf_reciprocal_lossless"):
        raise ValueError(
            "expected a joint SVWF reciprocal-lossless experiment or forward validation"
        )
    if method == "forward_svwf_reciprocal_lossless" and solution != "forward":
        raise ValueError(
            "forward-only validation has no inverse candidate; select --solution forward"
        )
    if experiment.get("status") != "completed":
        raise ValueError("the saved experiment must be completed before preparing plots")
    if experiment.get("coefficient_time_convention") != "exp(-j omega t)":
        raise ValueError("unsupported saved coefficient time convention")
    configuration = load_workflow_configuration(experiment_dir / "configuration.json")
    mapping = experiment.get("state_cases", [])
    if not isinstance(mapping, list) or any(
        not isinstance(pair, list) or len(pair) != 2
        or not all(isinstance(value, str) for value in pair) for pair in mapping
    ):
        raise ValueError("saved state/case mapping must contain transformation/case pairs")
    state_ids = [pair[0] for pair in mapping]
    case_ids = [pair[1] for pair in mapping]
    if (
        len(state_ids) != len(set(state_ids))
        or set(state_ids) != {state.id for state in configuration.transformations}
        or len(case_ids) != len(set(case_ids))
        or set(case_ids) != {case.id for case in configuration.simulation.cases}
    ):
        raise ValueError("saved state/case mapping disagrees with its configuration snapshot")
    n_max = experiment.get("n_max")
    if isinstance(n_max, bool) or not isinstance(n_max, int) or n_max < 1:
        raise ValueError("saved n_max must be a positive integer")
    expected_modes = {(m, n) for n in range(1, n_max + 1) for m in range(-n, n + 1)}
    coefficients = {}
    for case in configuration.simulation.cases:
        archive = experiment_dir / "cases" / case.id / f"{solution}_coefficients.npz"
        coefficients[case.id] = load_coefficients(archive)
        if set(coefficients[case.id][0]) != expected_modes:
            raise ValueError(f"coefficient modes disagree with saved n_max: {archive}")
    return configuration, dict(zip(case_ids, state_ids)), coefficients


def plot_manifest(configuration, experiment_dir, solution, state_ids):
    """Describe SVWF post-processing explicitly, preserving the plotting data contract."""
    cases = configuration.to_dict()["simulation"]["cases"]
    for case in cases:
        prefix = f"cases/{case['id']}"
        case.update(
            status="pending", reference_transformation=state_ids[case["id"]],
            files={
                "computed_radial": f"{prefix}/fields/svwf_radial.csv",
                "computed_zx": f"{prefix}/fields/svwf_zx.csv",
                "plots": f"{prefix}/plots",
            },
        )
    return {
        "schema_version": 2, "application": "anisotropic_gstc_sphere",
        "artifact": ARTIFACT, "backend": "svwf", "solution": solution,
        "case_id": configuration.case_id, "status": "preparing",
        "phasor_convention": "exp(+j omega t)",
        "frequency_hz": configuration.case.frequency_hz,
        "radius_m": configuration.case.radius_m,
        "simulation": {"cases": cases}, "sampling": asdict(configuration.sampling),
        "field_data": {
            "coordinate_system": "Cartesian", "coordinate_unit": "m",
            "electric_field_unit": "V/m", "magnetic_field_unit": "A/m",
            "regions": {"1": "interior", "2": "exterior"},
            "physical_construction": (
                "incident + regular c,d inside; incident + outgoing a,b outside"
            ),
        },
        "post_processing": {
            "source_experiment": str(experiment_dir),
            "coefficient_archive": f"{solution}_coefficients.npz",
            "source": "saved SVWF coefficients; no synthesis or forward solve",
            "interior_reference_enforced": False,
            "exterior_candidate_prescribed": solution == "inverse",
        },
        "files": {"configuration": "configuration.json"},
    }


def prepare_plot_fields(experiment_dir, *, solution="forward"):
    """Write separate forward/inverse plotting datasets without modifying synthesis artifacts."""
    experiment_dir = Path(experiment_dir).resolve()
    configuration, state_ids, coefficients = load_saved_inputs(experiment_dir, solution)
    points, zx_count = observation_points(configuration)
    directory = experiment_dir / "plot_data" / solution
    if directory.exists() and any(directory.iterdir()):
        with (directory / "run.json").open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if (
            previous.get("artifact") != ARTIFACT or previous.get("solution") != solution
            or previous.get("post_processing", {}).get("source_experiment") != str(experiment_dir)
        ):
            raise FileExistsError("output directory contains data from another run")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "configuration.json", configuration.to_dict())
    manifest = plot_manifest(configuration, experiment_dir, solution, state_ids)
    manifest_path = directory / "run.json"
    write_json(manifest_path, manifest)
    try:
        for case, saved_case in zip(
            configuration.simulation.cases, manifest["simulation"]["cases"]
        ):
            regions, fields = total_fields(configuration, case, points, coefficients[case.id])
            for cut, selection in (("zx", slice(0, zx_count)), ("radial", slice(zx_count, None))):
                write_field_csv(
                    directory / saved_case["files"][f"computed_{cut}"],
                    points[selection], regions[selection],
                    tuple(field[:, selection] for field in fields),
                )
            saved_case["status"] = "completed"
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(manifest_path, manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--solution", choices=SOLUTIONS, default="forward")
    args = parser.parse_args()
    try:
        directory = prepare_plot_fields(args.experiment_dir, solution=args.solution)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
