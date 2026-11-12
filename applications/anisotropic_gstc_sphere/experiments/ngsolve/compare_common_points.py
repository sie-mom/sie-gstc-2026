#!/usr/bin/env python3
"""Compare checkpointed refinements on one baseline's physical radial points."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import model as m
import ngsolve as ng
import numpy as np
from workflow import load_workflow_configuration


def relative(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def compare(baseline, runs, output):
    baseline_report = json.loads((baseline / "metrics.json").read_text())
    result = {"baseline": str(baseline.resolve()), "runs": {}}
    for run_dir in runs:
        report = json.loads((run_dir / "metrics.json").read_text())
        config = load_workflow_configuration(run_dir / "configuration.json")
        if report["physical"] != baseline_report["physical"]:
            raise ValueError("physical parameters differ")
        if report["inputs"]["tensors_sha256"] != baseline_report["inputs"]["tensors_sha256"]:
            raise ValueError("synthesized tensor inputs differ")
        if report["ngsolve"] != ng.__version__:
            raise ValueError("checkpoint requires matching NGSolve version")
        mesh = ng.Mesh(str(run_dir / "mesh.vol.gz"))
        system = SimpleNamespace(mesh=mesh, config=config)
        space = m.make_space(mesh, report["settings"]["order"])
        values = {"path": str(run_dir.resolve()), "dofs": space.ndof, "cases": {}}
        for case_id in baseline_report["cases"]:
            solution = ng.GridFunction(space)
            solution.Load(str(run_dir / case_id / "solution.sol"))
            fields = m.total_fields(system, solution, *m.incident_cf(config, case_id))
            with np.load(baseline / case_id / "radial_fields.npz") as data:
                xyz = data["xyz_m"]
                radius = data["radius_m"]
                inside = data["interior"]
                sampled = {key: np.empty((len(xyz), 3), complex) for key in ("E", "H")}
                for index, mask in ((0, inside), (1, ~inside)):
                    for field, cf, scale in zip(
                        ("E", "H"),
                        fields[index],
                        (m.k0_of(config) ** 3, m.k0_of(config) ** 3 / m.ETA0),
                    ):
                        sampled[field][mask] = (
                            m.evaluate(cf, mesh, xyz[mask] * m.k0_of(config)) * scale
                        )
                metrics = {}
                for region, mask in [
                    ("interior_r_ge_0p2a", inside & (radius >= 0.2 * config.case.radius_m)),
                    ("exterior", ~inside),
                ]:
                    for field in ("E", "H"):
                        metrics[f"{region}_{field}_target_relative_difference"] = relative(
                            sampled[field][mask], data[f"target_{field}_cartesian"][mask]
                        )
                        metrics[f"{region}_{field}_baseline_relative_difference"] = relative(
                            sampled[field][mask], data[f"{field}_cartesian"][mask]
                        )
                values["cases"][case_id] = metrics
                print(
                    run_dir.name,
                    case_id,
                    "exterior target E/H",
                    metrics["exterior_E_target_relative_difference"],
                    metrics["exterior_H_target_relative_difference"],
                    flush=True,
                )
        result["runs"][run_dir.name] = values
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.runs, args.output)
