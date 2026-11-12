"""Evaluate saved physical currents at prepared observations, without a new solve."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from applications.anisotropic_gstc_sphere.sie_gstc_sphere import _solver_cases_arguments
from sie_mom.postprocessing.field_io import load_field_csv

from .observations import read_observations
from .sie_gstc_surface import (
    REPO_ROOT,
    _read_json,
    _sha256,
    _validate_outputs,
    _write_json,
    load_solved_run,
)

DEFAULT_EXECUTABLE = REPO_ROOT / "build/applications/gstc_surface" / (
    "evaluate_surface_fields.exe" if os.name == "nt" else "evaluate_surface_fields"
)


def field_command(configuration, case, *, executable, mesh, currents, observations, output):
    command = [str(executable)]
    physical = configuration.case
    options = {
        "mesh": mesh, "currents": currents, "observations": observations, "output": output,
        "frequency-hz": physical.frequency_hz,
        "epsilon-r-interior": physical.relative_permittivity_interior,
        "mu-r-interior": physical.relative_permeability_interior,
        "epsilon-r-exterior": physical.relative_permittivity_exterior,
        "mu-r-exterior": physical.relative_permeability_exterior,
        "quadrature-rule": configuration.sie.quadrature_rule,
    }
    for name, value in options.items():
        command.extend(("--" + name, str(value)))
    selected = replace(configuration, simulation=replace(configuration.simulation, cases=(case,)))
    command.extend(_solver_cases_arguments(selected))
    return command


def run_field_evaluation(
    *, solve_dir, observations_dir, output_dir=None, executable=DEFAULT_EXECUTABLE, case_id=None
):
    solve_dir, observations_dir = Path(solve_dir).resolve(), Path(observations_dir).resolve()
    configuration, solve, mesh_path = load_solved_run(solve_dir)
    observations = _read_json(observations_dir / "observations.json")
    if (
        observations.get("artifact") != "gstc_surface_observations"
        or observations["mesh_sha256"] != solve["mesh"]["sha256"]
        or observations["coordinate_unit"] != "m"
    ):
        raise ValueError("observations must refer to the solve mesh and use metres")
    for case in solve["simulation"]["cases"]:
        if _sha256(solve_dir / case["files"]["currents"]) != case["currents_sha256"]:
            raise ValueError(f"saved currents SHA-256 mismatch: {case['id']}")
    _validate_outputs(solve_dir, solve)
    grids = {}
    for name, grid in observations["grids"].items():
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid observation grid name")
        path = observations_dir / grid["path"]
        if Path(grid["path"]).name != grid["path"] or _sha256(path) != grid["sha256"]:
            raise ValueError("observation CSV SHA-256 mismatch")
        points, regions = read_observations(path)
        if len(points) != grid["point_count"]:
            raise ValueError("observation count disagrees with its manifest")
        grids[name] = points, regions
    if not grids:
        raise ValueError("at least one observation grid is required")
    cases = [case for case in configuration.simulation.cases if case_id in (None, case.id)]
    if not cases:
        raise ValueError(f"unknown simulation case: {case_id}")
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"field evaluator not found: {executable}; build target evaluate_surface_fields"
        )
    output_dir = Path(output_dir or solve_dir / "fields/run01").resolve()
    if output_dir.exists():
        raise FileExistsError(f"field directory already exists: {output_dir}")
    manifest = {
        "schema_version": 1,
        "application": "gstc_surface",
        "stage": "fields",
        "status": "running",
        "solve_directory": str(solve_dir),
        "mesh_sha256": solve["mesh"]["sha256"],
        "executable_sha256": _sha256(executable),
        "sampling": observations["sampling"],
        "physical_construction": "F_incident(region) - Q(region); physical J and M",
        "coordinate_unit": "m",
        "electric_field_unit": "V/m",
        "magnetic_field_unit": "A/m",
        "warnings": solve["warnings"],
        "cases": [],
    }
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "fields.json"
    _write_json(manifest_path, manifest)
    try:
        shutil.copy2(solve_dir / "configuration.json", output_dir / "configuration.json")
        shutil.copy2(observations_dir / "observations.json", output_dir / "observations.json")
        for grid in observations["grids"].values():
            shutil.copy2(observations_dir / grid["path"], output_dir / grid["path"])
        for case in cases:
            saved = next(c for c in solve["simulation"]["cases"] if c["id"] == case.id)
            currents = solve_dir / saved["files"]["currents"]
            case_result = {"id": case.id, "currents_sha256": saved["currents_sha256"], "grids": {}}
            manifest["cases"].append(case_result)
            directory = output_dir / case.id
            directory.mkdir()
            for name, (points, regions) in grids.items():
                output = directory / f"sie_physical_{name}.csv"
                command = field_command(
                    configuration, case, executable=executable, mesh=mesh_path, currents=currents,
                    observations=output_dir / observations["grids"][name]["path"], output=output,
                )
                result = {"path": output.relative_to(output_dir).as_posix(), "command": command}
                case_result["grids"][name] = result
                _write_json(manifest_path, manifest)
                start = time.perf_counter()
                with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                data = load_field_csv(output)
                actual = np.column_stack([data[axis] for axis in "xyz"])
                if (
                    actual.shape != points.shape
                    or not np.allclose(actual, points, rtol=2e-7, atol=1e-12)
                    or not np.array_equal(data["region"], regions)
                    or not np.array_equal(data["point_index"], np.arange(len(points)))
                ):
                    raise ValueError("field output observations/regions differ from the input")
                result.update(sha256=_sha256(output), seconds=time.perf_counter() - start)
                print(
                    f"{case.id}/{name}: {len(points)} samples, {result['seconds']:.3f} s",
                    flush=True,
                )
        manifest["status"] = "evaluated"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-dir", required=True, type=Path)
    parser.add_argument("--observations-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--case", dest="case_id")
    print(run_field_evaluation(**vars(parser.parse_args())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
