"""Solve prepared general-surface GSTC inputs in one CUDA batch, without computing fields.

Run with ``python -m applications.gstc_surface.sie_gstc_surface``. Expects
``--inputs-dir`` containing ``configuration.json``, ``sie_inputs.json``, and
``face_tensors.bin``. Preparation and observation sampling remain separate
steps. ``--dry-run`` is read-only.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.constants import epsilon_0, mu_0

from applications.anisotropic_gstc_sphere.sie_gstc_sphere import (
    DEFAULT_EXECUTABLE as SPHERE_EXECUTABLE,
)
from applications.anisotropic_gstc_sphere.sie_gstc_sphere import (
    _solver_cases_arguments,
)
from sie_mom.mesh.obj_mesh import load_triangle_obj

from .artifacts import (
    load_prepared_inputs as _validated_inputs,
)
from .artifacts import (
    read_json as _read_json,
)
from .artifacts import (
    sha256 as _sha256,
)
from .artifacts import (
    write_json as _write_json,
)
from .workflow import APPLICATION_DIR, load_workflow_configuration

REPO_ROOT = APPLICATION_DIR.parents[1]
DEFAULT_EXECUTABLE = SPHERE_EXECUTABLE.with_suffix(".exe") if os.name == "nt" else SPHERE_EXECUTABLE


def choose_assembly_singularity_threshold(
    mesh, frequency_hz, configured_lambda0, *, safety_factor=1.05
):
    """Cover every touching triangle pair using the actual ordered solve mesh.

    The Calderon assembly kernel branches on centroid distance, not edge length:
    ``float distance < float threshold``. Enumerate all distinct face pairs with
    at least one common vertex index, including vertex-only and edge neighbors.
    Use max(configured_lambda0 * lambda0, safety_factor * maximum_distance).
    Self pairs have distance zero and are covered by the positive config floor.

    Coverage comes from measuring the maximum over ALL touching pairs. The 5%
    factor is a numerical margin, not a quadrature accuracy criterion. Check the
    strict comparison after float conversion for both float/double coordinate
    arithmetic, and refuse to launch if any touching pair remains uncovered.
    Vertex incidence avoids a dense, all-face-pairs distance matrix.

    The caller validates topology first. Returned values describe assembly only;
    they do not change the independent field-evaluation kernel threshold. The
    source configuration and prepared tensors are never modified by this helper.
    """
    for name, value in (
        ("frequency_hz", frequency_hz),
        ("configured_lambda0", configured_lambda0),
        ("safety_factor", safety_factor),
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if safety_factor <= 1:
        raise ValueError("safety_factor must be greater than one for strict coverage")

    incident = [[] for _ in mesh.vertices]
    for face, vertices in enumerate(mesh.faces):
        for vertex in vertices:
            incident[int(vertex)].append(face)
    pairs = np.asarray(
        sorted({pair for faces in incident for pair in combinations(faces, 2)}), dtype=np.int64
    ).reshape(-1, 2)
    distances = []
    speeds = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for dtype in (np.float64, np.float32):
            triangles = np.asarray(mesh.vertices, dtype=dtype)[mesh.faces]
            centroids = (triangles[:, 0] + triangles[:, 1] + triangles[:, 2]) / dtype(3)
            delta = centroids[pairs[:, 0]] - centroids[pairs[:, 1]]
            distances.append(np.sqrt(np.sum(delta * delta, axis=1)))
            speeds.append(dtype(1) / np.sqrt(dtype(epsilon_0) * dtype(mu_0)))
        maximum = max(float(d.max(initial=0)) for d in distances)
        lambda0 = float(speeds[0] / frequency_hz)
        effective = max(float(configured_lambda0), safety_factor * maximum / lambda0)

        def kernel_threshold(value):
            # Match C++ evaluation order before the kernel's float argument cast.
            return min(
                float(np.float32(dtype(dtype(value) * speed) / dtype(frequency_hz)))
                for dtype, speed in zip((np.float64, np.float32), speeds)
            )

        configured_kernel = kernel_threshold(configured_lambda0)
        effective_kernel = kernel_threshold(effective)
        kernel_distances = np.maximum(
            distances[0].astype(np.float32), distances[1].astype(np.float32)
        )
    if (
        not np.isfinite(kernel_distances).all()
        or not np.isfinite([lambda0, effective, effective_kernel]).all()
        or effective_kernel <= 0
    ):
        raise ValueError(
            "mesh/frequency cannot be represented by the assembly threshold arithmetic"
        )
    uncovered = int(np.count_nonzero(kernel_distances >= effective_kernel))
    if uncovered:
        raise ValueError(f"computed assembly threshold leaves {uncovered} touching pairs uncovered")
    return {
        "policy": "configured_floor_and_maximum_touching_centroid_distance",
        "configured_lambda0": float(configured_lambda0),
        "effective_lambda0": effective,
        "effective_m": effective * lambda0,
        "lambda0_m": lambda0,
        "safety_factor": float(safety_factor),
        "maximum_touching_centroid_distance_m": maximum,
        "touching_pair_count": len(pairs),
        "configured_uncovered_pair_count": int(
            np.count_nonzero(kernel_distances >= configured_kernel)
        ),
        "effective_uncovered_pair_count": uncovered,
        "minimum_checked_float32_kernel_threshold_m": effective_kernel,
        "arithmetic_checked": ["float64_geometry", "float32_geometry"],
    }


def load_solved_run(solve_dir):
    """Validate the saved geometry/configuration identity for postprocessing."""
    solve_dir = Path(solve_dir).resolve()
    run = _read_json(solve_dir / "run.json")
    if run.get("application") != "gstc_surface" or run.get("status") != "solved":
        raise ValueError("expected a solved gstc_surface current run")
    mesh_path = solve_dir / run["files"]["mesh"]
    if _sha256(mesh_path) != run["mesh"]["sha256"]:
        raise ValueError("solve mesh snapshot SHA-256 mismatch")
    inputs_path = solve_dir / run["files"]["sie_inputs"]
    if _sha256(inputs_path) != run["input_preparation"]["manifest_sha256"]:
        raise ValueError("preparation manifest SHA-256 mismatch")
    configuration = load_workflow_configuration(solve_dir / "configuration.json")
    if configuration.to_dict() != _read_json(inputs_path)["configuration"]:
        raise ValueError("solve configuration differs from the prepared snapshot")
    return configuration, run, mesh_path


def solver_command(configuration, *, run_dir, executable, singularity_threshold_lambda0):
    """Reuse source serialization, omitting the sphere sampler unconditionally."""
    case = configuration.case
    command = [str(executable)]
    options = {
        "mesh": run_dir / "surface.obj",
        "tensors": run_dir / "face_tensors.bin",
        "output-dir": run_dir,
        "frequency-hz": case.frequency_hz,
        "epsilon-r-interior": case.relative_permittivity_interior,
        "mu-r-interior": case.relative_permeability_interior,
        "epsilon-r-exterior": case.relative_permittivity_exterior,
        "mu-r-exterior": case.relative_permeability_exterior,
        "quadrature-rule": configuration.sie.quadrature_rule,
        "singularity-threshold-lambda0": singularity_threshold_lambda0,
        "basis": "face",
    }
    for name, value in options.items():
        command.extend(("--" + name, str(value)))
    command.extend(_solver_cases_arguments(configuration))
    command.append("--no-fields")
    if not configuration.sie.compute_residual:
        command.append("--no-residual")
    return command


def _validate_outputs(run_dir, manifest):
    """A successful process must also produce matching, finite physical currents."""
    batch = _read_json(run_dir / "batch_metrics.json")
    cases = manifest["simulation"]["cases"]
    if batch["case_ids"] != [case["id"] for case in cases]:
        raise ValueError("solver output case IDs disagree with the requested batch")
    for name in ("vertex_count", "face_count", "edge_count"):
        if batch["mesh"][name] != manifest["mesh"][name]:
            raise ValueError(f"solver output mesh {name} disagrees with the prepared mesh")
    expected = manifest["solver"]
    for name in ("method", "basis", "scaling", "case_count", "matrix_dimension"):
        if batch["solver"][name] != expected[name]:
            raise ValueError(f"solver output {name} disagrees with the requested batch")
    for case in cases:
        metrics = _read_json(run_dir / case["files"]["solver_metrics"])
        if metrics["case_id"] != case["id"] or metrics["basis"] != "face":
            raise ValueError(f"solver output identity disagrees for {case['id']}")
        data = metrics["current_data"]
        if (
            data["physical_currents"] is not True
            or data["precision"] != "complex128"
            or data["byte_order"] != "little-endian"
            or data["ordering"] != "J1, M1, J2, M2; one block per RWG edge"
            or data["complex_value_count"] != expected["matrix_dimension"]
        ):
            raise ValueError(f"unsupported current layout for {case['id']}")
        current_path = run_dir / case["files"]["currents"]
        if current_path.stat().st_size != 16 * expected["matrix_dimension"]:
            raise ValueError(f"current binary size disagrees for {case['id']}")
        if not np.isfinite(np.fromfile(current_path, dtype="<c16")).all():
            raise ValueError(f"non-finite currents for {case['id']}")
        if metrics["solver"]["residual_computed"] != expected["compute_residual"]:
            raise ValueError(f"residual output disagrees for {case['id']}")
        if expected["compute_residual"]:
            residual = metrics["solver"]["scaled_relative_residual"]
            if not np.isfinite(residual) or residual < 0:
                raise ValueError(f"invalid algebraic residual for {case['id']}")
            case["scaled_relative_residual"] = residual
        case["currents_sha256"] = _sha256(current_path)


def run_sie_solve(*, inputs_dir, run_dir=None, executable=DEFAULT_EXECUTABLE, dry_run=False):
    """Snapshot prepared inputs, solve all cases together, and retain failure logs.

    Dry runs validate inputs and return the planned manifest without creating
    directories or invoking the solver. A real run always requires a fresh path.
    """
    inputs_dir = Path(inputs_dir).resolve()
    configuration, inputs, report, prepared_files = _validated_inputs(inputs_dir)
    run_dir = Path(
        run_dir or APPLICATION_DIR / "results" / configuration.case_id / "sie/run01"
    ).resolve()
    executable = Path(executable).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose a fresh run directory"
        )
    if not dry_run and not executable.is_file():
        raise FileNotFoundError(
            f"SIE executable does not exist: {executable}\n"
            "Build with: cmake --build build --target anisotropic_sphere_app --parallel 4"
        )
    threshold = choose_assembly_singularity_threshold(
        load_triangle_obj(configuration.mesh_path),
        configuration.case.frequency_hz,
        configuration.sie.singularity_threshold_lambda0,
    )
    command = solver_command(
        configuration,
        run_dir=run_dir,
        executable=executable,
        singularity_threshold_lambda0=threshold["effective_lambda0"],
    )
    simulation = configuration.to_dict()["simulation"]
    for case in simulation["cases"]:
        directory = f"cases/{case['id']}"
        case.update(
            status="pending",
            files={
                "currents": f"{directory}/currents.bin",
                "solver_metrics": f"{directory}/solver_metrics.json",
            },
        )
    manifest = {
        "schema_version": 1,
        "application": "gstc_surface",
        "backend": "sie",
        "stage": "currents",
        "case_id": configuration.case_id,
        "status": "planned",
        "solver_executed": False,
        "command": command,
        "executable_sha256": _sha256(executable) if executable.is_file() else None,
        "input_preparation": {
            "directory": str(inputs_dir),
            "manifest_sha256": _sha256(inputs_dir / "sie_inputs.json"),
            "status": inputs["status"],
        },
        "warnings": inputs.get("warnings", []).copy(),
        "mesh": {**report, "path": "surface.obj", "sha256": inputs["mesh"]["sha256"]},
        "tensors_sha256": _sha256(inputs_dir / "face_tensors.bin"),
        "solver": {
            "method": "dense_lu_batch",
            "basis": "face",
            "scaling": "eta0",
            "case_count": len(simulation["cases"]),
            "matrix_dimension": inputs["matrix_dimension"],
            "dense_matrix_bytes": inputs["dense_matrix_bytes"],
            "compute_residual": configuration.sie.compute_residual,
            "evaluate_fields": False,
            "singularity_threshold": threshold,
        },
        "simulation": simulation,
        "files": {
            "configuration": "configuration.json",
            "sie_inputs": "sie_inputs.json",
            "mesh": "surface.obj",
            "tensors": "face_tensors.bin",
            "batch_metrics": "batch_metrics.json",
            "solver_log": "solver.log",
        },
    }
    for warning in manifest["warnings"]:
        print(f"Preparation warning: {warning}", flush=True)
    print(
        "Using assembly singularity threshold: "
        f"{threshold['effective_lambda0']:.9g} lambda0 "
        f"({threshold['effective_m']:.9g} m)",
        flush=True,
    )
    print(
        f"  Configured minimum: {threshold['configured_lambda0']:.9g} lambda0; "
        "touching triangle pairs covered: "
        f"{threshold['touching_pair_count'] - threshold['effective_uncovered_pair_count']}"
        f"/{threshold['touching_pair_count']}",
        flush=True,
    )
    formatter = subprocess.list2cmdline if os.name == "nt" else shlex.join
    print(f"Batch command: {formatter(command)}", flush=True)
    if dry_run:
        return manifest

    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run.json"
    try:
        for name in sorted(prepared_files):
            shutil.copy2(inputs_dir / name, run_dir / name)
        shutil.copy2(configuration.mesh_path, run_dir / "surface.obj")
        if (
            _sha256(run_dir / "surface.obj") != manifest["mesh"]["sha256"]
            or _sha256(run_dir / "face_tensors.bin") != manifest["tensors_sha256"]
        ):
            raise ValueError("prepared inputs changed while making the run snapshot")
        manifest["status"] = "running"
        for case in simulation["cases"]:
            case["status"] = "running"
        _write_json(manifest_path, manifest)
        print(f"Solving; output is saved in {run_dir / 'solver.log'}", flush=True)
        with (run_dir / "solver.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        _validate_outputs(run_dir, manifest)
        manifest["solver_executed"] = True
        manifest["status"] = "solved"
        for case in simulation["cases"]:
            case["status"] = "solved"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        for case in simulation["cases"]:
            case["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise
    _write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-dir", required=True, type=Path, help="Completed preparation directory"
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Fresh output directory (default: results/CASE/sie/run01)"
    )
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the batch command without writing files",
    )
    args = parser.parse_args()
    try:
        manifest = run_sie_solve(**vars(args))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SIE solve failed: {error}\n")
    print(f"Status: {manifest['status']}; solver_executed={manifest['solver_executed']}")
    for case in manifest["simulation"]["cases"]:
        if "scaled_relative_residual" in case:
            print(
                f"  {case['id']}: scaled relative algebraic residual = "
                f"{case['scaled_relative_residual']:.6g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
