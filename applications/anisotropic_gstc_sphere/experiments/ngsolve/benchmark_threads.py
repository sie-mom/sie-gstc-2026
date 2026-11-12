#!/usr/bin/env python3
"""Time PARDISO thread counts on exactly one assembled checkpoint-mesh matrix."""

import argparse
import gc
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import model as m
import ngsolve as ng
import numpy as np
from threadpoolctl import threadpool_info


def benchmark(run_dir, output_dir, counts, ordering="minimum_degree"):
    m.direct_solver("pardiso")
    if any(n < 1 for n in counts):
        raise ValueError("thread counts must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("benchmark output must be a new or empty directory")
    baseline = json.loads((run_dir / "metrics.json").read_text())
    if baseline["ngsolve"] != ng.__version__:
        raise ValueError("checkpoint requires the original NGSolve version")
    config, sheet, inputs = m.load_benchmark(
        run_dir / "configuration.json", Path(baseline["inputs"]["tensors"])
    )
    mesh = ng.Mesh(str(run_dir / "mesh.vol.gz"))
    # Sampling is restricted to physical regions; assembly also needs the PML transform.
    settings = baseline["settings"]
    mesh.SetPML(
        ng.pml.Radial(
            origin=(0, 0, 0),
            rad=2 * np.pi * settings["air_lambda"],
            alpha=-1j * settings["pml_strength"],
        ),
        "pml",
    )
    ng.SetNumThreads(1)
    ng.ngsglobals.msg_level = 0
    with ng.TaskManager():
        system = m.assemble(config, sheet, mesh, order=settings["order"])
        cases = {}
        for case_id in baseline["cases"]:
            rhs = system.rhs(*m.incident_cf(config, case_id))
            reference = ng.GridFunction(system.space)
            reference.Load(str(run_dir / case_id / "solution.sol"))
            cases[case_id] = (rhs, reference)
    if system.space.ndof != baseline["dofs"]:
        raise ValueError("checkpoint DOF count does not match")
    digest = hashlib.sha256()
    for item in system.matrix.mat.CSR():
        digest.update(np.asarray(item.NumPy()).tobytes())
    report = {
        "source_run": str(run_dir.resolve()),
        "inputs": inputs,
        "dofs": system.space.ndof,
        "matrix_sha256": digest.hexdigest(),
        "ngsolve": ng.__version__,
        "mkl": version("mkl"),
        "ordering": ordering,
        "assembly_seconds": system.assembly_seconds,
        "environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "MKL_DYNAMIC")
        },
        "measurements": [],
        "status": "running",
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    def save():
        (output_dir / "benchmark.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    save()
    try:
        for nthreads in counts:
            ng.SetNumThreads(nthreads)
            with ng.TaskManager():
                wall, cpu = time.perf_counter(), time.process_time()
                inverse = system.matrix.mat.Inverse(
                    system.space.FreeDofs(), inverse="pardiso", flags={"ordering": ordering}
                )
                elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
                row = {
                    "threads_requested": nthreads,
                    "ngsolve_threads": ng.GetNumThreads(),
                    "analysis_factorization_seconds": elapsed,
                    "process_cpu_seconds": cpu_elapsed,
                    "cpu_seconds_per_wall_second": cpu_elapsed / elapsed,
                    "cases": {},
                }
                for case_id, (rhs, reference) in cases.items():
                    solution = ng.GridFunction(system.space)
                    start = time.perf_counter()
                    solution.vec.data = inverse * rhs.vec
                    values = {
                        "backsolve_seconds": time.perf_counter() - start,
                        "linear_relative_residual": m.residual(system, solution, rhs),
                    }
                    values["coefficient_relative_difference_from_saved_umfpack"] = float(
                        np.linalg.norm(solution.vec.FV().NumPy() - reference.vec.FV().NumPy())
                        / np.linalg.norm(reference.vec.FV().NumPy())
                    )
                    if values["linear_relative_residual"] > 1e-8:
                        raise RuntimeError("unacceptable linear residual")
                    row["cases"][case_id] = values
                report["measurements"].append(row)
                report["threadpools"] = threadpool_info()
                save()
                print(
                    f"PARDISO {nthreads} threads: {elapsed:.3f} s analysis/LU; "
                    f"CPU/wall={cpu_elapsed / elapsed:.2f}",
                    flush=True,
                )
                del inverse
                gc.collect()
        report["status"] = "completed"
        save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16, 1])
    parser.add_argument(
        "--ordering",
        choices=("minimum_degree", "metis", "parallel_metis"),
        default="minimum_degree",
    )
    args = parser.parse_args()
    benchmark(args.run_dir, args.output_dir, args.threads, args.ordering)
