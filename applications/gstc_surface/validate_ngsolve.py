"""Validate prepared physical GSTC tensors using the matching FEM interface.

Run with python -m applications.gstc_surface.validate_ngsolve. NGSolve is an
optional dependency. Preparation, SIE solving and FEM validation are separate.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import ngsolve as ng
import numpy as np

from sie_mom.mesh.obj_mesh import load_triangle_obj
from sie_mom.postprocessing import complex_vector, load_field_csv, write_field_csv
from sie_mom.validation.ngsolve import model as fem
from sie_mom.validation.ngsolve.tensor_transfer import shared_interface_coefficients
from sie_mom.validation.ngsolve.volume_mesh import load_shared_mesh, make_shared_mesh

from .artifacts import (
    artifact_path,
    read_observations,
)
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


def _observations(directory, mesh_sha256, air_lambda, k0):
    if directory is None:
        return {}, None
    directory = Path(directory).resolve()
    manifest = _read_json(directory / "observations.json")
    if (
        manifest.get("artifact") != "gstc_surface_observations"
        or manifest["mesh_sha256"] != mesh_sha256
        or manifest["coordinate_unit"] != "m"
    ):
        raise ValueError("observations must use the prepared mesh and metres")
    grids = {}
    for name, entry in manifest["grids"].items():
        path = directory / entry["path"]
        if Path(entry["path"]).name != entry["path"] or _sha256(path) != entry["sha256"]:
            raise ValueError("observation CSV SHA-256 mismatch")
        points, regions = read_observations(path)
        if len(points) != entry["point_count"]:
            raise ValueError("observation count disagrees with manifest")
        if np.any(np.linalg.norm(points * k0, axis=1) >= 2 * np.pi * air_lambda):
            raise ValueError("increase --air-lambda to place all observations in physical air")
        grids[name] = points, regions
    return grids, manifest


def sample_physical_fields(system, fields, points, regions):
    """Follow saved region labels; verify them against the actual FEM materials."""
    k0 = fem.k0_of(system.config)
    mapped = system.mesh(*(points * k0).T, ng.VOL, tol=1e-10)
    material = system.mesh.MaterialCF({"core": 1, "air": 2, "pml": 3})
    if not np.array_equal(np.asarray(material(mapped)).reshape(-1), regions):
        raise ValueError("saved observation regions disagree with FEM core/air locations")
    electric, magnetic = np.empty((len(points), 3), complex), np.empty((len(points), 3), complex)
    for region, pair in enumerate(fields, 1):
        mask = regions == region
        if np.any(mask):
            electric[mask] = np.asarray(pair[0](mapped[mask])) * k0**3
            magnetic[mask] = np.asarray(pair[1](mapped[mask])) * k0**3 / fem.ETA0
    if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
        raise ValueError("nonfinite FEM observation fields")
    return electric, magnetic


def compare_sie_fields(electric, magnetic, points, regions, path):
    saved = load_field_csv(path)
    if not np.array_equal(saved["region"], regions) or not np.allclose(
        np.column_stack([saved[axis] for axis in "xyz"]), points, rtol=0, atol=1e-14
    ):
        raise ValueError("SIE and FEM comparison samples/regions differ")
    result = {}
    for region, name in ((1, "interior"), (2, "exterior")):
        mask = regions == region
        if not np.any(mask):
            continue
        result[name] = {}
        for field, values in (("E", electric), ("H", magnetic)):
            reference = complex_vector(saved, field)[mask]
            denominator = np.linalg.norm(reference)
            result[name][field] = {
                "point_count": int(mask.sum()),
                "relative_complex_l2": float(np.linalg.norm(values[mask] - reference) / denominator)
                if denominator > 0
                else None,
            }
    return result


def _comparison_paths(directory, manifest, case_ids, grids):
    """Check every requested SIE reference before meshing or factorization."""
    cases = {case["id"]: case for case in manifest["cases"]}
    paths = {}
    for case_id in case_ids:
        if case_id not in cases or not set(grids).issubset(cases[case_id]["grids"]):
            raise ValueError(f"SIE field run is missing requested case/grids: {case_id}")
        paths[case_id] = {}
        for name in grids:
            entry = cases[case_id]["grids"][name]
            path = artifact_path(directory, entry["path"])
            if _sha256(path) != entry["sha256"]:
                raise ValueError("SIE comparison CSV SHA-256 mismatch")
            paths[case_id][name] = path, entry["sha256"]
    return paths


def _reusable_mesh(directory, config, mesh_sha256, settings):
    """Accept a completed matching run, preserving its volume-mesh provenance."""
    directory = Path(directory).resolve()
    report = _read_json(directory / "metrics.json")
    if (
        report.get("backend") != "ngsolve"
        or report.get("status") != "completed"
        or report["mesh_sha256"] != mesh_sha256
        or _read_json(directory / "configuration.json") != config.to_dict()
    ):
        raise ValueError(
            "mesh reuse requires a completed run with identical configuration and interface"
        )
    for name in ("ppw", "interior_ppw", "air_lambda", "pml_lambda", "pml_strength"):
        previous = report["settings"][name]
        current = getattr(settings, name)
        if name == "interior_ppw":
            previous = report["settings"]["ppw"] if previous is None else previous
            current = settings.ppw if current is None else current
        if previous != current:
            raise ValueError(f"mesh reuse requires identical {name}")
    path = directory / "mesh.vol.gz"
    digest = _sha256(path)
    if report.get("volume_mesh_sha256", digest) != digest:
        raise ValueError("saved volume mesh SHA-256 mismatch")
    return path, {"run_directory": str(directory), "volume_mesh_sha256": digest}


def run(args):
    if args.order < 1 or args.threads < 1:
        raise ValueError("field order and thread count must be positive")
    inputs_dir = args.inputs_dir.resolve()
    config, prepared, _, files = _validated_inputs(inputs_dir)
    mesh_path = config.mesh_path.resolve()
    source = load_triangle_obj(mesh_path)
    physical_maps = np.fromfile(inputs_dir / "face_tensors.bin", dtype="<c16").reshape(-1, 32)
    physical_maps = physical_maps[:, :16].reshape(-1, 4, 4)
    case_ids = [case.id for case in config.simulation.cases if args.case_id in (None, case.id)]
    if not case_ids:
        raise ValueError("unknown simulation case")
    sources = {case_id: fem.incident_cf(config, case_id) for case_id in case_ids}
    k0 = fem.k0_of(config)
    grids, observations = _observations(
        args.observations_dir, prepared["mesh"]["sha256"], args.air_lambda, k0
    )
    sie_directory = Path(args.sie_fields_dir).resolve() if args.sie_fields_dir else None
    sie_manifest = _read_json(sie_directory / "fields.json") if sie_directory else None
    if sie_manifest is not None and (
        sie_manifest.get("status") != "evaluated"
        or sie_manifest["mesh_sha256"] != prepared["mesh"]["sha256"]
        or _read_json(sie_directory / "configuration.json") != config.to_dict()
    ):
        raise ValueError("SIE field run must use identical prepared configuration and mesh")
    if sie_manifest is not None and not grids:
        raise ValueError("--sie-fields-dir requires --observations-dir")
    comparison_paths = (
        _comparison_paths(sie_directory, sie_manifest, case_ids, grids)
        if sie_manifest is not None
        else {}
    )
    solver = fem.direct_solver(args.inverse)
    reused_mesh = (
        _reusable_mesh(args.mesh_run_dir, config, prepared["mesh"]["sha256"], args)
        if args.mesh_run_dir
        else None
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for filename in files:
        shutil.copy2(inputs_dir / filename, output / filename)
    shutil.copy2(mesh_path, output / "surface.obj")
    field_dir = output / "fields/run01"
    field_dir.mkdir(parents=True)
    shutil.copy2(inputs_dir / "configuration.json", field_dir / "configuration.json")
    settings = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    report = {
        "application": "gstc_surface",
        "backend": "ngsolve",
        "status": "running",
        "ngsolve": ng.__version__,
        "settings": settings,
        "mesh_sha256": prepared["mesh"]["sha256"],
        "tensor_sha256": _sha256(inputs_dir / "face_tensors.bin"),
        "input_manifest_sha256": _sha256(inputs_dir / "sie_inputs.json"),
        "inverse": solver,
        "cases": {},
        "warnings": prepared.get("warnings", []),
        "accuracy_note": "Independent FEM comparison; mesh/order/PML convergence remains pending.",
    }
    fields_manifest = {
        "schema_version": 1,
        "application": "gstc_surface",
        "stage": "fields",
        "solver_label": "NGSolve",
        "status": "running",
        "solve_directory": str(output),
        "mesh_sha256": prepared["mesh"]["sha256"],
        "sampling": observations["sampling"] if observations else config.raw["sampling"],
        "physical_construction": "interior analytic incident plus FEM correction; exterior total",
        "coordinate_unit": "m",
        "electric_field_unit": "V/m",
        "magnetic_field_unit": "A/m",
        "warnings": report["warnings"],
        "cases": [],
    }

    def save():
        _write_json(output / "metrics.json", report)
        _write_json(field_dir / "fields.json", fields_manifest)

    save()
    ng.SetNumThreads(args.threads)
    ng.ngsglobals.msg_level = 0
    try:
        with ng.TaskManager():
            mesh_settings = dict(
                air_lambda=args.air_lambda,
                pml_lambda=args.pml_lambda,
                pml_strength=args.pml_strength,
            )
            if reused_mesh:
                path, provenance = reused_mesh
                mesh, mesh_metrics = load_shared_mesh(config, source, path, **mesh_settings)
                report["mesh_reused_from"] = provenance
                shutil.copy2(path, output / "mesh.vol.gz")
                print(f"Reused volume mesh: {path}", flush=True)
            else:
                mesh, mesh_metrics = make_shared_mesh(
                    config, source, ppw=args.ppw, interior_ppw=args.interior_ppw, **mesh_settings
                )
                mesh.ngmesh.Save(str(output / "mesh.vol.gz"))
            report["mesh"] = mesh_metrics
            report["volume_mesh_sha256"] = _sha256(output / "mesh.vol.gz")
            coefficients = shared_interface_coefficients(mesh, source, physical_maps, k0)
            np.savez_compressed(
                output / "interface.npz",
                boundary_element=coefficients.boundary_elements,
                original_sie_face=coefficients.source_faces,
            )
            report["maximum_magnetic_condition_number"] = (
                coefficients.maximum_magnetic_condition_number
            )
            system = fem.assemble(
                config,
                mesh,
                frame=coefficients.frame,
                admittance=coefficients.admittance,
                order=args.order,
            )
            report.update(
                dofs=system.space.ndof,
                free_dofs=sum(system.space.FreeDofs()),
                assembly_seconds=system.assembly_seconds,
            )
            print(
                f"Mesh: {mesh.ne} tetrahedra; assembly: {report['free_dofs']} free DOFs, "
                f"{system.assembly_seconds:.2f} s",
                flush=True,
            )
            save()
            started = time.perf_counter()
            cpu_started = time.process_time()
            inverse = system.matrix.mat.Inverse(
                system.space.FreeDofs(),
                inverse=solver,
                flags={"ordering": "metis"} if solver == "pardiso" else {},
            )
            report["factorization_seconds"] = time.perf_counter() - started
            report["factorization_cpu_seconds"] = time.process_time() - cpu_started
            report["factorization_average_cpu_cores"] = (
                report["factorization_cpu_seconds"] / report["factorization_seconds"]
            )
            print(
                f"Factorization ({solver}): {report['factorization_seconds']:.2f} s; "
                f"shared by {len(case_ids)} RHS",
                flush=True,
            )
            save()
            for case_id, incident in sources.items():
                directory = output / case_id
                directory.mkdir()
                started = time.perf_counter()
                rhs = system.rhs(*incident)
                solution = ng.GridFunction(system.space)
                solution.vec.data = inverse * rhs.vec
                error = fem.residual(system, solution, rhs)
                result = {
                    "linear_relative_residual": error,
                    "rhs_and_solve_seconds": time.perf_counter() - started,
                    "grids": {},
                }
                report["cases"][case_id] = result
                if not np.isfinite(error) or error > 1e-8:
                    raise RuntimeError(f"{case_id}: unacceptable linear residual {error}")
                solution.Save(str(directory / "solution.sol"))
                fields = fem.total_fields(system, solution, *incident)
                field_case = {"id": case_id, "grids": {}}
                fields_manifest["cases"].append(field_case)
                (field_dir / case_id).mkdir()
                for name, (points, regions) in grids.items():
                    electric, magnetic = sample_physical_fields(system, fields, points, regions)
                    path = field_dir / case_id / f"ngsolve_physical_{name}.csv"
                    write_field_csv(path, *points.T, regions, electric, magnetic)
                    field_case["grids"][name] = {
                        "path": path.relative_to(field_dir).as_posix(),
                        "sha256": _sha256(path),
                    }
                    if sie_manifest is not None:
                        reference_path, expected_hash = comparison_paths[case_id][name]
                        if _sha256(reference_path) != expected_hash:
                            raise ValueError("SIE comparison CSV SHA-256 mismatch")
                        result["grids"][name] = compare_sie_fields(
                            electric, magnetic, points, regions, reference_path
                        )
                print(f"{case_id}: linear relative residual {error:.3e}", flush=True)
                save()
        report["status"] = "completed"
        fields_manifest["status"] = "evaluated" if grids else "not_sampled"
        save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        fields_manifest["status"] = "failed"
        save()
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh result directory")
    parser.add_argument("--observations-dir", type=Path, help="Reuse saved SIE point/region grids")
    parser.add_argument(
        "--sie-fields-dir", type=Path, help="Compare complex fields at identical points"
    )
    parser.add_argument("--case-id")
    parser.add_argument(
        "--mesh-run-dir",
        type=Path,
        help="Reuse mesh.vol.gz from a completed run with identical geometry and mesh/PML settings",
    )
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--ppw", type=float, default=4)
    parser.add_argument("--interior-ppw", type=float)
    parser.add_argument("--air-lambda", type=float, default=1.8)
    parser.add_argument("--pml-lambda", type=float, default=0.25)
    parser.add_argument("--pml-strength", type=float, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--inverse", choices=("auto", "pardiso", "umfpack"), default="auto")
    args = parser.parse_args()
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
