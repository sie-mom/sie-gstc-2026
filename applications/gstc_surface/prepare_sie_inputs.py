"""CPU-only geometry, prescribed-field evaluation and joint GSTC extraction.

Run as ``python -m applications.gstc_surface.prepare_sie_inputs`` from the
repository root. This phase never launches the CUDA solver or NGSolve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import gmsh
import numpy as np

from applications.anisotropic_gstc_sphere.prescribed_fields import (
    prescribed_boundary_states_cartesian,
)
from sie_mom.gstc import (
    extract_reciprocal_complex_least_squares,
    extract_reciprocal_lossless_least_squares,
    form_boundary_quantities,
)
from sie_mom.gstc.tensor_io import prepare_tensor_records
from sie_mom.mesh.face_frames import FaceFrames, compute_face_frames, project_to_face_frame
from sie_mom.mesh.obj_mesh import load_triangle_obj

from .geometry import generate_mesh, validate_mesh
from .workflow import APPLICATION_DIR, DEFAULT_CONFIGURATION, load_workflow_configuration


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _power(electric, magnetic):
    return 0.5 * np.real(electric[0] * magnetic[1].conj() - electric[1] * magnetic[0].conj())


def _power_diagnostic(interior, exterior, areas):
    difference = exterior - interior
    scale = np.max(np.abs(interior))
    norm = np.sqrt(np.sum(areas * interior**2))
    return {
        "interior_integrated_power_w": float(np.sum(areas * interior)),
        "exterior_integrated_power_w": float(np.sum(areas * exterior)),
        "area_weighted_relative_l2_defect": float(
            np.sqrt(np.sum(areas * difference**2)) / max(norm, np.finfo(float).tiny)
        ),
        "maximum_defect_over_peak_interior_flux": float(
            np.max(np.abs(difference)) / max(scale, np.finfo(float).tiny)
        ),
    }


def prepare_inputs(configuration, *, run_dir, use_existing_mesh=False):
    """Preserve mesh-face order, save diagnostics, then attempt full-map export."""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose a new run directory"
        )
    if configuration.mesh_path.exists() and not use_existing_mesh:
        raise FileExistsError(
            f"mesh already exists: {configuration.mesh_path}; use --use-existing-mesh"
        )
    if use_existing_mesh and not configuration.mesh_path.is_file():
        raise FileNotFoundError(configuration.mesh_path)
    print("[1/4] Surface mesh generation and geometry checks", flush=True)
    if not use_existing_mesh:
        generate_mesh(configuration)
    mesh = load_triangle_obj(configuration.mesh_path)
    report = validate_mesh(mesh, configuration.geometry)
    frames = compute_face_frames(mesh)
    print(
        f"  {report.face_count} faces, {report.edge_count} RWG edges; "
        f"Euler characteristic {report.euler_characteristic}",
        flush=True,
    )
    print("[2/4] Cartesian prescribed fields and intrinsic face tangents", flush=True)
    states = prescribed_boundary_states_cartesian(frames.centroids, configuration)
    tangential = [
        tuple(
            project_to_face_frame(getattr(state, name), frames)
            for name in (
                "electric_interior",
                "magnetic_interior",
                "electric_exterior",
                "magnetic_exterior",
            )
        )
        for state in states
    ]
    quantities = [form_boundary_quantities(*fields) for fields in tangential]
    print("[3/4] Joint reciprocal tensor extraction", flush=True)
    extractor = (
        extract_reciprocal_lossless_least_squares
        if configuration.extraction_method == "reciprocal_lossless"
        else extract_reciprocal_complex_least_squares
    )
    eta = configuration.case.eta_exterior
    extraction = extractor(
        quantities, reference_impedance=eta, structure=configuration.tensor_structure
    )
    power_in = np.stack([_power(e, h) for e, h, _, _ in tangential])
    power_out = np.stack([_power(e, h) for _, _, e, h in tangential])
    source_norm_sq = np.sum(
        [
            np.sum(np.abs(q.electric_current) ** 2 + np.abs(q.magnetic_current / eta) ** 2, axis=0)
            for q in quantities
        ],
        axis=0,
    )
    diagnostics = {
        "minimum_rank": int(extraction.rank.min()),
        "maximum_rank": int(extraction.rank.max()),
        "unique_face_count": int(np.count_nonzero(extraction.unique)),
        "non_unique_face_count": int(np.count_nonzero(~extraction.unique)),
        "maximum_condition_number": float(extraction.condition_number.max())
        if np.isfinite(extraction.condition_number).all()
        else None,
        "infinite_condition_number_face_count": int(
            np.count_nonzero(~np.isfinite(extraction.condition_number))
        ),
        "relative_residual_median": float(np.median(extraction.relative_residual)),
        "relative_residual_p95": float(np.percentile(extraction.relative_residual, 95)),
        "maximum_relative_residual": float(extraction.relative_residual.max()),
        "area_weighted_relative_residual": float(
            np.sqrt(
                np.sum(frames.areas * extraction.residual_norm**2)
                / max(np.sum(frames.areas * source_norm_sq), np.finfo(float).tiny)
            )
        ),
        "maximum_state_relative_residual": {
            state.state_id: float(extraction.state_relative_residual[i].max())
            for i, state in enumerate(states)
        },
        "prescribed_normal_power": {
            state.state_id: _power_diagnostic(power_in[i], power_out[i], frames.areas)
            for i, state in enumerate(states)
        },
        "coherent_superposition_normal_power": {},
    }
    # Diagnostic control only. Export remains in the actual triangle frames.
    smooth_normals = configuration.geometry.outward_normals(frames.centroids)
    smooth_t1 = frames.t1 - np.sum(frames.t1 * smooth_normals, axis=1)[:, None] * smooth_normals
    smooth_t1 /= np.linalg.norm(smooth_t1, axis=1)[:, None]
    smooth_frames = FaceFrames(
        frames.centroids,
        smooth_t1,
        np.cross(smooth_normals, smooth_t1),
        smooth_normals,
        frames.areas,
    )
    smooth_quantities = [
        form_boundary_quantities(
            *[
                project_to_face_frame(getattr(state, name), smooth_frames)
                for name in (
                    "electric_interior",
                    "magnetic_interior",
                    "electric_exterior",
                    "magnetic_exterior",
                )
            ]
        )
        for state in states
    ]
    control = extractor(
        smooth_quantities, reference_impedance=eta, structure=configuration.tensor_structure
    )
    normal_angles = np.degrees(
        np.arccos(np.clip(np.sum(frames.normals * smooth_normals, axis=1), -1, 1))
    )
    diagnostics["smooth_normal_control"] = {
        "used_for_export": False,
        "relative_residual_median": float(np.median(control.relative_residual)),
        "maximum_relative_residual": float(control.relative_residual.max()),
        "triangle_to_smooth_normal_angle_degrees_median": float(np.median(normal_angles)),
        "triangle_to_smooth_normal_angle_degrees_maximum": float(normal_angles.max()),
    }
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            for phase, name in ((1, "+"), (1j, "+j*")):
                fields = [a + phase * b for a, b in zip(tangential[i], tangential[j])]
                diagnostics["coherent_superposition_normal_power"][
                    f"{states[i].state_id}{name}{states[j].state_id}"
                ] = _power_diagnostic(_power(*fields[:2]), _power(*fields[2:]), frames.areas)
    warnings = []
    if diagnostics["maximum_relative_residual"] > 1e-8:
        warnings.append(
            "The prescribed fields are not fitted exactly on the flat-triangle tangent planes; "
            "inspect the per-face residuals before a solve."
        )
    if diagnostics["non_unique_face_count"]:
        warnings.append("Some local fits are non-unique; minimum-norm coefficients were retained.")
    run_dir.mkdir(parents=True)
    snapshot = configuration.to_dict()
    _write_json(run_dir / "configuration.json", snapshot)
    np.savez_compressed(
        run_dir / "surface_samples.npz",
        face_index=np.arange(report.face_count),
        centroids=frames.centroids,
        t1=frames.t1,
        t2=frames.t2,
        normals=frames.normals,
        areas=frames.areas,
        state_ids=np.array([state.state_id for state in states]),
        **{
            name: np.stack([getattr(state, name) for state in states])
            for name in (
                "electric_interior",
                "magnetic_interior",
                "electric_exterior",
                "magnetic_exterior",
            )
        },
        surface_maps=np.moveaxis(extraction.parameters.as_matrix(), -1, 0),
        rank=extraction.rank,
        condition_number=extraction.condition_number,
        singular_values=extraction.singular_values,
        relative_residual=extraction.relative_residual,
        state_relative_residual=extraction.state_relative_residual,
        normal_power_interior=power_in,
        normal_power_exterior=power_out,
        smooth_normals=smooth_normals,
        smooth_normal_control_relative_residual=control.relative_residual,
    )
    with (run_dir / "face_diagnostics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "face_index",
                "x",
                "y",
                "z",
                "area_m2",
                "rank",
                "condition_number",
                "relative_residual",
            ]
            + [
                f"{state.state_id}_{name}"
                for state in states
                for name in ("relative_residual", "power_interior_w_m2", "power_exterior_w_m2")
            ]
        )
        for face in range(report.face_count):
            writer.writerow(
                [
                    face,
                    *frames.centroids[face],
                    frames.areas[face],
                    extraction.rank[face],
                    extraction.condition_number[face],
                    extraction.relative_residual[face],
                ]
                + [
                    value
                    for i in range(len(states))
                    for value in (
                        extraction.state_relative_residual[i, face],
                        power_in[i, face],
                        power_out[i, face],
                    )
                ]
            )
    print("[4/4] Full-map checks and SIE binary export", flush=True)
    manifest = {
        "schema_version": 1,
        "artifact": "gstc_surface_inputs",
        "status": "prepared",
        "configuration": snapshot,
        "phasor_convention": "exp(+j omega t)",
        "basis": "face",
        "tangent_basis": "face_edge: t1, t2; outward triangle normal",
        "mesh": {
            "path": str(configuration.mesh_path.resolve()),
            "sha256": hashlib.sha256(configuration.mesh_path.read_bytes()).hexdigest(),
            **asdict(report),
        },
        "geometry_volume_m3": configuration.geometry.volume,
        "mesh_relative_volume_error": abs(report.signed_volume / configuration.geometry.volume - 1),
        "target_edge_length_m": configuration.target_edge_length,
        "matrix_dimension": 4 * report.edge_count,
        "dense_matrix_bytes": 16 * (4 * report.edge_count) ** 2,
        "reference_impedance_ohm": eta,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gmsh": gmsh.__version__,
        },
        "diagnostics": diagnostics,
        "warnings": warnings,
        "files": {
            "configuration": "configuration.json",
            "samples": "surface_samples.npz",
            "face_diagnostics": "face_diagnostics.csv",
            "diagnostics": "diagnostics.json",
        },
        "solver_executed": False,
    }
    try:
        records, map_diagnostics = prepare_tensor_records(
            extraction.parameters,
            report.face_count,
            reference_impedance=eta,
            extraction_method=configuration.extraction_method,
            tensor_structure=configuration.tensor_structure,
        )
    except ValueError as error:
        manifest["status"] = "diagnostics_only"
        manifest["tensor_export_error"] = str(error)
        _write_json(run_dir / "diagnostics.json", diagnostics)
        _write_json(run_dir / "sie_inputs.json", manifest)
        raise ValueError(f"{error}; diagnostics retained in {run_dir}") from error
    diagnostics.update(map_diagnostics)
    records.tofile(run_dir / "face_tensors.bin")
    manifest["files"]["tensors"] = "face_tensors.bin"
    manifest["tensors"] = {
        "path": "face_tensors.bin",
        "face_count": report.face_count,
        "ordering": "per face: row-major fields-to-currents map, then its inverse",
        "precision": "complex128",
        "byte_order": "little-endian",
    }
    if warnings:
        manifest["status"] = "prepared_with_warnings"
    _write_json(run_dir / "diagnostics.json", diagnostics)
    _write_json(run_dir / "sie_inputs.json", manifest)
    print(f"  rank range: {diagnostics['minimum_rank']}..{diagnostics['maximum_rank']}", flush=True)
    print(
        f"  median/max relative fit residual: {diagnostics['relative_residual_median']:.6g} / "
        f"{diagnostics['maximum_relative_residual']:.6g}",
        flush=True,
    )
    print(f"  saved: {run_dir}", flush=True)
    for warning in warnings:
        print(f"  WARNING: {warning}", flush=True)
    return run_dir / "sie_inputs.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--use-existing-mesh", action="store_true")
    args = parser.parse_args()
    config = load_workflow_configuration(args.config)
    run_dir = args.run_dir or APPLICATION_DIR / "results" / config.case_id / "prep01"
    prepare_inputs(config, run_dir=run_dir, use_existing_mesh=args.use_existing_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
