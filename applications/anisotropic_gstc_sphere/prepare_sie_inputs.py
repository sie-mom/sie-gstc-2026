"""Prepare spherical surface maps and export SIE inputs in mesh-face order.

Preparation validates the spherical mesh and evaluates prescribed total E/H
at the sphere radius in face-centroid directions. For each transformation,
it selects the theta/phi components and calls ``form_boundary_quantities``
to form tangential averages and equivalent sheet currents from the jumps::

    E_avg = (E_exterior + E_interior) / 2
    H_avg = (H_exterior + H_interior) / 2
    J_s = n x (H_exterior - H_interior)
    M_s = -n x (E_exterior - E_interior)

Here n points from interior to exterior. Each ``BoundaryQuantities`` object
represents one transformation in the form required by extraction::

    [J_s; M_s] = Lambda [E_avg; H_avg]

The currents are the known left-hand side, and the averages multiply the
unknown surface parameter map Lambda on the right. The configured shared
extractor fits one map to all transformations, with the selected reciprocal
complex or reciprocal passive-lossless constraints. Preparation exports that
map and its full inverse in OBJ face order, together with mesh identity,
configuration, and extraction diagnostics. The current SIE interface requires
the full map to be invertible. An optional ``parameter_evaluator(theta, phi)``
can instead supply physical ``SurfaceParameters`` at those same face directions,
without evaluating prescribed fields or performing local extraction. Both paths
share material checks, full-map inversion, and binary/metadata export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sie_mom.gstc import (
    SurfaceParameters,
    extract_reciprocal_complex_least_squares,
    extract_reciprocal_lossless_least_squares,
    form_boundary_quantities,
)
from sie_mom.gstc.tensor_io import prepare_tensor_records
from sie_mom.mesh.face_frames import compute_face_frames, project_to_face_frame
from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh

if __package__:
    from .prescribed_fields import (
        prescribed_boundary_states,
        prescribed_boundary_states_cartesian,
    )
    from .workflow import (
        DEFAULT_CONFIGURATION,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
        write_json,
    )
else:
    from prescribed_fields import (
        prescribed_boundary_states,
        prescribed_boundary_states_cartesian,
    )
    from workflow import (
        DEFAULT_CONFIGURATION,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
        write_json,
    )


def face_centroid_spherical_coordinates(mesh) -> tuple[np.ndarray, np.ndarray]:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    centroid_radii = np.linalg.norm(centroids, axis=1)
    if np.any(centroid_radii == 0.0):
        raise ValueError("Zero-radius face centroid")
    directions = centroids / centroid_radii[:, np.newaxis]
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    return theta, phi


ParameterEvaluator = Callable[[np.ndarray, np.ndarray], SurfaceParameters]


def resolve_parameter_source(configuration, parameter_evaluator, parameter_source):
    """Validate the optional evaluator and return JSON-safe source metadata."""
    if parameter_evaluator is not None and not callable(parameter_evaluator):
        raise ValueError("parameter_evaluator must be callable")
    if parameter_evaluator is None and parameter_source is not None:
        raise ValueError("parameter_source requires a parameter_evaluator")
    if parameter_evaluator is None:
        source = {
            "method": "local_extraction", "extraction_method": configuration.extraction_method,
        }
    else:
        source = dict(parameter_source or {"method": "provided_parameters"})
    if not isinstance(source.get("method"), str) or not source["method"].strip():
        raise ValueError("parameter_source.method must be a nonempty string")
    json.dumps(source, allow_nan=False)
    return source


def _extract_prescribed_parameters(theta, phi, configuration, radius, mesh=None):
    """Fit one local map to all configured prescribed field transformations."""

    if configuration.extraction_method == "reciprocal_complex":
        extractor = extract_reciprocal_complex_least_squares
    elif configuration.extraction_method == "reciprocal_lossless":
        extractor = extract_reciprocal_lossless_least_squares
    else:
        raise ValueError("unknown surface parameter extraction method")

    if configuration.sie.basis == "face":
        if mesh is None:
            raise ValueError("mesh is required to extract parameters in face basis")
        frames = compute_face_frames(mesh)
        prescribed = prescribed_boundary_states_cartesian(
            frames.centroids, configuration
        )
        quantities = [
            form_boundary_quantities(
                project_to_face_frame(state.electric_interior, frames),
                project_to_face_frame(state.magnetic_interior, frames),
                project_to_face_frame(state.electric_exterior, frames),
                project_to_face_frame(state.magnetic_exterior, frames),
            )
            for state in prescribed
        ]
    else:
        prescribed = prescribed_boundary_states(
            theta, phi, configuration, radius=radius
        )
        quantities = [
            form_boundary_quantities(
                state.electric_interior[1:],
                state.magnetic_interior[1:],
                state.electric_exterior[1:],
                state.magnetic_exterior[1:],
            )
            for state in prescribed
        ]
    extraction = extractor(
        quantities,
        reference_impedance=configuration.case.eta_exterior,
        structure=configuration.tensor_structure,
    )
    return extraction, [state.state_id for state in prescribed]


def _prepare_tensor_records(parameters, face_count, configuration):
    """Use the shared physical-map checks and SIE binary layout."""
    return prepare_tensor_records(
        parameters,
        face_count,
        reference_impedance=configuration.case.eta_exterior,
        extraction_method=configuration.extraction_method,
        tensor_structure=configuration.tensor_structure,
    )


def export_sie_inputs(
    configuration: WorkflowConfiguration,
    *,
    mesh_path: Path,
    run_dir: Path,
    parameter_evaluator: ParameterEvaluator | None = None,
    parameter_source: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Prepare maps on face directions and export Lambda and its full inverse.

    By default, extract parameters from prescribed total E/H. Alternatively,
    ``parameter_evaluator(theta, phi)`` returns physical ``SurfaceParameters``
    with shape (2, 2, face_count), in exp(+j omega t) and the existing outward
    theta/phi frame. It is called once, after mesh validation, instead of local
    extraction. ``parameter_source`` records its provenance in sie_inputs.json.
    Incident excitations do not enter preparation. Both paths share all export
    checks; existing tensor/manifest files are never overwritten.
    """
    source = resolve_parameter_source(configuration, parameter_evaluator, parameter_source)
    for name in ("face_tensors.bin", "sie_inputs.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"SIE input already exists: {run_dir / name}")
    case = configuration.case
    mesh_path = mesh_path.resolve()
    print(f"Preparing inputs for {mesh_path} at {case.frequency_hz / 1e6} MHz")
    mesh = load_triangle_obj(mesh_path)
    report = validate_triangle_mesh(mesh)
    report.require_valid(expected_euler_characteristic=2)
    theta, phi = face_centroid_spherical_coordinates(mesh)
    radii = np.linalg.norm(mesh.vertices, axis=1)
    radius = float(radii[0])
    if not np.allclose(radii, case.radius_m, rtol=1e-4, atol=0.0):
        raise ValueError("mesh vertices must lie on the configured sphere radius about the origin")
    if (np.max(radii) - np.min(radii)) / np.mean(radii) > 1e-4:
        raise ValueError("mesh vertices must lie on a common spherical radius")

    diagnostics = {}
    state_ids = []
    if parameter_evaluator is None:
        extraction, state_ids = _extract_prescribed_parameters(
            theta, phi, configuration, radius, mesh=mesh
        )
        parameters = extraction.parameters
        maximum_condition = float(np.max(extraction.condition_number))
        diagnostics.update(
            minimum_rank=int(np.min(extraction.rank)),
            maximum_rank=int(np.max(extraction.rank)),
            unique_face_count=int(np.count_nonzero(extraction.unique)),
            non_unique_face_count=int(np.count_nonzero(~extraction.unique)),
            maximum_condition_number=maximum_condition if np.isfinite(maximum_condition) else None,
            maximum_relative_residual=float(np.max(extraction.relative_residual)),
            maximum_state_relative_residual=(
                np.max(extraction.state_relative_residual, axis=1).tolist()
            ),
        )
    else:
        parameters = parameter_evaluator(theta, phi)
    records, map_diagnostics = _prepare_tensor_records(parameters, len(theta), configuration)
    diagnostics.update(map_diagnostics)
    manifest = {
        "schema_version": 1,
        "artifact": "anisotropic_gstc_sphere_inputs",
        "phasor_convention": "exp(+j omega t)",
        "tangent_basis": (
            "face_edge: t1, t2; outward normal"
            if configuration.sie.basis == "face"
            else "theta, phi; outward normal"
        ),
        "basis": configuration.sie.basis,
        "configuration": configuration.to_dict(),
        "mesh": {
            "path": str(mesh_path),
            "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "vertex_count": report.vertex_count,
            "face_count": report.face_count,
            "edge_count": report.edge_count,
        },
        "extraction_method": (
            configuration.extraction_method if parameter_evaluator is None else None
        ),
        "parameter_source": source,
        "tensor_structure": configuration.tensor_structure,
        "reference_impedance_ohm": float(case.eta_exterior),
        "frequency_hz": case.frequency_hz,
        "radius_m": float(radius),
        "prescribed_state_ids": state_ids,
        "tensors": {
            "path": "face_tensors.bin",
            "face_count": len(theta),
            "ordering": "per face: row-major fields-to-currents map, then its inverse",
            "precision": "complex128",
            "byte_order": "little-endian",
        },
        "diagnostics": diagnostics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = run_dir / "face_tensors.bin"
    records.tofile(tensor_path)
    manifest_path = run_dir / "sie_inputs.json"
    write_json(manifest_path, manifest)
    print(f"Saved {len(theta)} face tensors to {tensor_path}")
    print(f"Surface parameter source: {source['method']}")
    if parameter_evaluator is None:
        print(f"Non-unique faces: {diagnostics['non_unique_face_count']}/{len(theta)}")
        print(f"Max relative GSTC residual: {diagnostics['maximum_relative_residual']:.3e}")
        for state_id, residual in zip(state_ids, diagnostics["maximum_state_relative_residual"]):
            print(f"  {state_id}: {residual:.3e}")
    print(
        f"Max balanced inversion residual: {diagnostics['maximum_balanced_inverse_residual']:.3e}"
    )
    return tensor_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--mesh", type=Path, help="Override the configured existing mesh")
    parser.add_argument(
        "--basis",
        choices=["spherical", "face"],
        help="Override tangent basis for GSTC tensors ('spherical' or 'face')",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path)
    destination.add_argument("--run-name")
    args = parser.parse_args()
    configuration = load_workflow_configuration(args.config)
    if args.mesh is not None:
        configuration = replace(
            configuration,
            sie=replace(
                configuration.sie,
                mesh=replace(configuration.sie.mesh, path=str(args.mesh.resolve())),
            ),
        )
    if args.basis is not None:
        configuration = replace(
            configuration,
            sie=replace(
                configuration.sie,
                basis=args.basis,
            ),
        )
    run_dir = args.run_dir or default_run_directory(
        configuration.case_id, args.run_name or configuration.run_name
    )
    if __package__:
        from .sie_gstc_sphere import run_sie_workflow
    else:
        from sie_gstc_sphere import run_sie_workflow
    manifest = run_sie_workflow(
        configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True
    )
    print(f"Prepared SIE inputs: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
