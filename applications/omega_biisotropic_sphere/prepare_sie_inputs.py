#!/usr/bin/env python3
"""Prepare the canonical mesh and per-face tensors for the sphere SIE."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    from .dipole_impedance_matching import (
        apply_cap_perturbation,
        extract_case_parameters,
        prescribed_dipole_fields,
    )
    from .generate_sphere_mesh import generate_mesh, validate_sie_mesh
    from .workflow import (
        APPLICATION_DIR,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
    )
except ImportError:  # Support direct execution from the repository root.
    from dipole_impedance_matching import (  # type: ignore[no-redef]
        apply_cap_perturbation,
        extract_case_parameters,
        prescribed_dipole_fields,
    )
    from generate_sphere_mesh import (  # type: ignore[no-redef]
        generate_mesh,
        validate_sie_mesh,
    )
    from workflow import (  # type: ignore[no-redef]
        APPLICATION_DIR,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
    )

from sie_mom.gstc.omega_biisotropic import (
    OmegaParameters,
    normal_power_density,
    omega_tensor_blocks,
)
from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj

DEFAULT_CONFIGURATION = APPLICATION_DIR / "configurations" / "budhu_10ghz_eps7.json"


def _application_path(path: Path) -> Path:
    return path if path.is_absolute() else APPLICATION_DIR / path


def face_centroid_spherical_coordinates(
    mesh: TriangleMesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(theta, phi)`` in OBJ face order from centroid directions."""

    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    centroid_radii = np.linalg.norm(centroids, axis=1)
    if np.any(centroid_radii == 0.0):
        indices = np.flatnonzero(centroid_radii == 0.0)
        raise ValueError(
            "cannot define a spherical basis at zero-radius face centroid(s): "
            + ", ".join(map(str, indices[:10]))
        )
    directions = centroids / centroid_radii[:, np.newaxis]
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    return theta, phi


def assemble_omega_maps(
    parameters: OmegaParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble each local fields-to-currents map and its full inverse."""

    B, X, chi = np.broadcast_arrays(parameters.B, parameters.X, parameters.chi)
    if B.ndim != 1:
        raise ValueError("per-face omega parameter arrays must be one-dimensional")

    blocks = omega_tensor_blocks(parameters)
    forward = np.zeros((B.size, 4, 4), dtype=np.complex128)
    forward[:, :2, :2] = np.moveaxis(blocks["Y"], -1, 0)
    forward[:, :2, 2:] = np.moveaxis(blocks["Chi"], -1, 0)
    forward[:, 2:, :2] = np.moveaxis(blocks["Gamma"], -1, 0)
    forward[:, 2:, 2:] = np.moveaxis(blocks["Z"], -1, 0)
    inverse = np.linalg.inv(forward)
    return forward, inverse


def _relative_spread(values: np.ndarray) -> float:
    mean = np.mean(values)
    scale = max(float(np.abs(mean)), np.finfo(float).tiny)
    return float(np.max(np.abs(values - mean)) / scale)


def _mesh_manifest_path(mesh_path: Path) -> str:
    try:
        return str(mesh_path.resolve().relative_to(APPLICATION_DIR.resolve()))
    except ValueError:
        return str(mesh_path.resolve())


def export_sie_inputs(
    configuration: WorkflowConfiguration,
    *,
    mesh_path: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    """Validate one OBJ and export tensors in its unchanged face ordering."""

    mesh_path = mesh_path.resolve()
    report = validate_sie_mesh(mesh_path)
    mesh = load_triangle_obj(mesh_path)
    theta, phi = face_centroid_spherical_coordinates(mesh)

    extraction = extract_case_parameters(
        theta,
        phi,
        configuration.case,
        method=configuration.extraction_method,
        svwf_convention=False,
    )
    if not np.all(extraction.valid):
        invalid = np.flatnonzero(~extraction.valid)
        raise RuntimeError(
            "omega extraction is degenerate at OBJ face index/indices: "
            + ", ".join(map(str, invalid[:10]))
        )

    parameters, cap_mask = apply_cap_perturbation(
        extraction.parameters, theta, phi, configuration.cap_perturbation
    )
    cap = configuration.cap_perturbation
    if cap is not None:
        if cap.parameter_scale != 1.0 and not np.any(cap_mask):
            raise ValueError("no SIE face centroids sample the cap; refine the mesh")
        print(f"  cap faces: {np.count_nonzero(cap_mask)} / {theta.size}", flush=True)
    # Rebuild the inverse from the perturbed forward map, not from the baseline.
    forward, inverse = assemble_omega_maps(parameters)
    records = np.concatenate((forward.reshape((-1, 16)), inverse.reshape((-1, 16))), axis=1).astype(
        np.dtype("<c8")
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = run_dir / "face_tensors.bin"
    temporary_tensor_path = tensor_path.with_suffix(".bin.tmp")
    records.tofile(temporary_tensor_path)
    temporary_tensor_path.replace(tensor_path)

    identity = np.eye(4, dtype=np.complex128)
    inverse_residual = np.linalg.norm(forward @ inverse - identity, axis=(1, 2))
    exported_forward = records[:, :16].reshape((-1, 4, 4))
    exported_inverse = records[:, 16:].reshape((-1, 4, 4))
    exported_inverse_residual = np.linalg.norm(
        exported_forward @ exported_inverse - identity,
        axis=(1, 2),
    )

    fields = prescribed_dipole_fields(theta, phi, configuration.case)
    power_interior = normal_power_density(fields[0][1:], fields[1][1:])
    power_exterior = normal_power_density(fields[2][1:], fields[3][1:])
    power_scale = max(float(np.linalg.norm(power_interior)), np.finfo(float).tiny)
    power_error = float(np.linalg.norm(power_exterior - power_interior) / power_scale)

    manifest = {
        "schema_version": 1,
        "artifact": "omega_biisotropic_sie_inputs",
        "case_id": configuration.case_id,
        "phasor_convention": "exp(+j omega t)",
        "configuration": configuration.to_dict(),
        "mesh": {
            "path": _mesh_manifest_path(mesh_path),
            "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            "face_order": "unchanged OBJ order",
            "validation": asdict(report),
        },
        "sampling": {
            "angular_location": "planar face-centroid direction",
            "field_evaluation_radius_m": configuration.case.radius,
            "local_basis": "spherical (theta_hat, phi_hat)",
        },
        "tensor_data": {
            "path": tensor_path.name,
            "face_count": report.face_count,
            "complex_values_per_face": 32,
            "shape": [report.face_count, 32],
            "precision": "complex64",
            "byte_order": "little-endian",
            "complex_encoding": "interleaved real-imaginary float32",
            "layout": (
                "per face: 4x4 fields-to-currents map row-major, then its "
                "full 4x4 inverse row-major"
            ),
        },
        "parameters": {
            "B_mean_S": float(np.mean(parameters.B)),
            "X_mean_ohm": float(np.mean(parameters.X)),
            "chi_mean": float(np.mean(parameters.chi)),
            "B_relative_spread": _relative_spread(parameters.B),
            "X_relative_spread": _relative_spread(parameters.X),
            "chi_relative_spread": _relative_spread(parameters.chi),
        },
        "diagnostics": {
            "maximum_extraction_relative_residual": float(np.max(extraction.relative_residual)),
            "normal_power_relative_error": power_error,
            "maximum_map_condition_number": float(np.max(np.linalg.cond(forward))),
            "maximum_inverse_residual_complex128": float(np.max(inverse_residual)),
            "maximum_exported_inverse_residual_complex64": float(np.max(exported_inverse_residual)),
        },
    }
    if cap is not None:
        manifest["cap_perturbation"] = {
            **asdict(cap),
            "selected_face_count": int(np.count_nonzero(cap_mask)),
            "selected_face_indices": np.flatnonzero(cap_mask).tolist(),
        }
        manifest["diagnostics"]["extraction_and_power_scope"] = (
            "unperturbed prescribed transformation"
        )
    manifest_path = run_dir / "sie_inputs.json"
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_manifest_path.replace(manifest_path)
    return tensor_path, manifest_path


def prepare_sie_inputs(
    configuration: WorkflowConfiguration,
    *,
    run_dir: Path,
    use_existing_mesh: bool = False,
) -> tuple[Path, Path]:
    """Generate the configured sphere when needed, then export SIE tensors."""

    configured_mesh_path = _application_path(Path(configuration.sie.mesh.path))
    mesh_path = configured_mesh_path if use_existing_mesh else generate_mesh(configuration)
    if use_existing_mesh and not mesh_path.is_file():
        raise FileNotFoundError(f"configured SIE mesh does not exist: {mesh_path}")
    return export_sie_inputs(
        configuration,
        mesh_path=mesh_path,
        run_dir=run_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIGURATION,
        help="Shared application configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory (default: results/<case>/sie/<method>)",
    )
    parser.add_argument(
        "--use-existing-mesh",
        action="store_true",
        help="Validate and use the configured OBJ without regenerating it",
    )
    arguments = parser.parse_args()

    configuration = load_workflow_configuration(arguments.config)
    run_dir = arguments.run_dir or default_run_directory(
        configuration.case_id,
        "sie",
        configuration.extraction_method,
    )
    tensor_path, manifest_path = prepare_sie_inputs(
        configuration,
        run_dir=run_dir,
        use_existing_mesh=arguments.use_existing_mesh,
    )
    print("Prepared omega-biisotropic SIE inputs")
    print(f"  tensors: {tensor_path}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
