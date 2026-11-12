"""Read-only artifact validation shared by SIE and independent FEM runners.

This module does not import mesh generators, CUDA coordinators or NGSolve.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

import numpy as np

from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh

from .workflow import load_workflow_configuration


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(directory, name):
    """Resolve a run-relative manifest path written on either Windows or Linux.

    Preserve the manifest and file contents. Accept either separator while
    retaining the requirement that references remain inside the run directory.
    """
    root = Path(directory).resolve()
    windows = PureWindowsPath(name)
    path = Path(str(name).replace("\\", "/"))
    resolved = (root / path).resolve()
    if windows.drive or windows.root or path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("artifact path must remain inside its field directory")
    return resolved


def load_prepared_inputs(inputs_dir):
    """Check prepared provenance and binary layout without regenerating inputs."""
    configuration = load_workflow_configuration(inputs_dir / "configuration.json")
    inputs = read_json(inputs_dir / "sie_inputs.json")
    if (
        inputs.get("schema_version") != 1
        or inputs.get("artifact") != "gstc_surface_inputs"
        or inputs.get("status") not in ("prepared", "prepared_with_warnings")
        or inputs.get("basis") != "face"
    ):
        raise ValueError("expected prepared gstc_surface inputs in the face basis")
    if inputs["configuration"] != configuration.to_dict():
        raise ValueError("prepared manifest and configuration snapshot disagree")
    mesh_path = configuration.mesh_path.resolve()
    if sha256(mesh_path) != inputs["mesh"]["sha256"]:
        raise ValueError("mesh SHA-256 differs from the mesh used for tensor preparation")
    report = validate_triangle_mesh(load_triangle_obj(mesh_path))
    report.require_valid(expected_euler_characteristic=2)
    for name in ("vertex_count", "face_count", "edge_count"):
        if getattr(report, name) != inputs["mesh"][name]:
            raise ValueError(f"prepared mesh {name} disagrees with the actual mesh")
    dimension = 4 * report.edge_count
    if inputs["matrix_dimension"] != dimension or inputs["dense_matrix_bytes"] != 16 * dimension**2:
        raise ValueError("prepared matrix dimensions disagree with four currents per RWG edge")
    tensors = inputs["tensors"]
    if (
        tensors["face_count"] != report.face_count
        or tensors["precision"] != "complex128"
        or tensors["byte_order"] != "little-endian"
        or tensors["ordering"] != "per face: row-major fields-to-currents map, then its inverse"
        or tensors["path"] != "face_tensors.bin"
        or inputs["files"]["tensors"] != "face_tensors.bin"
    ):
        raise ValueError("unsupported prepared tensor layout")
    tensor_path = inputs_dir / "face_tensors.bin"
    if tensor_path.stat().st_size != report.face_count * 32 * 16:
        raise ValueError("tensor binary size disagrees with the prepared face count")
    if not np.isfinite(np.fromfile(tensor_path, dtype="<c16")).all():
        raise ValueError("tensor binary contains non-finite coefficients")
    files = {"sie_inputs.json", "configuration.json", "face_tensors.bin"}
    for name in inputs["files"].values():
        if Path(name).name != name or not (inputs_dir / name).is_file():
            raise ValueError(f"invalid or missing prepared artifact: {name}")
        files.add(name)
    return configuration, inputs, asdict(report), files


def read_observations(path):
    """Read the explicit, ordered coordinate/region contract used by C++."""
    data = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    if data.dtype.names != ("point_index", "x", "y", "z", "region") or not len(data):
        raise ValueError("expected nonempty point_index,x,y,z,region observations")
    if not all(np.isfinite(data[name]).all() for name in data.dtype.names):
        raise ValueError("observation data must be finite")
    if not np.array_equal(data["point_index"], np.arange(len(data))):
        raise ValueError("observation indices must be ordered and contiguous from zero")
    if not np.isin(data["region"], (1, 2)).all():
        raise ValueError("observation regions must be 1 or 2")
    return np.column_stack([data[name] for name in "xyz"]), data["region"].astype(np.int8)
