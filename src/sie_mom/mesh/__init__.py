"""Mesh loading, validation, generation, and coordinate frame utilities."""

from .face_frames import (
    FaceFrames,
    compute_face_frames,
    lift_2x2_to_3x3,
    project_3x3_to_2x2,
    project_to_face_frame,
)
from .obj_mesh import (
    MeshValidationReport,
    TriangleMesh,
    load_triangle_obj,
    validate_triangle_mesh,
)
from .spherical_surface import SphericalSurfaceMesh

__all__ = [
    "FaceFrames",
    "MeshValidationReport",
    "SphericalSurfaceMesh",
    "TriangleMesh",
    "compute_face_frames",
    "lift_2x2_to_3x3",
    "load_triangle_obj",
    "project_3x3_to_2x2",
    "project_to_face_frame",
    "validate_triangle_mesh",
]

