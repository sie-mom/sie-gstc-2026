"""Non-mutating OBJ loading and validation for triangular surface meshes.

The loader preserves vertex order, face order, and face winding.  Validation
reports problems but never repairs the mesh, which is important when another
artifact (such as per-face GSTC tensors) uses the OBJ face index as its key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


@dataclass(frozen=True)
class TriangleMesh:
    """Vertices and triangular faces in their original OBJ ordering."""

    vertices: FloatArray
    faces: IntegerArray


@dataclass(frozen=True)
class MeshValidationReport:
    """Topological and geometric checks for a triangular surface mesh."""

    vertex_count: int
    face_count: int
    edge_count: int
    euler_characteristic: int
    boundary_edges: int
    nonmanifold_edges: int
    connected_components: int
    orientation_failures: int
    duplicate_vertex_pairs: int
    orphan_vertices: int
    duplicate_faces: int
    degenerate_faces: int
    signed_volume: float
    volume_tolerance: float

    def problems(self, *, expected_euler_characteristic: int | None = None) -> list[str]:
        """Return descriptions of conditions that prevent strict SIE use."""

        problems: list[str] = []
        if self.vertex_count == 0:
            problems.append("the mesh has no vertices")
        if self.face_count == 0:
            problems.append("the mesh has no faces")
        if self.duplicate_vertex_pairs:
            problems.append(f"{self.duplicate_vertex_pairs} near-duplicate vertex pair(s)")
        if self.orphan_vertices:
            problems.append(f"{self.orphan_vertices} unreferenced vertex/vertices")
        if self.degenerate_faces:
            problems.append(f"{self.degenerate_faces} degenerate face(s)")
        if self.duplicate_faces:
            problems.append(f"{self.duplicate_faces} duplicate face(s)")
        if self.boundary_edges:
            problems.append(f"{self.boundary_edges} boundary edge(s)")
        if self.nonmanifold_edges:
            problems.append(f"{self.nonmanifold_edges} non-manifold edge(s)")
        if self.connected_components != 1:
            problems.append(f"{self.connected_components} connected component(s), expected 1")
        if self.orientation_failures:
            problems.append(f"{self.orientation_failures} inconsistently oriented interior edge(s)")
        if self.signed_volume <= self.volume_tolerance:
            problems.append(
                "the signed volume is not positive; the surface is not "
                "consistently outward-oriented"
            )
        if (
            expected_euler_characteristic is not None
            and self.euler_characteristic != expected_euler_characteristic
        ):
            problems.append(
                "Euler characteristic "
                f"{self.euler_characteristic}, expected "
                f"{expected_euler_characteristic}"
            )
        return problems

    def require_valid(self, *, expected_euler_characteristic: int | None = None) -> None:
        """Raise ``ValueError`` when the mesh is not a valid closed surface."""

        problems = self.problems(expected_euler_characteristic=expected_euler_characteristic)
        if problems:
            raise ValueError("invalid triangular surface mesh: " + "; ".join(problems))


def _parse_obj_index(token: str, vertex_count: int, path: Path, line_number: int) -> int:
    vertex_token = token.split("/", maxsplit=1)[0]
    if not vertex_token:
        raise ValueError(f"{path}:{line_number}: face has an empty vertex index")
    raw_index = int(vertex_token)
    if raw_index == 0:
        raise ValueError(f"{path}:{line_number}: OBJ vertex index zero is invalid")
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if index < 0 or index >= vertex_count:
        raise ValueError(f"{path}:{line_number}: vertex index {raw_index} is out of range")
    return index


def load_triangle_obj(path: str | Path) -> TriangleMesh:
    """Load a triangular OBJ without modifying its indices or winding."""

    obj_path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    with obj_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"{obj_path}:{line_number}: vertex requires three coordinates")
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(
                        f"{obj_path}:{line_number}: only triangular faces are supported"
                    )
                faces.append(
                    tuple(
                        _parse_obj_index(token, len(vertices), obj_path, line_number)
                        for token in fields[1:4]
                    )
                )

    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64).reshape((-1, 3)),
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
    )


def validate_triangle_mesh(
    mesh: TriangleMesh,
    *,
    relative_tolerance: float = 1e-10,
) -> MeshValidationReport:
    """Validate a closed triangular mesh while preserving the input unchanged."""

    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be nonnegative")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape (n_vertices, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("faces must have shape (n_faces, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite coordinates")
    if faces.size and (np.min(faces) < 0 or np.max(faces) >= len(vertices)):
        raise ValueError("faces contain an out-of-range vertex index")

    if len(vertices):
        scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        if scale == 0.0:
            scale = 1.0
        vertex_tolerance = relative_tolerance * scale
        duplicate_vertex_pairs = len(cKDTree(vertices).query_pairs(vertex_tolerance))
    else:
        scale = 1.0
        duplicate_vertex_pairs = 0

    area_tolerance = relative_tolerance * scale**2
    volume_tolerance = relative_tolerance * scale**3
    degenerate_faces = 0
    signed_volume = 0.0
    if len(faces):
        triangles = vertices[faces]
        doubled_areas = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        repeated_indices = (
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        )
        degenerate_faces = int(
            np.count_nonzero(repeated_indices | (doubled_areas <= area_tolerance))
        )
        signed_volume = float(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            ).sum()
            / 6.0
        )

    canonical_faces = [tuple(sorted(map(int, face))) for face in faces]
    duplicate_faces = len(canonical_faces) - len(set(canonical_faces))
    referenced_vertices = set(map(int, faces.ravel()))
    orphan_vertices = len(vertices) - len(referenced_vertices)

    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        cycle = (int(face[0]), int(face[1]), int(face[2]))
        for start, end in zip(cycle, cycle[1:] + cycle[:1]):
            edge = (min(start, end), max(start, end))
            direction = 1 if (start, end) == edge else -1
            edge_incidence.setdefault(edge, []).append((face_index, direction))

    boundary_edges = sum(len(incidences) == 1 for incidences in edge_incidence.values())
    nonmanifold_edges = sum(len(incidences) > 2 for incidences in edge_incidence.values())
    orientation_failures = sum(
        len(incidences) == 2 and incidences[0][1] == incidences[1][1]
        for incidences in edge_incidence.values()
    )

    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for incidences in edge_incidence.values():
        incident_faces = [face_index for face_index, _ in incidences]
        for face_index in incident_faces:
            adjacency[face_index].update(
                neighbor for neighbor in incident_faces if neighbor != face_index
            )

    connected_components = 0
    unvisited = set(range(len(faces)))
    while unvisited:
        connected_components += 1
        pending = [unvisited.pop()]
        while pending:
            face_index = pending.pop()
            new_neighbors = adjacency[face_index] & unvisited
            unvisited.difference_update(new_neighbors)
            pending.extend(new_neighbors)

    edge_count = len(edge_incidence)
    return MeshValidationReport(
        vertex_count=len(vertices),
        face_count=len(faces),
        edge_count=edge_count,
        euler_characteristic=len(vertices) - edge_count + len(faces),
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
        connected_components=connected_components,
        orientation_failures=int(orientation_failures),
        duplicate_vertex_pairs=duplicate_vertex_pairs,
        orphan_vertices=orphan_vertices,
        duplicate_faces=duplicate_faces,
        degenerate_faces=degenerate_faces,
        signed_volume=signed_volume,
        volume_tolerance=volume_tolerance,
    )
