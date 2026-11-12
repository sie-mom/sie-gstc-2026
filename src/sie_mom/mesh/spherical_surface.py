from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

__all__ = ["SphericalSurfaceMesh"]


@dataclass
class SphericalSurfaceMesh:
    """Generate a quasi-uniform triangular discretization of a sphere ("icosphere").

    This class refines a regular icosahedron by recursive triangle subdivision and
    projection of new vertices onto the sphere. The resulting mesh is far more
    uniform than lat/long (UV) parameterizations that suffer from tiny polar
    triangles, thus yielding better conditioning for Method of Moments (MoM)
    RWG basis functions.

    Returned topology arrays are suitable to build RWG functions:
      - vertices: (nV,3) float64 array of Cartesian coordinates.
      - edges: (nE,2) int32 array of unique undirected edges (va < vb) referencing vertex indices.
      - faces: (nF,3) int32 array CCW oriented (outward normal) referencing vertex indices.
      - faces_edges: (nF,3) int32 array mapping each face's local edges (same order as faces rows) into the global edge array indices.

    Parameters
    ----------
    radius : float
        Sphere radius.
    center : tuple[float,float,float]
        Sphere center coordinates.
    refinement_level : int
        Number of recursive subdivisions applied to each icosahedron face.
        Each level quadruples the number of faces. Level 0 is the base icosahedron (20 faces).
    enforce_min_angle_deg : Optional[float]
        If provided, will perform an (optional) simple edge-swap local optimization pass
        attempting to raise the minimum face angle toward this threshold. This is a light-weight
        improvement and does not guarantee success (a full spherical Delaunay / CVT optimization
        would be more elaborate). Set None to skip.
    dtype : np.dtype
        Floating dtype for coordinates (default float64 for numerical robustness).
    """

    radius: float = 1.0
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    refinement_level: int = 2
    # Optional: specify desired approximate mean edge length; if set, overrides refinement_level
    target_edge_length: Optional[float] = None
    enforce_min_angle_deg: Optional[float] = None
    dtype: Any = np.float64

    # Generated outputs
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    faces_edges: Optional[np.ndarray] = None
    """Mapping from local to global edge indices, shape (nF, 3).

    Each row corresponds to a face, and its three columns contain the global
    indices of the edges forming that face. The order of edges in a row
    `[e0, e1, e2]` corresponds to the vertex pairs `(v0, v1)`, `(v1, v2)`,
    and `(v2, v0)` for that face's vertices `[v0, v1, v2]`.
    """

    # Cached stats
    _angle_stats: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Generate the mesh immediately after the object is created."""
        self.generate()

    def generate(self) -> "SphericalSurfaceMesh":
        # Determine refinement level if target_edge_length provided
        level = self.refinement_level
        if self.target_edge_length is not None:
            level = self._resolve_level_for_target_edge(self.target_edge_length)
            self.refinement_level = level  # store chosen level
        self.vertices, self.faces = self._generate_icosphere(
            level, self.radius, np.asarray(self.center, dtype=self.dtype), self.dtype
        )
        self._orient_faces_ccw_outward()
        self.edges, self.faces_edges = self._build_edges_and_mapping(self.faces)

        if self.enforce_min_angle_deg is not None:
            self._local_edge_flip_optimization(self.enforce_min_angle_deg)
            # Recompute edges / mapping after potential flips
            self.edges, self.faces_edges = self._build_edges_and_mapping(self.faces)

        self._angle_stats = self._compute_angle_stats()
        return self

    @property
    def n_vertices(self) -> int:
        """int: The number of vertices in the mesh."""
        return 0 if self.vertices is None else self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        """int: The number of faces in the mesh."""
        return 0 if self.faces is None else self.faces.shape[0]

    @property
    def n_edges(self) -> int:
        """int: The number of edges in the mesh."""
        return 0 if self.edges is None else self.edges.shape[0]

    @property
    def angle_stats(self) -> Optional[Dict[str, float]]:
        """Optional[Dict[str, float]]: Statistics about the angles in the mesh."""
        return self._angle_stats

    # ------------------------------------------------------------------
    # Core generation (icosphere)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_icosphere(level: int, radius: float, center: np.ndarray, dtype) -> Tuple[np.ndarray, np.ndarray]:
        # Canonical icosahedron (12 unique vertices) around origin before scaling.
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        V = np.array([
            (-1,  phi,  0), (1,  phi,  0), (-1, -phi,  0), (1, -phi,  0),
            (0, -1,  phi), (0, 1,  phi), (0, -1, -phi), (0, 1, -phi),
            ( phi,  0, -1), ( phi,  0, 1), (-phi,  0, -1), (-phi,  0, 1)
        ], dtype=float)
        # Normalize + scale + translate
        V /= np.linalg.norm(V, axis=1)[:, None]
        V = (V * radius + center).astype(dtype)

        F = np.array([
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
        ], dtype=np.int32)

        # Subdivide
        for _ in range(level):
            midpoint_cache: Dict[Tuple[int, int], int] = {}
            new_faces = []
            def midpoint(i: int, j: int) -> int:
                key = (i, j) if i < j else (j, i)
                if key in midpoint_cache:
                    return midpoint_cache[key]
                m = (V[i] + V[j]) * 0.5
                # Project to sphere
                m_dir = m - center
                nrm = np.linalg.norm(m_dir)
                if nrm == 0.0:
                    # Degenerate midpoint (should not happen) - perturb slightly
                    m_dir = np.array([1e-9, 0.0, 0.0])
                    nrm = 1e-9
                m_dir /= nrm
                m = center + m_dir * radius
                idx = len(V_list)
                V_list.append(m)
                midpoint_cache[key] = idx
                return idx

            V_list = V.tolist()
            for (a, b, c) in F:
                ab = midpoint(a, b)
                bc = midpoint(b, c)
                ca = midpoint(c, a)
                new_faces.extend([
                    (a, ab, ca),
                    (ab, b, bc),
                    (ca, bc, c),
                    (ab, bc, ca),
                ])
            V = np.asarray(V_list, dtype=dtype)
            F = np.asarray(new_faces, dtype=np.int32)

        # Final deduplication (numerical duplicates) just in case
        V_rounded = np.round(V, decimals=12)
        uniq_map = {}
        new_index = 0
        remap = np.empty(V.shape[0], dtype=np.int32)
        unique_vertices = []
        for i, row in enumerate(V_rounded):
            key = tuple(row.tolist())
            idx = uniq_map.get(key)
            if idx is None:
                uniq_map[key] = new_index
                remap[i] = new_index
                unique_vertices.append(V[i])
                new_index += 1
            else:
                remap[i] = idx
        V_unique = np.asarray(unique_vertices, dtype=dtype)
        F_remapped = remap[F]
        return V_unique, F_remapped

    # ------------------------------------------------------------------
    def _orient_faces_ccw_outward(self):
        V = self.vertices
        F = self.faces
        center_est = np.mean(V, axis=0)
        for i, (a, b, c) in enumerate(F):
            va, vb, vc = V[a], V[b], V[c]
            n = np.cross(vb - va, vc - va)
            centroid = (va + vb + vc) / 3.0
            outward = centroid - center_est
            # If normal points inward, swap b and c for CCW
            if np.dot(n, outward) < 0:
                F[i] = (a, c, b)
        self.faces = F

    # ------------------------------------------------------------------
    @staticmethod
    def _build_edges_and_mapping(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Builds the global edge list and the face-to-edge mapping array.

        This method iterates through all triangles (faces) and identifies unique
        edges. Each unique edge is assigned a global index. It produces two
        critical arrays for MoM matrix assembly:

        1.  `edges`: An array of shape (nE, 2) listing unique global edges,
            where each edge is represented by its two vertex indices (va, vb)
            with va < vb.
        2.  `faces_edges`: An array of shape (nF, 3) that maps each face to its
            three global edge indices. The order of edges corresponds to the
            vertex order in the `faces` array (e.g., edge 0 of a face connects
            its vertices 0 and 1).

        Returns:
            A tuple containing the `edges` and `faces_edges` numpy arrays.
        """
        edge_dict: Dict[Tuple[int, int], int] = {}
        faces_edges = np.empty((faces.shape[0], 3), dtype=np.int32)
        next_id = 0
        for fi, (a, b, c) in enumerate(faces):
            tri = ((a, b), (b, c), (c, a))
            for ei, (u, v) in enumerate(tri):
                key = (u, v) if u < v else (v, u)
                idx = edge_dict.get(key)
                if idx is None:
                    idx = next_id
                    edge_dict[key] = idx
                    next_id += 1
                faces_edges[fi, ei] = idx
        # Build edges array sorted by index
        edges = np.empty((len(edge_dict), 2), dtype=np.int32)
        for (u, v), idx in edge_dict.items():
            edges[idx] = (u, v)
        return edges, faces_edges

    # ------------------------------------------------------------------
    def _compute_angle_stats(self) -> Dict[str, float]:
        V, F = self.vertices, self.faces
        angles = []
        for a, b, c in F:
            va, vb, vc = V[a], V[b], V[c]
            ab = vb - va
            bc = vc - vb
            ca = va - vc
            # Compute angles at each vertex using cosine law
            def angle(u, v):
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)
                if nu == 0 or nv == 0:
                    return 0.0
                cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
                return math.degrees(math.acos(cosang))
            A = angle(-ab, ca)
            B = angle(-bc, ab)
            C = 180.0 - A - B
            angles.extend([A, B, C])
        arr = np.asarray(angles)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }

    # ------------------------------------------------------------------
    def _local_edge_flip_optimization(self, target_min_angle_deg: float, max_passes: int = 2):
        # Simple heuristic: For each interior edge shared by two triangles, consider flipping.
        if self.faces is None or self.vertices is None:
            return
        V = self.vertices
        for _ in range(max_passes):
            improved = False
            adjacency = self._edge_to_faces()
            for e_idx, face_ids in adjacency.items():
                if len(face_ids) != 2:
                    continue  # boundary (shouldn't happen on closed sphere)
                f1, f2 = face_ids
                tri1 = self.faces[f1]
                tri2 = self.faces[f2]
                # Edge vertices
                a, b = self.edges[e_idx]
                # Opposite vertices
                c = tri1[~np.isin(tri1, [a, b])][0]
                d = tri2[~np.isin(tri2, [a, b])][0]
                # Current angles (min of both tris)
                old_min = self._min_angle_of_tris([tri1, tri2])
                if old_min >= target_min_angle_deg:
                    continue
                # Check if edge flip is valid (avoid duplicate or degenerate)
                if c == d or c in (a, b) or d in (a, b):
                    continue
                # Candidate new triangles: (c,a,d) and (c,d,b) (orientation later re-fixed)
                new_tris = [np.array([c, a, d]), np.array([c, d, b])]
                new_min = self._min_angle_of_tris(new_tris)
                if new_min > old_min:  # Accept improvement
                    self.faces[f1] = new_tris[0]
                    self.faces[f2] = new_tris[1]
                    improved = True
            if not improved:
                break
        # Re-orient after modifications
        self._orient_faces_ccw_outward()

    def _edge_to_faces(self) -> Dict[int, list]:
        mapping: Dict[int, list] = {i: [] for i in range(self.edges.shape[0])}
        for fi, row in enumerate(self.faces_edges):
            for e in row:
                mapping[e].append(fi)
        return mapping

    def _min_angle_of_tris(self, tris):
        V = self.vertices
        min_ang = 180.0
        for tri in tris:
            a, b, c = tri
            va, vb, vc = V[a], V[b], V[c]
            # Edge vectors
            ab = vb - va
            bc = vc - vb
            ca = va - vc
            def angle(u, v):
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)
                if nu == 0 or nv == 0:
                    return 0.0
                cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
                return math.degrees(math.acos(cosang))
            A = angle(-ab, ca)
            B = angle(-bc, ab)
            C = 180.0 - A - B
            min_ang = min(min_ang, A, B, C)
        return min_ang

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a summary string for the mesh."""
        if self.vertices is None:
            return "SphericalSurfaceMesh(not generated)"
        s = self.angle_stats or {}
        return (
            f"SphericalSurfaceMesh: V={self.n_vertices} E={self.n_edges} F={self.n_faces} "
            f"angle[min/mean/max]={s.get('min'):.2f}/{s.get('mean'):.2f}/{s.get('max'):.2f}"
        )


if __name__ == "__main__":
    mesh = SphericalSurfaceMesh(radius=1.0, refinement_level=2, enforce_min_angle_deg=25.0).generate()
    print(mesh.summary())
