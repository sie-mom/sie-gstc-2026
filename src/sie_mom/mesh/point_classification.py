"""Point containment against an unchanged, closed triangle mesh.

Install the optional ``sampling`` extra. Embree is an explicit optional backend;
the default uses Trimesh's existing NumPy/rtree implementation.
"""

from __future__ import annotations

import numpy as np

from .obj_mesh import TriangleMesh, validate_triangle_mesh


class MeshPointClassifier:
    """Build one spatial index and reuse it for independent point batches.

    Region 1 is inside and region 2 outside. Exact on-surface containment is
    undefined by the ray classifier; deliberate surface samples need a supplied
    side. Nearby volume samples are retained without a distance exclusion mask.
    This geometric operation is independent of field integration quadrature.
    """

    def __init__(self, mesh: TriangleMesh, *, backend: str = "numpy"):
        if backend not in ("numpy", "embree"):
            raise ValueError("classification backend must be numpy or embree")
        validate_triangle_mesh(mesh).require_valid()
        try:
            import trimesh
            if backend == "embree":
                from trimesh.ray.ray_pyembree import RayMeshIntersector
            else:
                import rtree  # noqa: F401
                from trimesh.ray.ray_triangle import RayMeshIntersector
        except ImportError as error:
            raise ImportError(
                'Install the sampling extra with: python -m pip install -e ".[sampling]"; '
                "the embree backend additionally requires embreex"
            ) from error
        self.backend = backend
        self.mesh = trimesh.Trimesh(
            vertices=np.array(mesh.vertices, copy=True),
            faces=np.array(mesh.faces, copy=True),
            process=False,
            validate=False,
            use_embree=False,
        )
        self._rays = RayMeshIntersector(self.mesh)

    def classify(self, points_xyz, *, batch_size: int = 1024):
        """Return one integer region label per finite Cartesian point."""
        points = np.asarray(points_xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("observation points must be a finite (N, 3) array")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        regions = np.empty(len(points), dtype=np.int8)
        for first in range(0, len(points), batch_size):
            inside = self._rays.contains_points(points[first:first + batch_size])
            regions[first:first + len(inside)] = np.where(inside, 1, 2)
        return regions
