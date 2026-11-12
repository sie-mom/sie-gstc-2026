# Mesh Utilities

This directory contains utilities to construct surface meshes for Method of Moments (MoM) RWG basis functions.

## ⚠️ Transition to Gmsh Workflow

**Note:** The procedural C++ meshing routines and their Python equivalents described below (such as the recursive icosahedron-based spherical mesh generator) are planned to be deprecated in favor of using **Gmsh** (via Python scripts).

Generating high-quality, uniform meshes for complex Computational Electromagnetics (CEM) benchmark geometries (like cone-spheres or NASA almonds) is much more robust using Gmsh. Gmsh provides exact control over the maximum edge lengths (e.g., specifying `\lambda / 10` sizing) without the rapid exponential growth of triangles seen in basic recursive subdivision.

The new unified workflow will be:
1. Generate the surface mesh using Python and the `gmsh` API (see `generate_sphere_gmsh.py` and `generate_conesphere_gmsh.py`).
2. Export the mesh to a standard format (`.obj`).
3. Load the mesh directly into the C++ MoM solver using a generic `ObjMeshLoader` class, which automatically builds the required `edges` and `faces_edges` connectivity mappings.

---

## SphericalSurfaceMesh

Generates a quasi-uniform spherical triangulation ("icosphere") via recursive subdivision of the canonical icosahedron:

```python
from sie_mom.mesh import SphericalSurfaceMesh
mesh = SphericalSurfaceMesh(radius=1.0, refinement_level=3, enforce_min_angle_deg=50).generate()
print(mesh.summary())
# Access arrays:
V = mesh.vertices       # (nV,3)
E = mesh.edges          # (nE,2) with va < vb
F = mesh.faces          # (nF,3) CCW oriented
FE = mesh.faces_edges   # (nF,3) edge indices per face
```

### Parameters
- `radius`: sphere radius.
- `center`: 3-tuple center.
- `refinement_level`: number of recursive subdivisions (each level *4 faces).
- `enforce_min_angle_deg`: optional heuristic local edge flips to raise minimal angle.

### Angle Quality
Typical icosphere triangles have angles clustered around 60°. The simple edge-flip pass can marginally improve the worst cases but is intentionally conservative to remain fast.

### Future Work Ideas
- Adaptive refinement based on user-supplied sizing field.
- Spherical Lloyd (CVT) relaxation for even more uniformity.
- Export to common formats (STL, OBJ, Gmsh msh2) or integration with pygmsh.
- Direct construction of RWG basis: oriented edge list + plus/minus triangle association.

---
Maintained for experimental MoM development.
