# Mesh Plotting and Generation Applications

**Note on Future Transition:**
The applications and procedural C++ meshing routines utilized in this folder (such as the recursive icosahedron-based spherical mesh generator) are planned to be deprecated and substituted in favor of using **Gmsh** (via Python scripts).

The primary reason for this transition is to avoid maintaining two separate directories/systems of meshing routines. Furthermore, generating high-quality, uniform meshes for complex CEM benchmark geometries (like cone-spheres or NASA almonds) is much more robust using Gmsh. Gmsh provides exact control over the maximum edge lengths (e.g., specifying $\lambda / 10$ sizing) without the rapid exponential growth of triangles seen in basic recursive subdivision.

In the future, the unified workflow for the MoM solver will be:
1. Generate the surface mesh using Python and the `gmsh` API (see the reusable generators in `src/sie_mom/mesh/mesh_generators.py`).
2. Export the mesh to a standard format (such as `.obj`).
3. Load the mesh directly into the C++ MoM solver using a generic `ObjMeshLoader` class, which will automatically build the required `edges` and `faces_edges` connectivity mappings.

Until that loader is fully implemented and tested, these procedural generation applications remain available for basic visualization and validation.
