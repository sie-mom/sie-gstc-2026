"""Generate and validate the canonical surface mesh before tensor synthesis."""

from pathlib import Path

import numpy as np

from sie_mom.mesh.circular_ogive import CircularOgive
from sie_mom.mesh.face_frames import compute_face_frames
from sie_mom.mesh.mesh_generators import (
    CircularOgiveGenerator,
    CommonMeshOptions,
    TangentConeSphereGenerator,
)
from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj, validate_triangle_mesh
from sie_mom.mesh.tangent_cone_sphere import TangentConeSphere


def validate_mesh(mesh, geometry):
    report = validate_triangle_mesh(mesh)
    report.require_valid(expected_euler_characteristic=2)
    axial, _ = geometry.axial_radial(mesh.vertices)
    lower, upper = geometry.axial_bounds
    error = np.abs(geometry.surface_residual(mesh.vertices))
    tolerance = max(1e-9 * max(abs(lower), abs(upper)), 1e-12)
    if (
        np.max(error) > tolerance
        or np.any(axial < lower - tolerance)
        or np.any(axial > upper + tolerance)
    ):
        raise ValueError("mesh vertices do not lie on the configured surface")
    frames = compute_face_frames(mesh)
    if np.any(np.sum(frames.normals * geometry.outward_normals(frames.centroids), axis=1) <= 0):
        raise ValueError("mesh face normals do not point outward from the configured body")
    return report


def generate_mesh(configuration):
    """Create an OBJ once; existing canonical meshes are never overwritten.

    Canonical export drops unused CAD point nodes and fixes triangle winding
    before any tensor is associated with a face. Subsequent reads never repair
    or reorder that canonical mesh.
    """
    path = configuration.mesh_path
    if path.exists():
        raise FileExistsError(f"mesh already exists: {path}; use --use-existing-mesh")
    path.parent.mkdir(parents=True, exist_ok=True)
    h = configuration.target_edge_length
    generators = {
        TangentConeSphere: TangentConeSphereGenerator,
        CircularOgive: CircularOgiveGenerator,
    }
    generator = generators[type(configuration.geometry)]
    options = (
        {"tip_grading": configuration.sie.mesh.tip_grading}
        if isinstance(configuration.geometry, CircularOgive)
        else {}
    )
    generator(
        configuration.geometry,
        CommonMeshOptions(
            output_obj=str(path),
            mesh_size_min=0.8 * h,
            mesh_size_max=h,
            headless=True,
            model_name=configuration.raw["geometry"]["type"],
            write_step=False,
        ),
        **options,
    ).generate()
    generated = load_triangle_obj(path)
    used, compact_faces = np.unique(generated.faces, return_inverse=True)
    faces = compact_faces.reshape(-1, 3)
    mesh = TriangleMesh(generated.vertices[used], faces.copy())
    frames = compute_face_frames(mesh)
    reverse = (
        np.sum(frames.normals * configuration.geometry.outward_normals(frames.centroids), axis=1)
        < 0
    )
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    canonical = TriangleMesh(mesh.vertices, faces)
    validate_mesh(canonical, configuration.geometry)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Canonical outward triangular surface\n")
        for vertex in canonical.vertices:
            stream.write("v " + " ".join(f"{value:.17g}" for value in vertex) + "\n")
        for face in canonical.faces:
            stream.write("f " + " ".join(str(int(index) + 1) for index in face) + "\n")
    validate_mesh(load_triangle_obj(path), configuration.geometry)
    print(
        f"Canonical mesh: {len(canonical.faces)} faces; "
        f"{np.count_nonzero(reverse)} triangles oriented outward before synthesis",
        flush=True,
    )
    return path
