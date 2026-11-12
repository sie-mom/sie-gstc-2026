from pathlib import Path

import numpy as np
import pytest

from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh


def _write_tetrahedron(path, *, reverse_one_face=False, add_orphan_vertex=False):
    nonzero_volume_face = "f 2 4 3" if reverse_one_face else "f 2 3 4"
    optional_vertex = ("v 2 2 2",) if add_orphan_vertex else ()
    path.write_text(
        "\n".join(
            (
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                *optional_vertex,
                "f 1 3 2",
                "f 1 2 4",
                nonzero_volume_face,
                "f 3 1 4",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_obj_loader_preserves_vertex_face_order_and_winding(tmp_path):
    path = tmp_path / "tetrahedron.obj"
    _write_tetrahedron(path)

    mesh = load_triangle_obj(path)

    np.testing.assert_array_equal(
        mesh.vertices,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        mesh.faces,
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )

    report = validate_triangle_mesh(mesh)
    report.require_valid(expected_euler_characteristic=2)
    assert report.edge_count == 6
    assert report.signed_volume > 0.0


def test_validation_reports_inconsistent_winding_without_repairing_it(tmp_path):
    path = tmp_path / "reversed_face.obj"
    _write_tetrahedron(path, reverse_one_face=True)
    mesh = load_triangle_obj(path)
    faces_before = mesh.faces.copy()

    report = validate_triangle_mesh(mesh)

    assert report.orientation_failures == 3
    assert report.signed_volume < 0.0
    np.testing.assert_array_equal(mesh.faces, faces_before)
    with pytest.raises(ValueError, match="inconsistently oriented"):
        report.require_valid(expected_euler_characteristic=2)


def test_validation_reports_orphan_vertices(tmp_path):
    path = tmp_path / "orphan_vertex.obj"
    _write_tetrahedron(path, add_orphan_vertex=True)

    report = validate_triangle_mesh(load_triangle_obj(path))

    assert report.orphan_vertices == 1
    assert report.euler_characteristic == 3
    with pytest.raises(ValueError, match="unreferenced vertex"):
        report.require_valid(expected_euler_characteristic=2)


def test_checked_in_gmsh_sphere_is_a_valid_canonical_mesh():
    path = (
        Path(__file__).parents[2]
        / "applications"
        / "omega_biisotropic_sphere"
        / "meshes"
        / "budhu_10ghz_eps7"
        / "sphere.obj"
    )
    report = validate_triangle_mesh(load_triangle_obj(path))

    report.require_valid(expected_euler_characteristic=2)
    assert report.face_count == 1790
    assert report.edge_count == 2685
