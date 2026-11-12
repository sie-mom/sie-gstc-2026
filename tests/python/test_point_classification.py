"""Mesh containment, unchanged connectivity, and the known cone-face control."""

from pathlib import Path

import numpy as np
import pytest

from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj
from sie_mom.mesh.point_classification import MeshPointClassifier

pytest.importorskip("trimesh")
pytest.importorskip("rtree")


def tetrahedron():
    return TriangleMesh(
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def test_known_regions_and_batches_preserve_mesh():
    mesh = tetrahedron()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    classifier = MeshPointClassifier(mesh)
    points = np.array([[.1, .1, .1], [.9, .9, .9], [-1, .1, .1], [.2, .3, .1]])
    np.testing.assert_array_equal(classifier.classify(points, batch_size=1), [1, 2, 2, 1])
    np.testing.assert_array_equal(classifier.classify(points, batch_size=4), [1, 2, 2, 1])
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)
    np.testing.assert_array_equal(classifier.mesh.vertices, vertices)
    np.testing.assert_array_equal(classifier.mesh.faces, faces)
    assert classifier.classify(np.empty((0, 3))).shape == (0,)


def test_rotation_translation_preserve_regions_and_invalid_winding_is_rejected():
    mesh = tetrahedron()
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1]])
    shift = np.array([10., -2, 3])
    points = np.array([[.1, .1, .1], [.9, .9, .9]])
    transformed = TriangleMesh(mesh.vertices @ rotation.T + shift, mesh.faces)
    np.testing.assert_array_equal(
        MeshPointClassifier(transformed).classify(points @ rotation.T + shift), [1, 2]
    )
    # Invalid winding must be rejected, rather than silently repaired.
    with pytest.raises(ValueError):
        MeshPointClassifier(TriangleMesh(mesh.vertices, mesh.faces[:, ::-1]))


@pytest.mark.parametrize("points", [[[0, 1]], [[np.nan, 0, 0]], [0, 1, 2]])
def test_rejects_invalid_observations(points):
    with pytest.raises(ValueError, match="finite"):
        MeshPointClassifier(tetrahedron()).classify(points)


def test_face_2218_nearby_points_use_triangle_geometry():
    path = (
        Path(__file__).resolve().parents[2]
        / "applications/gstc_surface/meshes/cone_sphere/surface.obj"
    )
    mesh = load_triangle_obj(path)
    face = mesh.vertices[mesh.faces[2218]]
    normal = np.cross(face[1] - face[0], face[2] - face[0])
    normal /= np.linalg.norm(normal)
    points = face.mean(axis=0) + np.array([-0.0001, 0.0001])[:, None] * normal
    np.testing.assert_array_equal(MeshPointClassifier(mesh).classify(points), [1, 2])


def test_rejects_open_surface():
    mesh = tetrahedron()
    with pytest.raises(ValueError):
        MeshPointClassifier(TriangleMesh(mesh.vertices, mesh.faces[:-1]))
