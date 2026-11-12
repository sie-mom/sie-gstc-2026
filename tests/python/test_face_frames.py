"""Tests for intrinsic face tangent frames, Cartesian field evaluators, and tensor lifting."""

import numpy as np
import pytest

from sie_mom.excitation import (
    evaluate_dipole_fields,
    evaluate_dipole_fields_cartesian,
    evaluate_plane_wave_cartesian,
    evaluate_plane_wave_fields,
)
from sie_mom.mesh.face_frames import (
    FaceFrames,
    compute_face_frames,
    lift_2x2_to_3x3,
    project_3x3_to_2x2,
    project_to_face_frame,
)
from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj


@pytest.fixture
def sample_mesh() -> TriangleMesh:
    """A simple closed tetrahedron with outward CCW faces."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],  # z = 0 face, normal is [0, 0, -1]
            [0, 1, 3],  # y = 0 face, normal is [0, -1, 0]
            [1, 2, 3],  # inclined face, normal is [1, 1, 1] / sqrt(3)
            [2, 0, 3],  # x = 0 face, normal is [-1, 0, 0]
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices=vertices, faces=faces)


def test_face_frames_orthonormality_and_handedness(sample_mesh):
    frames = compute_face_frames(sample_mesh)

    assert frames.face_count == 4
    assert frames.centroids.shape == (4, 3)
    assert frames.t1.shape == (4, 3)
    assert frames.t2.shape == (4, 3)
    assert frames.normals.shape == (4, 3)
    assert frames.areas.shape == (4,)

    # Unit lengths
    np.testing.assert_allclose(np.linalg.norm(frames.t1, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(frames.t2, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(frames.normals, axis=1), 1.0, atol=1e-14)

    # Orthogonality
    np.testing.assert_allclose(np.sum(frames.t1 * frames.t2, axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(frames.t1 * frames.normals, axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(frames.t2 * frames.normals, axis=1), 0.0, atol=1e-14)

    # Right-handedness: t1 x t2 == normal
    cross_t1_t2 = np.cross(frames.t1, frames.t2)
    np.testing.assert_allclose(cross_t1_t2, frames.normals, atol=1e-14)

    # Centroid definition: (v0 + v1 + v2) / 3
    for f in range(4):
        v = sample_mesh.vertices[sample_mesh.faces[f]]
        expected_centroid = v.mean(axis=0)
        np.testing.assert_allclose(frames.centroids[f], expected_centroid, atol=1e-14)


def test_face_frames_first_face_algebraic_exactness(sample_mesh):
    """Face 0 has v0=(0,0,0), v1=(0,1,0), v2=(1,0,0)."""
    frames = compute_face_frames(sample_mesh)

    # e1 = v1 - v0 = [0, 1, 0] -> t1 = [0, 1, 0]
    np.testing.assert_allclose(frames.t1[0], [0.0, 1.0, 0.0], atol=1e-14)
    # e2 = v2 - v0 = [1, 0, 0]
    # e1 x e2 = [0, 1, 0] x [1, 0, 0] = [0, 0, -1] -> normal = [0, 0, -1]
    np.testing.assert_allclose(frames.normals[0], [0.0, 0.0, -1.0], atol=1e-14)
    # t2 = normal x t1 = [0, 0, -1] x [0, 1, 0] = [1, 0, 0]
    np.testing.assert_allclose(frames.t2[0], [1.0, 0.0, 0.0], atol=1e-14)
    # Area = 0.5 * base * height = 0.5 * 1 * 1 = 0.5
    assert np.isclose(frames.areas[0], 0.5, atol=1e-14)


def test_face_frames_degenerate_raises():
    degenerate_vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    degenerate_faces = np.array([[0, 1, 2]])  # Collinear vertices -> zero area
    mesh = TriangleMesh(vertices=degenerate_vertices, faces=degenerate_faces)
    with pytest.raises(ValueError, match="zero area"):
        compute_face_frames(mesh)


def test_cartesian_plane_wave_vs_spherical():
    k = 2.0 * np.pi * 1e9 / 3e8
    eta = 376.730313668
    radius = np.array([1.0, 1.5, 2.0])
    theta = np.array([0.2, 0.8, 1.5])
    phi = np.array([0.1, 1.2, 2.4])

    # Spherical evaluator
    E_sph, H_sph = evaluate_plane_wave_fields(k, radius, theta, phi, eta)

    # Cartesian points
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack([x, y, z], axis=-1)

    E_cart, H_cart = evaluate_plane_wave_cartesian(k, points, eta)

    # Project Cartesian fields to spherical (r, theta, phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    def to_sph(cart):
        fx, fy, fz = cart[0], cart[1], cart[2]
        return np.stack(
            [
                fx * sin_theta * cos_phi + fy * sin_theta * sin_phi + fz * cos_theta,
                fx * cos_theta * cos_phi + fy * cos_theta * sin_phi - fz * sin_theta,
                -fx * sin_phi + fy * cos_phi,
            ]
        )

    np.testing.assert_allclose(to_sph(E_cart), E_sph, atol=1e-15)
    np.testing.assert_allclose(to_sph(H_cart), H_sph, atol=1e-15)


def test_cartesian_dipole_vs_spherical():
    k = 2.0 * np.pi * 1e9 / 3e8
    eta = 376.730313668
    radius = np.array([1.2, 1.8, 2.5])
    theta = np.array([0.3, 1.1, 2.2])
    phi = np.array([0.5, 1.7, 3.0])

    for dipole_type in ("electric", "magnetic"):
        E_sph, H_sph = evaluate_dipole_fields(
            k, radius, theta, phi, eta, dipole_type=dipole_type, effective_moment=(1.0, 2.0, 3.0)
        )

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        points = np.stack([x, y, z], axis=-1)

        E_cart, H_cart = evaluate_dipole_fields_cartesian(
            k, points, eta, dipole_type=dipole_type, effective_moment=(1.0, 2.0, 3.0)
        )

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        def to_sph(cart):
            fx, fy, fz = cart[0], cart[1], cart[2]
            return np.stack(
                [
                    fx * sin_theta * cos_phi + fy * sin_theta * sin_phi + fz * cos_theta,
                    fx * cos_theta * cos_phi + fy * cos_theta * sin_phi - fz * sin_theta,
                    -fx * sin_phi + fy * cos_phi,
                ]
            )

        np.testing.assert_allclose(to_sph(E_cart), E_sph, atol=1e-15)
        np.testing.assert_allclose(to_sph(H_cart), H_sph, atol=1e-15)


def test_project_to_face_frame_and_lifting(sample_mesh):
    frames = compute_face_frames(sample_mesh)
    nf = frames.face_count

    # Test lifting an arbitrary 2x2 complex tensor
    t2x2 = np.array(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [-2.0 + 0.5j, 4.0 + 1.5j],
        ],
        dtype=complex,
    )
    t2x2_all = np.repeat(t2x2[:, :, np.newaxis], nf, axis=2)

    t3x3 = lift_2x2_to_3x3(t2x2_all, frames)
    assert t3x3.shape == (3, 3, nf)

    # Recover 2x2 via projection
    recovered = project_3x3_to_2x2(t3x3, frames)
    np.testing.assert_allclose(recovered, t2x2_all, atol=1e-14)

    # Verify normal vector is annihilated: T_3x3 @ n == 0
    for f in range(nf):
        n = frames.normals[f]
        T = t3x3[:, :, f]
        np.testing.assert_allclose(T @ n, [0.0, 0.0, 0.0], atol=1e-14)

    # Verify action on in-plane vector: T_3x3 @ (a*t1 + b*t2) == (T_2x2 @ [a, b]) in (t1, t2)
    for f in range(nf):
        a, b = 2.5, -1.3
        v_tangent_3d = a * frames.t1[f] + b * frames.t2[f]
        w_3d = t3x3[:, :, f] @ v_tangent_3d
        w_expected = (t2x2 @ np.array([a, b]))[0] * frames.t1[f] + (t2x2 @ np.array([a, b]))[1] * frames.t2[f]
        np.testing.assert_allclose(w_3d, w_expected, atol=1e-14)


def test_face_basis_gauge_invariance_under_vertex_permutation(sample_mesh):
    """Permuting vertices (v0, v1, v2) -> (v1, v2, v0) produces the exact same

    3D Cartesian tensor when the 2x2 block is rotated by Q = [t1.t1' t1.t2'; t2.t1' t2.t2'].
    """
    frames1 = compute_face_frames(sample_mesh)

    # Permuted mesh: cyclic shift of face vertex indices
    perm_faces = np.column_stack([sample_mesh.faces[:, 1], sample_mesh.faces[:, 2], sample_mesh.faces[:, 0]])
    perm_mesh = TriangleMesh(vertices=sample_mesh.vertices, faces=perm_faces)
    frames2 = compute_face_frames(perm_mesh)

    # Normals must be identical (same CCW plane)
    np.testing.assert_allclose(frames1.normals, frames2.normals, atol=1e-14)

    # Arbitrary 2x2 tensor
    T1 = np.array([[2.0 + 1.0j, -0.5j], [1.5j, 3.0 - 2.0j]], dtype=complex)
    T1_all = np.repeat(T1[:, :, np.newaxis], frames1.face_count, axis=2)
    T3x3_from_1 = lift_2x2_to_3x3(T1_all, frames1)

    # For each face, compute 2D rotation matrix Q = R2^T R1
    T2_all = np.zeros_like(T1_all)
    for f in range(frames1.face_count):
        t1_old, t2_old = frames1.t1[f], frames1.t2[f]
        t1_new, t2_new = frames2.t1[f], frames2.t2[f]
        Q = np.array(
            [
                [np.dot(t1_new, t1_old), np.dot(t1_new, t2_old)],
                [np.dot(t2_new, t1_old), np.dot(t2_new, t2_old)],
            ],
            dtype=float,
        )
        # Transform 2x2 tensor: T' = Q T Q^T
        T2_all[:, :, f] = Q @ T1 @ Q.T

    T3x3_from_2 = lift_2x2_to_3x3(T2_all, frames2)

    # Both 3x3 Cartesian tensors must match to machine precision
    np.testing.assert_allclose(T3x3_from_1, T3x3_from_2, atol=1e-14)

