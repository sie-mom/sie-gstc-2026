"""Scalar face colors retain values/order independently of geometry and GSTCs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

from sie_mom.mesh.obj_mesh import TriangleMesh
from sie_mom.postprocessing.mesh_plotting import face_value_limits, plot_mesh_face_values


def test_color_limits_handle_signed_sparse_zero_and_nonnegative_data():
    assert face_value_limits([-8, 1, 2], percentile=100) == (-8, 8)
    assert face_value_limits(np.zeros(5)) == (-1, 1)
    assert face_value_limits([0, 0, 0, 4], percentile=20) == (-4, 4)
    assert face_value_limits([0, 1, 4], percentile=50, symmetric=False) == (0, 1)
    for values in ([], [np.nan], [np.inf], [1j]):
        with pytest.raises(ValueError, match="finite and real"):
            face_value_limits(values)
    with pytest.raises(ValueError, match="nonnegative"):
        face_value_limits([-1, 3], symmetric=False)
    for percentile in (0, 101, np.nan):
        with pytest.raises(ValueError, match="percentile"):
            face_value_limits([1], percentile=percentile)


def test_renderer_retains_face_association_and_equal_spatial_scale():
    # A translated, rotated planar mesh exercises geometry-independent rendering.
    vertices = np.array([[3.0, 7, 2], [3, 9, 2], [3, 9, 3], [3, 7, 3]])
    faces = np.array([[0, 1, 2], [2, 3, 0]])
    values = np.array([-5.0, 2.0])
    before = vertices.copy(), faces.copy(), values.copy()
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        norm = Normalize(-3, 3)
        image = plot_mesh_face_values(
            TriangleMesh(vertices, faces), values, ax=ax, norm=norm, show_edges=True, view=(20, 15)
        )
        fig.canvas.draw()
        np.testing.assert_array_equal(image.get_array(), values)
        assert image.norm is norm
        spans = np.array([np.diff(getattr(ax, f"get_{c}lim")())[0] for c in "xyz"])
        np.testing.assert_allclose(
            spans / spans.max(), ax.get_box_aspect() / ax.get_box_aspect().max()
        )
        for actual, original in zip((vertices, faces, values), before):
            np.testing.assert_array_equal(actual, original)
        with pytest.raises(ValueError, match="one scalar"):
            plot_mesh_face_values(TriangleMesh(vertices, faces), [1], ax=ax)
        with pytest.raises(ValueError, match="integer faces"):
            plot_mesh_face_values(TriangleMesh(vertices, faces + 20), values, ax=ax)
    finally:
        plt.close(fig)


def test_equal_values_have_equal_colors_despite_different_normals_and_views():
    # Perpendicular triangles would receive different brightness under lighting.
    mesh = TriangleMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2], [0, 3, 1]])
    )
    fig = plt.figure()
    try:
        ax = fig.add_subplot(projection="3d")
        image = plot_mesh_face_values(mesh, [0.5, 0.5], ax=ax, norm=Normalize(-1, 1))
        expected = image.cmap(image.norm([0.5, 0.5]))
        for elevation, azimuth in ((20, -40), (-30, 130)):
            ax.view_init(elev=elevation, azim=azimuth)
            fig.canvas.draw()
            np.testing.assert_allclose(image.get_facecolor(), expected, rtol=0, atol=0)
    finally:
        plt.close(fig)
