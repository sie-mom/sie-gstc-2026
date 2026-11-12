"""Geometry-independent, flat face colors on triangular surface meshes.

Matplotlib is an optional plotting dependency. Import this module explicitly;
ordinary postprocessing imports do not load it or select a plotting backend.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _real_values(values):
    values = np.asarray(values)
    if values.size == 0 or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError("face values must be nonempty, finite and real")
    return values


def face_value_limits(values, *, percentile=98, symmetric=True):
    """Limits shared across any collection of scalar arrays, without changing data.

    Signed maps use +/- the percentile of absolute values. Nonnegative maps
    start at zero. A zero percentile falls back to the actual maximum so sparse
    nonzero values remain visible; identically zero maps use a unit display span.
    """
    values = _real_values(values)
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if not symmetric and np.any(values < 0):
        raise ValueError("nonnegative color limits require nonnegative values")
    magnitude = np.abs(values)
    limit = float(np.percentile(magnitude, percentile))
    if limit == 0:
        limit = float(magnitude.max()) or 1.0
    return (-limit if symmetric else 0.0), limit


def plot_mesh_face_values(
    mesh, values, *, ax, norm=None, cmap="RdBu_r", view=(0, -90), show_edges=False
):
    """Draw one real scalar per original triangle on a supplied Matplotlib 3D axis.

    ``mesh`` supplies ``vertices`` (N,3) and integer ``faces`` (F,3). No topology
    repair, face reordering, interpolation, geometry classification or material
    interpretation is performed. ``view`` is (elevation, azimuth) in degrees;
    projections are orthographic, with equal physical lengths on every axis.
    Matplotlib sorts opaque triangle polygons by depth for display. The returned
    collection is a scalar mappable suitable for a shared figure colorbar.
    Lighting/shading is disabled: face interiors use only cmap(norm(value)).
    """
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or vertices.dtype.kind not in "fiu"
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or faces.dtype.kind not in "iu"
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("expected finite vertices and nonempty triangular integer faces")
    values = _real_values(values)
    if values.shape != (len(faces),):
        raise ValueError("expected exactly one scalar per mesh face")
    if norm is None:
        norm = Normalize(*face_value_limits(values, percentile=100))
    triangles = vertices[faces]
    if np.any(
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        == 0
    ):
        raise ValueError("mesh contains degenerate triangles")
    collection = Poly3DCollection(
        triangles,
        cmap=cmap,
        norm=norm,
        shade=False,
        edgecolors=(0, 0, 0, 0.15) if show_edges else "none",
        linewidths=0.15 if show_edges else 0,
        antialiaseds=show_edges,
    )
    collection.set_array(values)
    ax.add_collection3d(collection)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    span = upper - lower
    # Give planar meshes a small depth extent while retaining equal spatial scale.
    span = np.maximum(span, max(float(span.max()), 1e-12) * 1e-3)
    center = (upper + lower) / 2
    for index, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(
            center[index] - 0.52 * span[index], center[index] + 0.52 * span[index]
        )
        getattr(ax, f"set_{axis}label")(axis)
    ax.set_box_aspect(span)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    return collection
