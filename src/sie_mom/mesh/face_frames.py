"""Intrinsic local face coordinate frames for triangular surface meshes.

This module computes the local orthogonal tangent frame :math:`(\\hat{\\mathbf t}_1,
\\hat{\\mathbf t}_2, \\hat{\\mathbf n})` for every face in a triangular surface mesh
strictly from the ordered vertex coordinates :math:`(\\mathbf v_0, \\mathbf v_1, \\mathbf v_2)`:

.. math::

    \\mathbf e_1 = \\mathbf v_1 - \\mathbf v_0,
    \\qquad
    \\mathbf e_2 = \\mathbf v_2 - \\mathbf v_0,
    \\qquad
    \\hat{\\mathbf n} = \\frac{\\mathbf e_1 \\times \\mathbf e_2}{\\|\\mathbf e_1 \\times \\mathbf e_2\\|},

.. math::

    \\hat{\\mathbf t}_1 = \\frac{\\mathbf e_1}{\\|\\mathbf e_1\\|},
    \\qquad
    \\hat{\\mathbf t}_2 = \\hat{\\mathbf n} \\times \\hat{\\mathbf t}_1.

By construction:
1. :math:`\\hat{\\mathbf t}_1 \\cdot \\hat{\\mathbf t}_2 = 0`
2. :math:`\\hat{\\mathbf t}_1 \\cdot \\hat{\\mathbf n} = 0`
3. :math:`\\hat{\\mathbf t}_2 \\cdot \\hat{\\mathbf n} = 0`
4. :math:`\\hat{\\mathbf t}_1 \\times \\hat{\\mathbf t}_2 = \\hat{\\mathbf n}`

The orientation of :math:`\\hat{\\mathbf n}` follows the right-hand rule with
respect to the cyclic ordering of :math:`(\\mathbf v_0, \\mathbf v_1, \\mathbf v_2)`.
For meshes with counter-clockwise outward winding (the standard in ``sie_mom``),
:math:`\\hat{\\mathbf n}` points outward.

This frame is algebraically identical to the C++/CUDA implementation in
``include/sie_mom/cuda/gstc_tensor_utils.cuh`` (``get_face_basis_vectors``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .obj_mesh import TriangleMesh

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FaceFrames:
    """Intrinsic face tangent frames and geometric attributes.

    All arrays have leading axis corresponding to the mesh faces (length :math:`N_F`).
    """

    centroids: FloatArray
    """Cartesian coordinates of face centroids, shape ``(N_F, 3)``."""

    t1: FloatArray
    """First in-plane unit tangent vector directed along edge :math:`\\mathbf v_1 - \\mathbf v_0`, shape ``(N_F, 3)``."""

    t2: FloatArray
    """Second in-plane unit tangent vector completing the right-handed triad :math:`\\hat{\\mathbf t}_2 = \\hat{\\mathbf n} \\times \\hat{\\mathbf t}_1`, shape ``(N_F, 3)``."""

    normals: FloatArray
    """Outward unit normal vector :math:`\\hat{\\mathbf n} = \\hat{\\mathbf t}_1 \\times \\hat{\\mathbf t}_2`, shape ``(N_F, 3)``."""

    areas: FloatArray
    """Triangle surface area, shape ``(N_F,)``."""

    @property
    def face_count(self) -> int:
        """Return the number of faces."""
        return len(self.areas)


def compute_face_frames(mesh: TriangleMesh) -> FaceFrames:
    """Compute intrinsic face tangent frames from ordered triangle vertices.

    Parameters
    ----------
    mesh : TriangleMesh
        Mesh with vertices of shape ``(N_V, 3)`` and faces of shape ``(N_F, 3)``.
        Vertex indexing within each face row must follow counter-clockwise
        outward winding :math:`(v_0, v_1, v_2)`.

    Returns
    -------
    FaceFrames
        Dataclass containing centroids, t1, t2, normals, and areas.

    Raises
    ------
    ValueError
        If the mesh has no faces or contains degenerate faces with zero area
        or zero edge length.
    """
    if len(mesh.faces) == 0:
        raise ValueError("Cannot compute face frames for an empty mesh.")

    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0

    cross_prod = np.cross(e1, e2)
    cross_norm = np.linalg.norm(cross_prod, axis=-1, keepdims=True)

    if np.any(cross_norm <= 0.0):
        degenerate_indices = np.flatnonzero(cross_norm.squeeze(-1) <= 0.0)
        raise ValueError(
            f"Mesh contains degenerate faces with zero area (e.g. face {degenerate_indices[0]})."
        )

    e1_norm = np.linalg.norm(e1, axis=-1, keepdims=True)
    if np.any(e1_norm <= 0.0):
        degenerate_indices = np.flatnonzero(e1_norm.squeeze(-1) <= 0.0)
        raise ValueError(
            f"Mesh contains degenerate faces with zero e1 edge length (e.g. face {degenerate_indices[0]})."
        )

    normals = cross_prod / cross_norm
    areas = 0.5 * cross_norm.squeeze(-1)

    t1 = e1 / e1_norm
    t2 = np.cross(normals, t1)

    return FaceFrames(
        centroids=centroids,
        t1=t1,
        t2=t2,
        normals=normals,
        areas=areas,
    )


def project_to_face_frame(
    cartesian_field: NDArray[Any],
    frames: FaceFrames,
    *,
    axis: int = 0,
) -> NDArray[Any]:
    """Project 3D Cartesian vector fields onto 2D local face tangent frames (t1, t2).

    Parameters
    ----------
    cartesian_field : NDArray
        Array containing 3D Cartesian field vectors.
        By default (``axis=0``), shape is ``(3, N_F)`` where component 0 is x,
        1 is y, 2 is z.
        If ``axis=-1``, shape is ``(N_F, 3)``.
    frames : FaceFrames
        Face frames containing ``t1`` and ``t2`` arrays of shape ``(N_F, 3)``.
    axis : int, optional
        Axis corresponding to the 3 Cartesian components (0 or -1, default 0).

    Returns
    -------
    NDArray
        Tangential field components in the ``(t1, t2)`` frame.
        If ``axis=0``, returned shape is ``(2, N_F)`` where index 0 is the
        :math:`t_1` component and index 1 is the :math:`t_2` component.
        If ``axis=-1``, returned shape is ``(N_F, 2)``.
    """
    if axis not in (0, -1):
        raise ValueError("axis must be 0 (component-first) or -1 (component-last)")

    if axis == 0:
        if cartesian_field.shape[0] != 3:
            raise ValueError(
                f"Expected cartesian_field.shape[0] == 3 for axis=0, got {cartesian_field.shape}"
            )
        fx, fy, fz = cartesian_field[0], cartesian_field[1], cartesian_field[2]
        f_t1 = fx * frames.t1[:, 0] + fy * frames.t1[:, 1] + fz * frames.t1[:, 2]
        f_t2 = fx * frames.t2[:, 0] + fy * frames.t2[:, 1] + fz * frames.t2[:, 2]
        return np.stack([f_t1, f_t2], axis=0)
    else:
        if cartesian_field.shape[-1] != 3:
            raise ValueError(
                f"Expected cartesian_field.shape[-1] == 3 for axis=-1, got {cartesian_field.shape}"
            )
        fx = cartesian_field[..., 0]
        fy = cartesian_field[..., 1]
        fz = cartesian_field[..., 2]
        f_t1 = fx * frames.t1[:, 0] + fy * frames.t1[:, 1] + fz * frames.t1[:, 2]
        f_t2 = fx * frames.t2[:, 0] + fy * frames.t2[:, 1] + fz * frames.t2[:, 2]
        return np.stack([f_t1, f_t2], axis=-1)


def lift_2x2_to_3x3(
    local_block: NDArray[Any],
    frames: FaceFrames,
) -> NDArray[Any]:
    """Lift 2x2 tangent-plane tensors to 3x3 Cartesian tensors.

    For each face :math:`m`, given the local :math:`2\\times 2` tensor :math:`T`:

    .. math::

        T_{3\\times 3} = R_{3\\times 2} T_{2\\times 2} R_{3\\times 2}^T

    where columns of :math:`R_{3\\times 2}` are :math:`\\hat{\\mathbf t}_1`
    and :math:`\\hat{\\mathbf t}_2`:

    .. math::

        T_{3\\times 3} = T_{11} (\\hat{\\mathbf t}_1 \\otimes \\hat{\\mathbf t}_1)
                       + T_{12} (\\hat{\\mathbf t}_1 \\otimes \\hat{\\mathbf t}_2)
                       + T_{21} (\\hat{\\mathbf t}_2 \\otimes \\hat{\\mathbf t}_1)
                       + T_{22} (\\hat{\\mathbf t}_2 \\otimes \\hat{\\mathbf t}_2).

    Parameters
    ----------
    local_block : NDArray
        Local 2x2 tensor block with shape ``(2, 2, N_F)`` or ``(2, 2)``.
    frames : FaceFrames
        Face frames containing ``t1`` and ``t2``.

    Returns
    -------
    NDArray
        Cartesian 3x3 tensor with shape ``(3, 3, N_F)``.
    """
    block = np.asarray(local_block)
    nf = frames.face_count

    if block.shape == (2, 2):
        t11 = block[0, 0]
        t12 = block[0, 1]
        t21 = block[1, 0]
        t22 = block[1, 1]
    elif block.shape == (2, 2, nf):
        t11 = block[0, 0, :]
        t12 = block[0, 1, :]
        t21 = block[1, 0, :]
        t22 = block[1, 1, :]
    else:
        raise ValueError(
            f"local_block must have shape (2, 2) or (2, 2, {nf}), got {block.shape}"
        )

    t1 = frames.t1  # (N_F, 3)
    t2 = frames.t2  # (N_F, 3)

    out = np.zeros((3, 3, nf), dtype=block.dtype)
    for i in range(3):
        for j in range(3):
            out[i, j, :] = (
                t11 * (t1[:, i] * t1[:, j])
                + t12 * (t1[:, i] * t2[:, j])
                + t21 * (t2[:, i] * t1[:, j])
                + t22 * (t2[:, i] * t2[:, j])
            )
    return out


def project_3x3_to_2x2(
    cartesian_tensor: NDArray[Any],
    frames: FaceFrames,
) -> NDArray[Any]:
    """Project 3x3 Cartesian tensors onto 2x2 local face tangent frames (t1, t2).

    .. math::

        T_{2\\times 2} = R_{3\\times 2}^T T_{3\\times 3} R_{3\\times 2}

    Parameters
    ----------
    cartesian_tensor : NDArray
        Cartesian 3x3 tensor with shape ``(3, 3, N_F)`` or ``(3, 3)``.
    frames : FaceFrames
        Face frames containing ``t1`` and ``t2``.

    Returns
    -------
    NDArray
        Local 2x2 tensor block with shape ``(2, 2, N_F)``.
    """
    tensor = np.asarray(cartesian_tensor)
    nf = frames.face_count

    if tensor.shape == (3, 3):
        tensor = np.repeat(tensor[:, :, np.newaxis], nf, axis=2)
    elif tensor.shape != (3, 3, nf):
        raise ValueError(
            f"cartesian_tensor must have shape (3, 3) or (3, 3, {nf}), got {tensor.shape}"
        )

    t1 = frames.t1  # (N_F, 3)
    t2 = frames.t2  # (N_F, 3)

    out = np.zeros((2, 2, nf), dtype=tensor.dtype)
    basis = [t1, t2]
    for a in range(2):
        for b in range(2):
            va = basis[a]
            vb = basis[b]
            term = np.zeros(nf, dtype=tensor.dtype)
            for i in range(3):
                for j in range(3):
                    term += va[:, i] * tensor[i, j, :] * vb[:, j]
            out[a, b, :] = term
    return out

