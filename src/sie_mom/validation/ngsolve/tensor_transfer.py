"""Frames and direct, facewise-constant tensor transfer for matching meshes."""

from __future__ import annotations

from dataclasses import dataclass

import ngsolve as ng
import numpy as np
from scipy.spatial import cKDTree

from sie_mom.mesh.face_frames import compute_face_frames

from .model import normalized_map, sheet_admittance


def spherical_tangent_frame(mesh):
    """Right-handed theta/phi frame projected onto the curved mesh interface."""
    n = ng.specialcf.normal(mesh.Materials("core"))
    rho = ng.sqrt(ng.x**2 + ng.y**2 + 1e-30)
    phi = ng.IfPos(rho - 1e-12, ng.CF((-ng.y / rho, ng.x / rho, 0)), ng.CF((0, 1, 0)))
    phi = phi - n * (n * phi)
    phi = phi / ng.sqrt(phi * phi)
    theta = ng.Cross(phi, n)
    return theta, phi


def match_interface(mesh, source, k0):
    """Map every ``sheet`` boundary element to one original SIE triangle.

    Match physical geometry, not boundary ordering or vertex numbering. Curving,
    splitting, moving or duplicating the imported interface is rejected. The
    local tensor basis always comes from the original ordered SIE vertices.
    """
    xyz = np.asarray(source.vertices) * k0
    tree = cKDTree(xyz)
    keys = {tuple(sorted(face)): i for i, face in enumerate(source.faces)}
    elements = [e for e in mesh.Elements(ng.BND) if e.mat == "sheet"]
    ids, maximum_error = [], 0.0
    for element in elements:
        points = np.asarray([mesh[v].point for v in element.vertices])
        distance, vertex_ids = tree.query(points)
        maximum_error = max(maximum_error, float(distance.max()))
        key = tuple(sorted(vertex_ids))
        if key not in keys:
            raise ValueError("GSTC interface connectivity differs from the SIE mesh")
        ids.append(keys[key])
    if sorted(ids) != list(range(len(source.faces))):
        raise ValueError("GSTC interface is not a bijection with original SIE faces")
    tolerance = 1e-12 * max(float(np.ptp(xyz, axis=0).max()), 1e-12)
    if maximum_error > tolerance:
        raise ValueError("GSTC interface vertices moved from the original SIE mesh")
    if mesh.GetCurveOrder() != 1:
        raise ValueError("shared GSTC interface requires planar geometry (curve order 1)")
    return elements, np.asarray(ids), maximum_error / k0


@dataclass
class InterfaceCoefficients:
    """Native, piecewise-constant coefficient fields and their source-face map."""

    frame: tuple
    admittance: object
    boundary_elements: np.ndarray
    source_faces: np.ndarray
    maximum_vertex_error_m: float
    maximum_magnetic_condition_number: float
    coefficient_fields: tuple


def shared_interface_coefficients(mesh, source, physical_maps, k0):
    """Assign original physical maps and frames directly to matching triangles.

    SurfaceL2(order=0) stores constants on each boundary element. Evaluation at
    quadrature points runs in NGSolve's compiled coefficient machinery; there
    is no nearest-neighbor search or Python callback during assembly. Frame
    vectors are original SIE t1,t2, independent of FEM edge orientations.
    """
    physical_maps = np.asarray(physical_maps)
    if physical_maps.shape != (len(source.faces), 4, 4) or not np.isfinite(physical_maps).all():
        raise ValueError("expected finite physical 4x4 maps in source face order")
    elements, source_ids, error = match_interface(mesh, source, k0)
    frames = compute_face_frames(source)
    normalized = normalized_map(physical_maps)
    maximum_condition = float(np.linalg.cond(normalized[:, 2:, 2:]).max())
    operators = np.stack([sheet_admittance(face_map) for face_map in normalized])

    def field(values, *, complex):
        # Scalar SurfaceL2 supports complex evaluation in the tested NGSolve
        # build; its vector-valued BlockDifferentialOperator does not. Compose
        # native scalar fields, retaining constant values on each triangle.
        space = ng.SurfaceL2(mesh, order=0, complex=complex, definedon=mesh.Boundaries("sheet"))
        dofs = []
        for element in elements:
            element_dofs = space.GetDofNrs(ng.ElementId(ng.BND, element.nr))
            if len(element_dofs) != 1 or element_dofs[0] < 0:
                raise ValueError("expected one constant coefficient DOF per GSTC triangle")
            dofs.append(element_dofs[0])
        fields = tuple(ng.GridFunction(space) for _ in range(values.shape[1]))
        for index, result in enumerate(fields):
            result.vec.FV().NumPy()[dofs] = values[source_ids, index]
        return ng.CF(fields), fields

    t1, t1_fields = field(frames.t1, complex=False)
    t2, t2_fields = field(frames.t2, complex=False)
    operator, operator_fields = field(operators.reshape(-1, 16), complex=True)
    admittance = ng.CF(tuple(operator[i] for i in range(16)), dims=(4, 4))
    return InterfaceCoefficients(
        (t1, t2),
        admittance,
        np.asarray([e.nr for e in elements]),
        source_ids,
        error,
        maximum_condition,
        t1_fields + t2_fields + operator_fields,
    )
