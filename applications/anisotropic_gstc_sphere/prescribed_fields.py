"""Evaluate prescribed interior/exterior total E/H for GSTC transformations.

Each JSON transformation specifies a pair of field references, one for each
region. This module evaluates them and returns a ``PrescribedFieldState``:
``(E_interior, H_interior, E_exterior, H_exterior)`` at the supplied points.
These are the target total fields for the transformation. An explicit null
reference prescribes zero E and H in that region. States retain the order and
IDs of ``configuration.transformations``.

Returned arrays are component-first ``(3, ...)`` in the local ``(r, theta,
phi)`` basis, including the normal component. The frame has an outward radial
normal, and the time convention is ``exp(+j omega t)``. Field references are
evaluated using the wavenumber and impedance of their respective regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sie_mom.excitation import (
    evaluate_dipole_fields,
    evaluate_dipole_fields_cartesian,
    evaluate_plane_wave_fields,
    evaluate_plane_wave_cartesian,
)

if __package__:
    from .workflow import (
        DipoleFieldDefinition,
        FieldReference,
        PlaneWaveFieldDefinition,
        WorkflowConfiguration,
    )
else:
    from workflow import (
        DipoleFieldDefinition,
        FieldReference,
        PlaneWaveFieldDefinition,
        WorkflowConfiguration,
    )


@dataclass(frozen=True)
class PrescribedFieldState:
    """Prescribed total E/H on both sides for one configured transformation.

    ``state_id`` is the transformation's JSON ``id``. Each field array has
    component-first shape ``(3, ...)``, ordered as ``(r, theta, phi)``, at the
    same observation points. The four arrays contain the prescribed total
    electric and magnetic fields in their respective regions.
    """

    state_id: str
    electric_interior: NDArray[np.complex128]
    magnetic_interior: NDArray[np.complex128]
    electric_exterior: NDArray[np.complex128]
    magnetic_exterior: NDArray[np.complex128]


def evaluate_prescribed_fields(
    reference: FieldReference | None,
    region: Literal["interior", "exterior"],
    configuration: WorkflowConfiguration,
    *,
    radius: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate one regional total-field reference at arbitrary observation points.

    Return component-first E/H in the sphere-centered (r, theta, phi) basis.
    The region selects the medium, not the source parameters. Null prescribes
    zero fields; complex amplitudes and the exp(+j omega t) convention are kept.
    """

    if region not in ("interior", "exterior"):
        raise ValueError("prescribed field region must be interior or exterior")
    shape = (3,) + np.broadcast_shapes(np.shape(radius), np.shape(theta), np.shape(phi))
    if reference is None:
        return np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex)
    definition = configuration.fields[reference.field_id]
    k = getattr(configuration.case, f"k_{region}")
    eta = getattr(configuration.case, f"eta_{region}")
    if isinstance(definition, PlaneWaveFieldDefinition):
        electric, magnetic = evaluate_plane_wave_fields(
            k,
            radius,
            theta,
            phi,
            eta,
            electric_field=np.conjugate(definition.electric_amplitude),
            propagation_direction=definition.propagation_direction,
        )
        # Reverse the shared helper's spatial phase without conjugating E0.
        return np.conjugate(electric), np.conjugate(magnetic)
    if isinstance(definition, DipoleFieldDefinition):
        return evaluate_dipole_fields(
            k,
            radius,
            theta,
            phi,
            eta,
            effective_moment=definition.effective_moment,
            source_position=definition.source_position,
            dipole_type=definition.dipole_type,
        )
    raise ValueError("unsupported prescribed field definition")


def evaluate_prescribed_fields_cartesian(
    reference: FieldReference | None,
    region: Literal["interior", "exterior"],
    configuration: WorkflowConfiguration,
    *,
    points_xyz: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate one regional total-field reference at arbitrary Cartesian observation points.

    Return component-first E/H in the Cartesian (x, y, z) basis.
    """
    if region not in ("interior", "exterior"):
        raise ValueError("prescribed field region must be interior or exterior")
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[-1] == 3:
        leading_shape = pts.shape[:-1]
    elif pts.shape[0] == 3:
        leading_shape = pts.shape[1:]
    else:
        raise ValueError("points_xyz must have 3 components along first or last axis")
    shape = (3,) + leading_shape
    if reference is None:
        return np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex)
    definition = configuration.fields[reference.field_id]
    k = getattr(configuration.case, f"k_{region}")
    eta = getattr(configuration.case, f"eta_{region}")
    if isinstance(definition, PlaneWaveFieldDefinition):
        electric, magnetic = evaluate_plane_wave_cartesian(
            k,
            points_xyz,
            eta,
            electric_field=np.conjugate(definition.electric_amplitude),
            propagation_direction=definition.propagation_direction,
        )
        return np.conjugate(electric), np.conjugate(magnetic)
    if isinstance(definition, DipoleFieldDefinition):
        return evaluate_dipole_fields_cartesian(
            k,
            points_xyz,
            eta,
            effective_moment=definition.effective_moment,
            source_position=definition.source_position,
            dipole_type=definition.dipole_type,
        )
    raise ValueError("unsupported prescribed field definition")


def prescribed_boundary_states_cartesian(
    points_xyz: ArrayLike,
    configuration: WorkflowConfiguration,
) -> tuple[PrescribedFieldState, ...]:
    """Return one total-field state per transformation at Cartesian observation points.

    Returned arrays have component-first shape (3, ...) in the (x, y, z) Cartesian basis.
    """
    states = []
    for transformation in configuration.transformations:
        electric_interior, magnetic_interior = evaluate_prescribed_fields_cartesian(
            transformation.gstc.interior,
            "interior",
            configuration,
            points_xyz=points_xyz,
        )
        electric_exterior, magnetic_exterior = evaluate_prescribed_fields_cartesian(
            transformation.gstc.exterior,
            "exterior",
            configuration,
            points_xyz=points_xyz,
        )
        states.append(
            PrescribedFieldState(
                transformation.id,
                electric_interior,
                magnetic_interior,
                electric_exterior,
                magnetic_exterior,
            )
        )
    return tuple(states)


def prescribed_boundary_states(
    theta: ArrayLike,
    phi: ArrayLike,
    configuration: WorkflowConfiguration,
    *,
    radius: float | None = None,
) -> tuple[PrescribedFieldState, ...]:
    """Return one total-field state per transformation, in configuration order.

    ``theta`` and ``phi`` are broadcast-compatible observation angles in
    radians. ``radius`` defaults to the configured sphere radius and can be
    overridden explicitly. Observation points are evaluated at that radius.

    Each ``gstc.interior``/``gstc.exterior`` reference selects a named field
    definition; null yields zero E/H with the same shape. Source positions
    belong to the field definitions, while the chosen region determines the
    evaluation medium. Complex source amplitudes are preserved, and plane
    waves use spatial phase ``exp(-j k.r)`` under the application's convention.

    The result includes normal components and is determined by the field
    definitions, transformations, and physical settings in the configuration.
    """

    case = configuration.case
    radius = case.radius_m if radius is None else radius

    states = []
    for transformation in configuration.transformations:
        electric_interior, magnetic_interior = evaluate_prescribed_fields(
            transformation.gstc.interior,
            "interior",
            configuration,
            radius=radius,
            theta=theta,
            phi=phi,
        )
        electric_exterior, magnetic_exterior = evaluate_prescribed_fields(
            transformation.gstc.exterior,
            "exterior",
            configuration,
            radius=radius,
            theta=theta,
            phi=phi,
        )
        states.append(
            PrescribedFieldState(
                transformation.id,
                electric_interior,
                magnetic_interior,
                electric_exterior,
                magnetic_exterior,
            )
        )
    return tuple(states)
