"""Analytical electromagnetic source fields shared by SIE-MoM applications.

The routines in this module define source fields independently of any
particular GSTC synthesis problem. Both Cartesian ``(x, y, z)`` and spherical
``(r, theta, phi)`` field component evaluators are provided.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


DipoleType = Literal["electric", "magnetic"]


def _parse_cartesian_points(
    points_xyz: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
    """Validate and unpack Cartesian observation coordinates (x, y, z)."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[-1] == 3:
        x = pts[..., 0]
        y = pts[..., 1]
        z = pts[..., 2]
        leading_shape = pts.shape[:-1]
    elif pts.shape[0] == 3:
        x = pts[0]
        y = pts[1]
        z = pts[2]
        leading_shape = pts.shape[1:]
    else:
        raise ValueError(
            f"points_xyz must have 3 components along first or last axis, got shape {pts.shape}"
        )
    return x, y, z, leading_shape


def evaluate_plane_wave_cartesian(
    k: float,
    points_xyz: ArrayLike,
    eta: float,
    *,
    electric_field: ArrayLike = (1.0, 0.0, 0.0),
    propagation_direction: ArrayLike = (0.0, 0.0, 1.0),
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate a transverse plane wave in Cartesian components.

    Parameters
    ----------
    k:
        Wavenumber of the medium.
    points_xyz:
        Observation coordinates with shape ``(..., 3)`` or ``(3, ...)``.
    eta:
        Intrinsic impedance of the medium.
    electric_field:
        Constant Cartesian phasor amplitude :math:`\\mathbf E_0`.
    propagation_direction:
        Cartesian propagation vector :math:`\\hat{\\mathbf k}` (normalized internally).

    Returns
    -------
    (E, H):
        Complex arrays of Cartesian fields with component-first shape ``(3, ...)``
        ordered as ``(x, y, z)``. Spatial convention is :math:`\\exp(+j\\mathbf k\\cdot\\mathbf r)`.
    """
    if eta == 0.0:
        raise ValueError("eta must be nonzero")

    electric_amplitude = np.asarray(electric_field, dtype=complex)
    direction = np.asarray(propagation_direction, dtype=float)
    if electric_amplitude.shape != (3,):
        raise ValueError("electric_field must contain three Cartesian components")
    if direction.shape != (3,):
        raise ValueError(
            "propagation_direction must contain three Cartesian components"
        )

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0.0:
        raise ValueError("propagation_direction must be nonzero")
    direction = direction / direction_norm

    longitudinal_amplitude = np.dot(direction, electric_amplitude)
    transverse_tolerance = 1e-12 * max(1.0, np.linalg.norm(electric_amplitude))
    if abs(longitudinal_amplitude) > transverse_tolerance:
        raise ValueError("electric_field must be transverse to propagation_direction")

    x, y, z, leading_shape = _parse_cartesian_points(points_xyz)
    phase_argument = direction[0] * x + direction[1] * y + direction[2] * z
    phase = np.exp(1j * k * phase_argument)

    magnetic_amplitude = np.cross(direction, electric_amplitude) / eta
    vector_shape = (3,) + (1,) * phase.ndim
    electric_cartesian = electric_amplitude.reshape(vector_shape) * phase
    magnetic_cartesian = magnetic_amplitude.reshape(vector_shape) * phase

    return electric_cartesian, magnetic_cartesian


def evaluate_plane_wave_fields(
    k: float,
    radius: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    eta: float,
    *,
    electric_field: ArrayLike = (1.0, 0.0, 0.0),
    propagation_direction: ArrayLike = (0.0, 0.0, 1.0),
    grid: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate a transverse plane wave in spherical components.

    ``electric_field`` is the constant Cartesian phasor amplitude and
    ``propagation_direction`` is normalized internally.  The magnetic-field
    amplitude is obtained from
    :math:`\\mathbf H_0=\\hat{\\mathbf k}\\times\\mathbf E_0/\\eta`, and the
    spatial convention is :math:`\\exp(+j\\mathbf k\\cdot\\mathbf r)`.

    When ``grid`` is false, ``radius``, ``theta``, and ``phi`` describe
    broadcast-compatible observation points.  When it is true, ``theta`` and
    ``phi`` are treated as independent one-dimensional grid axes.  The two
    returned arrays have component-first shape ``(3, ...)``, ordered as
    ``(r, theta, phi)``.
    """
    if grid:
        theta_array, phi_array = np.meshgrid(
            np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float),
            indexing="ij",
        )
        radius_array = np.asarray(radius, dtype=float)
    else:
        radius_array, theta_array, phi_array = np.broadcast_arrays(
            np.asarray(radius, dtype=float),
            np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float),
        )

    radius_array, theta_array, phi_array = np.broadcast_arrays(
        radius_array, theta_array, phi_array
    )
    sin_theta = np.sin(theta_array)
    cos_theta = np.cos(theta_array)
    sin_phi = np.sin(phi_array)
    cos_phi = np.cos(phi_array)

    x = radius_array * sin_theta * cos_phi
    y = radius_array * sin_theta * sin_phi
    z = radius_array * cos_theta

    points = np.stack([x, y, z], axis=-1)
    electric_cartesian, magnetic_cartesian = evaluate_plane_wave_cartesian(
        k,
        points,
        eta,
        electric_field=electric_field,
        propagation_direction=propagation_direction,
    )

    def to_spherical(cartesian: NDArray[np.complex128]) -> NDArray[np.complex128]:
        field_x, field_y, field_z = cartesian
        return np.stack(
            [
                field_x * sin_theta * cos_phi
                + field_y * sin_theta * sin_phi
                + field_z * cos_theta,
                field_x * cos_theta * cos_phi
                + field_y * cos_theta * sin_phi
                - field_z * sin_theta,
                -field_x * sin_phi + field_y * cos_phi,
            ]
        )

    return to_spherical(electric_cartesian), to_spherical(magnetic_cartesian)


def evaluate_dipole_fields_cartesian(
    k: float,
    points_xyz: ArrayLike,
    eta: float,
    *,
    effective_moment: ArrayLike = (0.0, 0.0, 1.0),
    source_position: ArrayLike = (0.0, 0.0, 0.0),
    dipole_type: DipoleType = "electric",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate electric- or magnetic-dipole fields in Cartesian components.

    Parameters
    ----------
    k:
        Wavenumber of the homogeneous medium.
    points_xyz:
        Observation coordinates with shape ``(..., 3)`` or ``(3, ...)``.
    eta:
        Intrinsic impedance of the medium.
    effective_moment:
        Effective Cartesian source amplitude ``(p_x, p_y, p_z)``.
    source_position:
        Cartesian source position ``(x_0, y_0, z_0)``.
    dipole_type:
        ``"electric"`` or ``"magnetic"``.

    Returns
    -------
    (E, H):
        Complex arrays with component-first shape ``(3, ...)`` ordered as
        ``(x, y, z)``.
    """
    if dipole_type not in ("electric", "magnetic"):
        raise ValueError("dipole_type must be 'electric' or 'magnetic'")
    if eta == 0.0:
        raise ValueError("eta must be nonzero")

    effective_moment_array = np.asarray(effective_moment, dtype=complex)
    position_array = np.asarray(source_position, dtype=float)

    if effective_moment_array.shape != (3,):
        raise ValueError(
            "effective_moment must contain exactly three Cartesian components"
        )
    if position_array.shape != (3,):
        raise ValueError("source_position must contain exactly three Cartesian components")

    x, y, z, _ = _parse_cartesian_points(points_xyz)

    displacement_x = x - position_array[0]
    displacement_y = y - position_array[1]
    displacement_z = z - position_array[2]
    distance = np.sqrt(
        displacement_x**2 + displacement_y**2 + displacement_z**2
    )

    if np.any(distance == 0.0):
        raise ValueError("dipole fields are singular at the source position")

    radial_x = displacement_x / distance
    radial_y = displacement_y / distance
    radial_z = displacement_z / distance

    effective_moment_x, effective_moment_y, effective_moment_z = (
        effective_moment_array
    )
    radial_dot_effective_moment = (
        radial_x * effective_moment_x
        + radial_y * effective_moment_y
        + radial_z * effective_moment_z
    )

    radial_cross_effective_moment_x = (
        radial_y * effective_moment_z - radial_z * effective_moment_y
    )
    radial_cross_effective_moment_y = (
        radial_z * effective_moment_x - radial_x * effective_moment_z
    )
    radial_cross_effective_moment_z = (
        radial_x * effective_moment_y - radial_y * effective_moment_x
    )

    transverse_x = (
        radial_cross_effective_moment_y * radial_z
        - radial_cross_effective_moment_z * radial_y
    )
    transverse_y = (
        radial_cross_effective_moment_z * radial_x
        - radial_cross_effective_moment_x * radial_z
    )
    transverse_z = (
        radial_cross_effective_moment_x * radial_y
        - radial_cross_effective_moment_y * radial_x
    )

    phase = np.exp(-1j * k * distance)
    radiation_coefficient = k**2 / distance
    near_field_coefficient = 1.0 / distance**3 + 1j * k / distance**2

    electric_x = phase * (
        radiation_coefficient * transverse_x
        + near_field_coefficient
        * (3.0 * radial_x * radial_dot_effective_moment - effective_moment_x)
    )
    electric_y = phase * (
        radiation_coefficient * transverse_y
        + near_field_coefficient
        * (3.0 * radial_y * radial_dot_effective_moment - effective_moment_y)
    )
    electric_z = phase * (
        radiation_coefficient * transverse_z
        + near_field_coefficient
        * (3.0 * radial_z * radial_dot_effective_moment - effective_moment_z)
    )

    magnetic_coefficient = (
        1j * (k / eta) * (1j * k / distance + 1.0 / distance**2) * phase
    )
    magnetic_x = -magnetic_coefficient * radial_cross_effective_moment_x
    magnetic_y = -magnetic_coefficient * radial_cross_effective_moment_y
    magnetic_z = -magnetic_coefficient * radial_cross_effective_moment_z

    electric = np.stack([electric_x, electric_y, electric_z], axis=0)
    magnetic = np.stack([magnetic_x, magnetic_y, magnetic_z], axis=0)

    if dipole_type == "magnetic":
        return -eta * magnetic, electric / eta

    return electric, magnetic


def evaluate_dipole_fields(
    k: float,
    radius: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    eta: float,
    *,
    effective_moment: ArrayLike = (0.0, 0.0, 1.0),
    source_position: ArrayLike = (0.0, 0.0, 0.0),
    dipole_type: DipoleType = "electric",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate electric- or magnetic-dipole fields in spherical components.

    Parameters
    ----------
    k:
        Wavenumber of the homogeneous medium.
    radius, theta, phi:
        Spherical coordinates of the observation points.  The inputs must be
        broadcast-compatible.
    eta:
        Intrinsic impedance of the medium.
    effective_moment:
        Effective Cartesian source amplitude ``(p_x, p_y, p_z)``.  For an
        electric dipole with physical SI moment :math:`\\mathbf p`, this
        argument is :math:`\\mathbf p/(4\\pi\\epsilon)`.  It is therefore not
        the physical electric dipole moment in C m.
    source_position:
        Cartesian source position ``(x_0, y_0, z_0)``.
    dipole_type:
        ``"electric"`` or ``"magnetic"``.  The magnetic result is obtained
        from the electric result by electromagnetic duality.

    Returns
    -------
    (E, H):
        Complex arrays with shape ``(3, ...)`` ordered as
        ``(r, theta, phi)``.  The convention matches the existing SIE source
        model and uses the spatial phase factor :math:`\\exp(-j k R)`, which
        is outgoing under the :math:`\\exp(+j\\omega t)` time convention.
    """
    radius_array, theta_array, phi_array = np.broadcast_arrays(
        np.asarray(radius, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
    )

    sin_theta = np.sin(theta_array)
    cos_theta = np.cos(theta_array)
    sin_phi = np.sin(phi_array)
    cos_phi = np.cos(phi_array)

    x = radius_array * sin_theta * cos_phi
    y = radius_array * sin_theta * sin_phi
    z = radius_array * cos_theta

    points = np.stack([x, y, z], axis=-1)
    electric_cartesian, magnetic_cartesian = evaluate_dipole_fields_cartesian(
        k,
        points,
        eta,
        effective_moment=effective_moment,
        source_position=source_position,
        dipole_type=dipole_type,
    )

    electric_x, electric_y, electric_z = electric_cartesian
    magnetic_x, magnetic_y, magnetic_z = magnetic_cartesian

    electric = np.stack(
        [
            electric_x * sin_theta * cos_phi
            + electric_y * sin_theta * sin_phi
            + electric_z * cos_theta,
            electric_x * cos_theta * cos_phi
            + electric_y * cos_theta * sin_phi
            - electric_z * sin_theta,
            -electric_x * sin_phi + electric_y * cos_phi,
        ]
    )
    magnetic = np.stack(
        [
            magnetic_x * sin_theta * cos_phi
            + magnetic_y * sin_theta * sin_phi
            + magnetic_z * cos_theta,
            magnetic_x * cos_theta * cos_phi
            + magnetic_y * cos_theta * sin_phi
            - magnetic_z * sin_theta,
            -magnetic_x * sin_phi + magnetic_y * cos_phi,
        ]
    )

    return electric, magnetic
