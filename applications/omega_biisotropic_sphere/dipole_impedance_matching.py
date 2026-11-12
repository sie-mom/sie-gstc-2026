"""Solver-independent physics for the spherical dipole-matching example.

This module constructs the prescribed interior and exterior dipole fields and
extracts the corresponding omega-sheet parameters directly from those fields.
It does not call the SIE or SVWF solver; solver-specific field representations and comparisons are handled by
the surrounding application workflow.
"""

from __future__ import annotations

import numpy as np

try:
    from .workflow import (
        CapPerturbationConfiguration,
        DipoleMatchingCase,
        ExtractionMethod,
    )
except ImportError:  # Support direct imports by application scripts.
    from workflow import (  # type: ignore[no-redef]
        CapPerturbationConfiguration,
        DipoleMatchingCase,
        ExtractionMethod,
    )

from sie_mom.excitation import evaluate_dipole_fields
from sie_mom.postprocessing import spherical_to_cartesian
from sie_mom.gstc.omega_biisotropic import (
    LeastSquaresExtraction,
    OmegaParameters,
    ProjectedExtraction,
    extract_lossless_omega_least_squares,
    extract_lossless_omega_projected,
    form_boundary_quantities,
)

ExtractionResult = ProjectedExtraction | LeastSquaresExtraction


def apply_cap_perturbation(
    parameters: OmegaParameters,
    theta: np.ndarray,
    phi: np.ndarray,
    cap: CapPerturbationConfiguration | None,
) -> tuple[OmegaParameters, np.ndarray]:
    """Return scaled parameters and the cap mask at the supplied angles.

    Angles are in radians. The mask includes the cap boundary and handles
    azimuthal wrapping through the dot product of unit radial directions.
    Extraction results are not mutated. This operation is independent of
    extraction method and commutes with the existing phasor conjugation.
    With no cap, the parameters are unchanged and the mask is all false.
    """

    B, X, chi, theta, phi = np.broadcast_arrays(
        parameters.B, parameters.X, parameters.chi, theta, phi
    )
    mask = np.zeros(theta.shape, dtype=bool)
    if cap is not None:
        theta0, phi0, alpha = np.deg2rad(
            [cap.theta_degrees, cap.phi_degrees, cap.angular_radius_degrees]
        )
        cosine_distance = (
            np.cos(theta) * np.cos(theta0)
            + np.sin(theta) * np.sin(theta0) * np.cos(phi - phi0)
        )
        mask = cosine_distance >= np.cos(alpha) - 8 * np.finfo(float).eps
    scale = np.where(mask, cap.parameter_scale if cap is not None else 1.0, 1.0)
    return OmegaParameters(B=B * scale, X=X * scale, chi=chi * scale), mask


def prescribed_dipole_fields(
    theta: np.ndarray,
    phi: np.ndarray,
    case: DipoleMatchingCase,
    *,
    radius: float | np.ndarray | None = None,
    svwf_convention: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the prescribed interior and power-matched exterior fields.

    Fields use the physical ``exp(+j omega t)`` convention unless
    ``svwf_convention`` is true, in which case all four phasor arrays are
    conjugated for the current SVWF implementation.
    """

    observation_radius = case.radius if radius is None else radius
    electric_interior, magnetic_interior = evaluate_dipole_fields(
        case.k_interior,
        observation_radius,
        theta,
        phi,
        case.eta_interior,
        effective_moment=(0.0, 0.0, case.effective_moment_interior),
    )
    electric_exterior, magnetic_exterior = evaluate_dipole_fields(
        case.k_exterior,
        observation_radius,
        theta,
        phi,
        case.eta_exterior,
        effective_moment=(0.0, 0.0, case.effective_moment_exterior),
    )

    fields = (
        electric_interior,
        magnetic_interior,
        electric_exterior,
        magnetic_exterior,
    )
    if svwf_convention:
        return tuple(np.conjugate(field) for field in fields)
    return fields


def prescribed_total_fields(
    radius: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    case: DipoleMatchingCase,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the piecewise prescribed transformation in spherical components."""

    radius, theta, phi = np.broadcast_arrays(radius, theta, phi)
    radius = radius.ravel()
    theta = theta.ravel()
    phi = phi.ravel()
    electric = np.full((3, radius.size), np.nan + 0j, dtype=complex)
    magnetic = np.full((3, radius.size), np.nan + 0j, dtype=complex)
    valid = radius > 0.0
    interior = valid & (radius < case.radius)
    exterior = valid & ~interior

    if np.any(interior):
        electric_interior, magnetic_interior, _, _ = prescribed_dipole_fields(
            theta[interior],
            phi[interior],
            case,
            radius=radius[interior],
        )
        electric[:, interior] = electric_interior
        magnetic[:, interior] = magnetic_interior

    if np.any(exterior):
        _, _, electric_exterior, magnetic_exterior = prescribed_dipole_fields(
            theta[exterior],
            phi[exterior],
            case,
            radius=radius[exterior],
        )
        electric[:, exterior] = electric_exterior
        magnetic[:, exterior] = magnetic_exterior

    return electric, magnetic


def prescribed_total_fields_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    case: DipoleMatchingCase,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the piecewise prescribed transformation at Cartesian points."""

    x, y, z = np.broadcast_arrays(x, y, z)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    radius = np.sqrt(x**2 + y**2 + z**2)
    if np.any(radius == 0.0):
        raise ValueError("the centered dipole field is singular at the origin")
    theta = np.arccos(np.clip(z / radius, -1.0, 1.0))
    phi = np.arctan2(y, x)
    electric, magnetic = prescribed_total_fields(radius, theta, phi, case)
    return (
        spherical_to_cartesian(electric, theta, phi),
        spherical_to_cartesian(magnetic, theta, phi),
    )


def extract_case_parameters(
    theta: np.ndarray,
    phi: np.ndarray,
    case: DipoleMatchingCase,
    *,
    method: ExtractionMethod = "least_squares",
    svwf_convention: bool = False,
) -> ExtractionResult:
    """Extract lossless omega parameters for the prescribed dipole mapping."""

    fields = prescribed_dipole_fields(theta, phi, case, svwf_convention=svwf_convention)
    quantities = form_boundary_quantities(*(field[1:] for field in fields))
    if method == "projected":
        return extract_lossless_omega_projected(quantities)
    if method == "least_squares":
        return extract_lossless_omega_least_squares(quantities)
    raise ValueError("method must be 'projected' or 'least_squares'")
