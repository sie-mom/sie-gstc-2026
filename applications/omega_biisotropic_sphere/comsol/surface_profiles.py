"""Compile shared surface profiles into COMSOL coefficient expressions."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any, Protocol

import numpy as np


class SphericalCapConfiguration(Protocol):
    """Structural type shared with ``workflow.CapPerturbationConfiguration``."""

    theta_degrees: float
    phi_degrees: float
    angular_radius_degrees: float
    parameter_scale: float


@dataclass(frozen=True)
class ScalarOmegaProfileExpressions:
    """COMSOL expressions for one scalar reciprocal-omega surface profile."""

    variables: dict[str, str]
    center_direction: tuple[float, float, float]
    cosine_threshold: float
    parameter_scale: float

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "spherical_cap",
            "coordinate_inputs": "unit_radial_direction",
            "center_direction": list(self.center_direction),
            "cosine_threshold": self.cosine_threshold,
            "parameter_scale": self.parameter_scale,
            "edge_convention": "inclusive",
            "radial_extension": "constant_along_radial_lines_through_shell",
            "comsol_evaluation": "domain_integration_points",
        }


def _clean_direction_component(value: float) -> float:
    """Remove roundoff-sized components from cardinal cap directions."""
    return 0.0 if abs(value) < 16.0 * np.finfo(float).eps else value


def spherical_cap_geometry(
    cap: SphericalCapConfiguration,
) -> tuple[tuple[float, float, float], float]:
    """Return cap-center unit direction and inclusive cosine threshold."""
    theta = cap.theta_degrees * pi / 180.0
    phi = cap.phi_degrees * pi / 180.0
    center = (
        _clean_direction_component(sin(theta) * cos(phi)),
        _clean_direction_component(sin(theta) * sin(phi)),
        _clean_direction_component(cos(theta)),
    )
    threshold = cos(cap.angular_radius_degrees * pi / 180.0)
    # Match apply_cap_perturbation(), which includes points within a small
    # floating-point tolerance of the analytical cap boundary.
    threshold -= 8.0 * np.finfo(float).eps
    return center, threshold


def compile_scalar_omega_cap_profile(
    cap: SphericalCapConfiguration,
    *,
    nx: str,
    ny: str,
    nz: str,
) -> ScalarOmegaProfileExpressions:
    """Compile a spherical cap into shell-scoped COMSOL variables.

    The same real scale multiplies ``Yee``, ``Zmm``, and ``Xem`` so the
    reciprocal lossless scalar-omega structure is preserved pointwise.
    """
    center, threshold = spherical_cap_geometry(cap)
    terms = [
        f"({coefficient:.17g})*({normal})"
        for coefficient, normal in zip(center, (nx, ny, nz), strict=True)
        if coefficient != 0.0
    ]
    dot_expression = "+".join(terms) if terms else "0"
    scale_expression = (
        f"if(gstc_cap_dot>={threshold:.17g},"
        f"{cap.parameter_scale:.17g},1)"
    )
    return ScalarOmegaProfileExpressions(
        variables={
            "gstc_cap_dot": dot_expression,
            "gstc_profile_scale": scale_expression,
            "gstc_Yee": "gstc_profile_scale*Yee",
            "gstc_Zmm": "gstc_profile_scale*Zmm",
            "gstc_Xem": "gstc_profile_scale*Xem",
        },
        center_direction=center,
        cosine_threshold=threshold,
        parameter_scale=cap.parameter_scale,
    )
