"""Circular-arc ogives of revolution, including the spherical limit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CircularOgive:
    """Two circular arcs meeting tangentially at the maximum-radius plane.

    ``center_m`` is the center of that plane; the axis points toward the right
    tip. Left/right lengths are measured from this plane, not from a tip.
    Equal lengths give a symmetric ogive. Lengths equal to ``max_radius_m``
    give hemispheres, and two such halves give a sphere. Dimensions are metres.
    """

    max_radius_m: float
    left_length_m: float
    right_length_m: float
    center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("max_radius_m", "left_length_m", "right_length_m"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"geometry.{name} must be finite and positive")
        for name in ("left_length_m", "right_length_m"):
            if getattr(self, name) < self.max_radius_m:
                raise ValueError(f"geometry.{name} must be at least max_radius_m")
        for name in ("center_m", "axis_direction"):
            values = np.asarray(getattr(self, name))
            if (
                values.shape != (3,)
                or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"geometry.{name} must contain three finite real components")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if not np.isclose(np.linalg.norm(self.axis_direction), 1.0, rtol=0, atol=1e-12):
            raise ValueError("geometry.axis_direction must be a unit vector")

    def circle_radius(self, half_length):
        """Radius of the generating circle for one axial half-length."""
        return (half_length**2 + self.max_radius_m**2) / (2 * self.max_radius_m)

    @property
    def axial_bounds(self):
        return -self.left_length_m, self.right_length_m

    def axial_radial(self, points_xyz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xyz, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_xyz must have finite Cartesian components on its last axis")
        displacement = points - np.asarray(self.center_m)
        axial = displacement @ np.asarray(self.axis_direction)
        return axial, displacement - axial[..., None] * np.asarray(self.axis_direction)

    def radius_profile(self, axial: ArrayLike) -> np.ndarray:
        """Meridional radius at an axial coordinate; zero beyond either tip.

        Rationalizing the circular-arc equation avoids subtracting similar
        circle-radius terms near an elongated tip.
        """
        x = np.asarray(axial, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("axial coordinates must be finite")
        length = np.where(x < 0, self.left_length_m, self.right_length_m)
        offset = self.circle_radius(length) - self.max_radius_m
        numerator = np.maximum((length - np.abs(x)) * (length + np.abs(x)), 0)
        denominator = np.sqrt(offset**2 + numerator) + offset
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )

    def surface_residual(self, points_xyz: ArrayLike) -> np.ndarray:
        """Radial profile residual in metres; not a signed distance function."""
        axial, radial = self.axial_radial(points_xyz)
        return np.linalg.norm(radial, axis=-1) - self.radius_profile(axial)

    def contains(self, points_xyz: ArrayLike) -> np.ndarray:
        """Strict analytic interior; the boundary and tips are excluded."""
        axial, radial = self.axial_radial(points_xyz)
        return (
            (axial > -self.left_length_m)
            & (axial < self.right_length_m)
            & (np.linalg.norm(radial, axis=-1) < self.radius_profile(axial))
        )

    def outward_normals(self, points_xyz: ArrayLike) -> np.ndarray:
        """Normal directions near the smooth surface, for geometry diagnostics.

        A pointed tip has no unique normal. Axis points are rejected for an
        elongated half, and the sphere's center is also undefined. GSTC export
        uses actual triangle frames, not these analytic directions.
        """
        axial, radial = self.axial_radial(points_xyz)
        length = np.where(axial < 0, self.left_length_m, self.right_length_m)
        offset = self.circle_radius(length) - self.max_radius_m
        rho = np.linalg.norm(radial, axis=-1, keepdims=True)
        if np.any((rho[..., 0] == 0) & (offset > 0)):
            raise ValueError("ogive normal is undefined on the axis of an elongated half")
        radial_unit = np.divide(radial, rho, out=np.zeros_like(radial), where=rho > 0)
        normal = (
            axial[..., None] * np.asarray(self.axis_direction)
            + (rho + offset[..., None]) * radial_unit
        )
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError("ogive normal is undefined at the center")
        return normal / norm

    @property
    def volume(self) -> float:
        def half(length):
            radius = self.circle_radius(length)
            offset = radius - self.max_radius_m
            return math.pi * (
                radius**2 * length
                - length**3 / 3
                - offset * radius**2 * math.asin(min(length / radius, 1.0))
            )

        return half(self.left_length_m) + half(self.right_length_m)


@dataclass(frozen=True)
class OgiveTipGrading:
    """Cubic target-size transition over a fraction of each axial half-length."""

    size_factor: float
    region_fraction: float

    def __post_init__(self):
        for name in ("size_factor", "region_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"tip_grading.{name} must be finite and in (0, 1]")

    def target_edge_length(self, geometry: CircularOgive, points_xyz, body_edge_length):
        """Evaluate h_tip + (h_body-h_tip)*(3*t^2-2*t^3) at Cartesian points."""
        axial, _ = geometry.axial_radial(points_xyz)
        length = np.where(axial < 0, geometry.left_length_m, geometry.right_length_m)
        t = np.clip((length - np.abs(axial)) / (self.region_fraction * length), 0, 1)
        return body_edge_length * (self.size_factor + (1 - self.size_factor) * t**2 * (3 - 2 * t))
