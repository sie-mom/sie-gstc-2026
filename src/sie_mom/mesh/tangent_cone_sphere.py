"""Geometry of a sphere joined tangentially to a sharp cone."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TangentConeSphere:
    """The axis points from the spherical-end center toward the cone apex.

    Coordinates and radii are in metres. The spherical cap ends at axial
    coordinate ``R*sin(alpha)``; the apex is at ``R/sin(alpha)``.
    """

    sphere_radius_m: float
    cone_half_angle_deg: float
    sphere_center_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("sphere_radius_m", "cone_half_angle_deg"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"geometry.{name} must be finite")
        if self.sphere_radius_m <= 0:
            raise ValueError("geometry.sphere_radius_m must be positive")
        if not 0 < self.cone_half_angle_deg < 90:
            raise ValueError("geometry.cone_half_angle_deg must lie between 0 and 90")
        for name in ("sphere_center_m", "axis_direction"):
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

    @property
    def half_angle(self) -> float:
        return math.radians(self.cone_half_angle_deg)

    @property
    def junction(self) -> float:
        return self.sphere_radius_m * math.sin(self.half_angle)

    @property
    def junction_radius(self) -> float:
        return self.sphere_radius_m * math.cos(self.half_angle)

    @property
    def apex(self) -> float:
        return self.sphere_radius_m / math.sin(self.half_angle)

    @property
    def axial_bounds(self):
        return -self.sphere_radius_m, self.apex

    @property
    def volume(self) -> float:
        r, junction = self.sphere_radius_m, self.junction
        cap = math.pi * (r * r * junction - junction**3 / 3 + 2 * r**3 / 3)
        cone = math.pi * self.junction_radius**2 * (self.apex - junction) / 3
        return cap + cone

    def axial_radial(self, points_xyz: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xyz, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_xyz must have finite Cartesian components on its last axis")
        displacement = points - np.asarray(self.sphere_center_m)
        axial = displacement @ np.asarray(self.axis_direction)
        radial = displacement - axial[..., None] * np.asarray(self.axis_direction)
        return axial, radial

    def contains(self, points_xyz: ArrayLike) -> np.ndarray:
        """Strict interior of the analytic body; boundary points are excluded."""
        axial, radial = self.axial_radial(points_xyz)
        rho = np.linalg.norm(radial, axis=-1)
        sphere = axial**2 + rho**2 < self.sphere_radius_m**2
        cone = (axial < self.apex) & (rho < (self.apex - axial) * math.tan(self.half_angle))
        return np.where(axial <= self.junction, sphere, cone)

    def surface_residual(self, points_xyz: ArrayLike) -> np.ndarray:
        """Surface-equation residual in metres; not a signed distance function."""
        axial, radial = self.axial_radial(points_xyz)
        distance = np.linalg.norm(
            np.asarray(points_xyz) - np.asarray(self.sphere_center_m), axis=-1
        )
        return np.where(
            axial <= self.junction,
            distance - self.sphere_radius_m,
            np.linalg.norm(radial, axis=-1) - (self.apex - axial) * math.tan(self.half_angle),
        )

    def outward_normals(self, points_xyz: ArrayLike) -> np.ndarray:
        """Smooth surface normal directions, also defined near the surface.

        These are useful for checking CAD/mesh orientation. GSTC export uses
        the actual flat-triangle normals from the ordered mesh vertices.
        """
        axial, radial = self.axial_radial(points_xyz)
        axis = np.asarray(self.axis_direction)
        rho = np.linalg.norm(radial, axis=-1, keepdims=True)
        displacement = np.asarray(points_xyz) - np.asarray(self.sphere_center_m)
        distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
        sphere = np.divide(
            displacement, distance, out=np.zeros_like(displacement), where=distance > 0
        )
        radial_unit = np.divide(radial, rho, out=np.zeros_like(radial), where=rho > 0)
        cone = math.cos(self.half_angle) * radial_unit + math.sin(self.half_angle) * axis
        return np.where((axial <= self.junction)[..., None], sphere, cone)
