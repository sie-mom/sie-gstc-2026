"""Reusable Gmsh mesh generators for MoM workflows.

This module keeps geometry generation logic in one place so application-level
scripts can vary physical parameters (e.g. wavelength-dependent mesh size)
without duplicating CAD code.

Design:
- One abstract base class with common Gmsh lifecycle and output handling.
- One dataclass per geometry parameters.
- One concrete class per geometry.

All generators output a surface triangular mesh in OBJ format.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import gmsh

from .circular_ogive import CircularOgive, OgiveTipGrading
from .tangent_cone_sphere import TangentConeSphere


@dataclass
class CommonMeshOptions:
    output_obj: str
    mesh_size_min: float
    mesh_size_max: float
    headless: bool = False
    model_name: str = "Mesh"
    write_step: bool = True


@dataclass
class SphereParams:
    radius: float = 1.0


@dataclass
class SphericalPatchRefinement:
    """Smooth local surface refinement around one point on a sphere."""

    center: tuple[float, float, float]
    inner_chord_radius: float
    outer_chord_radius: float
    mesh_size: float


@dataclass
class ConeSphereParams:
    sphere_radius: float = 0.1
    cone_length: float = 0.6


@dataclass
class BluntedConeSphereParams:
    sphere_radius: float = 0.1
    cone_length: float = 0.6
    nose_radius: float = 0.02


class BaseGmshGenerator(ABC):
    """Template method for mesh generation using gmsh OCC."""

    def __init__(self, options: CommonMeshOptions) -> None:
        self.options = options

    def generate(self) -> str:
        gmsh.initialize()
        try:
            gmsh.model.add(self.options.model_name)
            self._build_geometry()
            gmsh.model.occ.synchronize()

            if self.options.write_step:
                # Optional analytical geometry for external tools such as HFSS.
                step_path = str(self.options.output_obj).replace(".obj", ".step")
                gmsh.write(step_path)

            gmsh.option.setNumber("Mesh.MeshSizeMin", self.options.mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.options.mesh_size_max)
            self._configure_mesh()
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.removeDuplicateNodes()
            gmsh.write(self.options.output_obj)
            print(f"Saved mesh to {self.options.output_obj}")
            if not self.options.headless:
                gmsh.fltk.run()
            return self.options.output_obj
        finally:
            gmsh.finalize()

    @abstractmethod
    def _build_geometry(self) -> None:
        pass

    def _configure_mesh(self) -> None:
        """Install optional mesh fields after OCC synchronization."""


class SphereGenerator(BaseGmshGenerator):
    def __init__(
        self,
        params: SphereParams,
        options: CommonMeshOptions,
        patch_refinement: SphericalPatchRefinement | None = None,
    ) -> None:
        super().__init__(options)
        self.params = params
        self.patch_refinement = patch_refinement

    def _build_geometry(self) -> None:
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, self.params.radius, tag=1)

    def _configure_mesh(self) -> None:
        if self.patch_refinement is None:
            return
        refinement = self.patch_refinement
        field = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(field, "XCenter", refinement.center[0])
        gmsh.model.mesh.field.setNumber(field, "YCenter", refinement.center[1])
        gmsh.model.mesh.field.setNumber(field, "ZCenter", refinement.center[2])
        gmsh.model.mesh.field.setNumber(
            field, "Radius", refinement.inner_chord_radius
        )
        gmsh.model.mesh.field.setNumber(
            field,
            "Thickness",
            refinement.outer_chord_radius - refinement.inner_chord_radius,
        )
        gmsh.model.mesh.field.setNumber(field, "VIn", refinement.mesh_size)
        gmsh.model.mesh.field.setNumber(field, "VOut", self.options.mesh_size_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(field)


class ConeSphereGenerator(BaseGmshGenerator):
    """Sharp-apex cone-sphere using two half-revolves to avoid seam artifacts."""

    def __init__(self, params: ConeSphereParams, options: CommonMeshOptions) -> None:
        super().__init__(options)
        self.params = params

    def _build_geometry(self) -> None:
        R = self.params.sphere_radius
        L = self.params.cone_length
        h = self.options.mesh_size_max

        p_south = gmsh.model.occ.addPoint(0, 0, -R, h)
        p_center = gmsh.model.occ.addPoint(0, 0, 0, h)
        p_equator = gmsh.model.occ.addPoint(R, 0, 0, h)
        p_apex = gmsh.model.occ.addPoint(0, 0, L, h)

        arc = gmsh.model.occ.addCircleArc(p_south, p_center, p_equator)
        line = gmsh.model.occ.addLine(p_equator, p_apex)

        profile = [(1, arc), (1, line)]
        gmsh.model.occ.revolve(profile, 0, 0, 0, 0, 0, 1, math.pi)
        gmsh.model.occ.revolve(profile, 0, 0, 0, 0, 0, 1, -math.pi)


class TangentConeSphereGenerator(BaseGmshGenerator):
    """Tangent spherical cap and cone, with a configurable center and axis."""

    def __init__(self, params: TangentConeSphere, options: CommonMeshOptions) -> None:
        super().__init__(options)
        self.params = params

    def _build_geometry(self) -> None:
        import numpy as np

        geometry = self.params
        center = np.asarray(geometry.sphere_center_m)
        axis = np.asarray(geometry.axis_direction)
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        radial = seed - np.dot(seed, axis)*axis
        radial /= np.linalg.norm(radial)

        def point(position):
            return gmsh.model.occ.addPoint(*position, self.options.mesh_size_max)

        back = point(center - geometry.sphere_radius_m*axis)
        middle = point(center)
        junction = point(center + geometry.junction*axis + geometry.junction_radius*radial)
        apex = point(center + geometry.apex*axis)
        arc = gmsh.model.occ.addCircleArc(back, middle, junction)
        line = gmsh.model.occ.addLine(junction, apex)
        profile = [(1, arc), (1, line)]
        gmsh.model.occ.revolve(profile, *center, *axis, math.pi)
        gmsh.model.occ.revolve(profile, *center, *axis, -math.pi)


class CircularOgiveGenerator(BaseGmshGenerator):
    """Circular ogive with independent half-lengths and a configurable axis."""

    def __init__(
        self, params: CircularOgive, options: CommonMeshOptions,
        tip_grading: OgiveTipGrading | None = None,
    ) -> None:
        super().__init__(options)
        self.params = params
        self.tip_grading = tip_grading

    def _configure_mesh(self) -> None:
        grading = self.tip_grading
        if grading is None or grading.size_factor == 1:
            return
        body = self.options.mesh_size_max
        tip = grading.size_factor * body
        # The global minimum otherwise clips a requested local refinement.
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(self.options.mesh_size_min, 0.8 * tip))
        # Let the spatial field control the transition instead of extending
        # the finest meridian-curve sizes across an entire surface patch.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        axial = "(" + "+".join(
            f"({coordinate}-({center:.17g}))*({axis:.17g})"
            for coordinate, center, axis in zip(
                ("x", "y", "z"), self.params.center_m, self.params.axis_direction
            )
        ) + ")"
        fields = []
        for length, sign in ((self.params.left_length_m, "+"),
                             (self.params.right_length_m, "-")):
            transition = grading.region_fraction * length
            t = f"Min(1,Max(0,(({length:.17g}){sign}{axial})/({transition:.17g})))"
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(
                field, "F", f"({tip:.17g})+({body-tip:.17g})*({t})^2*(3-2*({t}))"
            )
            fields.append(field)
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)

    def _build_geometry(self) -> None:
        import numpy as np

        geometry = self.params
        center = np.asarray(geometry.center_m)
        axis = np.asarray(geometry.axis_direction)
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        radial = seed - np.dot(seed, axis) * axis
        radial /= np.linalg.norm(radial)

        def point(position):
            return gmsh.model.occ.addPoint(*position, self.options.mesh_size_max)

        left = point(center - geometry.left_length_m * axis)
        equator = point(center + geometry.max_radius_m * radial)
        right = point(center + geometry.right_length_m * axis)
        left_center = point(
            center + (geometry.max_radius_m - geometry.circle_radius(geometry.left_length_m))
            * radial
        )
        right_center = point(
            center + (geometry.max_radius_m - geometry.circle_radius(geometry.right_length_m))
            * radial
        )
        arcs = [
            (1, gmsh.model.occ.addCircleArc(left, left_center, equator)),
            (1, gmsh.model.occ.addCircleArc(equator, right_center, right)),
        ]
        for angle in (math.pi, -math.pi):
            gmsh.model.occ.revolve(arcs, *center, *axis, angle)


class BluntedConeSphereGenerator(BaseGmshGenerator):
    """Cone-sphere with spherical nose cap tangent to the cone (G1 continuity)."""

    def __init__(self, params: BluntedConeSphereParams, options: CommonMeshOptions) -> None:
        super().__init__(options)
        self.params = params

    def _nose_geometry(self) -> Tuple[float, float, float, float, float]:
        R = self.params.sphere_radius
        L = self.params.cone_length
        Rn = self.params.nose_radius

        slant = math.sqrt(L * L + R * R)
        max_nose_radius = R * L / slant
        if Rn >= max_nose_radius:
            raise ValueError(
                f"nose_radius={Rn} is too large; must be < {max_nose_radius:.6f}"
            )

        z_center = L - Rn * slant / R
        z_tip = z_center + Rn
        r_tan = L * Rn / slant
        z_tan = L - (L * L * Rn) / (R * slant)
        return slant, z_center, z_tip, r_tan, z_tan

    def _build_geometry(self) -> None:
        R = self.params.sphere_radius
        h = self.options.mesh_size_max
        Rn = self.params.nose_radius

        _, z_center, z_tip, r_tan, z_tan = self._nose_geometry()
        nose_h = min(h * 0.5, Rn)

        p_south = gmsh.model.occ.addPoint(0, 0, -R, h)
        p_center_aft = gmsh.model.occ.addPoint(0, 0, 0, h)
        p_equator = gmsh.model.occ.addPoint(R, 0, 0, h)
        p_tangent = gmsh.model.occ.addPoint(r_tan, 0, z_tan, nose_h)
        p_center_nose = gmsh.model.occ.addPoint(0, 0, z_center, nose_h)
        p_tip = gmsh.model.occ.addPoint(0, 0, z_tip, nose_h)

        arc_aft = gmsh.model.occ.addCircleArc(p_south, p_center_aft, p_equator)
        line_cone = gmsh.model.occ.addLine(p_equator, p_tangent)
        arc_nose = gmsh.model.occ.addCircleArc(p_tangent, p_center_nose, p_tip)

        profile = [(1, arc_aft), (1, line_cone), (1, arc_nose)]
        gmsh.model.occ.revolve(profile, 0, 0, 0, 0, 0, 1, math.pi)
        gmsh.model.occ.revolve(profile, 0, 0, 0, 0, 0, 1, -math.pi)
