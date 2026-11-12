"""Configuration adapter for the first non-spherical GSTC preparation workflow.

Reuse the sphere application's field/case parsers and SIE settings during this
phase. Geometry and observation bounds belong to this application. Common
configuration types can move together when the full shared workflow is defined.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import MappingProxyType

from scipy.constants import c, epsilon_0, mu_0

from applications.anisotropic_gstc_sphere.workflow import (
    DipoleFieldDefinition,
    FieldDefinition,
    GSTCTransformationConfiguration,
    SIEConfiguration,
    SIEMeshConfiguration,
    SimulationConfiguration,
    _identifier,
    _json_array,
    _parse_field_definition,
    _parse_simulation_case,
    _parse_transformation,
    _positive,
    _required_object,
    _unique_json_object,
)
from sie_mom.mesh.circular_ogive import CircularOgive, OgiveTipGrading
from sie_mom.mesh.tangent_cone_sphere import TangentConeSphere

APPLICATION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGURATION = APPLICATION_DIR / "configs/cone_sphere/ppw6.json"


@dataclass(frozen=True)
class SurfaceMeshConfiguration(SIEMeshConfiguration):
    tip_grading: OgiveTipGrading | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.tip_grading is not None and not isinstance(self.tip_grading, OgiveTipGrading):
            raise ValueError("sie.mesh.tip_grading must be an OgiveTipGrading configuration")


@dataclass(frozen=True)
class PhysicalConfiguration:
    frequency_hz: float
    relative_permittivity_interior: float
    relative_permeability_interior: float
    relative_permittivity_exterior: float
    relative_permeability_exterior: float

    def __post_init__(self):
        for item in dataclass_fields(self):
            _positive(getattr(self, item.name), f"physical.{item.name}")

    @property
    def k_interior(self):
        return (
            2
            * math.pi
            * self.frequency_hz
            * math.sqrt(
                epsilon_0
                * mu_0
                * self.relative_permittivity_interior
                * self.relative_permeability_interior
            )
        )

    @property
    def k_exterior(self):
        return (
            2
            * math.pi
            * self.frequency_hz
            * math.sqrt(
                epsilon_0
                * mu_0
                * self.relative_permittivity_exterior
                * self.relative_permeability_exterior
            )
        )

    @property
    def eta_interior(self):
        return math.sqrt(
            mu_0
            * self.relative_permeability_interior
            / (epsilon_0 * self.relative_permittivity_interior)
        )

    @property
    def eta_exterior(self):
        return math.sqrt(
            mu_0
            * self.relative_permeability_exterior
            / (epsilon_0 * self.relative_permittivity_exterior)
        )


@dataclass(frozen=True)
class WorkflowConfiguration:
    case_id: str
    case: PhysicalConfiguration
    geometry: TangentConeSphere | CircularOgive
    fields: Mapping[str, FieldDefinition]
    transformations: tuple[GSTCTransformationConfiguration, ...]
    simulation: SimulationConfiguration
    extraction_method: str
    tensor_structure: str
    sie: SIEConfiguration
    raw: dict

    @property
    def mesh_path(self):
        path = Path(self.sie.mesh.path)
        return path if path.is_absolute() else APPLICATION_DIR / path

    @property
    def target_edge_length(self):
        index = max(
            math.sqrt(
                self.case.relative_permittivity_interior * self.case.relative_permeability_interior
            ),
            math.sqrt(
                self.case.relative_permittivity_exterior * self.case.relative_permeability_exterior
            ),
        )
        return c / self.case.frequency_hz / index / self.sie.mesh.points_per_wavelength

    def to_dict(self):
        return copy.deepcopy(self.raw)


def load_workflow_configuration(path=DEFAULT_CONFIGURATION):
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream, object_pairs_hook=_unique_json_object)
    keys = {
        "schema_version",
        "application",
        "case_id",
        "physical",
        "geometry",
        "fields",
        "transformations",
        "simulation",
        "extraction_method",
        "tensor_structure",
        "sampling",
        "sie",
    }
    raw = _required_object(raw, "configuration", keys, keys)
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise ValueError("configuration schema_version must be 2")
    if raw["application"] != "gstc_surface":
        raise ValueError("configuration is for a different application")
    _identifier(raw["case_id"], "case_id")
    physical = _required_object(
        raw["physical"],
        "physical",
        {"frequency_hz", "interior", "exterior"},
        {"frequency_hz", "interior", "exterior"},
    )
    media = {
        region: _required_object(
            physical[region], region, {"epsilon_r", "mu_r"}, {"epsilon_r", "mu_r"}
        )
        for region in ("interior", "exterior")
    }
    case = PhysicalConfiguration(
        physical["frequency_hz"],
        media["interior"]["epsilon_r"],
        media["interior"]["mu_r"],
        media["exterior"]["epsilon_r"],
        media["exterior"]["mu_r"],
    )
    geometry_raw = raw["geometry"]
    geometry_types = {"tangent_cone_sphere": TangentConeSphere, "circular_ogive": CircularOgive}
    if not isinstance(geometry_raw, dict) or not isinstance(geometry_raw.get("type"), str):
        raise ValueError("geometry must be an object with a type string")
    if geometry_raw["type"] not in geometry_types:
        raise ValueError("unsupported geometry.type")
    geometry_type = geometry_types[geometry_raw["type"]]
    geometry_keys = {item.name for item in dataclass_fields(geometry_type)} | {"type"}
    geometry_raw = _required_object(geometry_raw, "geometry", geometry_keys, geometry_keys)
    geometry = geometry_type(
        **{name: value for name, value in geometry_raw.items() if name != "type"}
    )
    if not isinstance(raw["fields"], dict):
        raise ValueError("fields must be a JSON object")
    catalog = {}
    for name, definition in raw["fields"].items():
        _identifier(name, "field id")
        catalog[name] = _parse_field_definition(definition, f"fields.{name}")
    transforms = tuple(
        _parse_transformation(value, str(i))
        for i, value in enumerate(_json_array(raw["transformations"], "transformations"))
    )
    if not transforms or len({item.id for item in transforms}) != len(transforms):
        raise ValueError("transformations must be nonempty with unique ids")
    simulation_raw = _required_object(raw["simulation"], "simulation", {"cases"}, {"cases"})
    simulation = SimulationConfiguration(
        tuple(
            _parse_simulation_case(value, str(i))
            for i, value in enumerate(_json_array(simulation_raw["cases"], "simulation.cases"))
        )
    )
    for transform in transforms:
        for reference in (transform.gstc.interior, transform.gstc.exterior):
            if reference is not None and reference.field_id not in catalog:
                raise ValueError(f"unknown transformation field {reference.field_id}")
    for simulation_case in simulation.cases:
        for excitation in simulation_case.excitations:
            if excitation.field.field_id not in catalog:
                raise ValueError(f"unknown excitation field {excitation.field.field_id}")
            definition = catalog[excitation.field.field_id]
            if isinstance(definition, DipoleFieldDefinition):
                inside = bool(geometry.contains(definition.source_position))
                if inside != (excitation.region == "interior"):
                    raise ValueError(
                        f"dipole source position disagrees with region in {simulation_case.id}"
                    )
    if raw["extraction_method"] not in ("reciprocal_lossless", "reciprocal_complex"):
        raise ValueError("unsupported extraction_method")
    if raw["tensor_structure"] not in ("general", "diagonal"):
        raise ValueError("unsupported tensor_structure")
    sie_keys = {item.name for item in dataclass_fields(SIEConfiguration)}
    sie_raw = _required_object(raw["sie"], "sie", sie_keys, sie_keys)
    mesh_required = {item.name for item in dataclass_fields(SIEMeshConfiguration)}
    mesh_raw = _required_object(
        sie_raw["mesh"], "sie.mesh", mesh_required | {"tip_grading"}, mesh_required
    )
    grading = None
    if "tip_grading" in mesh_raw:
        grading_keys = {item.name for item in dataclass_fields(OgiveTipGrading)}
        grading = OgiveTipGrading(
            **_required_object(
                mesh_raw["tip_grading"], "sie.mesh.tip_grading", grading_keys, grading_keys
            )
        )
        if not isinstance(geometry, CircularOgive):
            raise ValueError("sie.mesh.tip_grading requires circular_ogive geometry")
    mesh = SurfaceMeshConfiguration(
        **{name: mesh_raw[name] for name in mesh_required}, tip_grading=grading
    )
    sie = SIEConfiguration(
        mesh=mesh, **{name: value for name, value in sie_raw.items() if name != "mesh"}
    )
    if sie.basis != "face":
        raise ValueError("gstc_surface preparation requires sie.basis=face")
    sampling_keys = {
        "zx_grid_size",
        "zx_x_limits_m",
        "zx_z_limits_m",
        "radial_points",
        "radial_theta_degrees",
        "radial_phi_degrees",
        "radial_min_m",
        "radial_max_m",
    }
    sampling = _required_object(raw["sampling"], "sampling", sampling_keys, sampling_keys)
    for name in ("zx_grid_size", "radial_points"):
        if type(sampling[name]) is not int or sampling[name] < 2:
            raise ValueError(f"sampling.{name} must be an integer of at least two")
    if sampling["zx_grid_size"] % 2:
        raise ValueError("sampling.zx_grid_size must be even")
    for name in ("zx_x_limits_m", "zx_z_limits_m"):
        bounds = sampling[name]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"sampling.{name} must contain two increasing finite bounds")
    for name in ("radial_min_m", "radial_max_m"):
        _positive(sampling[name], f"sampling.{name}")
    if sampling["radial_min_m"] >= sampling["radial_max_m"]:
        raise ValueError("sampling radial bounds must be increasing")
    for name in ("radial_theta_degrees", "radial_phi_degrees"):
        if type(sampling[name]) not in (int, float) or not math.isfinite(sampling[name]):
            raise ValueError(f"sampling.{name} must be finite")
    if not 0 <= sampling["radial_theta_degrees"] <= 180:
        raise ValueError("sampling.radial_theta_degrees must lie in [0, 180]")
    return WorkflowConfiguration(
        raw["case_id"],
        case,
        geometry,
        MappingProxyType(catalog),
        transforms,
        simulation,
        raw["extraction_method"],
        raw["tensor_structure"],
        sie,
        copy.deepcopy(raw),
    )
