"""Configuration, validation, and paths for the spherical GSTC application."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from scipy.constants import epsilon_0, mu_0

from sie_mom.excitation import DipoleType

APPLICATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = APPLICATION_DIR.parents[1]
DEFAULT_CASE_ID = "cloaking_100mhz_radius1m_ppw8"
DEFAULT_CONFIGURATION = (
    APPLICATION_DIR / "configurations" / f"{DEFAULT_CASE_ID}_reciprocal_complex.json"
)
ExtractionMethod = Literal["reciprocal_complex", "reciprocal_lossless"]
TensorStructure = Literal["general", "diagonal"]
BasisType = Literal["spherical", "face"]


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"{name} must contain only lowercase letters, digits, '-' and '_'")


def _real_vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three Cartesian components")
    for component in value:
        _finite(component, name)
    return (float(value[0]), float(value[1]), float(value[2]))


def _complex_vector3(value: Any, name: str) -> tuple[complex, complex, complex]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three Cartesian components")
    for component in value:
        if (
            isinstance(component, bool)
            or not isinstance(component, (int, float, complex))
            or not math.isfinite(component.real)
            or not math.isfinite(component.imag)
        ):
            raise ValueError(f"{name} must contain finite numeric components")
    return (complex(value[0]), complex(value[1]), complex(value[2]))


@dataclass(frozen=True)
class SphereCase:
    frequency_hz: float = 1e8
    radius_m: float = 1.0
    relative_permittivity_interior: float = 1.0
    relative_permeability_interior: float = 1.0
    relative_permittivity_exterior: float = 1.0
    relative_permeability_exterior: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive(value, f"physical.{name}")

    @property
    def k_exterior(self) -> float:
        return (
            2.0
            * math.pi
            * self.frequency_hz
            * math.sqrt(
                epsilon_0
                * self.relative_permittivity_exterior
                * mu_0
                * self.relative_permeability_exterior
            )
        )

    @property
    def eta_exterior(self) -> float:
        return math.sqrt(
            mu_0
            * self.relative_permeability_exterior
            / (epsilon_0 * self.relative_permittivity_exterior)
        )

    @property
    def k_interior(self) -> float:
        return (
            2.0
            * math.pi
            * self.frequency_hz
            * math.sqrt(
                epsilon_0
                * self.relative_permittivity_interior
                * mu_0
                * self.relative_permeability_interior
            )
        )

    @property
    def eta_interior(self) -> float:
        return math.sqrt(
            mu_0
            * self.relative_permeability_interior
            / (epsilon_0 * self.relative_permittivity_interior)
        )


@dataclass(frozen=True)
class PlaneWaveFieldDefinition:
    """Region-independent Cartesian parameters for a transverse plane wave."""

    electric_amplitude: tuple[complex, complex, complex]
    propagation_direction: tuple[float, float, float]

    def __post_init__(self) -> None:
        electric = _complex_vector3(self.electric_amplitude, "plane_wave.electric_amplitude")
        direction = _real_vector3(self.propagation_direction, "plane_wave.propagation_direction")
        direction_norm = math.hypot(*direction)
        if direction_norm == 0.0 or not math.isfinite(direction_norm):
            raise ValueError(
                "plane_wave.propagation_direction must have a finite, nonzero magnitude"
            )
        electric_norm = math.hypot(*(abs(component) for component in electric))
        _finite(electric_norm, "plane_wave.electric_amplitude magnitude")
        longitudinal = sum(
            component * (axis / direction_norm) for component, axis in zip(electric, direction)
        )
        if abs(longitudinal) > 1e-12 * max(1.0, electric_norm):
            raise ValueError(
                "plane_wave.electric_amplitude must be transverse to propagation_direction"
            )
        object.__setattr__(self, "electric_amplitude", electric)
        object.__setattr__(self, "propagation_direction", direction)


@dataclass(frozen=True)
class DipoleFieldDefinition:
    """Region-independent Cartesian dipole parameters.

    The effective moment follows the shared excitation evaluator's
    normalization, rather than representing a physical SI dipole moment.
    """

    dipole_type: DipoleType
    effective_moment: tuple[complex, complex, complex]
    source_position: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.dipole_type not in ("electric", "magnetic"):
            raise ValueError("dipole.dipole_type must be 'electric' or 'magnetic'")
        moment = _complex_vector3(self.effective_moment, "dipole.effective_moment")
        position = _real_vector3(self.source_position, "dipole.source_position")
        object.__setattr__(self, "effective_moment", moment)
        object.__setattr__(self, "source_position", position)


FieldDefinition = PlaneWaveFieldDefinition | DipoleFieldDefinition


# fields
@dataclass(frozen=True)
class FieldReference:
    field_id: str

    def __post_init__(self) -> None:
        _identifier(self.field_id, "field reference")


# Each gstc tranformation, i.e pair of interior and exterior fields
@dataclass(frozen=True)
class GSTCFieldsConfiguration:
    interior: FieldReference | None
    exterior: FieldReference | None

    def __post_init__(self) -> None:
        for region in ("interior", "exterior"):
            reference = getattr(self, region)
            if reference is not None and not isinstance(reference, FieldReference):
                raise ValueError(f"gstc.{region} must be a FieldReference or None")


# Transformations
@dataclass(frozen=True)
class GSTCTransformationConfiguration:
    id: str
    gstc: GSTCFieldsConfiguration

    def __post_init__(self) -> None:
        _identifier(self.id, "transformation.id")
        if not isinstance(self.gstc, GSTCFieldsConfiguration):
            raise ValueError("transformation.gstc must be a GSTCFieldsConfiguration")


# excitation configuration
@dataclass(frozen=True)
class SimulationExcitationConfiguration:
    field: FieldReference
    region: Literal["interior", "exterior"]

    def __post_init__(self) -> None:
        if not isinstance(self.field, FieldReference):
            raise ValueError("simulation excitation.field must be a FieldReference")
        if self.region not in ("interior", "exterior"):
            raise ValueError("simulation excitation.region must be 'interior' or 'exterior'")


# SimulationCaseConfiguration
@dataclass(frozen=True)
class SimulationCaseConfiguration:
    id: str
    excitations: tuple[SimulationExcitationConfiguration, ...]

    def __post_init__(self) -> None:
        _identifier(self.id, "simulation case.id")
        if not isinstance(self.excitations, tuple) or not all(
            isinstance(item, SimulationExcitationConfiguration) for item in self.excitations
        ):
            raise ValueError(
                "simulation case.excitations must be a tuple of SimulationExcitationConfiguration"
            )

        if not 1 <= len(self.excitations) <= 2:
            raise ValueError("simulation case must contain one or two excitations")

        regions = {item.region for item in self.excitations}
        if len(regions) != len(self.excitations):
            raise ValueError("simulation case allows only one excitation per region")


# SimulationConfiguration
@dataclass(frozen=True)
class SimulationConfiguration:
    cases: tuple[SimulationCaseConfiguration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cases, tuple) or not all(
            isinstance(item, SimulationCaseConfiguration) for item in self.cases
        ):
            raise ValueError("simulation.cases must be a tuple of SimulationCaseConfiguration")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("simulation case IDs must be unique")


@dataclass(frozen=True)
class SamplingConfiguration:
    zx_grid_size: int = 100
    zx_span_radii: float = 3.0
    radial_points: int = 100
    radial_theta_degrees: float = 45.0
    radial_phi_degrees: float = 0.0
    radial_min_radii: float = 0.05
    radial_max_radii: float = 2.0

    def __post_init__(self) -> None:
        for name in ("zx_grid_size", "radial_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"sampling.{name} must be an integer of at least two")
        if self.zx_grid_size % 2:
            raise ValueError("sampling.zx_grid_size must be even")
        for name, value in asdict(self).items():
            _finite(value, f"sampling.{name}")
        if self.zx_span_radii <= 1.0 or self.radial_max_radii <= 1.0:
            raise ValueError("sampling.zx_span_radii and radial_max_radii must exceed one")
        if not 0.0 < self.radial_min_radii < 1.0:
            raise ValueError("sampling.radial_min_radii must lie strictly between zero and one")
        if not 0.0 <= self.radial_theta_degrees <= 180.0:
            raise ValueError("sampling.radial_theta_degrees must be in [0, 180]")


@dataclass(frozen=True)
class SIEMeshConfiguration:
    path: str = f"meshes/{DEFAULT_CASE_ID}/sphere.obj"
    points_per_wavelength: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("sie.mesh.path must be a nonempty string")
        _positive(self.points_per_wavelength, "sie.mesh.points_per_wavelength")


@dataclass(frozen=True)
class SIEConfiguration:
    mesh: SIEMeshConfiguration = field(default_factory=SIEMeshConfiguration)
    quadrature_rule: int = 17
    singularity_threshold_lambda0: float = 0.1
    basis: BasisType = "spherical"
    compute_residual: bool = True
    evaluate_fields: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.quadrature_rule, bool)
            or not isinstance(self.quadrature_rule, int)
            or not 1 <= self.quadrature_rule <= 20
        ):
            raise ValueError("sie.quadrature_rule must be an integer in [1, 20]")
        _positive(self.singularity_threshold_lambda0, "sie.singularity_threshold_lambda0")
        if self.basis not in ("spherical", "face"):
            raise ValueError("sie.basis must be 'spherical' or 'face'")
        if not isinstance(self.compute_residual, bool) or not isinstance(
            self.evaluate_fields, bool
        ):
            raise ValueError("sie.compute_residual and sie.evaluate_fields must be booleans")


@dataclass(frozen=True)
class WorkflowConfiguration:
    """The application configuration, with explicit fields and simulation cases."""

    fields: Mapping[str, FieldDefinition]
    transformations: tuple[GSTCTransformationConfiguration, ...]
    simulation: SimulationConfiguration

    case_id: str = DEFAULT_CASE_ID
    case: SphereCase = field(default_factory=SphereCase)
    extraction_method: ExtractionMethod = "reciprocal_complex"
    tensor_structure: TensorStructure = "general"
    sampling: SamplingConfiguration = field(default_factory=SamplingConfiguration)
    sie: SIEConfiguration = field(default_factory=SIEConfiguration)

    def __post_init__(self) -> None:
        _identifier(self.case_id, "case_id")
        if self.extraction_method not in ("reciprocal_complex", "reciprocal_lossless"):
            raise ValueError(
                "extraction_method must be 'reciprocal_complex' or 'reciprocal_lossless'"
            )
        if self.tensor_structure not in ("general", "diagonal"):
            raise ValueError("tensor_structure must be 'general' or 'diagonal'")

        for name, expected_type in (
            ("case", SphereCase),
            ("sampling", SamplingConfiguration),
            ("sie", SIEConfiguration),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise ValueError(f"{name} must be a {expected_type.__name__}")

        if not isinstance(self.fields, Mapping):
            raise ValueError("fields must be a mapping")
        catalog = dict(self.fields)
        for field_id, definition in catalog.items():
            _identifier(field_id, "fields identifier")
            if not isinstance(definition, FieldDefinition):
                raise ValueError(f"fields.{field_id} must be a supported field definition")
        if not isinstance(self.transformations, tuple) or not self.transformations:
            raise ValueError("transformations must be a nonempty tuple")

        if not all(
            isinstance(item, GSTCTransformationConfiguration) for item in self.transformations
        ):
            raise ValueError("transformations must contain GSTCTransformationConfiguration objects")

        transformation_ids = [item.id for item in self.transformations]
        if len(transformation_ids) != len(set(transformation_ids)):
            raise ValueError("transformation IDs must be unique")

        if not isinstance(self.simulation, SimulationConfiguration):
            raise ValueError("simulation must be a SimulationConfiguration")

        for transformation in self.transformations:
            for region in ("interior", "exterior"):
                reference = getattr(transformation.gstc, region)
                if reference is not None and reference.field_id not in catalog:
                    raise ValueError(
                        f"transformation '{transformation.id}' gstc.{region} "
                        f"references unknown field '{reference.field_id}'"
                    )

        for simulation_case in self.simulation.cases:
            for excitation in simulation_case.excitations:
                if excitation.field.field_id not in catalog:
                    raise ValueError(
                        f"simulation case '{simulation_case.id}' "
                        f"{excitation.region} excitation references unknown "
                        f"field '{excitation.field.field_id}'"
                    )

        object.__setattr__(self, "fields", MappingProxyType(catalog))

    @property
    def run_name(self) -> str:
        return f"{self.extraction_method}_{self.tensor_structure}"

    @property
    def mesh_path(self) -> Path:
        path = Path(self.sie.mesh.path)
        return path if path.is_absolute() else APPLICATION_DIR / path

    @property
    def shortest_wavelength(self) -> float:
        """Return the shortest wavelength between interior and exterior media."""
        n_max = math.sqrt(
            max(
                self.case.relative_permittivity_interior * self.case.relative_permeability_interior,
                self.case.relative_permittivity_exterior * self.case.relative_permeability_exterior,
            )
        )
        return 3e8 / self.case.frequency_hz / n_max

    @property
    def target_edge_length(self) -> float:
        return self.shortest_wavelength / self.sie.mesh.points_per_wavelength

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "application": "anisotropic_gstc_sphere",
            "case_id": self.case_id,
            "physical": _physical_to_dict(self.case),
            "fields": {
                field_id: _field_to_dict(definition) for field_id, definition in self.fields.items()
            },
            "transformations": [
                {
                    "id": transformation.id,
                    "gstc": {
                        "interior": _reference_to_json(transformation.gstc.interior),
                        "exterior": _reference_to_json(transformation.gstc.exterior),
                    },
                }
                for transformation in self.transformations
            ],
            "simulation": {
                "cases": [
                    {
                        "id": case.id,
                        "excitations": [
                            {"field": excitation.field.field_id, "region": excitation.region}
                            for excitation in case.excitations
                        ],
                    }
                    for case in self.simulation.cases
                ]
            },
            "extraction_method": self.extraction_method,
            "tensor_structure": self.tensor_structure,
            "sampling": asdict(self.sampling),
            "sie": asdict(self.sie),
        }


def _physical_to_dict(case: SphereCase) -> dict[str, Any]:
    return {
        "frequency_hz": case.frequency_hz,
        "radius_m": case.radius_m,
        "interior": {
            "epsilon_r": case.relative_permittivity_interior,
            "mu_r": case.relative_permeability_interior,
        },
        "exterior": {
            "epsilon_r": case.relative_permittivity_exterior,
            "mu_r": case.relative_permeability_exterior,
        },
    }


def _complex_vector_to_dict(value: tuple[complex, complex, complex]) -> dict[str, list[float]]:
    return {"real": [item.real for item in value], "imag": [item.imag for item in value]}


def _reference_to_json(reference: FieldReference | None) -> str | None:
    return reference.field_id if reference is not None else None


def _field_to_dict(definition: FieldDefinition) -> dict[str, Any]:
    if isinstance(definition, PlaneWaveFieldDefinition):
        return {
            "type": "plane_wave",
            "electric_amplitude": _complex_vector_to_dict(definition.electric_amplitude),
            "propagation_direction": list(definition.propagation_direction),
        }
    if isinstance(definition, DipoleFieldDefinition):
        return {
            "type": "dipole",
            "dipole_type": definition.dipole_type,
            "effective_moment": _complex_vector_to_dict(definition.effective_moment),
            "source_position": list(definition.source_position),
        }
    raise ValueError("cannot serialize an unsupported field definition")


def default_run_directory(case_id: str, run_name: str) -> Path:
    """Return the canonical ``results/<case_id>/sie/<run_name>`` path."""

    _identifier(case_id, "case_id")
    _identifier(run_name, "run_name")
    return APPLICATION_DIR / "results" / case_id / "sie" / run_name


def create_run_directory(run_dir: Path) -> None:
    """Protect saved runs from accidental overwrites and stale artifacts."""

    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; choose a new run name or path"
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _object(parent: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = parent.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} configuration field(s): {', '.join(unknown)}")
    return value


def _names(configuration_type: type) -> set[str]:
    return {item.name for item in fields(configuration_type)}


def _parse_settings(raw: dict[str, Any]) -> dict[str, Any]:
    physical = _object(raw, "physical", {"frequency_hz", "radius_m", "interior", "exterior"})
    interior = _object(physical, "interior", {"epsilon_r", "mu_r"})
    exterior = _object(physical, "exterior", {"epsilon_r", "mu_r"})
    sie = _object(raw, "sie", _names(SIEConfiguration))
    mesh = SIEMeshConfiguration(**_object(sie, "mesh", _names(SIEMeshConfiguration)))
    return {
        "case_id": raw.get("case_id", DEFAULT_CASE_ID),
        "case": SphereCase(
            frequency_hz=physical.get("frequency_hz", 1e8),
            radius_m=physical.get("radius_m", 1.0),
            relative_permittivity_interior=interior.get("epsilon_r", 1.0),
            relative_permeability_interior=interior.get("mu_r", 1.0),
            relative_permittivity_exterior=exterior.get("epsilon_r", 1.0),
            relative_permeability_exterior=exterior.get("mu_r", 1.0),
        ),
        "extraction_method": raw.get("extraction_method", "reciprocal_complex"),
        "tensor_structure": raw.get("tensor_structure", "general"),
        "sampling": SamplingConfiguration(
            **_object(raw, "sampling", _names(SamplingConfiguration))
        ),
        "sie": SIEConfiguration(
            mesh=mesh, **{name: value for name, value in sie.items() if name != "mesh"}
        ),
    }


def _required_object(
    value: Any, name: str, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    result = _object({name: value}, name, allowed)
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")
    return result


def _json_array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _parse_complex_vector(value: Any, name: str) -> tuple[complex, complex, complex]:
    components = _required_object(value, name, {"real", "imag"}, {"real", "imag"})
    real = _real_vector3(components["real"], f"{name}.real")
    imag = _real_vector3(components["imag"], f"{name}.imag")
    return (complex(real[0], imag[0]), complex(real[1], imag[1]), complex(real[2], imag[2]))


def _parse_field_definition(value: Any, name: str) -> FieldDefinition:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if value.get("type") == "plane_wave":
        keys = {"type"} | _names(PlaneWaveFieldDefinition)
        raw = _required_object(value, name, keys, keys)
        return PlaneWaveFieldDefinition(
            electric_amplitude=_parse_complex_vector(
                raw["electric_amplitude"], f"{name}.electric_amplitude"
            ),
            propagation_direction=_real_vector3(
                raw["propagation_direction"], f"{name}.propagation_direction"
            ),
        )
    if value.get("type") == "dipole":
        keys = {"type"} | _names(DipoleFieldDefinition)
        raw = _required_object(value, name, keys, keys)
        return DipoleFieldDefinition(
            dipole_type=raw["dipole_type"],
            effective_moment=_parse_complex_vector(
                raw["effective_moment"], f"{name}.effective_moment"
            ),
            source_position=_real_vector3(raw["source_position"], f"{name}.source_position"),
        )
    raise ValueError(f"{name}.type must be 'plane_wave' or 'dipole'")


def _parse_field_reference(value: Any, name: str) -> FieldReference:
    _identifier(value, name)
    return FieldReference(value)


def _parse_transformation(value: Any, name: str) -> GSTCTransformationConfiguration:
    raw = _required_object(value, name, {"id", "gstc"}, {"id", "gstc"})
    regions = {"interior", "exterior"}
    gstc = _required_object(raw["gstc"], f"{name}.gstc", regions, regions)
    return GSTCTransformationConfiguration(
        id=raw["id"],
        gstc=GSTCFieldsConfiguration(
            **{
                region: None
                if gstc[region] is None
                else _parse_field_reference(gstc[region], f"{name}.gstc.{region}")
                for region in ("interior", "exterior")
            }
        ),
    )


def _parse_simulation_case(value: Any, name: str) -> SimulationCaseConfiguration:
    raw = _required_object(value, name, {"id", "excitations"}, {"id", "excitations"})
    excitations = []
    for index, value in enumerate(_json_array(raw["excitations"], f"{name}.excitations")):
        path = f"{name}.excitations[{index}]"
        source = _required_object(value, path, {"field", "region"}, {"field", "region"})
        excitations.append(
            SimulationExcitationConfiguration(
                field=_parse_field_reference(source["field"], f"{path}.field"),
                region=source["region"],
            )
        )
    return SimulationCaseConfiguration(id=raw["id"], excitations=tuple(excitations))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_workflow_configuration(path: Path) -> WorkflowConfiguration:
    """Load and validate the sole supported configuration schema (version 2)."""

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_unique_json_object)
    raw = _required_object(
        value,
        "configuration",
        {
            "schema_version",
            "application",
            "case_id",
            "physical",
            "fields",
            "transformations",
            "simulation",
            "extraction_method",
            "tensor_structure",
            "sampling",
            "sie",
        },
        {"schema_version", "fields", "transformations", "simulation"},
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise ValueError("configuration schema_version must be 2")
    if raw.get("application", "anisotropic_gstc_sphere") != "anisotropic_gstc_sphere":
        raise ValueError("configuration is for a different application")
    if not isinstance(raw["fields"], dict):
        raise ValueError("fields must be a JSON object")
    simulation = _required_object(raw["simulation"], "simulation", {"cases"}, {"cases"})
    return WorkflowConfiguration(
        fields={
            field_id: _parse_field_definition(definition, f"fields.{field_id}")
            for field_id, definition in raw["fields"].items()
        },
        transformations=tuple(
            _parse_transformation(value, f"transformations[{index}]")
            for index, value in enumerate(_json_array(raw["transformations"], "transformations"))
        ),
        simulation=SimulationConfiguration(
            tuple(
                _parse_simulation_case(value, f"simulation.cases[{index}]")
                for index, value in enumerate(_json_array(simulation["cases"], "simulation.cases"))
            )
        ),
        **_parse_settings(raw),
    )
