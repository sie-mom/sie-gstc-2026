"""Shared configuration and result paths for the omega-sphere workflows.

This module contains only solver-independent application inputs plus the
numerical settings for each backend.  Both the SVWF and future SIE drivers use
the same physical case and sampling configuration.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from scipy.constants import c, epsilon_0, mu_0

ExtractionMethod = Literal["projected", "least_squares"]
SIESolver = Literal["dense_lu", "gmres"]
SIEScaling = Literal["none", "eta0"]

APPLICATION_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_ID = "budhu_10ghz_eps7"

FIELD_COLUMNS = (
    "point_index",
    "x",
    "y",
    "z",
    "region",
    "Ex_real",
    "Ex_imag",
    "Ey_real",
    "Ey_imag",
    "Ez_real",
    "Ez_imag",
    "Hx_real",
    "Hx_imag",
    "Hy_real",
    "Hy_imag",
    "Hz_real",
    "Hz_imag",
)


@dataclass(frozen=True)
class DipoleMatchingCase:
    """Physical parameters shared by every solver for this application."""

    frequency: float = 10.0e9
    radius_in_free_space_wavelengths: float = 0.7
    relative_permittivity_interior: float = 7.0
    relative_permeability_interior: float = 1.0
    relative_permittivity_exterior: float = 1.0
    relative_permeability_exterior: float = 1.0
    effective_moment_interior: float = 1.0

    @property
    def angular_frequency(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def radius(self) -> float:
        return self.radius_in_free_space_wavelengths * c / self.frequency

    @property
    def epsilon_interior(self) -> float:
        return self.relative_permittivity_interior * epsilon_0

    @property
    def mu_interior(self) -> float:
        return self.relative_permeability_interior * mu_0

    @property
    def epsilon_exterior(self) -> float:
        return self.relative_permittivity_exterior * epsilon_0

    @property
    def mu_exterior(self) -> float:
        return self.relative_permeability_exterior * mu_0

    @property
    def k_interior(self) -> float:
        return self.angular_frequency * np.sqrt(
            self.mu_interior * self.epsilon_interior
        )

    @property
    def k_exterior(self) -> float:
        return self.angular_frequency * np.sqrt(
            self.mu_exterior * self.epsilon_exterior
        )

    @property
    def eta_interior(self) -> float:
        return np.sqrt(self.mu_interior / self.epsilon_interior)

    @property
    def eta_exterior(self) -> float:
        return np.sqrt(self.mu_exterior / self.epsilon_exterior)

    @property
    def effective_moment_ratio(self) -> float:
        """Exterior/interior ratio that conserves radial real power locally."""

        return (self.k_interior / self.k_exterior) ** 2 * np.sqrt(
            self.eta_exterior / self.eta_interior
        )

    @property
    def effective_moment_exterior(self) -> float:
        return self.effective_moment_interior * self.effective_moment_ratio


@dataclass(frozen=True)
class SamplingConfiguration:
    """Observation points shared by the SVWF and SIE comparisons."""

    zx_grid_size: int = 200
    zx_span_radii: float = 2.0
    radial_points: int = 300
    radial_theta_degrees: float = 45.0
    radial_phi_degrees: float = 0.0
    radial_min_radii: float = 0.05
    radial_max_radii: float = 2.0


@dataclass(frozen=True)
class SVWFConfiguration:
    """Numerical settings specific to the SVWF collocation backend."""

    n_max: int = 4
    n_theta: int = 18
    n_phi: int = 12


@dataclass(frozen=True)
class SIECapMeshRefinementConfiguration:
    """Local Gmsh sizing around a configured cap perturbation."""

    points_per_wavelength: float = 32.0
    fully_refined_radius_degrees: float = 5.5
    transition_radius_degrees: float = 10.0


@dataclass(frozen=True)
class SIEMeshConfiguration:
    """Gmsh/OBJ mesh settings for the SIE backend.

    Relative paths are interpreted from the omega application directory.
    Mesh sizing uses the shortest wavelength of the two media.
    """

    path: str = "meshes/budhu_10ghz_eps7/sphere.obj"
    points_per_wavelength: float = 4.0
    cap_refinement: SIECapMeshRefinementConfiguration | None = None


@dataclass(frozen=True)
class SIEConfiguration:
    """Numerical settings reserved for the SIE backend."""

    mesh: SIEMeshConfiguration = field(default_factory=SIEMeshConfiguration)
    quadrature_rule: int = 17
    solver: SIESolver = "dense_lu"
    scaling: SIEScaling = "eta0"


@dataclass(frozen=True)
class CapPerturbationConfiguration:
    """Scale all three omega parameters within a spherical cap.

    The angular radius is measured from the cap center, in degrees; the
    diameter is twice this value. A real scale in (0, 1] preserves the
    reciprocal lossless structure and permits the SIE matrix inverse.
    """

    theta_degrees: float = 90.0
    phi_degrees: float = 0.0
    angular_radius_degrees: float = 5.0
    parameter_scale: float = 0.1

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("cap_perturbation values must be finite")
        if not 0.0 <= self.theta_degrees <= 180.0:
            raise ValueError("cap_perturbation.theta_degrees must be in [0, 180]")
        if not 0.0 < self.angular_radius_degrees <= 180.0:
            raise ValueError(
                "cap_perturbation.angular_radius_degrees must be in (0, 180]"
            )
        if not 0.0 < self.parameter_scale <= 1.0:
            raise ValueError("cap_perturbation.parameter_scale must be in (0, 1]")


@dataclass(frozen=True)
class WorkflowConfiguration:
    """Complete shared configuration for one physical application case."""

    case_id: str = DEFAULT_CASE_ID
    case: DipoleMatchingCase = field(default_factory=DipoleMatchingCase)
    extraction_method: ExtractionMethod = "least_squares"
    sampling: SamplingConfiguration = field(default_factory=SamplingConfiguration)
    svwf: SVWFConfiguration = field(default_factory=SVWFConfiguration)
    sie: SIEConfiguration = field(default_factory=SIEConfiguration)
    cap_perturbation: CapPerturbationConfiguration | None = None

    @property
    def reference_label(self) -> str:
        if (
            self.cap_perturbation is not None
            and self.cap_perturbation.parameter_scale != 1.0
        ):
            return "Unperturbed reference"
        return "Prescribed"

    def __post_init__(self) -> None:
        _validate_identifier(self.case_id, "case_id")
        if self.extraction_method not in ("projected", "least_squares"):
            raise ValueError(
                "extraction_method must be 'projected' or 'least_squares'"
            )
        if self.case.frequency <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.case.radius_in_free_space_wavelengths <= 0.0:
            raise ValueError("radius_lambda0 must be positive")
        material_values = (
            self.case.relative_permittivity_interior,
            self.case.relative_permeability_interior,
            self.case.relative_permittivity_exterior,
            self.case.relative_permeability_exterior,
        )
        if any(value <= 0.0 for value in material_values):
            raise ValueError(
                "relative permittivities and permeabilities must be positive"
            )
        if self.svwf.n_max < 1:
            raise ValueError("svwf.n_max must be at least one")
        if self.svwf.n_theta < 2 or self.svwf.n_phi < 2:
            raise ValueError("svwf.n_theta and svwf.n_phi must be at least two")
        if not self.sie.mesh.path:
            raise ValueError("sie.mesh.path must not be empty")
        if self.sie.mesh.points_per_wavelength <= 0.0:
            raise ValueError("sie.mesh.points_per_wavelength must be positive")
        cap_refinement = self.sie.mesh.cap_refinement
        if cap_refinement is not None:
            if self.cap_perturbation is None:
                raise ValueError(
                    "sie.mesh.cap_refinement requires cap_perturbation"
                )
            refinement_values = asdict(cap_refinement).values()
            if not all(np.isfinite(value) for value in refinement_values):
                raise ValueError("sie.mesh.cap_refinement values must be finite")
            if (
                cap_refinement.points_per_wavelength
                < self.sie.mesh.points_per_wavelength
            ):
                raise ValueError(
                    "sie.mesh.cap_refinement.points_per_wavelength must be at "
                    "least the global mesh value"
                )
            if (
                cap_refinement.fully_refined_radius_degrees
                < self.cap_perturbation.angular_radius_degrees
            ):
                raise ValueError(
                    "sie.mesh.cap_refinement.fully_refined_radius_degrees must "
                    "include the complete physical cap"
                )
            if not (
                cap_refinement.fully_refined_radius_degrees
                < cap_refinement.transition_radius_degrees
                <= 180.0
            ):
                raise ValueError(
                    "sie.mesh.cap_refinement.transition_radius_degrees must be "
                    "greater than the fully refined radius and at most 180"
                )
        if self.sie.quadrature_rule < 1:
            raise ValueError("sie.quadrature_rule must be positive")
        if self.sie.solver not in ("dense_lu", "gmres"):
            raise ValueError("sie.solver must be 'dense_lu' or 'gmres'")
        if self.sie.scaling not in ("none", "eta0"):
            raise ValueError("sie.scaling must be 'none' or 'eta0'")
        if self.sampling.zx_grid_size < 2:
            raise ValueError("sampling.zx_grid_size must be at least two")
        if self.sampling.radial_points < 2:
            raise ValueError("sampling.radial_points must be at least two")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation written into each run."""

        result = {
            "schema_version": 1,
            "application": "dipole_impedance_matching",
            "case_id": self.case_id,
            "physical": {
                "frequency_hz": self.case.frequency,
                "radius_lambda0": self.case.radius_in_free_space_wavelengths,
                "interior": {
                    "epsilon_r": self.case.relative_permittivity_interior,
                    "mu_r": self.case.relative_permeability_interior,
                },
                "exterior": {
                    "epsilon_r": self.case.relative_permittivity_exterior,
                    "mu_r": self.case.relative_permeability_exterior,
                },
                "source": {
                    "effective_moment_interior": self.case.effective_moment_interior,
                },
            },
            "extraction_method": self.extraction_method,
            "sampling": {
                "zx_grid_size": self.sampling.zx_grid_size,
                "zx_span_radii": self.sampling.zx_span_radii,
                "radial_points": self.sampling.radial_points,
                "radial_theta_degrees": self.sampling.radial_theta_degrees,
                "radial_phi_degrees": self.sampling.radial_phi_degrees,
                "radial_min_radii": self.sampling.radial_min_radii,
                "radial_max_radii": self.sampling.radial_max_radii,
            },
            "svwf": {
                "n_max": self.svwf.n_max,
                "n_theta": self.svwf.n_theta,
                "n_phi": self.svwf.n_phi,
            },
            "sie": {
                "mesh": {
                    "path": self.sie.mesh.path,
                    "points_per_wavelength": self.sie.mesh.points_per_wavelength,
                },
                "quadrature_rule": self.sie.quadrature_rule,
                "solver": self.sie.solver,
                "scaling": self.sie.scaling,
            },
        }
        if self.sie.mesh.cap_refinement is not None:
            result["sie"]["mesh"]["cap_refinement"] = asdict(
                self.sie.mesh.cap_refinement
            )
        if self.cap_perturbation is not None:
            result["cap_perturbation"] = asdict(self.cap_perturbation)
        return result


def default_run_directory(
    case_id: str,
    backend: Literal["svwf", "sie", "comsol", "comparisons"],
    run_name: str,
) -> Path:
    """Return ``results/<case>/<backend>/<run>`` after validating each name."""

    _validate_identifier(case_id, "case_id")
    _validate_identifier(run_name, "run_name")
    return APPLICATION_DIR / "results" / case_id / backend / run_name


def load_field_csv(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the Cartesian complex-field CSV contract."""

    table = np.genfromtxt(path, delimiter=",", names=True)
    if table.dtype.names != FIELD_COLUMNS:
        raise ValueError(
            f"{path} has columns {table.dtype.names}; expected {FIELD_COLUMNS}"
        )
    return {name: np.atleast_1d(table[name]) for name in FIELD_COLUMNS}


def write_field_csv(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    region: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
) -> None:
    """Write Cartesian complex fields using the shared backend-neutral schema."""

    x, y, z, region = np.broadcast_arrays(x, y, z, region)
    expected_shape = (3, x.size)
    if electric.shape != expected_shape or magnetic.shape != expected_shape:
        raise ValueError(
            "electric and magnetic fields must have shape "
            f"{expected_shape}; received {electric.shape} and {magnetic.shape}"
        )

    columns = np.column_stack(
        (
            np.arange(x.size),
            x.ravel(),
            y.ravel(),
            z.ravel(),
            region.ravel(),
            electric[0].real,
            electric[0].imag,
            electric[1].real,
            electric[1].imag,
            electric[2].real,
            electric[2].imag,
            magnetic[0].real,
            magnetic[0].imag,
            magnetic[1].real,
            magnetic[1].imag,
            magnetic[2].real,
            magnetic[2].imag,
        )
    )
    formats = ["%d"] + ["%.17e"] * 3 + ["%d"] + ["%.17e"] * 12
    np.savetxt(
        path,
        columns,
        delimiter=",",
        header=",".join(FIELD_COLUMNS),
        comments="",
        fmt=formats,
    )


def _validate_identifier(value: str, name: str) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(
            f"{name} must contain only lowercase letters, digits, '-' and '_'"
        )


def _validate_configuration_keys(
    values: dict[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown {context} configuration field(s): {names}")


def _object_section(
    parent: dict[str, Any], name: str, allowed: set[str], context: str = ""
) -> dict[str, Any]:
    section = parent.get(name, {})
    qualified = f"{context}.{name}" if context else name
    if not isinstance(section, dict):
        raise ValueError(f"{qualified} must be a JSON object")
    _validate_configuration_keys(section, allowed, qualified)
    return section


def load_workflow_configuration(path: Path) -> WorkflowConfiguration:
    """Load a possibly partial shared JSON configuration over the defaults."""

    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")

    _validate_configuration_keys(
        raw,
        {
            "schema_version",
            "application",
            "case_id",
            "physical",
            "extraction_method",
            "sampling",
            "svwf",
            "sie",
            "cap_perturbation",
        },
        "top-level",
    )
    if raw.get("schema_version", 1) != 1:
        raise ValueError("only configuration schema_version 1 is supported")
    if raw.get("application", "dipole_impedance_matching") != (
        "dipole_impedance_matching"
    ):
        raise ValueError("configuration is for a different application")

    defaults = WorkflowConfiguration()
    cap = None
    if raw.get("cap_perturbation") is not None:
        cap_raw = _object_section(
            raw,
            "cap_perturbation",
            {
                "theta_degrees", "phi_degrees",
                "angular_radius_degrees", "parameter_scale",
            },
        )
        cap = CapPerturbationConfiguration(
            **{name: float(value) for name, value in cap_raw.items()}
        )
    physical = _object_section(
        raw,
        "physical",
        {"frequency_hz", "radius_lambda0", "interior", "exterior", "source"},
    )
    interior = _object_section(
        physical, "interior", {"epsilon_r", "mu_r"}, "physical"
    )
    exterior = _object_section(
        physical, "exterior", {"epsilon_r", "mu_r"}, "physical"
    )
    source = _object_section(
        physical, "source", {"effective_moment_interior"}, "physical"
    )
    case = DipoleMatchingCase(
        frequency=float(physical.get("frequency_hz", defaults.case.frequency)),
        radius_in_free_space_wavelengths=float(
            physical.get(
                "radius_lambda0", defaults.case.radius_in_free_space_wavelengths
            )
        ),
        relative_permittivity_interior=float(
            interior.get(
                "epsilon_r", defaults.case.relative_permittivity_interior
            )
        ),
        relative_permeability_interior=float(
            interior.get(
                "mu_r", defaults.case.relative_permeability_interior
            )
        ),
        relative_permittivity_exterior=float(
            exterior.get(
                "epsilon_r", defaults.case.relative_permittivity_exterior
            )
        ),
        relative_permeability_exterior=float(
            exterior.get(
                "mu_r", defaults.case.relative_permeability_exterior
            )
        ),
        effective_moment_interior=float(
            source.get(
                "effective_moment_interior", defaults.case.effective_moment_interior
            )
        ),
    )

    sampling_raw = _object_section(
        raw,
        "sampling",
        {
            "zx_grid_size",
            "zx_span_radii",
            "radial_points",
            "radial_theta_degrees",
            "radial_phi_degrees",
            "radial_min_radii",
            "radial_max_radii",
        },
    )
    sampling = SamplingConfiguration(
        zx_grid_size=int(
            sampling_raw.get("zx_grid_size", defaults.sampling.zx_grid_size)
        ),
        zx_span_radii=float(
            sampling_raw.get("zx_span_radii", defaults.sampling.zx_span_radii)
        ),
        radial_points=int(
            sampling_raw.get("radial_points", defaults.sampling.radial_points)
        ),
        radial_theta_degrees=float(
            sampling_raw.get(
                "radial_theta_degrees", defaults.sampling.radial_theta_degrees
            )
        ),
        radial_phi_degrees=float(
            sampling_raw.get(
                "radial_phi_degrees", defaults.sampling.radial_phi_degrees
            )
        ),
        radial_min_radii=float(
            sampling_raw.get(
                "radial_min_radii", defaults.sampling.radial_min_radii
            )
        ),
        radial_max_radii=float(
            sampling_raw.get(
                "radial_max_radii", defaults.sampling.radial_max_radii
            )
        ),
    )

    svwf_raw = _object_section(
        raw, "svwf", {"n_max", "n_theta", "n_phi"}
    )
    svwf = SVWFConfiguration(
        n_max=int(svwf_raw.get("n_max", defaults.svwf.n_max)),
        n_theta=int(svwf_raw.get("n_theta", defaults.svwf.n_theta)),
        n_phi=int(svwf_raw.get("n_phi", defaults.svwf.n_phi)),
    )

    sie_raw = _object_section(
        raw, "sie", {"mesh", "quadrature_rule", "solver", "scaling"}
    )
    mesh_raw = _object_section(
        sie_raw,
        "mesh",
        {"path", "points_per_wavelength", "cap_refinement"},
        "sie",
    )
    cap_refinement = None
    if mesh_raw.get("cap_refinement") is not None:
        refinement_raw = _object_section(
            mesh_raw,
            "cap_refinement",
            {
                "points_per_wavelength",
                "fully_refined_radius_degrees",
                "transition_radius_degrees",
            },
            "sie.mesh",
        )
        cap_refinement = SIECapMeshRefinementConfiguration(
            **{name: float(value) for name, value in refinement_raw.items()}
        )
    sie = SIEConfiguration(
        mesh=SIEMeshConfiguration(
            path=str(mesh_raw.get("path", defaults.sie.mesh.path)),
            points_per_wavelength=float(
                mesh_raw.get(
                    "points_per_wavelength",
                    defaults.sie.mesh.points_per_wavelength,
                )
            ),
            cap_refinement=cap_refinement,
        ),
        quadrature_rule=int(
            sie_raw.get("quadrature_rule", defaults.sie.quadrature_rule)
        ),
        solver=cast(SIESolver, sie_raw.get("solver", defaults.sie.solver)),
        scaling=cast(SIEScaling, sie_raw.get("scaling", defaults.sie.scaling)),
    )

    return WorkflowConfiguration(
        case_id=str(raw.get("case_id", defaults.case_id)),
        case=case,
        extraction_method=cast(
            ExtractionMethod,
            raw.get("extraction_method", defaults.extraction_method),
        ),
        sampling=sampling,
        svwf=svwf,
        sie=sie,
        cap_perturbation=cap,
    )
