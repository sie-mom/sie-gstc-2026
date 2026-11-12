"""Joint SVWF synthesis and independent forward checks of saved surfaces.

Fit the prescribed exterior scattered E/H with outgoing a,b, then keep those
coefficients fixed while optimizing a shared surface and regular interior c,d.
All optimization fields use exp(-j omega t), as in spherical_expansion. Saved
surface blocks and sampled fields use the application's exp(+j omega t).
The forward-only mode solves all a,b,c,d with a fixed saved material, without
constructing an inverse problem or running extraction or optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares

from sie_mom.gstc import extract_reciprocal_lossless_least_squares, form_boundary_quantities
from sie_mom.gstc.omega_biisotropic import normal_power_density
from sie_mom.gstc.surface_parameters import SurfaceParameters
from spherical_expansion.coefficients import save_coefficients
from spherical_expansion.solver import solve_collocation_general
from spherical_expansion.svwf_sphere import Y_nm, compute_fields, get_M_N, get_radial_functions

if __package__:
    from ...prescribed_fields import evaluate_prescribed_fields
    from ...workflow import (
        APPLICATION_DIR,
        create_run_directory,
        load_workflow_configuration,
        write_json,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from prescribed_fields import evaluate_prescribed_fields
    from workflow import (
        APPLICATION_DIR,
        create_run_directory,
        load_workflow_configuration,
        write_json,
    )


# Edit numerical settings here; physical fields and media remain in the JSON.
DEFAULT_CONFIG = (
    APPLICATION_DIR
    / "configurations/moment_rot/rot90_eps7.json"
)
N_MAX = 6
N_THETA = 30
N_PHI = 60
ANGULAR_DEGREE = 12  # 0: constant local components; >0: allow azimuthal variation.
MAX_NFEV = 300
STATE_CASES = (
    ("electric_z_to_x", "electric_incidence"),
    ("magnetic_z_to_x", "magnetic_incidence"),
)
PARAMETER_NAMES = (
    "eta*Im(Y_theta_theta)", "eta*Im(Y_phi_phi)", "eta*Im(Y_theta_phi)",
    "Im(Z_theta_theta)/eta", "Im(Z_phi_phi)/eta", "Im(Z_theta_phi)/eta",
    "chi_theta_theta", "chi_theta_phi", "chi_phi_theta", "chi_phi_phi",
)
# [E_theta, E_phi, H_theta, H_phi] -> [J_theta, J_phi, M_theta, M_phi].
JUMP_TO_CURRENT = np.array(
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], float
)


@dataclass(frozen=True)
class SurfaceGrid:
    """Flattened spherical angles and solid-angle quadrature weights."""

    theta: np.ndarray
    phi: np.ndarray
    weights: np.ndarray


def require_integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def surface_grid(n_theta, n_phi, *, phi_offset=0.0):
    """Use Gauss-Legendre quadrature in cos(theta) and uniform azimuthal samples."""
    require_integer("n_theta", n_theta, 1)
    require_integer("n_phi", n_phi, 1)
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    theta, phi = np.meshgrid(
        np.arccos(nodes[::-1]),
        2 * np.pi * (np.arange(n_phi) + phi_offset) / n_phi,
        indexing="ij",
    )
    return SurfaceGrid(
        theta.ravel(), phi.ravel(), np.repeat(weights[::-1], n_phi) * 2 * np.pi / n_phi
    )


def angular_basis(theta, phi, degree):
    """Real spherical-harmonic profiles, with the first profile exactly one."""
    require_integer("angular_degree", degree, 0)
    theta, phi = np.broadcast_arrays(theta, phi)
    values = [np.ones_like(theta)]
    names = ["constant"]
    for n in range(1, degree + 1):
        values.append(np.sqrt(4 * np.pi) * Y_nm(n, 0, theta, phi).real)
        names.append(f"Y_{n}_0")
        for m in range(1, n + 1):
            harmonic = np.sqrt(8 * np.pi) * Y_nm(n, m, theta, phi)
            values.extend((harmonic.real, harmonic.imag))
            names.extend((f"Re_Y_{n}_{m}", f"Im_Y_{n}_{m}"))
    return np.asarray(values), tuple(names)


def surface_parameters(coefficients, profiles, reference_impedance):
    """Evaluate Y, chi, gamma, Z in exp(+j omega t), preserving material constraints.

    Ten rows of real angular coefficients define symmetric imaginary Y/Z and
    real chi, with gamma = -chi.T. Numerical coefficients are dimensionless:
    Y is divided by eta_ref, Z multiplied by eta_ref. Source amplitudes are
    unchanged. Output block shapes are (2, 2, number_of_points).
    """
    coefficients = np.asarray(coefficients)
    profiles = np.asarray(profiles)
    if np.iscomplexobj(coefficients) or not np.all(np.isfinite(coefficients)):
        raise ValueError("material coefficients must be finite and real")
    if profiles.ndim != 2 or np.iscomplexobj(profiles) or not np.all(np.isfinite(profiles)):
        raise ValueError("angular profiles must be a finite real two-dimensional array")
    if coefficients.shape != (10, profiles.shape[0]):
        raise ValueError("material coefficients must have shape (10, number_of_profiles)")
    if not np.isfinite(reference_impedance) or reference_impedance <= 0:
        raise ValueError("reference_impedance must be finite and positive")
    q = coefficients @ profiles
    electric = 1j * np.array([[q[0], q[2]], [q[2], q[1]]]) / reference_impedance
    magnetic = 1j * np.array([[q[3], q[5]], [q[5], q[4]]]) * reference_impedance
    coupling = np.array([[q[6], q[7]], [q[8], q[9]]], dtype=complex)
    return SurfaceParameters(electric, coupling, -coupling.swapaxes(0, 1), magnetic)


def load_parameter_evaluator(experiment_dir):
    """Load a saved inverse surface; return its configuration, evaluator, and provenance.

    The evaluator returns physical (2, 2, point_count) blocks in exp(+j omega t).
    Reconstruct from angular coefficients, not interpolation or another local fit.
    """
    experiment_dir = Path(experiment_dir).resolve()
    with (experiment_dir / "experiment.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("method") != "joint_svwf_reciprocal_lossless":
        raise ValueError("expected a joint SVWF reciprocal-lossless experiment")
    if metadata.get("status") != "completed":
        raise ValueError("the saved experiment must be completed before validation")
    if metadata.get("field_and_material_time_convention") != "exp(+j omega t)":
        raise ValueError("surface parameters must use exp(+j omega t)")
    degree = metadata.get("angular_degree")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("saved angular_degree must be a nonnegative integer")
    eta = metadata.get("reference_impedance_ohm")
    if (isinstance(eta, bool) or not isinstance(eta, (int, float))
            or not np.isfinite(eta) or eta <= 0):
        raise ValueError("saved reference impedance must be finite and positive")
    _, names = angular_basis(np.array([np.pi / 2]), np.array([0.0]), degree)
    if metadata.get("angular_profile_names") != list(names):
        raise ValueError("saved angular profile order is unsupported")
    if metadata.get("material_parameter_names") != list(PARAMETER_NAMES):
        raise ValueError("saved material parameter order is unsupported")
    with np.load(experiment_dir / "surface_parameters.npz", allow_pickle=False) as archive:
        if "angular_coefficients" not in archive.files:
            raise ValueError("surface archive is missing angular_coefficients")
        coefficients = np.asarray(archive["angular_coefficients"])
    if (coefficients.shape != (10, len(names)) or np.iscomplexobj(coefficients)
            or not np.isfinite(coefficients).all()):
        raise ValueError(
            f"angular_coefficients must be finite and real with shape (10, {len(names)})"
        )
    configuration = load_workflow_configuration(experiment_dir / "configuration.json")

    def evaluate_saved_parameters(theta, phi):
        profiles, _ = angular_basis(theta, phi, degree)
        return surface_parameters(coefficients, profiles, eta)

    source = {
        "method": "svwf_inverse", "experiment_dir": str(experiment_dir),
        "angular_degree": degree, "reference_impedance_ohm": eta,
        "material_constraints": "reciprocal_lossless",
        "phasor_convention": "exp(+j omega t)",
        "sha256": {
            name: hashlib.sha256((experiment_dir / name).read_bytes()).hexdigest()
            for name in ("configuration.json", "experiment.json", "surface_parameters.npz")
        },
    }
    return configuration, evaluate_saved_parameters, source


def validate_state_cases(configuration, state_cases):
    """Resolve explicit transformation/case pairs, preserving their saved order."""
    if not isinstance(state_cases, (tuple, list)) or any(
        not isinstance(pair, (tuple, list)) or len(pair) != 2
        or not all(isinstance(value, str) for value in pair) for pair in state_cases
    ):
        raise ValueError("STATE_CASES must contain transformation/case pairs")
    state_ids = [state for state, _ in state_cases]
    case_ids = [case for _, case in state_cases]
    if not state_ids or len(set(state_ids)) != len(state_ids) or set(state_ids) != {
        state.id for state in configuration.transformations
    }:
        raise ValueError("STATE_CASES must list every transformation exactly once")
    if len(set(case_ids)) != len(case_ids) or set(case_ids) != {
        case.id for case in configuration.simulation.cases
    }:
        raise ValueError("STATE_CASES must list every simulation case exactly once")
    return tuple(tuple(pair) for pair in state_cases)


def tangential_basis(n_max, k, eta, radius, grid, kind):
    """Cache existing SVWFs as (4, point, coefficient), interleaving M/N per mode."""
    require_integer("n_max", n_max, 1)
    modes = tuple((m, n) for n in range(1, n_max + 1) for m in range(-n, n + 1))
    matrix = np.empty((4, grid.theta.size, 2 * len(modes)), complex)
    radial = get_radial_functions(n_max, k * radius, kind)
    for index, (m, n) in enumerate(modes):
        electric_m, electric_n = get_M_N(
            n, m, k, radius, grid.theta, grid.phi, kind=kind, grid=False,
            radial_functions=radial,
        )
        matrix[:, :, 2 * index] = np.array(
            [electric_m[1], electric_m[2], electric_n[1] / (1j * eta),
             electric_n[2] / (1j * eta)]
        )
        matrix[:, :, 2 * index + 1] = np.array(
            [electric_n[1], electric_n[2], electric_m[1] / (1j * eta),
             electric_m[2] / (1j * eta)]
        )
    return modes, matrix


def field_weights(grid, eta):
    return np.array([1, 1, eta, eta])[:, None] * np.sqrt(grid.weights)[None, :]


def field_norm(field, grid, eta):
    return float(np.linalg.norm(field * field_weights(grid, eta)))


def relative_field_error(actual, target, grid, eta, reference):
    """Normalize by the target, or by incident fields when the target is zero."""
    denominator = field_norm(target, grid, eta)
    if denominator == 0:
        denominator = field_norm(reference, grid, eta)
    return field_norm(actual - target, grid, eta) / max(denominator, np.finfo(float).tiny)


def fit_trace(basis, target, grid, eta):
    """Fit both tangential E and H using balanced, area-weighted complex LS."""
    weights = field_weights(grid, eta)
    matrix = (basis * weights[:, :, None]).reshape(-1, basis.shape[-1])
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("SVWF basis has zero or nonfinite column norms")
    values, _, rank, _ = lstsq(
        matrix / norms, (target * weights).ravel(), cond=1e-12, lapack_driver="gelsy"
    )
    coefficients = values / norms
    return coefficients, np.einsum("ipk,k->ip", basis, coefficients), int(rank)


def reference_trace(reference, region, configuration, grid):
    """Convert configured total E/H to SVWF phasors; retain theta/phi components."""
    electric, magnetic = evaluate_prescribed_fields(
        reference, region, configuration, radius=configuration.case.radius_m,
        theta=grid.theta, phi=grid.phi,
    )
    return np.concatenate((electric[1:], magnetic[1:])).conj()


def incident_trace(configuration, simulation_case, region, grid):
    """Sum only the configured incident fields belonging to this solver region."""
    result = np.zeros((4, grid.theta.size), complex)
    for excitation in simulation_case.excitations:
        if excitation.region == region:
            result += reference_trace(excitation.field, region, configuration, grid)
    return result


@dataclass(frozen=True)
class InverseProblem:
    """Cached field traces and a real-variable joint GSTC residual problem."""

    configuration: object
    state_cases: tuple
    n_max: int
    angular_degree: int
    grid: SurfaceGrid
    modes: tuple
    profiles: np.ndarray
    profile_names: tuple
    material_basis: np.ndarray
    interior_basis: np.ndarray
    interior_column_norms: np.ndarray
    incident_interior: np.ndarray
    incident_exterior: np.ndarray
    prescribed_interior: np.ndarray
    target_exterior: np.ndarray
    fitted_exterior: np.ndarray
    target_coefficients: np.ndarray
    target_fit_errors: tuple
    residual_weights: np.ndarray

    @property
    def material_size(self):
        return 10 * self.profiles.shape[0]

    @property
    def interior_shape(self):
        return (len(self.state_cases), self.interior_basis.shape[-1])

    def pack(self, material, interior):
        return np.concatenate((np.ravel(material), interior.real.ravel(), interior.imag.ravel()))

    def unpack(self, values):
        material_size = self.material_size
        interior_size = int(np.prod(self.interior_shape))
        if np.shape(values) != (material_size + 2 * interior_size,):
            raise ValueError("incorrect optimization vector shape")
        material = values[:material_size].reshape(10, -1)
        interior = (
            values[material_size:material_size + interior_size]
            + 1j * values[material_size + interior_size:]
        ).reshape(self.interior_shape)
        return material, interior

    def fields_and_map(self, values):
        material, coefficients = self.unpack(values)
        surface_map = np.einsum("ta,ap,tij->ijp", material, self.profiles, self.material_basis)
        interior = self.incident_interior + np.einsum(
            "ipk,sk->sip", self.interior_basis, coefficients
        )
        return interior, self.fitted_exterior, surface_map

    def complex_residual(self, values):
        """Return weighted J/M minus the material map applied to average E/H."""
        interior, exterior, surface_map = self.fields_and_map(values)
        currents = np.einsum("ij,sjp->sip", JUMP_TO_CURRENT, exterior - interior)
        reconstructed = np.einsum("ijp,sjp->sip", surface_map, (exterior + interior) / 2)
        return (currents - reconstructed) * self.residual_weights

    def residual(self, values):
        residual = self.complex_residual(values).ravel()
        return np.concatenate((residual.real, residual.imag))

    def jacobian(self, values):
        """Differentiate the bilinear residual exactly in real optimization variables."""
        interior, exterior, surface_map = self.fields_and_map(values)
        state_count, _, point_count = interior.shape
        coefficient_count = self.interior_basis.shape[-1]
        jacobian = np.zeros((state_count, 4, point_count, values.size), complex)
        jacobian[:, :, :, :self.material_size] = -np.einsum(
            "tij,sjp,ap->sipta", self.material_basis, (exterior + interior) / 2,
            self.profiles,
        ).reshape(state_count, 4, point_count, self.material_size)
        interior_derivative = -np.einsum(
            "ij,jpk->ipk", JUMP_TO_CURRENT, self.interior_basis
        ) - 0.5 * np.einsum("ijp,jpk->ipk", surface_map, self.interior_basis)
        for state in range(state_count):
            start = self.material_size + state * coefficient_count
            stop = start + coefficient_count
            jacobian[state, :, :, start:stop] = interior_derivative
            imaginary_start = start + state_count * coefficient_count
            jacobian[state, :, :, imaginary_start:imaginary_start + coefficient_count] = (
                1j * interior_derivative
            )
        jacobian *= self.residual_weights[:, :, :, None]
        matrix = jacobian.reshape(-1, values.size)
        return np.concatenate((matrix.real, matrix.imag), axis=0)

    def initial_guess(self):
        """Project the existing local lossless extraction onto the angular material model."""
        states = [
            form_boundary_quantities(inside[:2].conj(), inside[2:].conj(),
                                     outside[:2].conj(), outside[2:].conj())
            for inside, outside in zip(self.prescribed_interior, self.fitted_exterior)
        ]
        eta = self.configuration.case.eta_exterior
        parameters = extract_reciprocal_lossless_least_squares(
            states, reference_impedance=eta
        ).parameters
        local_values = np.array(
            [eta * parameters.Y_ee[0, 0].imag, eta * parameters.Y_ee[1, 1].imag,
             eta * parameters.Y_ee[0, 1].imag, parameters.Y_mm[0, 0].imag / eta,
             parameters.Y_mm[1, 1].imag / eta, parameters.Y_mm[0, 1].imag / eta,
             parameters.Y_em[0, 0].real, parameters.Y_em[0, 1].real,
             parameters.Y_em[1, 0].real, parameters.Y_em[1, 1].real]
        )
        weights = np.sqrt(self.grid.weights)
        material = lstsq(
            self.profiles.T * weights[:, None], local_values.T * weights[:, None],
            lapack_driver="gelsy",
        )[0].T
        interior = np.stack([
            fit_trace(self.interior_basis, target - incident, self.grid, eta)[0]
            for target, incident in zip(self.prescribed_interior, self.incident_interior)
        ])
        return self.pack(material, interior)


def prepare_problem(
    configuration, *, state_cases=STATE_CASES, n_max=N_MAX, n_theta=N_THETA,
    n_phi=N_PHI, angular_degree=ANGULAR_DEGREE,
):
    """Fit target scattered fields after subtracting the exterior incident fields."""
    require_integer("n_max", n_max, 1)
    require_integer("angular_degree", angular_degree, 0)
    require_integer("n_theta", n_theta, max(n_max, angular_degree) + 1)
    require_integer("n_phi", n_phi, 2 * max(n_max, angular_degree) + 1)
    state_cases = validate_state_cases(configuration, state_cases)
    transformations = {state.id: state for state in configuration.transformations}
    cases = {case.id: case for case in configuration.simulation.cases}
    case = configuration.case
    grid = surface_grid(n_theta, n_phi)
    profiles, profile_names = angular_basis(grid.theta, grid.phi, angular_degree)
    modes, exterior_basis = tangential_basis(
        n_max, case.k_exterior, case.eta_exterior, case.radius_m, grid, 3
    )
    _, interior_basis = tangential_basis(
        n_max, case.k_interior, case.eta_interior, case.radius_m, grid, 1
    )
    interior_norms = np.linalg.norm(
        (interior_basis * field_weights(grid, case.eta_exterior)[:, :, None]).reshape(
            -1, interior_basis.shape[-1]
        ), axis=0,
    )
    if np.any(interior_norms == 0) or not np.all(np.isfinite(interior_norms)):
        raise ValueError("interior SVWF basis has zero or nonfinite column norms")
    interior_basis = interior_basis / interior_norms
    incident_interior, incident_exterior, prescribed_interior, target_exterior = [], [], [], []
    fitted_exterior, target_coefficients, target_fit_errors, residual_weights = [], [], [], []
    for state_id, case_id in state_cases:
        inside = incident_trace(configuration, cases[case_id], "interior", grid)
        outside = incident_trace(configuration, cases[case_id], "exterior", grid)
        target = reference_trace(transformations[state_id].gstc.exterior,
                                 "exterior", configuration, grid)
        reference = np.stack((inside, outside))
        scale = float(np.sqrt(
            field_norm(inside, grid, case.eta_exterior) ** 2
            + field_norm(outside, grid, case.eta_exterior) ** 2
        ))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"simulation case '{case_id}' must have finite nonzero incidence")
        coefficients, fitted_scattered, _ = fit_trace(
            exterior_basis, target - outside, grid, case.eta_exterior
        )
        fitted = outside + fitted_scattered
        target_fit_errors.append(relative_field_error(
            fitted, target, grid, case.eta_exterior, reference
        ))
        incident_interior.append(inside)
        incident_exterior.append(outside)
        prescribed_interior.append(reference_trace(
            transformations[state_id].gstc.interior, "interior", configuration, grid
        ))
        target_exterior.append(target)
        fitted_exterior.append(fitted)
        target_coefficients.append(coefficients)
        residual_weights.append(
            np.array([case.eta_exterior, case.eta_exterior, 1, 1])[:, None]
            * np.sqrt(grid.weights)[None, :] / scale
        )
    material_basis = surface_parameters(
        np.eye(10), np.eye(10), case.eta_exterior
    ).as_matrix().transpose(2, 0, 1).conj()
    return InverseProblem(
        configuration, tuple(state_cases), n_max, angular_degree, grid, modes, profiles,
        profile_names, material_basis, interior_basis, interior_norms,
        np.stack(incident_interior), np.stack(incident_exterior), np.stack(prescribed_interior),
        np.stack(target_exterior), np.stack(fitted_exterior), np.stack(target_coefficients),
        tuple(target_fit_errors), np.stack(residual_weights),
    )


def coefficient_maps(modes, exterior, interior):
    return tuple(
        dict(zip(modes, values))
        for values in (exterior[0::2], exterior[1::2], interior[0::2], interior[1::2])
    )


def reconstructed_trace(configuration, first, second, region, grid):
    case = configuration.case
    electric, magnetic = compute_fields(
        getattr(case, f"k_{region}"), np.full(grid.theta.shape, case.radius_m),
        grid.theta, grid.phi, first, second, kind=1 if region == "interior" else 3,
        eta=getattr(case, f"eta_{region}"), grid=False,
    )
    return np.concatenate((np.asarray(electric)[1:], np.asarray(magnetic)[1:]))


def power_diagnostics(interior, exterior, grid, radius):
    inside = normal_power_density(interior[:2], interior[2:])
    outside = normal_power_density(exterior[:2], exterior[2:])
    power_interior = float(radius ** 2 * np.dot(grid.weights, inside))
    power_exterior = float(radius ** 2 * np.dot(grid.weights, outside))
    local_norm = float(np.sqrt(np.dot(grid.weights, (outside - inside) ** 2)))
    reference_norm = max(
        float(np.sqrt(np.dot(grid.weights, inside ** 2))),
        float(np.sqrt(np.dot(grid.weights, outside ** 2))), np.finfo(float).tiny,
    )
    return {
        "interior_outward_power_w": power_interior,
        "exterior_outward_power_w": power_exterior,
        "global_power_difference_w": power_exterior - power_interior,
        "local_power_relative_mismatch": local_norm / reference_norm,
    }


def forward_validation(problem, material, n_theta, n_phi):
    """Check a newly optimized material with the shared independent forward solve."""
    def evaluator(theta, phi):
        profiles, _ = angular_basis(theta, phi, problem.angular_degree)
        return surface_parameters(material, profiles, problem.configuration.case.eta_exterior)

    return solve_forward_response(
        problem.configuration, evaluator, state_cases=problem.state_cases,
        n_max=problem.n_max, n_theta=n_theta, n_phi=n_phi,
    )


def solve_forward_response(
    configuration, parameter_evaluator, *, state_cases, n_max, n_theta, n_phi,
):
    """Solve all a,b,c,d for a fixed material and unchanged regional incidence.

    ``parameter_evaluator`` returns physical +j SurfaceParameters. Conjugate
    their blocks for the existing -j SVWF solver. Targets enter diagnostics
    only, not the linear system. Total fields include each regional incident
    field once; simultaneous interior/exterior sources use superposition.
    """
    require_integer("n_max", n_max, 1)
    require_integer("n_theta", n_theta, n_max + 1)
    require_integer("n_phi", n_phi, 2 * n_max + 1)
    state_cases = validate_state_cases(configuration, state_cases)
    case = configuration.case
    modes = tuple((m, n) for n in range(1, n_max + 1) for m in range(-n, n + 1))
    cases = {entry.id: entry for entry in configuration.simulation.cases}
    tensor_attributes = {"Y": "Y_ee", "Chi": "Y_em", "Gamma": "Y_me", "Z": "Y_mm"}

    def tensor(theta, phi, attribute):
        return getattr(parameter_evaluator(theta, phi), attribute).conj()

    tensors = {
        name: (lambda theta, phi, attribute=attribute: tensor(theta, phi, attribute))
        for name, attribute in tensor_attributes.items()
    }
    validation_grid = surface_grid(n_theta + 2, n_phi + 4, phi_offset=0.5)
    output = []
    for state_id, case_id in state_cases:
        simulation_case = cases[case_id]
        solved = [dict.fromkeys(modes, 0j) for _ in range(4)]
        solver_metrics = {}
        for region in ("interior", "exterior"):
            if not any(entry.region == region for entry in simulation_case.excitations):
                continue

            def incident(theta, phi, region=region):
                points = SurfaceGrid(theta, phi, np.ones_like(theta))
                return tuple(incident_trace(configuration, simulation_case, region, points))

            result = solve_collocation_general(
                n_max=n_max, k0=case.k_exterior, eta0=case.eta_exterior,
                k1=case.k_interior, eta1=case.eta_interior, radius=case.radius_m,
                gstc_params=tensors, n_theta=n_theta, n_phi=n_phi, return_metrics=True,
                incident_field_func=incident, is_interior_source=region == "interior",
            )
            for accumulated, contribution in zip(solved, result[:4]):
                for mode in modes:
                    accumulated[mode] += contribution[mode]
            solver_metrics[region] = result[4]
        interior = incident_trace(configuration, simulation_case, "interior", validation_grid)
        exterior = incident_trace(configuration, simulation_case, "exterior", validation_grid)
        interior += reconstructed_trace(configuration, solved[2], solved[3],
                                        "interior", validation_grid)
        exterior += reconstructed_trace(configuration, solved[0], solved[1],
                                        "exterior", validation_grid)
        transformation = next(entry for entry in configuration.transformations
                              if entry.id == state_id)
        target = reference_trace(transformation.gstc.exterior, "exterior", configuration,
                                 validation_grid)
        reference = np.stack((
            incident_trace(configuration, simulation_case, "interior", validation_grid),
            incident_trace(configuration, simulation_case, "exterior", validation_grid),
        ))
        output.append({
            "coefficients": tuple(solved), "grid": validation_grid,
            "interior": interior, "exterior": exterior, "target": target,
            "metrics": {
                "exterior_relative_field_error": relative_field_error(
                    exterior, target, validation_grid, case.eta_exterior, reference
                ),
                "power": power_diagnostics(interior, exterior, validation_grid, case.radius_m),
                "regional_solves": solver_metrics,
            },
        })
    return output


def save_forward_case(case_directory, validation):
    """Save forward coefficients (-j) and held-out tangential total fields (+j)."""
    save_coefficients(case_directory / "forward_coefficients.npz", *validation["coefficients"])
    np.savez(
        case_directory / "validation_fields.npz", theta=validation["grid"].theta,
        phi=validation["grid"].phi, solid_angle_weights=validation["grid"].weights,
        interior_total=validation["interior"].conj(),
        exterior_total=validation["exterior"].conj(), exterior_target=validation["target"].conj(),
    )


def run_forward_validation(
    experiment_dir, output_dir=None, *, n_max=None, n_theta=None, n_phi=None,
):
    """Re-solve a saved surface at independent resolution, without inverse synthesis."""
    started = perf_counter()
    experiment_dir = Path(experiment_dir).resolve()
    configuration, evaluator, source = load_parameter_evaluator(experiment_dir)
    with (experiment_dir / "experiment.json").open(encoding="utf-8") as stream:
        original = json.load(stream)
    n_max = original.get("n_max") if n_max is None else n_max
    n_theta = original.get("n_theta") if n_theta is None else n_theta
    n_phi = original.get("n_phi") if n_phi is None else n_phi
    degree = source["angular_degree"]
    require_integer("n_max", n_max, 1)
    require_integer("n_theta", n_theta, max(n_max, degree) + 1)
    require_integer("n_phi", n_phi, 2 * max(n_max, degree) + 1)
    state_cases = validate_state_cases(configuration, original.get("state_cases"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output_dir).resolve() if output_dir is not None else (
        experiment_dir / "forward_validation" / timestamp
    )
    create_run_directory(directory)
    write_json(directory / "configuration.json", configuration.to_dict())
    manifest = {
        "status": "running", "method": "forward_svwf_reciprocal_lossless",
        "n_max": n_max, "n_theta": n_theta, "n_phi": n_phi, "angular_degree": degree,
        "state_cases": [list(pair) for pair in state_cases], "parameter_source": source,
        "reference_impedance_ohm": source["reference_impedance_ohm"],
        "coefficient_time_convention": "exp(-j omega t)",
        "field_and_material_time_convention": "exp(+j omega t)",
        "field_components": ["E_theta", "E_phi", "H_theta", "H_phi"],
    }
    write_json(directory / "experiment.json", manifest)
    try:
        forward = solve_forward_response(
            configuration, evaluator, state_cases=state_cases, n_max=n_max,
            n_theta=n_theta, n_phi=n_phi,
        )
        case_metrics = {}
        for (state_id, case_id), validation in zip(state_cases, forward):
            save_forward_case(directory / "cases" / case_id, validation)
            case_metrics[case_id] = {
                "transformation_id": state_id, "forward_validation": validation["metrics"],
            }
        write_json(directory / "metrics.json", {
            "cases": case_metrics, "total_seconds": perf_counter() - started,
        })
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(directory / "experiment.json", manifest)
    return directory


def run_experiment(
    configuration, output_dir=None, *, state_cases=STATE_CASES, n_max=N_MAX,
    n_theta=N_THETA, n_phi=N_PHI, angular_degree=ANGULAR_DEGREE, max_nfev=MAX_NFEV,
    verbose=1,
):
    """Save the joint fit and independent forward checks in a new experiment directory."""
    require_integer("max_nfev", max_nfev, 1)
    started = perf_counter()
    problem = prepare_problem(
        configuration, state_cases=state_cases, n_max=n_max, n_theta=n_theta, n_phi=n_phi,
        angular_degree=angular_degree,
    )
    initial = problem.initial_guess()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output_dir) if output_dir is not None else (
        APPLICATION_DIR / "results" / configuration.case_id / "experiments/svwf_inverse" / timestamp
    )
    create_run_directory(directory)
    write_json(directory / "configuration.json", configuration.to_dict())
    manifest = {
        "status": "running", "method": "joint_svwf_reciprocal_lossless",
        "n_max": n_max, "n_theta": n_theta, "n_phi": n_phi,
        "angular_degree": angular_degree, "max_nfev": max_nfev,
        "state_cases": [list(pair) for pair in state_cases],
        "material_parameter_names": list(PARAMETER_NAMES),
        "angular_profile_names": list(problem.profile_names),
        "reference_impedance_ohm": configuration.case.eta_exterior,
        "coefficient_time_convention": "exp(-j omega t)",
        "field_and_material_time_convention": "exp(+j omega t)",
        "field_components": ["E_theta", "E_phi", "H_theta", "H_phi"],
    }
    write_json(directory / "experiment.json", manifest)
    history = []

    def residual(values):
        result = problem.residual(values)
        history.append((len(history), 0.5 * float(result @ result), float(np.linalg.norm(result))))
        return result

    try:
        # for or while loop
        # 1) run the least square for some amount of iterations
        optimized = least_squares(
            residual, initial, jac=problem.jacobian, method="trf", x_scale="jac",
            ftol=1e-4, xtol=1e-4,
            tr_solver="lsmr",
            max_nfev=max_nfev, verbose=verbose,
        )
        # 2) reconstruct interior and exterior fields
        # 3) use them to compute the lossless GSTC surface paramters
        # 4) use the updated surface parameters to refine the optimization
        # as maybe the new initial guess for the next iterations
        # find a criterion to stop the loop

        material, interior_coefficients = problem.unpack(optimized.x)
        parameters = surface_parameters(material, problem.profiles, configuration.case.eta_exterior)
        np.savez(
            directory / "surface_parameters.npz", theta=problem.grid.theta, phi=problem.grid.phi,
            solid_angle_weights=problem.grid.weights, Y=parameters.Y_ee, Chi=parameters.Y_em,
            Gamma=parameters.Y_me, Z=parameters.Y_mm, angular_coefficients=material,
        )
        inside, outside, _ = problem.fields_and_map(optimized.x)
        np.savez(
            directory / "synthesis.npz", initial_variables=initial, optimized_variables=optimized.x,
            interior_total=inside.conj(), exterior_total=outside.conj(),
            exterior_target=problem.target_exterior.conj(),
            incident_interior=problem.incident_interior.conj(),
            incident_exterior=problem.incident_exterior.conj(),
            weighted_gstc_residual=problem.complex_residual(optimized.x).conj(),
        )
        with (directory / "history.csv").open("w", newline="", encoding="ascii") as stream:
            writer = csv.writer(stream)
            writer.writerow(("function_evaluation", "cost", "weighted_gstc_norm"))
            writer.writerows(history)
        forward = forward_validation(problem, material, n_theta, n_phi)
        case_metrics = {}
        for state_index, ((state_id, case_id), validation) in enumerate(zip(state_cases, forward)):
            case_directory = directory / "cases" / case_id
            candidate_coefficients = coefficient_maps(
                problem.modes, problem.target_coefficients[state_index],
                interior_coefficients[state_index] / problem.interior_column_norms,
            )
            save_coefficients(case_directory / "inverse_coefficients.npz", *candidate_coefficients)
            save_forward_case(case_directory, validation)
            case_metrics[case_id] = {
                "transformation_id": state_id,
                "target_expansion_relative_error": problem.target_fit_errors[state_index],
                "inverse_weighted_gstc_norm": float(np.linalg.norm(
                    problem.complex_residual(optimized.x)[state_index]
                )),
                "inverse_power": power_diagnostics(
                    inside[state_index], outside[state_index], problem.grid,
                    configuration.case.radius_m,
                ),
                "forward_validation": validation["metrics"],
            }
        metrics = {
            "optimizer_success": bool(optimized.success), "optimizer_status": int(optimized.status),
            "optimizer_message": optimized.message, "function_evaluations": int(optimized.nfev),
            "initial_cost": 0.5 * float(problem.residual(initial) @ problem.residual(initial)),
            "final_cost": float(optimized.cost), "optimality": float(optimized.optimality),
            "material_unknown_count": problem.material_size,
            "real_unknown_count": optimized.x.size, "real_equation_count": optimized.fun.size,
            "cases": case_metrics, "total_seconds": perf_counter() - started,
        }
        write_json(directory / "metrics.json", metrics)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(directory / "experiment.json", manifest)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--config", type=Path)
    inputs.add_argument("--experiment-dir", type=Path, help="Saved inverse run for --forward-only")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--forward-only", action="store_true", help="Solve a saved surface; no optimization"
    )
    parser.add_argument("--n-max", type=int, help="Forward-only truncation; default: saved n_max")
    parser.add_argument("--n-theta", type=int, help="Forward-only grid; default: saved n_theta")
    parser.add_argument("--n-phi", type=int, help="Forward-only grid; default: saved n_phi")
    args = parser.parse_args()
    if args.forward_only and args.experiment_dir is None:
        parser.error("--forward-only requires --experiment-dir")
    if not args.forward_only and any(
        value is not None for value in (args.experiment_dir, args.n_max, args.n_theta, args.n_phi)
    ):
        parser.error("--experiment-dir and resolution overrides require --forward-only")
    try:
        if args.forward_only:
            directory = run_forward_validation(
                args.experiment_dir, args.output_dir, n_max=args.n_max,
                n_theta=args.n_theta, n_phi=args.n_phi,
            )
        else:
            configuration = load_workflow_configuration(args.config or DEFAULT_CONFIG)
            directory = run_experiment(configuration, args.output_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
