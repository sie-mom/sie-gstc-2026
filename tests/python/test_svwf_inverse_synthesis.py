"""CPU contracts and numerical checks for the isolated SVWF inverse prototype."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import least_squares

from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    inverse_synthesis as experiment,
)
from applications.anisotropic_gstc_sphere.workflow import (
    FieldReference,
    GSTCFieldsConfiguration,
    GSTCTransformationConfiguration,
    PlaneWaveFieldDefinition,
    SimulationCaseConfiguration,
    SimulationConfiguration,
    SimulationExcitationConfiguration,
    load_workflow_configuration,
)
from spherical_expansion.coefficients import load_coefficients
from spherical_expansion.svwf_sphere import compute_fields


@pytest.fixture
def configuration():
    return load_workflow_configuration(experiment.DEFAULT_CONFIG)


@pytest.fixture
def problem(configuration):
    return experiment.prepare_problem(
        configuration, n_max=1, n_theta=4, n_phi=8, angular_degree=1
    )


def test_quadrature_integrates_sphere_and_avoids_poles():
    grid = experiment.surface_grid(5, 12)
    assert grid.theta.shape == grid.phi.shape == grid.weights.shape == (60,)
    assert np.all((grid.theta > 0) & (grid.theta < np.pi))
    assert grid.weights.sum() == pytest.approx(4 * np.pi)
    assert np.dot(grid.weights, np.cos(grid.theta) ** 2) == pytest.approx(4 * np.pi / 3)


def test_angular_basis_includes_constant_and_nonaxisymmetric_profiles():
    grid = experiment.surface_grid(5, 12)
    constant, _ = experiment.angular_basis(grid.theta, grid.phi, 0)
    np.testing.assert_array_equal(constant, np.ones((1, 60)))
    profiles, names = experiment.angular_basis(grid.theta, grid.phi, 2)
    assert profiles.shape == (9, 60)
    assert len(names) == 9
    assert np.ptp(profiles[names.index("Re_Y_1_1")].reshape(5, 12), axis=1).max() > 1
    np.testing.assert_allclose((profiles * grid.weights) @ profiles.T,
                               4 * np.pi * np.eye(9), atol=1e-12)


def test_parameter_model_is_reciprocal_lossless_at_every_sample():
    grid = experiment.surface_grid(4, 8)
    profiles, _ = experiment.angular_basis(grid.theta, grid.phi, 2)
    coefficients = np.random.default_rng(231).normal(size=(10, 9))
    result = experiment.surface_parameters(coefficients, profiles, 377)
    np.testing.assert_array_equal(result.Y_ee.real, 0)
    np.testing.assert_array_equal(result.Y_mm.real, 0)
    np.testing.assert_array_equal(result.Y_em.imag, 0)
    np.testing.assert_array_equal(result.Y_me.imag, 0)
    np.testing.assert_array_equal(result.Y_ee, result.Y_ee.swapaxes(0, 1))
    np.testing.assert_array_equal(result.Y_mm, result.Y_mm.swapaxes(0, 1))
    np.testing.assert_array_equal(result.Y_me, -result.Y_em.swapaxes(0, 1))
    matrix = result.as_matrix()
    np.testing.assert_array_equal(matrix, -matrix.conj().swapaxes(0, 1))


@pytest.mark.parametrize("kind", [1, 3])
def test_cached_basis_matches_existing_field_reconstruction(configuration, kind):
    grid = experiment.surface_grid(4, 8)
    case = configuration.case
    modes, basis = experiment.tangential_basis(
        2, case.k_exterior, case.eta_exterior, case.radius_m, grid, kind
    )
    rng = np.random.default_rng(36)
    coefficients = rng.normal(size=2 * len(modes)) + 1j * rng.normal(size=2 * len(modes))
    electric, magnetic = compute_fields(
        case.k_exterior, np.full(grid.theta.shape, case.radius_m), grid.theta, grid.phi,
        dict(zip(modes, coefficients[0::2])), dict(zip(modes, coefficients[1::2])),
        kind=kind, eta=case.eta_exterior,
    )
    expected = np.concatenate((np.asarray(electric)[1:], np.asarray(magnetic)[1:]))
    np.testing.assert_allclose(np.einsum("ipk,k->ip", basis, coefficients), expected, atol=1e-14)


def test_centered_electric_and_magnetic_targets_fit_with_nmax_one(problem):
    assert max(problem.target_fit_errors) < 2e-9
    np.testing.assert_array_equal(problem.incident_exterior, 0)
    np.testing.assert_allclose(problem.fitted_exterior, problem.target_exterior, atol=2e-9)


def test_real_packing_preserves_complex_phase_and_residual_norm(problem):
    rng = np.random.default_rng(289)
    material = rng.normal(size=(10, 4))
    interior = rng.normal(size=problem.interior_shape) + 1j * rng.normal(
        size=problem.interior_shape
    )
    values = problem.pack(material, interior)
    actual_material, actual_interior = problem.unpack(values)
    np.testing.assert_array_equal(actual_material, material)
    np.testing.assert_array_equal(actual_interior, interior)
    residual = problem.residual(values)
    assert residual @ residual == pytest.approx(
        np.sum(np.abs(problem.complex_residual(values)) ** 2)
    )


def test_analytic_jacobian_matches_central_finite_differences(problem):
    rng = np.random.default_rng(777)
    values = problem.initial_guess() + rng.normal(scale=0.1, size=64)
    jacobian = problem.jacobian(values)
    for column in range(values.size):
        step = np.zeros_like(values)
        step[column] = 1e-6
        difference = (problem.residual(values + step) - problem.residual(values - step)) / 2e-6
        np.testing.assert_allclose(jacobian[:, column], difference, atol=2e-9, rtol=2e-6)


def exterior_plane_wave_configuration(configuration):
    fields = {
        "px": PlaneWaveFieldDefinition((1, 0, 0), (0, 0, 1)),
        "py": PlaneWaveFieldDefinition((0, 1, 0), (0, 0, 1)),
    }
    transformations = tuple(
        GSTCTransformationConfiguration(
            "target_" + name, GSTCFieldsConfiguration(None, FieldReference(name))
        ) for name in fields
    )
    cases = tuple(
        SimulationCaseConfiguration(
            "incident_" + name,
            (SimulationExcitationConfiguration(FieldReference(name), "exterior"),),
        ) for name in fields
    )
    return replace(configuration, fields=fields, transformations=transformations,
                   simulation=SimulationConfiguration(cases))


def test_cloaking_subtracts_exterior_incidence_before_fitting(configuration):
    configuration = exterior_plane_wave_configuration(configuration)
    mapping = tuple(("target_" + name, "incident_" + name) for name in configuration.fields)
    problem = experiment.prepare_problem(
        configuration, state_cases=mapping, n_max=1, n_theta=4, n_phi=8, angular_degree=0
    )
    np.testing.assert_array_equal(problem.target_coefficients, 0)
    np.testing.assert_array_equal(problem.incident_interior, 0)
    np.testing.assert_array_equal(problem.fitted_exterior, problem.incident_exterior)
    assert problem.target_fit_errors == (0, 0)


def test_forward_validation_superposes_both_regional_incidents(configuration, monkeypatch):
    fields = dict(configuration.fields)
    fields["plane"] = PlaneWaveFieldDefinition((1, 0, 0), (0, 0, 1))
    cases = list(configuration.simulation.cases)
    cases[0] = replace(cases[0], excitations=cases[0].excitations + (
        SimulationExcitationConfiguration(FieldReference("plane"), "exterior"),
    ))
    configuration = replace(configuration, fields=fields, simulation=SimulationConfiguration(
        tuple(cases)
    ))
    problem = experiment.prepare_problem(configuration, n_max=1, n_theta=4, n_phi=8,
                                         angular_degree=0)
    calls = []

    def fake_solve(**kwargs):
        calls.append(kwargs["is_interior_source"])
        coefficient = 1 if kwargs["is_interior_source"] else 2
        assert callable(kwargs["gstc_params"]["Y"])
        return (*[dict.fromkeys(problem.modes, coefficient) for _ in range(4)], {})

    monkeypatch.setattr(experiment, "solve_collocation_general", fake_solve)
    validation = experiment.forward_validation(problem, np.zeros((10, 1)), 4, 8)
    assert calls == [True, False, True]
    assert all(value == 3 for family in validation[0]["coefficients"] for value in family.values())
    assert validation[0]["grid"].theta.size == 6 * 12


def test_joint_optimizer_recovers_a_manufactured_compatible_trace(configuration):
    problem = experiment.prepare_problem(configuration, n_max=1, n_theta=4, n_phi=8,
                                         angular_degree=0)
    material = np.array([0.2, 0.3, 0.04, 0.5, 0.4, 0.06, 0.07, -0.1, 0.03, 0.08])[:, None]
    validation = experiment.forward_validation(problem, material, 6, 12)
    interior = []
    exterior = []
    for item in validation:
        exterior.append(experiment.reconstructed_trace(
            configuration, item["coefficients"][0], item["coefficients"][1],
            "exterior", problem.grid,
        ))
        interior.append(np.array([
            value for mode in problem.modes
            for value in (item["coefficients"][2][mode], item["coefficients"][3][mode])
        ]) * problem.interior_column_norms)
    problem = replace(problem, fitted_exterior=np.stack(exterior))
    exact = problem.pack(material, np.stack(interior))
    assert np.linalg.norm(problem.residual(exact)) < 2e-8
    initial = exact + np.random.default_rng(96).normal(scale=0.015, size=exact.size)
    result = least_squares(problem.residual, initial, jac=problem.jacobian, method="trf",
                           x_scale="jac", max_nfev=30, gtol=1e-11)
    assert np.linalg.norm(result.fun) < 2e-8
    assert np.linalg.norm(result.fun) < np.linalg.norm(problem.residual(initial)) / 1000


@pytest.mark.parametrize("settings", [
    {"n_max": 0}, {"n_max": True}, {"angular_degree": -1}, {"n_theta": 1},
    {"n_phi": 2}, {"angular_degree": 4, "n_theta": 4}, {"state_cases": ()},
    {"state_cases": (("electric_z_to_x", "unknown"), ("magnetic_z_to_x", "magnetic_incidence"))},
])
def test_preparation_rejects_invalid_settings(configuration, settings):
    with pytest.raises(ValueError):
        experiment.prepare_problem(configuration, **settings)


def test_zero_incidence_is_rejected(configuration):
    fields = dict(configuration.fields)
    fields["electric_z_centered"] = replace(
        fields["electric_z_centered"], effective_moment=(0, 0, 0)
    )
    with pytest.raises(ValueError, match="nonzero incidence"):
        experiment.prepare_problem(replace(configuration, fields=fields))


def test_saved_run_distinguishes_budget_stop_and_keeps_phase_conventions(configuration, tmp_path):
    directory = experiment.run_experiment(
        configuration, tmp_path / "experiment", n_max=1, n_theta=4, n_phi=8,
        angular_degree=1, max_nfev=1, verbose=0,
    )
    manifest = json.loads((directory / "experiment.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    assert manifest["status"] == "completed"
    assert not metrics["optimizer_success"]
    assert metrics["optimizer_status"] == 0
    assert metrics["real_unknown_count"] == 64
    assert metrics["real_equation_count"] == 512
    configuration_saved = json.loads((directory / "configuration.json").read_text())
    assert configuration_saved == configuration.to_dict()
    coefficients = load_coefficients(
        directory / "cases/electric_incidence/inverse_coefficients.npz"
    )
    fields = np.load(directory / "synthesis.npz")
    parameters = np.load(directory / "surface_parameters.npz")
    grid = experiment.SurfaceGrid(parameters["theta"], parameters["phi"],
                                  parameters["solid_angle_weights"])
    outside = experiment.reconstructed_trace(configuration, coefficients[0], coefficients[1],
                                             "exterior", grid)
    np.testing.assert_allclose(fields["exterior_total"][0], outside.conj(), atol=1e-12)
    np.testing.assert_array_equal(parameters["Y"].real, 0)
    for case in configuration.simulation.cases:
        saved = np.load(directory / "cases" / case.id / "validation_fields.npz")
        assert saved["exterior_total"].shape == (4, 72)
        assert np.isfinite(saved["interior_total"]).all()
        assert np.isfinite(metrics["cases"][case.id]["forward_validation"][
            "exterior_relative_field_error"
        ])
    with pytest.raises(FileExistsError):
        experiment.run_experiment(configuration, directory, n_max=1, n_theta=4, n_phi=8,
                                  angular_degree=1, max_nfev=1, verbose=0)
