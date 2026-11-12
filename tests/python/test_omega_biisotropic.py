import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

from applications.omega_biisotropic_sphere.comsol.surface_profiles import (
    compile_scalar_omega_cap_profile,
    spherical_cap_geometry,
)
from applications.omega_biisotropic_sphere.comsol.volumetric_impedance_matching import (
    _resolved_global_hmin_divisor,
    parse_arguments,
)
from applications.omega_biisotropic_sphere.dipole_impedance_matching import (
    apply_cap_perturbation,
    extract_case_parameters,
    prescribed_dipole_fields,
)
from applications.omega_biisotropic_sphere.generate_sphere_mesh import (
    shortest_wavelength,
    target_edge_length,
)
from applications.omega_biisotropic_sphere.plot_dipole_impedance_matching import (
    _assert_matching_coordinates,
    create_plots,
)
from applications.omega_biisotropic_sphere.prepare_sie_inputs import (
    export_sie_inputs,
)
from applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching import (
    DipoleMatchingCase,
    load_workflow_configuration,
    run_svwf_field_evaluation,
    run_svwf_validation,
)
from applications.omega_biisotropic_sphere.workflow import (
    CapPerturbationConfiguration,
    WorkflowConfiguration,
    default_run_directory,
)
from sie_mom.excitation import evaluate_dipole_fields
from sie_mom.gstc.omega_biisotropic import (
    BoundaryQuantities,
    OmegaParameters,
    absorbed_power_density,
    extract_lossless_omega_least_squares,
    extract_lossless_omega_projected,
    form_boundary_quantities,
    normal_power_density,
    omega_tensor_blocks,
    reconstruct_omega_currents,
)
from sie_mom.mesh.obj_mesh import load_triangle_obj


def normal_cross(field):
    return np.stack((-field[1], field[0]))


def boundary_fields_from_known_parameters(
    electric_average, magnetic_average, parameters
):
    empty = np.zeros_like(electric_average)
    averages = BoundaryQuantities(
        electric_average=electric_average,
        magnetic_average=magnetic_average,
        electric_current=empty,
        magnetic_current=empty,
    )
    electric_current, magnetic_current = reconstruct_omega_currents(
        averages, parameters
    )

    magnetic_jump = -normal_cross(electric_current)
    electric_jump = normal_cross(magnetic_current)
    electric_interior = electric_average - 0.5 * electric_jump
    electric_exterior = electric_average + 0.5 * electric_jump
    magnetic_interior = magnetic_average - 0.5 * magnetic_jump
    magnetic_exterior = magnetic_average + 0.5 * magnetic_jump
    return (
        electric_interior,
        magnetic_interior,
        electric_exterior,
        magnetic_exterior,
    )


def test_projected_and_svd_extraction_recover_known_lossless_parameters():
    electric_average = np.array(
        [
            [1.0 + 0.2j, -0.4 + 0.7j, 0.8 - 0.3j],
            [0.3 - 0.5j, 1.1 + 0.1j, -0.6 + 0.9j],
        ]
    )
    magnetic_average = np.array(
        [
            [0.7 - 0.1j, 0.2 + 0.8j, -0.9 - 0.2j],
            [-0.2 + 0.6j, 0.9 - 0.4j, 0.5 + 0.3j],
        ]
    )
    expected = OmegaParameters(
        B=np.array([0.3, -0.2, 0.5]),
        X=np.array([12.0, -7.0, 4.5]),
        chi=np.array([0.15, -0.35, 0.2]),
    )
    fields = boundary_fields_from_known_parameters(
        electric_average, magnetic_average, expected
    )
    quantities = form_boundary_quantities(*fields)

    projected = extract_lossless_omega_projected(quantities)
    least_squares = extract_lossless_omega_least_squares(quantities)

    assert np.all(projected.valid)
    assert np.all(least_squares.valid)
    assert np.all(least_squares.rank == 3)
    np.testing.assert_allclose(projected.parameters.B, expected.B, atol=2e-14)
    np.testing.assert_allclose(projected.parameters.X, expected.X, atol=2e-14)
    np.testing.assert_allclose(projected.parameters.chi, expected.chi, atol=2e-14)
    np.testing.assert_allclose(least_squares.parameters.B, expected.B, atol=2e-14)
    np.testing.assert_allclose(least_squares.parameters.X, expected.X, atol=2e-14)
    np.testing.assert_allclose(
        least_squares.parameters.chi, expected.chi, atol=2e-14
    )
    np.testing.assert_allclose(projected.chi_electric, projected.chi_magnetic)
    np.testing.assert_allclose(absorbed_power_density(quantities), 0.0, atol=2e-14)
    np.testing.assert_allclose(projected.relative_residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(least_squares.relative_residual, 0.0, atol=2e-14)


def test_centered_dipoles_are_locally_power_matched_for_paper_materials():
    frequency = 10.0e9
    omega = 2.0 * np.pi * frequency
    radius = 0.7 * c / frequency
    epsilon_interior = 7.0 * epsilon_0
    epsilon_exterior = epsilon_0
    k_interior = omega * np.sqrt(mu_0 * epsilon_interior)
    k_exterior = omega * np.sqrt(mu_0 * epsilon_exterior)
    eta_interior = np.sqrt(mu_0 / epsilon_interior)
    eta_exterior = np.sqrt(mu_0 / epsilon_exterior)

    theta = np.linspace(0.05, np.pi - 0.05, 41)
    phi = np.linspace(-np.pi, np.pi, theta.size)
    effective_moment_ratio = (k_interior / k_exterior) ** 2 * np.sqrt(
        eta_exterior / eta_interior
    )

    electric_interior, magnetic_interior = evaluate_dipole_fields(
        k_interior,
        radius,
        theta,
        phi,
        eta_interior,
        effective_moment=(0.0, 0.0, 1.0),
    )
    electric_exterior, magnetic_exterior = evaluate_dipole_fields(
        k_exterior,
        radius,
        theta,
        phi,
        eta_exterior,
        effective_moment=(0.0, 0.0, effective_moment_ratio),
    )

    electric_interior = electric_interior[1:]
    magnetic_interior = magnetic_interior[1:]
    electric_exterior = electric_exterior[1:]
    magnetic_exterior = magnetic_exterior[1:]
    power_interior = normal_power_density(electric_interior, magnetic_interior)
    power_exterior = normal_power_density(electric_exterior, magnetic_exterior)
    quantities = form_boundary_quantities(
        electric_interior,
        magnetic_interior,
        electric_exterior,
        magnetic_exterior,
    )
    projected = extract_lossless_omega_projected(quantities)
    least_squares = extract_lossless_omega_least_squares(quantities)

    np.testing.assert_allclose(effective_moment_ratio, 7.0 ** 1.25, rtol=2e-15)
    np.testing.assert_allclose(power_exterior, power_interior, rtol=2e-14)
    absorbed_power = absorbed_power_density(quantities)
    power_scale = np.max(np.abs(power_interior))
    assert np.max(np.abs(absorbed_power)) / power_scale < 2e-14
    np.testing.assert_allclose(
        absorbed_power,
        power_interior - power_exterior,
        atol=2e-14 * power_scale,
    )
    assert np.all(projected.valid)
    assert np.all(least_squares.valid)
    assert np.all(least_squares.rank == 3)
    np.testing.assert_allclose(
        projected.chi_electric, projected.chi_magnetic, rtol=2e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        projected.parameters.B, least_squares.parameters.B, rtol=2e-12
    )
    np.testing.assert_allclose(
        projected.parameters.X, least_squares.parameters.X, rtol=2e-12
    )
    np.testing.assert_allclose(
        projected.parameters.chi, least_squares.parameters.chi, rtol=2e-12
    )
    np.testing.assert_allclose(
        projected.parameters.B, projected.parameters.B[0], rtol=2e-12
    )
    np.testing.assert_allclose(
        projected.parameters.X, projected.parameters.X[0], rtol=2e-12
    )
    np.testing.assert_allclose(
        projected.parameters.chi, projected.parameters.chi[0], rtol=2e-12
    )
    assert np.max(projected.relative_residual) < 2e-12
    assert np.max(least_squares.relative_residual) < 2e-12


def test_solver_independent_dipole_physics_exposes_both_phasor_conventions():
    case = DipoleMatchingCase()
    theta = np.linspace(0.1, np.pi - 0.1, 17)
    phi = np.linspace(-np.pi, np.pi, theta.size)

    physical_fields = prescribed_dipole_fields(theta, phi, case)
    svwf_fields = prescribed_dipole_fields(
        theta, phi, case, svwf_convention=True
    )
    for physical, svwf in zip(physical_fields, svwf_fields):
        np.testing.assert_allclose(svwf, np.conjugate(physical))

    extraction = extract_case_parameters(theta, phi, case)
    assert np.all(extraction.valid)
    assert np.all(extraction.parameters.B < 0.0)
    assert np.all(extraction.parameters.X < 0.0)
    assert np.all(extraction.parameters.chi > 0.0)


def test_omega_tensor_blocks_have_reciprocal_omega_structure():
    parameters = OmegaParameters(
        B=np.array([2.0, 3.0]),
        X=np.array([5.0, 7.0]),
        chi=np.array([0.4, -0.2]),
    )

    blocks = omega_tensor_blocks(parameters)

    assert blocks["Y"].shape == (2, 2, 2)
    np.testing.assert_allclose(blocks["Y"][0, 0], 1j * parameters.B)
    np.testing.assert_allclose(blocks["Y"][1, 1], 1j * parameters.B)
    np.testing.assert_allclose(blocks["Z"][0, 0], 1j * parameters.X)
    np.testing.assert_allclose(blocks["Z"][1, 1], 1j * parameters.X)
    np.testing.assert_allclose(blocks["Chi"][0, 1], parameters.chi)
    np.testing.assert_allclose(blocks["Chi"][1, 0], -parameters.chi)
    np.testing.assert_allclose(blocks["Gamma"], blocks["Chi"])
    np.testing.assert_allclose(
        blocks["Chi"], -np.swapaxes(blocks["Gamma"], 0, 1)
    )


def test_svd_solution_minimizes_residual_for_incompatible_currents():
    electric_average = np.array([[1.0 + 0.3j], [0.4 - 0.2j]])
    magnetic_average = np.array([[0.2 + 0.1j], [-0.8 + 0.5j]])
    expected = OmegaParameters(
        B=np.array([0.25]),
        X=np.array([8.0]),
        chi=np.array([0.3]),
    )
    empty = np.zeros_like(electric_average)
    averages = BoundaryQuantities(
        electric_average=electric_average,
        magnetic_average=magnetic_average,
        electric_current=empty,
        magnetic_current=empty,
    )
    electric_current, magnetic_current = reconstruct_omega_currents(
        averages, expected
    )
    incompatible = BoundaryQuantities(
        electric_average=electric_average,
        magnetic_average=magnetic_average,
        electric_current=electric_current
        + np.array([[0.08 + 0.04j], [-0.03 + 0.02j]]),
        magnetic_current=magnetic_current
        + np.array([[0.01 - 0.02j], [0.05 + 0.03j]]),
    )

    projected = extract_lossless_omega_projected(incompatible)
    least_squares = extract_lossless_omega_least_squares(incompatible)

    assert projected.valid[0]
    assert least_squares.valid[0]
    assert least_squares.relative_residual[0] <= projected.relative_residual[0]


def test_zero_fields_are_reported_as_degenerate():
    zero = np.zeros((2, 3), dtype=complex)
    quantities = form_boundary_quantities(zero, zero, zero, zero)

    projected = extract_lossless_omega_projected(quantities)
    least_squares = extract_lossless_omega_least_squares(quantities)

    assert not np.any(projected.valid)
    assert not np.any(least_squares.valid)
    assert np.all(least_squares.rank == 0)
    assert np.all(np.isnan(projected.parameters.chi))
    assert np.all(np.isnan(least_squares.parameters.chi))


def test_extracted_sheet_reproduces_matching_transformation_in_svwf_solver():
    metrics = run_svwf_validation(
        case=DipoleMatchingCase(),
        n_max=3,
        n_theta=14,
        n_phi=10,
        extraction_method="least_squares",
    )

    np.testing.assert_allclose(
        metrics["effective_moment_ratio"], 7.0 ** 1.25, rtol=2e-15
    )
    assert metrics["B_exp_plus_jwt"] < 0.0
    assert metrics["X_exp_plus_jwt"] < 0.0
    assert 0.0 < metrics["chi"] < 1.0
    assert metrics["parameter_uniformity_error"] < 1e-11
    assert metrics["prescribed_power_error"] < 1e-13
    assert metrics["solver_residual"] < 1e-8
    assert metrics["interior_reflection_error"] < 1e-8
    assert metrics["exterior_transmission_error"] < 1e-8


def test_workflow_configuration_loads_partial_physical_and_numerical_inputs(
    tmp_path,
):
    configuration_path = tmp_path / "input.json"
    with configuration_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "physical": {
                    "frequency_hz": 12.0e9,
                    "interior": {"epsilon_r": 4.0},
                },
                "extraction_method": "projected",
                "svwf": {"n_max": 6},
            },
            stream,
        )

    configuration = load_workflow_configuration(configuration_path)

    assert configuration.case.frequency == 12.0e9
    assert configuration.case.relative_permittivity_interior == 4.0
    assert configuration.case.relative_permeability_interior == 1.0
    assert configuration.extraction_method == "projected"
    assert configuration.svwf.n_max == 6
    assert configuration.svwf.n_theta == 18
    assert configuration.sie.mesh.path == (
        "meshes/budhu_10ghz_eps7/sphere.obj"
    )
    assert configuration.sie.mesh.points_per_wavelength == 4.0
    assert configuration.sie.solver == "dense_lu"


def test_sie_mesh_size_uses_the_shortest_material_wavelength():
    case = DipoleMatchingCase()
    expected_wavelength = c / (
        case.frequency
        * np.sqrt(
            case.relative_permittivity_interior
            * case.relative_permeability_interior
        )
    )

    np.testing.assert_allclose(shortest_wavelength(case), expected_wavelength)
    np.testing.assert_allclose(
        target_edge_length(case, 4.0), expected_wavelength / 4.0
    )


def test_cap_refined_ppw8_configuration_keeps_physics_and_mesh_buffer_separate():
    configuration_path = (
        Path(__file__).parents[2]
        / "applications"
        / "omega_biisotropic_sphere"
        / "configurations"
        / "budhu_10ghz_eps7_cap5_ppw8_capref32.json"
    )
    configuration = load_workflow_configuration(configuration_path)

    assert configuration.cap_perturbation is not None
    assert configuration.cap_perturbation.angular_radius_degrees == 5.0
    assert configuration.sie.mesh.points_per_wavelength == 8.0
    refinement = configuration.sie.mesh.cap_refinement
    assert refinement is not None
    assert refinement.points_per_wavelength == 32.0
    assert refinement.fully_refined_radius_degrees == 5.5
    assert refinement.transition_radius_degrees == 10.0
    assert configuration.to_dict()["sie"]["mesh"]["cap_refinement"] == {
        "points_per_wavelength": 32.0,
        "fully_refined_radius_degrees": 5.5,
        "transition_radius_degrees": 10.0,
    }


def test_sie_mesh_size_rejects_nonpositive_points_per_wavelength():
    with pytest.raises(ValueError, match="points_per_wavelength must be positive"):
        target_edge_length(DipoleMatchingCase(), 0.0)


def test_sie_input_export_matches_existing_cuda_tensor_layout(tmp_path):
    configuration_path = (
        Path(__file__).parents[2]
        / "applications"
        / "omega_biisotropic_sphere"
        / "configurations"
        / "budhu_10ghz_eps7.json"
    )
    configuration = load_workflow_configuration(configuration_path)
    mesh_path = (
        Path(__file__).parents[2]
        / "applications"
        / "omega_biisotropic_sphere"
        / configuration.sie.mesh.path
    )

    tensor_path, manifest_path = export_sie_inputs(
        configuration,
        mesh_path=mesh_path,
        run_dir=tmp_path,
    )

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    face_count = manifest["tensor_data"]["face_count"]
    records = np.fromfile(tensor_path, dtype="<c8").reshape((face_count, 32))
    forward = records[:, :16].reshape((-1, 4, 4))
    inverse = records[:, 16:].reshape((-1, 4, 4))

    assert face_count == 1790
    assert manifest["phasor_convention"] == "exp(+j omega t)"
    assert manifest["mesh"]["face_order"] == "unchanged OBJ order"
    assert manifest["sampling"]["local_basis"] == "spherical (theta_hat, phi_hat)"
    assert manifest["tensor_data"]["precision"] == "complex64"
    assert tensor_path.stat().st_size == face_count * 32 * np.dtype("<c8").itemsize
    np.testing.assert_allclose(
        forward,
        np.broadcast_to(forward[0], forward.shape),
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        forward @ inverse,
        np.broadcast_to(np.eye(4), forward.shape),
        rtol=2e-6,
        atol=2e-6,
    )
    assert forward[0, 0, 0].imag < 0.0
    assert forward[0, 2, 2].imag < 0.0
    assert forward[0, 0, 3].real > 0.0
    assert forward[0, 1, 2].real < 0.0


def test_shared_result_directory_separates_case_backend_and_run():
    run_dir = default_run_directory(
        "budhu_10ghz_eps7", "sie", "gmsh_ppw4"
    )

    assert run_dir.parts[-4:] == (
        "results",
        "budhu_10ghz_eps7",
        "sie",
        "gmsh_ppw4",
    )


@pytest.mark.parametrize("method", ["projected", "least_squares"])
def test_cap_mask_scales_either_extraction_and_preserves_conventions(method):
    theta = np.deg2rad([90, 90, 90, 90, 90, 90, 95, 95.01])
    phi = np.deg2rad([0, 5, 355, 5.01, 359, 180, 0, 0])
    cap = CapPerturbationConfiguration()
    baseline = extract_case_parameters(theta, phi, DipoleMatchingCase(), method=method)
    snapshots = [getattr(baseline.parameters, name).copy() for name in ("B", "X", "chi")]
    perturbed, mask = apply_cap_perturbation(baseline.parameters, theta, phi, cap)
    np.testing.assert_array_equal(mask, [True, True, True, False, True, False, True, False])
    for name, before in zip(("B", "X", "chi"), snapshots, strict=True):
        np.testing.assert_array_equal(getattr(baseline.parameters, name), before)
        np.testing.assert_allclose(getattr(perturbed, name)[mask], before[mask] * 0.1)
        np.testing.assert_array_equal(getattr(perturbed, name)[~mask], before[~mask])

    conjugate_extraction = extract_case_parameters(
        theta, phi, DipoleMatchingCase(), method=method, svwf_convention=True
    )
    conjugate_parameters, _ = apply_cap_perturbation(
        conjugate_extraction.parameters, theta, phi, cap
    )
    for name, tensor in omega_tensor_blocks(perturbed).items():
        np.testing.assert_allclose(
            omega_tensor_blocks(conjugate_parameters)[name], tensor.conj(), atol=1e-12
        )
    np.testing.assert_allclose(perturbed.Y.real, 0.0)
    np.testing.assert_allclose(perturbed.Z.real, 0.0)


@pytest.mark.parametrize("cap", [None, CapPerturbationConfiguration(parameter_scale=1.0)])
def test_disabled_cap_preserves_parameters(cap):
    parameters = OmegaParameters(B=np.array([1., 2.]), X=np.array([3., 4.]), chi=np.array([5., 6.]))
    result, _ = apply_cap_perturbation(parameters, np.array([np.pi / 2, 1.]), 0., cap)
    for name in ("B", "X", "chi"):
        np.testing.assert_array_equal(getattr(result, name), getattr(parameters, name))


def test_comsol_cap_profile_matches_shared_spherical_mask():
    cap = CapPerturbationConfiguration()
    theta = np.deg2rad([90, 90, 90, 90, 95, 95.01])
    phi = np.deg2rad([0, 5, 5.01, 180, 0, 0])
    parameters = OmegaParameters(B=1.0, X=1.0, chi=1.0)
    _, shared_mask = apply_cap_perturbation(parameters, theta, phi, cap)

    center, threshold = spherical_cap_geometry(cap)
    directions = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(directions @ center >= threshold, shared_mask)

    profile = compile_scalar_omega_cap_profile(
        cap, nx="nx", ny="ny", nz="nz"
    )
    assert profile.variables["gstc_Yee"] == "gstc_profile_scale*Yee"
    assert profile.variables["gstc_Zmm"] == "gstc_profile_scale*Zmm"
    assert profile.variables["gstc_Xem"] == "gstc_profile_scale*Xem"
    assert "if(gstc_cap_dot>=" in profile.variables["gstc_profile_scale"]
    assert profile.metadata()["edge_convention"] == "inclusive"


def test_comsol_outer_shell_band_updates_global_minimum_size():
    divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=20.0,
        interior_hmax_divisor=35.0,
        outer_shell_band_hmax_divisor=100.0,
        shell_mesh="swept",
        shell_tangential_divisor=90.0,
    )
    assert divisor == 500.0


def test_comsol_cap_edge_updates_global_minimum_size():
    divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=20.0,
        interior_hmax_divisor=35.0,
        outer_shell_band_hmax_divisor=35.0,
        shell_mesh="swept",
        shell_tangential_divisor=90.0,
        cap_edge_hmax_divisor=240.0,
    )
    assert divisor == 1200.0


def test_comsol_inner_shell_band_updates_global_minimum_size():
    divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=20.0,
        interior_hmax_divisor=35.0,
        inner_shell_band_hmax_divisor=60.0,
        outer_shell_band_hmax_divisor=None,
        shell_mesh="swept",
        shell_tangential_divisor=45.0,
    )
    assert divisor == 300.0


def test_comsol_source_ball_updates_global_minimum_size():
    divisor = _resolved_global_hmin_divisor(
        global_hmax_divisor=20.0,
        interior_hmax_divisor=35.0,
        source_ball_hmax_divisor=120.0,
        outer_shell_band_hmax_divisor=None,
        shell_mesh="swept",
        shell_tangential_divisor=90.0,
    )
    assert divisor == 600.0


def test_comsol_outer_shell_band_cli_requires_width_and_size_together():
    with pytest.raises(SystemExit):
        parse_arguments(["--outer-shell-band-width", "0.05"])

    arguments = parse_arguments(
        [
            "--outer-shell-band-width",
            "0.05",
            "--outer-shell-band-hmax-divisor",
            "35",
        ]
    )
    assert arguments.outer_shell_band_width == 0.05
    assert arguments.outer_shell_band_hmax_divisor == 35.0


def test_comsol_inner_shell_band_cli_requires_width_and_size_together():
    with pytest.raises(SystemExit):
        parse_arguments(["--inner-shell-band-width", "0.05"])

    arguments = parse_arguments(
        [
            "--inner-shell-band-width",
            "0.05",
            "--inner-shell-band-hmax-divisor",
            "60",
        ]
    )
    assert arguments.inner_shell_band_width == 0.05
    assert arguments.inner_shell_band_hmax_divisor == 60.0


def test_comsol_source_ball_cli_requires_radius_and_size_together():
    with pytest.raises(SystemExit):
        parse_arguments(["--source-ball-radius", "0.15"])

    arguments = parse_arguments(
        [
            "--source-ball-radius",
            "0.15",
            "--source-ball-hmax-divisor",
            "120",
        ]
    )
    assert arguments.source_ball_radius == 0.15
    assert arguments.source_ball_hmax_divisor == 120.0


def test_comsol_cap_edge_cli_is_opt_in():
    assert parse_arguments([]).cap_edge_hmax_divisor is None
    arguments = parse_arguments(["--cap-edge-hmax-divisor", "240"])
    assert arguments.cap_edge_hmax_divisor == 240.0


@pytest.mark.parametrize("overrides", [
    {"parameter_scale": 0.0}, {"parameter_scale": -0.1}, {"parameter_scale": 1.1},
    {"angular_radius_degrees": 0.0}, {"angular_radius_degrees": 181.0},
    {"theta_degrees": -1.0}, {"phi_degrees": float("nan")},
])
def test_invalid_cap_configuration_is_rejected(overrides):
    with pytest.raises(ValueError, match="cap_perturbation"):
        CapPerturbationConfiguration(**overrides)


def test_cap_configuration_round_trip(tmp_path):
    configuration = WorkflowConfiguration(cap_perturbation=CapPerturbationConfiguration())
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(configuration.to_dict()))
    assert load_workflow_configuration(path) == configuration
    assert "cap_perturbation" not in WorkflowConfiguration().to_dict()


@pytest.mark.parametrize("method", ["projected", "least_squares"])
def test_sie_cap_export_changes_only_selected_forward_maps_and_inverses(tmp_path, method):
    application = Path(__file__).parents[2] / "applications" / "omega_biisotropic_sphere"
    configuration = load_workflow_configuration(application / "configurations/budhu_10ghz_eps7.json")
    configuration = replace(configuration, extraction_method=method)
    mesh_path = application / configuration.sie.mesh.path
    baseline_path, _ = export_sie_inputs(configuration, mesh_path=mesh_path, run_dir=tmp_path / "baseline")
    cap_configuration = replace(configuration, cap_perturbation=CapPerturbationConfiguration())
    cap_path, manifest_path = export_sie_inputs(cap_configuration, mesh_path=mesh_path, run_dir=tmp_path / "cap")
    baseline = np.fromfile(baseline_path, dtype="<c8").reshape((-1, 32))
    perturbed = np.fromfile(cap_path, dtype="<c8").reshape((-1, 32))
    mesh = load_triangle_obj(mesh_path)
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    # This cap is centered on +x: use Cartesian coordinates as an independent check.
    mask = centroids[:, 0] / np.linalg.norm(centroids, axis=1) >= np.cos(np.deg2rad(5.0))
    assert 0 < np.count_nonzero(mask) < mask.size
    np.testing.assert_array_equal(perturbed[~mask], baseline[~mask])
    np.testing.assert_allclose(perturbed[mask, :16], 0.1 * baseline[mask, :16], rtol=2e-6)
    np.testing.assert_allclose(perturbed[mask, 16:], 10.0 * baseline[mask, 16:], rtol=2e-6, atol=1e-12)
    forward = perturbed[:, :16].reshape((-1, 4, 4))
    inverse = perturbed[:, 16:].reshape((-1, 4, 4))
    np.testing.assert_allclose(forward @ inverse, np.broadcast_to(np.eye(4), forward.shape), atol=2e-6)
    np.testing.assert_allclose(forward + forward.conj().swapaxes(1, 2), 0.0, atol=1e-12)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cap_perturbation"]["selected_face_indices"] == np.flatnonzero(mask).tolist()


def test_svwf_cap_reaches_tensor_callback_and_manifest(tmp_path, monkeypatch):
    from applications.omega_biisotropic_sphere import svwf_dipole_impedance_matching as application

    original_solver = application.solve_collocation_general

    def inspect_then_solve(**kwargs):
        tensors = kwargs["gstc_params"]
        theta = np.array([np.pi / 2, np.pi / 2])
        phi = np.array([0., np.pi / 2])
        for name in ("Y", "Chi", "Gamma", "Z"):
            block = tensors[name](theta, phi)
            np.testing.assert_allclose(block[:, :, 0], 0.1 * block[:, :, 1], atol=1e-12)
        return original_solver(**kwargs)

    monkeypatch.setattr(application, "solve_collocation_general", inspect_then_solve)
    configuration = WorkflowConfiguration(cap_perturbation=CapPerturbationConfiguration())
    metrics = application.run_svwf_validation(
        case=configuration.case, n_max=1, n_theta=19, n_phi=36,
        run_dir=tmp_path, zx_grid_size=8, radial_points=10,
        workflow_configuration=configuration,
    )
    assert metrics["cap_collocation_point_count"] == 1
    assert "exterior_change_from_reference" in metrics
    assert "exterior_transmission_error" not in metrics
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["cap_perturbation"]["angular_radius_degrees"] == 5.0
    assert manifest["reference_label"] == "Unperturbed reference"
    assert load_workflow_configuration(tmp_path / "configuration.json").cap_perturbation == configuration.cap_perturbation


def test_svwf_grid_cannot_silently_miss_cap():
    configuration = WorkflowConfiguration(cap_perturbation=CapPerturbationConfiguration())
    with pytest.raises(ValueError, match="no SVWF collocation points"):
        run_svwf_validation(
            case=configuration.case, n_max=1, n_theta=18, n_phi=12,
            workflow_configuration=configuration,
        )


def test_sie_mesh_cannot_silently_miss_cap(tmp_path):
    application = Path(__file__).parents[2] / "applications" / "omega_biisotropic_sphere"
    configuration = WorkflowConfiguration(
        cap_perturbation=CapPerturbationConfiguration(angular_radius_degrees=0.001)
    )
    with pytest.raises(ValueError, match="no SIE face centroids"):
        export_sie_inputs(
            configuration,
            mesh_path=application / configuration.sie.mesh.path,
            run_dir=tmp_path,
        )
    assert not (tmp_path / "face_tensors.bin").exists()


def test_sie_rejects_reusing_unperturbed_tensors_for_cap(tmp_path):
    from applications.omega_biisotropic_sphere.sie_dipole_impedance_matching import (
        run_sie_validation,
    )

    baseline = WorkflowConfiguration()
    (tmp_path / "face_tensors.bin").touch()
    (tmp_path / "sie_inputs.json").write_text(
        json.dumps({"configuration": baseline.to_dict()})
    )
    with pytest.raises(ValueError, match="different cap_perturbation"):
        run_sie_validation(
            replace(baseline, cap_perturbation=CapPerturbationConfiguration()),
            run_dir=tmp_path,
            reuse_inputs=True,
        )


def test_complete_workflow_writes_self_describing_csv_data(tmp_path):
    run_svwf_validation(
        case=DipoleMatchingCase(),
        n_max=2,
        n_theta=10,
        n_phi=8,
        extraction_method="least_squares",
        run_dir=tmp_path,
        zx_grid_size=8,
        radial_points=10,
    )

    with (tmp_path / "run.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    assert manifest["schema_version"] == 1
    assert manifest["case_id"] == "budhu_10ghz_eps7"
    assert manifest["backend"] == "svwf"
    assert manifest["phasor_convention"] == "exp(+j omega t)"
    assert manifest["field_data"]["coordinate_system"] == "Cartesian"
    assert manifest["field_data"]["regions"] == {"1": "interior", "2": "exterior"}
    assert manifest["sampling"]["zx"]["shape"] == [8, 8]
    assert manifest["sampling"]["radial"]["points"] == 10
    assert manifest["performance_measurements"] == {
        "duration_unit": "seconds",
        "memory_unit": "bytes",
        "peak_resident_memory_bytes": (
            "measured process maximum resident set size"
        ),
        "peak_child_resident_memory_bytes": (
            "maximum resident set size reported for completed workers"
        ),
        "solver_matrix_bytes": "allocated collocation matrix Z only",
    }
    assert manifest["field_evaluation"] == {
        "workers": 1,
        "chunk_size": None,
        "default_chunk_policy": "ceil(points / (4 * workers))",
    }
    assert manifest["files"]["configuration"] == "configuration.json"
    assert manifest["files"]["coefficients"] == "svwf_coefficients.npz"
    assert manifest["files"]["reference_zx"] == "fields/prescribed_zx.csv"
    assert manifest["files"]["computed_zx"] == "fields/svwf_zx.csv"
    assert (tmp_path / manifest["files"]["coefficients"]).is_file()
    for name in (
        "solver_setup_seconds",
        "solver_basis_seconds",
        "solver_matrix_assembly_seconds",
        "solver_scaling_seconds",
        "solver_least_squares_seconds",
        "solver_residual_seconds",
        "solver_total_seconds",
        "coefficient_export_seconds",
        "zx_field_evaluation_seconds",
        "zx_field_export_seconds",
        "radial_field_evaluation_seconds",
        "radial_field_export_seconds",
        "output_workflow_seconds",
    ):
        assert manifest["metrics"][name] >= 0.0
    assert manifest["metrics"]["solver_equation_count"] == 320
    assert manifest["metrics"]["solver_unknown_count"] == 32
    assert manifest["metrics"]["solver_matrix_bytes"] == 320 * 32 * 16
    assert manifest["metrics"]["solver_full_size_matrix_arrays"] == 1
    if sys.platform != "win32":
        assert manifest["metrics"]["peak_resident_memory_bytes"] > 0

    with (tmp_path / "configuration.json").open(encoding="utf-8") as stream:
        configuration = json.load(stream)
    assert configuration["physical"]["frequency_hz"] == 10.0e9
    assert configuration["physical"]["interior"]["epsilon_r"] == 7.0
    assert configuration["svwf"] == {"n_max": 2, "n_phi": 8, "n_theta": 10}
    assert configuration["sie"]["mesh"] == {
        "path": "meshes/budhu_10ghz_eps7/sphere.obj",
        "points_per_wavelength": 4.0,
    }
    assert configuration["sampling"]["zx_grid_size"] == 8
    assert configuration["sampling"]["radial_points"] == 10

    expected_columns = (
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
    row_counts = {
        "prescribed_zx": 64,
        "svwf_zx": 64,
        "prescribed_radial": 10,
        "svwf_radial": 10,
    }
    tables = {}
    for name, expected_rows in row_counts.items():
        path = tmp_path / manifest["files"][name]
        assert path.is_file()
        table = np.genfromtxt(path, delimiter=",", names=True)
        assert table.dtype.names == expected_columns
        assert table.shape == (expected_rows,)
        assert np.all(np.isfinite(table["Ex_real"]))
        tables[name] = table

    for coordinate in ("x", "y", "z"):
        np.testing.assert_array_equal(
            tables["prescribed_zx"][coordinate], tables["svwf_zx"][coordinate]
        )
        np.testing.assert_array_equal(
            tables["prescribed_radial"][coordinate],
            tables["svwf_radial"][coordinate],
        )

    pytest.importorskip("matplotlib")
    plot_paths = create_plots(tmp_path)
    assert len(plot_paths) == 4
    assert all(path.is_file() for path in plot_paths)


def test_saved_coefficients_reproduce_field_outputs_without_solving(tmp_path):
    source_run = tmp_path / "solved"
    replay_run = tmp_path / "field_replay"
    run_svwf_validation(
        case=DipoleMatchingCase(),
        n_max=2,
        n_theta=10,
        n_phi=8,
        extraction_method="least_squares",
        run_dir=source_run,
        zx_grid_size=8,
        radial_points=10,
    )
    configuration = load_workflow_configuration(source_run / "configuration.json")

    metrics = run_svwf_field_evaluation(
        source_run_dir=source_run,
        run_dir=replay_run,
        workflow_configuration=configuration,
        field_workers=2,
        field_chunk_size=3,
    )

    assert metrics["coefficient_load_seconds"] >= 0.0
    assert metrics["coefficient_mode_count"] == 8
    assert metrics["field_workers"] == 2
    assert metrics["field_chunk_size"] == 3
    for relative_path in (
        "fields/prescribed_zx.csv",
        "fields/svwf_zx.csv",
        "fields/prescribed_radial.csv",
        "fields/svwf_radial.csv",
    ):
        assert (replay_run / relative_path).read_bytes() == (
            source_run / relative_path
        ).read_bytes()

    manifest = json.loads((replay_run / "run.json").read_text())
    assert manifest["workflow_mode"] == "field_evaluation"
    assert manifest["field_evaluation"]["workers"] == 2
    assert manifest["field_evaluation"]["chunk_size"] == 3
    assert manifest["coefficient_source"] == str(
        (source_run / "svwf_coefficients.npz").resolve()
    )
    assert "solver_total_seconds" not in manifest["metrics"]


def test_plot_coordinate_check_allows_roundoff_but_rejects_different_samples():
    first = {
        "point_index": np.arange(3), "region": np.array([1, 1, 2]),
        "x": np.array([0.01, 0.02, 0.03]), "y": np.zeros(3),
        "z": np.array([0.02, 0.02, 0.02]),
    }
    rounded = {name: values.copy() for name, values in first.items()}
    rounded["x"][1] = np.nextafter(rounded["x"][1], np.inf)
    _assert_matching_coordinates(first, rounded)
    for name, delta in (("x", 1e-8), ("region", 1), ("point_index", 1)):
        changed = {name: values.copy() for name, values in first.items()}
        changed[name][1] += delta
        with pytest.raises(AssertionError):
            _assert_matching_coordinates(first, changed)


def test_plotter_adds_sie_representation_diagnostics(tmp_path):
    run_svwf_validation(
        case=DipoleMatchingCase(),
        n_max=2,
        n_theta=10,
        n_phi=8,
        extraction_method="least_squares",
        run_dir=tmp_path,
        zx_grid_size=8,
        radial_points=10,
    )
    with (tmp_path / "run.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    manifest["backend"] = "sie"
    representation_files = {
        "interior_representation_zx": "fields/sie_interior_representation_zx.csv",
        "exterior_representation_zx": "fields/sie_exterior_representation_zx.csv",
        "interior_representation_radial": (
            "fields/sie_interior_representation_radial.csv"
        ),
        "exterior_representation_radial": (
            "fields/sie_exterior_representation_radial.csv"
        ),
    }
    source_files = {
        "interior_representation_zx": manifest["files"]["computed_zx"],
        "exterior_representation_zx": manifest["files"]["computed_zx"],
        "interior_representation_radial": manifest["files"]["computed_radial"],
        "exterior_representation_radial": manifest["files"]["computed_radial"],
    }
    for name, relative_path in representation_files.items():
        shutil.copyfile(tmp_path / source_files[name], tmp_path / relative_path)
    manifest["files"].update(representation_files)
    with (tmp_path / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream)

    plot_paths = create_plots(tmp_path)
    assert len(plot_paths) == 6
    assert {path.name for path in plot_paths} >= {
        "sie_representations_zx.png",
        "sie_representations_radial.png",
    }
    assert all(path.is_file() for path in plot_paths)
