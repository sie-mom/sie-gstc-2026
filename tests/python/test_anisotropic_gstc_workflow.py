import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from applications.anisotropic_gstc_sphere import prepare_sie_inputs as preparation
from applications.anisotropic_gstc_sphere import sie_gstc_sphere as coordinator
from applications.anisotropic_gstc_sphere.prepare_sie_inputs import export_sie_inputs
from applications.anisotropic_gstc_sphere.prescribed_fields import (
    PrescribedFieldState,
    prescribed_boundary_states,
)
from applications.anisotropic_gstc_sphere.workflow import (
    APPLICATION_DIR,
    DEFAULT_CASE_ID,
    DEFAULT_CONFIGURATION,
    DipoleFieldDefinition,
    FieldReference,
    GSTCFieldsConfiguration,
    GSTCTransformationConfiguration,
    PlaneWaveFieldDefinition,
    SamplingConfiguration,
    SimulationCaseConfiguration,
    SimulationConfiguration,
    SimulationExcitationConfiguration,
    SphereCase,
    default_run_directory,
    load_workflow_configuration,
)
from sie_mom.excitation import evaluate_dipole_fields, evaluate_plane_wave_fields
from sie_mom.gstc import SurfaceParameters

DISPLACEMENT_CONFIGURATION = (
    APPLICATION_DIR
    / "configurations"
    / "dipole_shift"
    / "refl_complex.json"
)


def test_dipole_transformation_configuration_uses_normalized_effective_moments():
    configuration = load_workflow_configuration(DISPLACEMENT_CONFIGURATION)
    assert configuration.case.frequency_hz == 1e8
    assert configuration.case.radius_m == 2.0
    assert configuration.extraction_method == "reciprocal_complex"
    assert configuration.tensor_structure == "general"
    for name, moment, position, dipole_type in (
        ("electric_z_shifted", 1.0, (0, 0, 0.5), "electric"),
        ("electric_z_centered", 1.0, (0, 0, 0), "electric"),
        ("magnetic_z_centered", 1.0, (0, 0, 0), "magnetic"),
        ("magnetic_z_centered_half", 0.5, (0, 0, 0), "magnetic"),
    ):
        definition = configuration.fields[name]
        assert definition.dipole_type == dipole_type
        assert definition.source_position == position
        np.testing.assert_allclose(definition.effective_moment, (0, 0, moment), rtol=1e-12)
    assert [state.id for state in configuration.transformations] == [
        "electric_displacement",
        "magnetic_attenuation",
    ]
    assert [case.excitations[0].field.field_id for case in configuration.simulation.cases] == [
        "electric_z_shifted",
        "magnetic_z_centered",
    ]
    assert all(case.excitations[0].region == "interior" for case in configuration.simulation.cases)


def test_dipole_transformations_prepare_reciprocal_complex_tensors(tmp_path, monkeypatch):
    mesh_path = tmp_path / "radius2.obj"
    mesh_path.write_text(
        "v 2 0 0\nv -2 0 0\nv 0 2 0\nv 0 -2 0\nv 0 0 2\nv 0 0 -2\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n",
        encoding="ascii",
    )
    configuration = load_workflow_configuration(DISPLACEMENT_CONFIGURATION)
    configuration = replace(
        configuration,
        sie=replace(
            configuration.sie,
            mesh=replace(configuration.sie.mesh, path=str(mesh_path)),
        ),
    )

    def unexpected_solver(*args, **kwargs):
        pytest.fail("dipole preparation must not invoke C++")

    monkeypatch.setattr(coordinator.subprocess, "run", unexpected_solver)
    run_dir = tmp_path / "dipole_preparation"
    path = coordinator.run_sie_workflow(
        configuration,
        run_dir=run_dir,
        use_existing_mesh=True,
        prepare_only=True,
    )
    manifest = json.loads(path.read_text())
    inputs = json.loads((run_dir / "sie_inputs.json").read_text())
    assert manifest["status"] == "prepared"
    cases = manifest["simulation"]["cases"]
    assert [case["id"] for case in cases] == ["electric_incidence", "magnetic_incidence"]
    assert all(case["status"] == "pending" and case["command"] is None for case in cases)
    assert all((run_dir / case["files"]["plots"]).is_dir() for case in cases)
    assert [case["excitations"] for case in cases] == [
        case["excitations"] for case in configuration.to_dict()["simulation"]["cases"]
    ]
    assert inputs["prescribed_state_ids"] == ["electric_displacement", "magnetic_attenuation"]
    assert inputs["radius_m"] == 2.0
    diagnostics = inputs["diagnostics"]
    assert diagnostics["maximum_relative_residual"] < 1e-10
    assert max(diagnostics["maximum_state_relative_residual"]) < 1e-10
    assert diagnostics["minimum_rank"] == diagnostics["maximum_rank"] == 7
    assert diagnostics["non_unique_face_count"] == 8
    assert diagnostics["maximum_inverse_residual"] < 1e-10
    records = np.fromfile(run_dir / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    matrix, inverse = records[:, :16].reshape(8, 4, 4), records[:, 16:].reshape(8, 4, 4)
    assert np.isfinite(records).all()
    np.testing.assert_allclose(matrix @ inverse, np.broadcast_to(np.eye(4), (8, 4, 4)), atol=1e-10)
    np.testing.assert_allclose(matrix[:, :2, :2], matrix[:, :2, :2].swapaxes(1, 2), atol=1e-14)
    np.testing.assert_allclose(matrix[:, 2:, 2:], matrix[:, 2:, 2:].swapaxes(1, 2), atol=1e-14)
    np.testing.assert_allclose(matrix[:, 2:, :2], -matrix[:, :2, 2:].swapaxes(1, 2), atol=1e-14)
    assert not (run_dir / "currents.bin").exists()
    assert load_workflow_configuration(run_dir / "configuration.json") == configuration


def test_dipole_targets_prescribe_displacement_and_half_amplitude():
    configuration = load_workflow_configuration(DISPLACEMENT_CONFIGURATION)
    theta, phi = np.array([0.3, 0.8, 1.3, 1.8, 2.3, 2.8]), np.array([0.1, 0.7, 1.2, 2.0, 3.0, 4.0])
    electric, magnetic = prescribed_boundary_states(theta, phi, configuration)
    assert np.any(electric.electric_interior != 0)
    assert np.any(electric.magnetic_interior != 0)
    for region, field_id in (
        ("interior", "electric_z_shifted"),
        ("exterior", "electric_z_centered"),
    ):
        definition = configuration.fields[field_id]
        expected_e, expected_h = evaluate_dipole_fields(
            getattr(configuration.case, f"k_{region}"),
            2.0,
            theta,
            phi,
            getattr(configuration.case, f"eta_{region}"),
            effective_moment=definition.effective_moment,
            source_position=definition.source_position,
            dipole_type="electric",
        )
        np.testing.assert_array_equal(getattr(electric, f"electric_{region}"), expected_e)
        np.testing.assert_array_equal(getattr(electric, f"magnetic_{region}"), expected_h)
    assert not np.allclose(electric.electric_interior, electric.electric_exterior)
    np.testing.assert_allclose(magnetic.electric_exterior, 0.5 * magnetic.electric_interior)
    np.testing.assert_allclose(magnetic.magnetic_exterior, 0.5 * magnetic.magnetic_interior)


@pytest.fixture
def sphere_mesh(tmp_path):
    path = tmp_path / "sphere.obj"
    path.write_text(
        "v 1 0 0\nv -1 0 0\nv 0 1 0\nv 0 -1 0\nv 0 0 1\nv 0 0 -1\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n",
        encoding="ascii",
    )
    return path


@pytest.fixture
def configuration(sphere_mesh):
    default = load_workflow_configuration(DEFAULT_CONFIGURATION)
    return replace(
        default,
        sie=replace(default.sie, mesh=replace(default.sie.mesh, path=str(sphere_mesh))),
    )


@pytest.mark.parametrize("method", ["reciprocal_complex", "reciprocal_lossless"])
def test_example_configurations_preserve_physical_defaults_and_separate_runs(method):
    configuration = load_workflow_configuration(
        APPLICATION_DIR / "configurations" / f"{DEFAULT_CASE_ID}_{method}.json"
    )
    default = load_workflow_configuration(DEFAULT_CONFIGURATION)
    assert configuration == replace(default, extraction_method=method)
    assert configuration.to_dict()["schema_version"] == 2
    assert [state.id for state in configuration.transformations] == ["plane_wave_x", "plane_wave_y"]
    assert configuration.target_edge_length == 0.375
    assert default_run_directory(configuration.case_id, configuration.run_name) == (
        APPLICATION_DIR / "results" / DEFAULT_CASE_ID / "sie" / f"{method}_general"
    )
    assert configuration.mesh_path == APPLICATION_DIR / configuration.sie.mesh.path


def test_configuration_round_trip_and_partial_defaults(tmp_path):
    default = load_workflow_configuration(DEFAULT_CONFIGURATION)
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(default.to_dict()))
    assert load_workflow_configuration(path) == default
    raw = default.to_dict()
    required = {
        key: raw[key] for key in ("schema_version", "fields", "transformations", "simulation")
    }
    path.write_text(json.dumps(required))
    assert load_workflow_configuration(path) == default
    path.write_text("{}")
    with pytest.raises(ValueError, match="missing required"):
        load_workflow_configuration(path)


@pytest.mark.parametrize(
    "raw, message",
    [
        ([], "JSON object"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 1}, "schema_version"),
        ({"application": "omega"}, "different application"),
        ({"case_id": "../escape"}, "case_id"),
        ({"unused": 1}, "unknown configuration"),
        ({"physical": []}, "physical"),
        ({"physical": {"unused": 1}}, "unknown physical"),
        ({"physical": {"radius_m": 0}}, "radius_m"),
        ({"physical": {"frequency_hz": -1}}, "frequency_hz"),
        ({"physical": {"frequency_hz": float("nan")}}, "finite"),
        ({"physical": {"frequency_hz": "100000000"}}, "finite"),
        ({"physical": {"exterior": {"epsilon_r": 0}}}, "permittivity"),
        ({"physical": {"interior": {"unused": 1}}}, "unknown interior"),
        ({"extraction_method": "unrestricted"}, "extraction_method"),
        ({"tensor_structure": "isotropic"}, "tensor_structure"),
        ({"transformation": {"type": "plane_wave_cloaking"}}, "unknown configuration"),
        ({"sampling": {"zx_grid_size": 3}}, "even"),
        ({"sampling": {"zx_grid_size": 2.5}}, "integer"),
        ({"sampling": {"radial_points": True}}, "integer"),
        ({"sampling": {"radial_min_radii": 1}}, "between zero and one"),
        ({"sampling": {"radial_max_radii": 1}}, "exceed one"),
        ({"sampling": {"zx_span_radii": float("inf")}}, "finite"),
        ({"sampling": {"radial_theta_degrees": 181}}, "radial_theta"),
        ({"sie": {"mesh": {"path": ""}}}, "nonempty"),
        ({"sie": {"mesh": {"points_per_wavelength": 0}}}, "points_per_wavelength"),
        ({"sie": {"quadrature_rule": 21}}, "quadrature_rule"),
        ({"sie": {"quadrature_rule": 3.5}}, "quadrature_rule"),
        ({"sie": {"singularity_threshold_lambda0": 0}}, "singularity_threshold"),
        ({"sie": {"evaluate_fields": "false"}}, "booleans"),
        ({"sie": {"solver": "gmres"}}, "unknown sie"),
    ],
)
def test_invalid_configuration_is_rejected_without_creating_artifacts(tmp_path, raw, message):
    path = tmp_path / "invalid.json"
    if isinstance(raw, dict):
        raw = load_workflow_configuration(DEFAULT_CONFIGURATION).to_dict() | raw
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=message):
        load_workflow_configuration(path)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("identifier", ["../run", "/tmp/run", "Run", "", "two words"])
def test_result_paths_reject_unsafe_identifiers(identifier):
    with pytest.raises(ValueError, match="run_name"):
        default_run_directory("case", identifier)
    with pytest.raises(ValueError, match="case_id"):
        default_run_directory(identifier, "run")


@pytest.mark.parametrize("epsilon_r, mu_r, amplitude", [(1.0, 1.0, 1.0), (2.5, 1.3, -2.0)])
def test_prescribed_fields_have_unchanged_state_order_spherical_basis_and_phase(
    epsilon_r, mu_r, amplitude
):
    case = SphereCase(
        relative_permittivity_exterior=epsilon_r,
        relative_permeability_exterior=mu_r,
    )
    configuration = load_workflow_configuration(DEFAULT_CONFIGURATION)
    configuration = replace(
        configuration,
        case=case,
        fields={
            "plane_x": PlaneWaveFieldDefinition((amplitude, 0.0, 0.0), (0.0, 0.0, 1.0)),
            "plane_y": PlaneWaveFieldDefinition((0.0, amplitude, 0.0), (0.0, 0.0, 1.0)),
        },
    )
    theta = np.array([0.34, 1.13, 2.71])
    phi = np.array([0.2, -0.7, 1.1])
    states = prescribed_boundary_states(theta, phi, configuration)
    assert [state.state_id for state in states] == ["plane_wave_x", "plane_wave_y"]
    phase = np.exp(-1j * case.k_exterior * case.radius_m * np.cos(theta))

    def spherical(vector):
        x, y, z = vector
        return (
            np.stack(
                (
                    x * np.sin(theta) * np.cos(phi)
                    + y * np.sin(theta) * np.sin(phi)
                    + z * np.cos(theta),
                    x * np.cos(theta) * np.cos(phi)
                    + y * np.cos(theta) * np.sin(phi)
                    - z * np.sin(theta),
                    -x * np.sin(phi) + y * np.cos(phi),
                )
            )
            * phase
        )

    for state, vector in zip(states, ((amplitude, 0.0, 0.0), (0.0, amplitude, 0.0))):
        np.testing.assert_array_equal(state.electric_interior, np.zeros((3, 3)))
        np.testing.assert_array_equal(state.magnetic_interior, np.zeros((3, 3)))
        np.testing.assert_allclose(state.electric_exterior, spherical(vector), atol=2e-15)
        magnetic = np.cross((0.0, 0.0, 1.0), vector) / case.eta_exterior
        np.testing.assert_allclose(state.magnetic_exterior, spherical(magnetic), atol=2e-15)


def test_complex_plane_wave_preserves_amplitude_phase_and_uses_each_region_medium(configuration):
    case = SphereCase(
        relative_permittivity_interior=2.5,
        relative_permeability_interior=1.3,
        relative_permittivity_exterior=1.7,
        relative_permeability_exterior=1.1,
    )
    definition = PlaneWaveFieldDefinition((1 + 2j, -0.5j, 0), (0, 0, 2))
    configuration = replace(
        configuration,
        case=case,
        fields={"plane_x": definition, "plane_y": configuration.fields["plane_y"]},
        transformations=(
            GSTCTransformationConfiguration(
                "both_regions",
                GSTCFieldsConfiguration(FieldReference("plane_x"), FieldReference("plane_x")),
            ),
        ),
    )
    theta, phi = np.array([0.0, np.pi / 2]), np.array([0.0, 0.0])
    (state,) = prescribed_boundary_states(theta, phi, configuration, radius=0.7)
    # At these two points, Cartesian x/y project onto theta/phi and r/phi.
    amplitude = np.array([[0, 1 + 2j], [1 + 2j, 0], [-0.5j, -0.5j]])
    magnetic = np.array([[0, 0.5j], [0.5j, 0], [1 + 2j, 1 + 2j]])
    for region in ("interior", "exterior"):
        phase = np.exp(-1j * getattr(case, f"k_{region}") * 0.7 * np.cos(theta))
        np.testing.assert_allclose(
            getattr(state, f"electric_{region}"), amplitude * phase, atol=1e-14
        )
        np.testing.assert_allclose(
            getattr(state, f"magnetic_{region}"),
            magnetic * phase / getattr(case, f"eta_{region}"),
            atol=1e-14,
        )


@pytest.mark.parametrize("dipole_type", ["electric", "magnetic"])
@pytest.mark.parametrize("region", ["interior", "exterior"])
def test_prescribed_dipoles_allow_nonzero_interior_and_zero_exterior(
    configuration, dipole_type, region
):
    definition = DipoleFieldDefinition(dipole_type, (1 + 2j, -1j, 3), (0.1, -0.2, 0.3))
    other_region = "exterior" if region == "interior" else "interior"
    configuration = replace(
        configuration,
        case=replace(configuration.case, relative_permittivity_interior=2.5),
        fields={**configuration.fields, "dipole": definition},
        transformations=(
            GSTCTransformationConfiguration(
                "dipole_state",
                GSTCFieldsConfiguration(**{region: FieldReference("dipole"), other_region: None}),
            ),
        ),
    )
    theta, phi = np.array([0.3, 1.2, 2.7]), 0.5
    (state,) = prescribed_boundary_states(theta, phi, configuration)
    expected_e, expected_h = evaluate_dipole_fields(
        getattr(configuration.case, f"k_{region}"),
        configuration.case.radius_m,
        theta,
        phi,
        getattr(configuration.case, f"eta_{region}"),
        effective_moment=definition.effective_moment,
        source_position=definition.source_position,
        dipole_type=dipole_type,
    )
    np.testing.assert_array_equal(getattr(state, f"electric_{region}"), expected_e)
    np.testing.assert_array_equal(getattr(state, f"magnetic_{region}"), expected_h)
    np.testing.assert_array_equal(getattr(state, f"electric_{other_region}"), np.zeros((3, 3)))
    np.testing.assert_array_equal(getattr(state, f"magnetic_{other_region}"), np.zeros((3, 3)))


def test_prescribed_fields_do_not_depend_on_simulation_cases(configuration):
    theta, phi = np.array([0.3, 1.2]), 0.5
    expected = prescribed_boundary_states(theta, phi, configuration)
    configuration = replace(configuration, simulation=SimulationConfiguration(()))
    actual = prescribed_boundary_states(theta, phi, configuration)
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first.electric_exterior, second.electric_exterior)
        np.testing.assert_array_equal(first.magnetic_exterior, second.magnetic_exterior)


@pytest.mark.parametrize(
    "kind", ["complex_plane", "zero_plane", "complex_dipole", "zero_dipole", "empty"]
)
def test_unsupported_solver_excitations_fail_before_creating_artifacts(
    tmp_path, configuration, kind
):
    if kind == "empty":
        configuration = replace(configuration, simulation=SimulationConfiguration(()))
    else:
        amplitude = (1 + 1j, 0, 0) if kind.startswith("complex") else (0, 0, 0)
        if kind.endswith("dipole"):
            definition = DipoleFieldDefinition("magnetic", amplitude, (0, 0, 0.1))
        else:
            definition = PlaneWaveFieldDefinition(amplitude, (0, 0, 1))
        configuration = replace(
            configuration,
            fields={**configuration.fields, "incident": definition},
            simulation=SimulationConfiguration(
                (SimulationCaseConfiguration("test", (
                    SimulationExcitationConfiguration(FieldReference("incident"), "interior"),
                )),)
            ),
        )
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "unsupported"
    with pytest.raises(ValueError, match=r"C\+\+"):
        coordinator.run_sie_workflow(
            configuration,
            run_dir=run_dir,
            executable=executable,
            use_existing_mesh=True,
        )
    assert not run_dir.exists()


def test_solver_command_uses_independent_interior_dipole_excitation(tmp_path, configuration):
    configuration = replace(
        configuration,
        fields={
            **configuration.fields,
            "dipole_z": DipoleFieldDefinition("electric", (0, 0, -2), (0, 0, 0.5)),
        },
        simulation=SimulationConfiguration(
            (
                SimulationCaseConfiguration(
                    "dipole_incidence",
                    (SimulationExcitationConfiguration(FieldReference("dipole_z"), "interior"),),
                ),
            )
        ),
    )
    command = coordinator.solver_command(
        configuration, run_dir=tmp_path, executable=tmp_path / "solver"
    )
    index = command.index("--case")
    assert command[index:index + 11] == [
        "--case", "dipole_incidence", "--dipole", "interior", "electric",
        "0.0", "0.0", "-2.0", "0.0", "0.0", "0.5",
    ]
    assert command[command.index("--tensors") + 1] == str(tmp_path / "face_tensors.bin")
    assert command[command.index("--output-dir") + 1] == str(tmp_path)


@pytest.mark.parametrize("threshold", [None, 0.05])
def test_cli_threshold_override_is_saved_and_forwarded(
    tmp_path, configuration, monkeypatch, threshold,
):
    configuration = replace(configuration, sie=replace(configuration.sie, evaluate_fields=False))
    config_path = tmp_path / "case.json"
    config_path.write_text(json.dumps(configuration.to_dict()))
    original_bytes = config_path.read_bytes()
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "run"
    argv = [
        "sie_gstc_sphere.py", "--config", str(config_path), "--run-dir", str(run_dir),
        "--executable", str(executable), "--use-existing-mesh",
    ]
    if threshold is not None:
        argv.extend(("--singularity-threshold-lambda0", str(threshold)))
    expected = configuration.sie.singularity_threshold_lambda0 if threshold is None else threshold

    def fake_solver(command, *, check):
        assert check
        assert float(command[command.index("--singularity-threshold-lambda0") + 1]) == expected
        manifest = json.loads((run_dir / "run.json").read_text())
        assert manifest["solver"]["singularity_threshold_lambda0"] == expected
        (run_dir / "batch_metrics.json").write_text("{}")
        for case in manifest["simulation"]["cases"]:
            (run_dir / case["files"]["currents"]).touch()
            (run_dir / case["files"]["solver_metrics"]).write_text("{}")

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(coordinator.subprocess, "run", fake_solver)
    assert coordinator.main() == 0
    saved = load_workflow_configuration(run_dir / "configuration.json")
    assert saved == replace(
        configuration, sie=replace(configuration.sie, singularity_threshold_lambda0=expected)
    )
    inputs = json.loads((run_dir / "sie_inputs.json").read_text())
    assert inputs["configuration"]["sie"]["singularity_threshold_lambda0"] == expected
    assert config_path.read_bytes() == original_bytes


@pytest.mark.parametrize("threshold", ["0", "-0.1", "nan", "inf"])
def test_cli_rejects_invalid_threshold_override_before_creating_artifacts(
    tmp_path, configuration, monkeypatch, capsys, threshold
):
    config_path = tmp_path / "case.json"
    config_path.write_text(json.dumps(configuration.to_dict()))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "sie_gstc_sphere.py", "--config", str(config_path), "--run-dir", str(run_dir),
        f"--singularity-threshold-lambda0={threshold}",
    ])
    with pytest.raises(SystemExit) as error:
        coordinator.main()
    assert error.value.code == 2
    assert "sie.singularity_threshold_lambda0" in capsys.readouterr().err
    assert not run_dir.exists()


def test_prepare_only_accepts_complex_incident_fields(tmp_path, configuration):
    (simulation_case,) = configuration.simulation.cases
    configuration = replace(
        configuration,
        fields={
            **configuration.fields,
            "plane_x": replace(configuration.fields["plane_x"], electric_amplitude=(1j, 0, 0)),
        },
        simulation=SimulationConfiguration(
            (simulation_case, replace(simulation_case, id="another_case")),
        ),
    )
    path = coordinator.run_sie_workflow(
        configuration,
        run_dir=tmp_path / "prepared",
        use_existing_mesh=True,
        prepare_only=True,
    )
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "prepared"
    cases = manifest["simulation"]["cases"]
    assert [case["id"] for case in cases] == [simulation_case.id, "another_case"]
    assert all(case["status"] == "pending" and case["command"] is None for case in cases)
    assert cases[0]["files"]["currents"] != cases[1]["files"]["currents"]
    assert [case["excitations"] for case in cases] == [
        case["excitations"] for case in configuration.to_dict()["simulation"]["cases"]
    ]


@pytest.mark.parametrize("case_count", [0, 2])
def test_preparation_shares_one_extraction_for_empty_or_dual_region_cases(
    tmp_path, configuration, monkeypatch, case_count
):
    (original_case,) = configuration.simulation.cases
    (exterior,) = original_case.excitations
    interior = SimulationExcitationConfiguration(FieldReference("plane_y"), "interior")
    cases = (
        SimulationCaseConfiguration("combined", (interior, exterior)),
        SimulationCaseConfiguration("independent", (interior,)),
    )[:case_count]
    configuration = replace(configuration, simulation=SimulationConfiguration(cases))
    snapshot = configuration.to_dict()
    calls = []

    def record_export(configuration, *, mesh_path, run_dir, **kwargs):
        calls.append((mesh_path, run_dir))
        return export_sie_inputs(configuration, mesh_path=mesh_path, run_dir=run_dir, **kwargs)

    def unexpected_solver(*args, **kwargs):
        pytest.fail("preparation must not invoke C++")

    monkeypatch.setattr(coordinator, "export_sie_inputs", record_export)
    monkeypatch.setattr(coordinator.subprocess, "run", unexpected_solver)
    run_dir = tmp_path / "shared_preparation"
    path = coordinator.run_sie_workflow(
        configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True
    )
    manifest = json.loads(path.read_text())
    assert calls == [(configuration.mesh_path, run_dir)]
    assert configuration.to_dict() == snapshot
    assert json.loads((run_dir / "configuration.json").read_text()) == snapshot
    assert manifest["fields"] == snapshot["fields"]
    assert manifest["transformations"] == snapshot["transformations"]
    assert manifest["status"] == "prepared"
    assert len(manifest["simulation"]["cases"]) == case_count
    assert all(
        (run_dir / path).is_file()
        for name, path in manifest["files"].items() if name != "batch_metrics"
    )
    assert not (run_dir / manifest["files"]["batch_metrics"]).exists()
    for case, definition in zip(manifest["simulation"]["cases"], snapshot["simulation"]["cases"]):
        assert case["id"] == definition["id"]
        assert case["excitations"] == definition["excitations"]
        assert case["status"] == "pending" and case["command"] is None
        assert all(path.startswith(f"cases/{case['id']}/") for path in case["files"].values())
        assert not (run_dir / case["files"]["currents"]).exists()
    if case_count == 0:
        assert not (run_dir / "cases").exists()


def test_preparation_failure_leaves_simulation_cases_pending(tmp_path, configuration, monkeypatch):
    def failed_export(*args, **kwargs):
        raise ValueError("invalid prepared surface map")

    monkeypatch.setattr(coordinator, "export_sie_inputs", failed_export)
    run_dir = tmp_path / "preparation_failed"
    with pytest.raises(ValueError, match="invalid prepared surface map"):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True
        )
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"
    for case in manifest["simulation"]["cases"]:
        assert case["status"] == "pending"
        assert case["command"] is None
        assert "error" not in case
    assert not (run_dir / "cases").exists()


@pytest.mark.parametrize("method", ["reciprocal_complex", "reciprocal_lossless"])
def test_prepare_only_records_configuration_inputs_and_mesh_but_never_runs_solver(
    tmp_path, configuration, monkeypatch, method
):
    configuration = replace(configuration, extraction_method=method)

    def unexpected_solver(*args, **kwargs):
        pytest.fail("prepare-only must not start a solver")

    monkeypatch.setattr(coordinator.subprocess, "run", unexpected_solver)
    run_dir = tmp_path / method
    path = coordinator.run_sie_workflow(
        configuration,
        run_dir=run_dir,
        executable=tmp_path / "missing_solver",
        use_existing_mesh=True,
        prepare_only=True,
    )
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "prepared"
    assert manifest["tangent_basis"] == "theta, phi; outward radial normal"
    assert manifest["prescribed_state_ids"] == ["plane_wave_x", "plane_wave_y"]
    assert manifest["mesh"]["face_count"] == 8
    assert len(manifest["mesh"]["sha256"]) == 64
    assert load_workflow_configuration(run_dir / "configuration.json") == configuration
    inputs = json.loads((run_dir / "sie_inputs.json").read_text())
    assert inputs["configuration"] == configuration.to_dict()
    assert inputs["mesh"] == manifest["mesh"]
    assert len(inputs["diagnostics"]["maximum_state_relative_residual"]) == 2
    assert manifest["schema_version"] == 2
    assert manifest["files"] == {
        "configuration": "configuration.json",
        "sie_inputs": "sie_inputs.json",
        "face_tensors": "face_tensors.bin",
        "batch_metrics": "batch_metrics.json",
    }
    for case in manifest["simulation"]["cases"]:
        assert (run_dir / case["files"]["plots"]).is_dir()
        assert case["status"] == "pending"
        assert case["command"] is None
        assert not (run_dir / case["files"]["currents"]).exists()
    assert not (run_dir / "currents.bin").exists()
    records = np.fromfile(run_dir / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    if method == "reciprocal_lossless":
        matrix = records[:, :16].reshape(8, 4, 4)
        np.testing.assert_allclose(matrix + matrix.conj().transpose(0, 2, 1), 0.0)
        assert inputs["diagnostics"]["maximum_relative_residual"] > 0.1


@pytest.mark.parametrize("method", ["reciprocal_complex", "reciprocal_lossless"])
def test_cloaking_tensor_bytes_match_previous_prescribed_fields(
    tmp_path, configuration, monkeypatch, method
):
    configuration = replace(configuration, extraction_method=method)
    current, _ = export_sie_inputs(
        configuration,
        mesh_path=configuration.mesh_path,
        run_dir=tmp_path / "current",
    )

    def previous_fields(theta, phi, configuration, *, radius):
        case = configuration.case
        states = []
        for state_id, vector in (
            ("plane_wave_x", (1.0, 0.0, 0.0)),
            ("plane_wave_y", (0.0, 1.0, 0.0)),
        ):
            electric, magnetic = evaluate_plane_wave_fields(
                case.k_exterior,
                radius,
                theta,
                phi,
                case.eta_exterior,
                electric_field=vector,
                propagation_direction=(0.0, 0.0, 1.0),
            )
            zero = np.zeros_like(electric)
            states.append(
                PrescribedFieldState(
                    state_id,
                    zero,
                    zero,
                    np.conjugate(electric),
                    np.conjugate(magnetic),
                )
            )
        return tuple(states)

    monkeypatch.setattr(preparation, "prescribed_boundary_states", previous_fields)
    reference, _ = export_sie_inputs(
        configuration,
        mesh_path=configuration.mesh_path,
        run_dir=tmp_path / "reference",
    )
    assert current.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("case_count", [1, 2])
def test_workflow_forwards_supported_settings_and_records_completion(
    tmp_path, configuration, monkeypatch, case_count
):
    configuration = replace(
        configuration,
        case=replace(configuration.case, relative_permittivity_interior=2.0),
        fields={
            name: replace(
                definition, electric_amplitude=tuple(2 * c for c in definition.electric_amplitude)
            )
            for name, definition in configuration.fields.items()
        },
        sampling=SamplingConfiguration(zx_grid_size=8, radial_points=9, radial_phi_degrees=30.0),
        sie=replace(configuration.sie, quadrature_rule=13, singularity_threshold_lambda0=0.2),
    )
    (original_case,) = configuration.simulation.cases
    configuration = replace(
        configuration,
        simulation=SimulationConfiguration((
            original_case,
            SimulationCaseConfiguration("combined", (
                SimulationExcitationConfiguration(FieldReference("plane_y"), "interior"),
                original_case.excitations[0],
            )),
        )[:case_count]),
    )
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "completed"
    calls = []

    def fake_solver(command, *, check):
        assert check
        calls.append(command)
        running = json.loads((run_dir / "run.json").read_text())
        assert running["status"] == "running"
        (run_dir / "batch_metrics.json").write_text("{}")
        for case in running["simulation"]["cases"]:
            assert case["status"] == "running"
            assert case["command"] == command
            for name, file_name in case["files"].items():
                if name != "plots":
                    output = run_dir / file_name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"test output")

    monkeypatch.setattr(coordinator.subprocess, "run", fake_solver)
    manifest_path = coordinator.run_sie_workflow(
        configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    cases = manifest["simulation"]["cases"]
    assert all(case["status"] == "completed" for case in cases)
    assert calls == [cases[0]["command"]]
    command = calls[0]
    assert command.count("--case") == case_count
    index = command.index("--plane-wave")
    assert command[index:index + 8] == [
        "--plane-wave", "exterior", "2.0", "0.0", "0.0", "0.0", "0.0", "1.0",
    ]
    if case_count == 2:
        index = command.index("combined")
        assert command[index + 1:index + 9] == [
            "--plane-wave", "interior", "0.0", "2.0", "0.0", "0.0", "0.0", "1.0",
        ]
    for name, value in {
        "--epsilon-r-interior": "2.0",
        "--quadrature-rule": "13",
        "--singularity-threshold-lambda0": "0.2",
        "--zx-grid-size": "8",
        "--radial-points": "9",
        "--radial-phi-degrees": "30.0",
        "--tensors": str(run_dir / "face_tensors.bin"),
        "--output-dir": str(run_dir),
    }.items():
        assert command[command.index(name) + 1] == value
    for case in cases:
        assert case["command"] == command
        assert case["files"]["computed_zx"] == f"cases/{case['id']}/fields/sie_physical_zx.csv"
    assert not (run_dir / "currents.bin").exists()


def test_disabled_field_and_residual_settings_are_forwarded(tmp_path, configuration, monkeypatch):
    configuration = replace(
        configuration, sie=replace(configuration.sie, evaluate_fields=False, compute_residual=False)
    )
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "no_fields"

    def fake_solver(command, *, check):
        assert command[-2:] == ["--no-residual", "--no-fields"]
        output = run_dir / "cases" / configuration.simulation.cases[0].id
        (output / "currents.bin").touch()
        (output / "solver_metrics.json").write_text("{}")
        (run_dir / "batch_metrics.json").write_text("{}")

    monkeypatch.setattr(coordinator.subprocess, "run", fake_solver)
    path = coordinator.run_sie_workflow(
        configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
    )
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "completed"
    (case,) = manifest["simulation"]["cases"]
    assert case["status"] == "completed"
    assert "computed_zx" not in case["files"]
    assert not (run_dir / "cases" / case["id"] / "fields").exists()


@pytest.mark.parametrize("case_count", [1, 2])
def test_workflow_records_solver_failure_without_claiming_completion(
    tmp_path, configuration, monkeypatch, case_count
):
    (simulation_case,) = configuration.simulation.cases
    configuration = replace(configuration, simulation=SimulationConfiguration((
        simulation_case, replace(simulation_case, id="another_case"),
    )[:case_count]))
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "failed"

    def failed_solver(command, *, check):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(coordinator.subprocess, "run", failed_solver)
    with pytest.raises(subprocess.CalledProcessError):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
        )
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "CalledProcessError"
    for case in manifest["simulation"]["cases"]:
        assert case["status"] == "failed"
        assert case["error"] == manifest["error"]


def test_missing_later_case_output_does_not_mark_validated_case_failed(
    tmp_path, configuration, monkeypatch
):
    (original_case,) = configuration.simulation.cases
    configuration = replace(configuration, simulation=SimulationConfiguration((
        original_case, replace(original_case, id="missing"),
    )))
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "partial"

    def fake_solver(command, *, check):
        (run_dir / "batch_metrics.json").write_text("{}")
        manifest = json.loads((run_dir / "run.json").read_text())
        for name, file_name in manifest["simulation"]["cases"][0]["files"].items():
            if name != "plots":
                output = run_dir / file_name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()

    monkeypatch.setattr(coordinator.subprocess, "run", fake_solver)
    with pytest.raises(FileNotFoundError, match="cases/missing/solver_metrics.json"):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
        )
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    first, second = manifest["simulation"]["cases"]
    assert first["status"] == "completed" and "error" not in first
    assert second["status"] == "failed" and second["error"] == manifest["error"]


def test_workflow_checks_solver_outputs_after_successful_exit(tmp_path, configuration, monkeypatch):
    executable = tmp_path / "solver"
    executable.touch()
    run_dir = tmp_path / "missing_output"
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="required output"):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
        )
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["simulation"]["cases"][0]["status"] == "failed"


def test_workflow_preserves_existing_runs_and_meshes(tmp_path, configuration):
    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    marker = run_dir / "currents.bin"
    marker.write_bytes(b"existing result")
    mesh_bytes = configuration.mesh_path.read_bytes()
    with pytest.raises(FileExistsError, match="run directory"):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True
        )
    assert marker.read_bytes() == b"existing result"
    with pytest.raises(FileExistsError, match="use --use-existing-mesh"):
        coordinator.run_sie_workflow(configuration, run_dir=tmp_path / "new", prepare_only=True)
    assert configuration.mesh_path.read_bytes() == mesh_bytes
    assert not (tmp_path / "new").exists()


def test_workflow_checks_executable_before_creating_run_or_mesh(tmp_path, configuration):
    configuration = replace(
        configuration,
        sie=replace(
            configuration.sie, mesh=replace(configuration.sie.mesh, path=str(tmp_path / "new.obj"))
        ),
    )
    with pytest.raises(FileNotFoundError, match="SIE executable"):
        coordinator.run_sie_workflow(
            configuration, run_dir=tmp_path / "new_run", executable=tmp_path / "missing_solver"
        )
    assert not (tmp_path / "new_run").exists()
    assert not configuration.mesh_path.exists()


def test_workflow_uses_configured_mesh_sizing_without_changing_generator(
    tmp_path, configuration, monkeypatch
):
    configuration = replace(
        configuration,
        sie=replace(
            configuration.sie, mesh=replace(configuration.sie.mesh, path=str(tmp_path / "new.obj"))
        ),
    )
    original = tmp_path / "sphere.obj"
    calls = []

    def fake_generator(radius, target_edge, output, *, repo_root):
        calls.append((radius, target_edge, output))
        output.write_bytes(original.read_bytes())

    monkeypatch.setattr(coordinator, "generate_mesh", fake_generator)
    coordinator.run_sie_workflow(configuration, run_dir=tmp_path / "generated", prepare_only=True)
    assert calls == [(1.0, 0.375, configuration.mesh_path)]


def test_preparation_rejects_mesh_radius_mismatch_before_export(tmp_path, configuration):
    configuration = replace(configuration, case=replace(configuration.case, radius_m=2.0))
    output = tmp_path / "wrong_radius"
    with pytest.raises(ValueError, match="configured sphere radius"):
        export_sie_inputs(configuration, mesh_path=configuration.mesh_path, run_dir=output)
    assert not output.exists()


def supplied_parameters(theta, phi, eta):
    """A full-rank reciprocal-lossless map varying with both face directions."""
    count = len(theta)
    electric = np.zeros((2, 2, count), complex)
    magnetic = np.zeros_like(electric)
    coupling = np.zeros_like(electric)
    electric[0, 0] = 1j * (2 + np.cos(theta)) / eta
    electric[1, 1] = 2j / eta
    magnetic[0, 0] = 3j * eta
    magnetic[1, 1] = 4j * eta
    coupling[0, 1] = 0.1 * np.sin(phi)
    return SurfaceParameters(electric, coupling, -coupling.swapaxes(0, 1), magnetic)


def test_parameter_evaluator_replaces_only_extraction(tmp_path, configuration, monkeypatch):
    configuration = replace(configuration, extraction_method="reciprocal_lossless")
    expected_configuration = configuration.to_dict()
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("provided parameters must not evaluate prescribed fields or extract")

    def evaluator(theta, phi):
        calls.append((theta.copy(), phi.copy()))
        return supplied_parameters(theta, phi, configuration.case.eta_exterior)

    monkeypatch.setattr(preparation, "prescribed_boundary_states", forbidden)
    monkeypatch.setattr(preparation, "extract_reciprocal_lossless_least_squares", forbidden)
    monkeypatch.setattr(coordinator.subprocess, "run", forbidden)
    source = {"method": "test_saved_surface", "angular_degree": 1}
    run_dir = tmp_path / "provided"
    path = coordinator.run_sie_workflow(
        configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True,
        parameter_evaluator=evaluator, parameter_source=source,
    )
    manifest = json.loads(path.read_text())
    inputs = json.loads((run_dir / "sie_inputs.json").read_text())
    assert len(calls) == 1
    mesh = preparation.load_triangle_obj(configuration.mesh_path)
    theta, phi = preparation.face_centroid_spherical_coordinates(mesh)
    np.testing.assert_array_equal(calls[0], (theta, phi))
    records = np.fromfile(run_dir / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    expected = supplied_parameters(theta, phi, configuration.case.eta_exterior).as_matrix()
    np.testing.assert_array_equal(records[:, :16].reshape(8, 4, 4), expected.transpose(2, 0, 1))
    np.testing.assert_allclose(
        records[:, :16].reshape(8, 4, 4) @ records[:, 16:].reshape(8, 4, 4),
        np.broadcast_to(np.eye(4), (8, 4, 4)), atol=1e-14,
    )
    for metadata in (manifest, inputs):
        assert metadata["parameter_source"] == source
        assert metadata["extraction_method"] is None
        assert metadata["prescribed_state_ids"] == []
    assert inputs["diagnostics"]["maximum_relative_reciprocity_residual"] == 0
    assert inputs["diagnostics"]["maximum_relative_skew_hermitian_residual"] == 0
    assert "minimum_rank" not in inputs["diagnostics"]
    assert manifest["status"] == "prepared"
    assert configuration.to_dict() == expected_configuration
    assert load_workflow_configuration(run_dir / "configuration.json") == configuration


@pytest.mark.parametrize("problem, message", [
    ("wrong_type", "return SurfaceParameters"),
    ("shape", "shape"),
    ("nonfinite", "finite"),
    ("nonreciprocal", "reciprocity"),
    ("nonlossless", "losslessness"),
    ("singular", "singular"),
    ("numerically_singular", "singular"),
    ("diagonal", "diagonal"),
])
def test_invalid_supplied_maps_fail_before_binary_export(tmp_path, configuration, problem, message):
    configuration = replace(configuration, extraction_method="reciprocal_lossless")
    if problem == "diagonal":
        configuration = replace(configuration, tensor_structure="diagonal")

    def evaluator(theta, phi):
        parameters = supplied_parameters(theta, phi, configuration.case.eta_exterior)
        if problem == "wrong_type":
            return parameters.as_matrix()
        if problem == "shape":
            return SurfaceParameters(*(np.zeros((2, 2, 1)) for _ in range(4)))
        if problem == "nonfinite":
            parameters.Y_ee[0, 0, 0] = np.nan
        if problem == "nonreciprocal":
            parameters.Y_ee[0, 1, 0] = 1j
        if problem == "nonlossless":
            parameters.Y_ee[0, 0, 0] += 1
        if problem in ("singular", "numerically_singular"):
            parameters.Y_ee[:, :, 0] = np.diag([0 if problem == "singular" else 1e-30j, 1j])
            parameters.Y_em[:, :, 0] = parameters.Y_me[:, :, 0] = 0
        return parameters

    run_dir = tmp_path / problem
    with pytest.raises(ValueError, match=message):
        coordinator.run_sie_workflow(
            configuration, run_dir=run_dir, use_existing_mesh=True, prepare_only=True,
            parameter_evaluator=evaluator,
        )
    assert not (run_dir / "face_tensors.bin").exists()
    assert not (run_dir / "sie_inputs.json").exists()
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"


def test_direct_export_refuses_to_overwrite_inputs(tmp_path, configuration):
    output = tmp_path / "inputs"
    export_sie_inputs(configuration, mesh_path=configuration.mesh_path, run_dir=output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="SIE input already exists"):
        export_sie_inputs(configuration, mesh_path=configuration.mesh_path, run_dir=output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_inaccurate_inverse_is_rejected_before_export(tmp_path, configuration, monkeypatch):
    monkeypatch.setattr(preparation.np.linalg, "inv", lambda matrix: np.zeros_like(matrix))
    output = tmp_path / "bad_inverse"
    with pytest.raises(ValueError, match="inversion residual"):
        export_sie_inputs(
            configuration, mesh_path=configuration.mesh_path, run_dir=output,
            parameter_evaluator=lambda theta, phi: supplied_parameters(
                theta, phi, configuration.case.eta_exterior
            ),
        )
    assert not output.exists()


@pytest.mark.parametrize("problem, message", [
    ("not_callable", "must be callable"),
    ("source_without_evaluator", "requires a parameter_evaluator"),
    ("empty_method", "nonempty string"),
    ("nonfinite_metadata", "JSON compliant"),
])
def test_bad_evaluator_arguments_fail_before_creating_run(
    tmp_path, configuration, problem, message,
):
    arguments = {"parameter_evaluator": lambda theta, phi: None}
    if problem == "not_callable":
        arguments["parameter_evaluator"] = 1
    if problem == "source_without_evaluator":
        arguments = {"parameter_source": {"method": "unused"}}
    if problem == "empty_method":
        arguments["parameter_source"] = {"method": ""}
    if problem == "nonfinite_metadata":
        arguments["parameter_source"] = {"method": "test", "bad_value": float("nan")}
    destination = tmp_path / "invalid_evaluator"
    with pytest.raises(ValueError, match=message):
        coordinator.run_sie_workflow(
            configuration, run_dir=destination, use_existing_mesh=True,
            prepare_only=True, **arguments,
        )
    assert not destination.exists()


def test_complex_evaluator_does_not_impose_losslessness(tmp_path, configuration):
    assert configuration.extraction_method == "reciprocal_complex"

    def evaluator(theta, phi):
        parameters = supplied_parameters(theta, phi, configuration.case.eta_exterior)
        parameters.Y_ee[0, 0] += 1 / configuration.case.eta_exterior
        return parameters

    _, path = export_sie_inputs(
        configuration, mesh_path=configuration.mesh_path, run_dir=tmp_path / "complex",
        parameter_evaluator=evaluator,
    )
    inputs = json.loads(path.read_text())
    assert inputs["diagnostics"]["maximum_relative_skew_hermitian_residual"] > 0.1
    assert inputs["parameter_source"] == {"method": "provided_parameters"}


def test_reciprocity_is_checked_at_each_face_not_against_largest_map(tmp_path, configuration):
    def evaluator(theta, phi):
        parameters = supplied_parameters(theta, phi, configuration.case.eta_exterior)
        for block in (parameters.Y_ee, parameters.Y_em, parameters.Y_me, parameters.Y_mm):
            block[:, :, 0] *= 1e15
        parameters.Y_ee[0, 1, 1] = 1j / configuration.case.eta_exterior
        return parameters

    with pytest.raises(ValueError, match="reciprocity"):
        export_sie_inputs(
            configuration, mesh_path=configuration.mesh_path, run_dir=tmp_path / "bad_face",
            parameter_evaluator=evaluator,
        )
