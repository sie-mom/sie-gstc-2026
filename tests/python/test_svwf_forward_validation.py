"""Independent forward resolution checks of a fixed saved inverse material."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from applications.anisotropic_gstc_sphere import plot_fields as plotting
from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    inverse_synthesis as experiment,
)
from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    prepare_plot_fields as postprocess,
)
from applications.anisotropic_gstc_sphere.workflow import (
    FieldReference,
    SamplingConfiguration,
    load_workflow_configuration,
)
from spherical_expansion.coefficients import load_coefficients


@pytest.fixture
def saved_surface(tmp_path):
    configuration = load_workflow_configuration(experiment.DEFAULT_CONFIG)
    configuration = replace(configuration, sampling=SamplingConfiguration(
        zx_grid_size=4, radial_points=6
    ))
    directory = tmp_path / "inverse"
    directory.mkdir()
    metadata = {
        "method": "joint_svwf_reciprocal_lossless", "status": "completed",
        "n_max": 1, "n_theta": 4, "n_phi": 8, "angular_degree": 0,
        "reference_impedance_ohm": 200.0,
        "angular_profile_names": ["constant"],
        "material_parameter_names": list(experiment.PARAMETER_NAMES),
        "field_and_material_time_convention": "exp(+j omega t)",
        "state_cases": [list(pair) for pair in reversed(experiment.STATE_CASES)],
    }
    coefficients = np.array([0.2, 0.3, 0.04, 0.5, 0.4, 0.06, 0.07, -0.1, 0.03, 0.08])[:, None]
    (directory / "configuration.json").write_text(json.dumps(configuration.to_dict()))
    (directory / "experiment.json").write_text(json.dumps(metadata))
    np.savez(directory / "surface_parameters.npz", angular_coefficients=coefficients)
    return directory, configuration, coefficients


def input_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.is_file()
    }


def test_forward_only_uses_saved_material_and_new_resolution_without_inverse_work(
    saved_surface, monkeypatch,
):
    directory, configuration, material = saved_surface
    before = input_hashes(directory)

    def forbidden(*args, **kwargs):
        pytest.fail("forward-only mode must not construct or solve an inverse problem")

    for name in ("prepare_problem", "least_squares", "extract_reciprocal_lossless_least_squares",
                 "fit_trace"):
        monkeypatch.setattr(experiment, name, forbidden)
    original_solve = experiment.solve_collocation_general
    calls = []

    def inspect_solve(**kwargs):
        calls.append(kwargs)
        assert kwargs["n_max"] == 2
        assert kwargs["n_theta"] == 5
        assert kwargs["n_phi"] == 10
        theta, phi = np.array([0.3, 1.4]), np.array([0.1, 2.0])
        profiles, _ = experiment.angular_basis(theta, phi, 0)
        expected = experiment.surface_parameters(material, profiles, 200.0)
        for name, attribute in (("Y", "Y_ee"), ("Chi", "Y_em"),
                                ("Gamma", "Y_me"), ("Z", "Y_mm")):
            np.testing.assert_array_equal(
                kwargs["gstc_params"][name](theta, phi), getattr(expected, attribute).conj()
            )
        return original_solve(**kwargs)

    monkeypatch.setattr(experiment, "solve_collocation_general", inspect_solve)
    output = experiment.run_forward_validation(directory, n_max=2, n_theta=5, n_phi=10)
    manifest = json.loads((output / "experiment.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert output.parent == directory / "forward_validation"
    assert manifest["method"] == "forward_svwf_reciprocal_lossless"
    assert manifest["status"] == "completed"
    assert manifest["n_max"] == 2 and manifest["angular_degree"] == 0
    assert manifest["state_cases"] == [list(pair) for pair in reversed(experiment.STATE_CASES)]
    assert manifest["parameter_source"]["sha256"] == before
    assert len(calls) == 2
    assert set(metrics) == {"cases", "total_seconds"}
    assert load_workflow_configuration(output / "configuration.json") == configuration
    for case in configuration.simulation.cases:
        coefficients = load_coefficients(output / "cases" / case.id / "forward_coefficients.npz")
        assert all(len(family) == 8 for family in coefficients)
        assert all(np.isfinite(list(family.values())).all() for family in coefficients)
        assert not (output / "cases" / case.id / "inverse_coefficients.npz").exists()
        with np.load(output / "cases" / case.id / "validation_fields.npz") as fields:
            assert fields["interior_total"].shape == fields["exterior_total"].shape == (4, 98)
            grid = experiment.SurfaceGrid(fields["theta"], fields["phi"],
                                          fields["solid_angle_weights"])
            simulation_case = next(item for item in configuration.simulation.cases
                                   if item.id == case.id)
            for region, families in (("interior", coefficients[2:]),
                                     ("exterior", coefficients[:2])):
                total = experiment.incident_trace(configuration, simulation_case, region, grid)
                total += experiment.reconstructed_trace(configuration, *families, region, grid)
                np.testing.assert_allclose(fields[f"{region}_total"], total.conj(), atol=1e-12)
    assert not (output / "history.csv").exists()
    assert not (output / "synthesis.npz").exists()
    assert input_hashes(directory) == before


def test_forward_defaults_are_saved_settings_not_current_constants(saved_surface, monkeypatch):
    directory, _, _ = saved_surface
    for name in ("N_MAX", "N_THETA", "N_PHI", "ANGULAR_DEGREE"):
        monkeypatch.setattr(experiment, name, 1000)
    output = experiment.run_forward_validation(directory)
    manifest = json.loads((output / "experiment.json").read_text())
    assert [manifest[name] for name in ("n_max", "n_theta", "n_phi")] == [1, 4, 8]


def test_same_resolution_reproduces_original_post_optimization_forward_result(saved_surface):
    directory, configuration, _ = saved_surface
    inverse = experiment.run_experiment(
        configuration, directory / "original_fit", n_max=1, n_theta=4, n_phi=8,
        angular_degree=1, max_nfev=1, verbose=0,
    )
    repeated = experiment.run_forward_validation(inverse)
    for case in configuration.simulation.cases:
        first = load_coefficients(inverse / "cases" / case.id / "forward_coefficients.npz")
        second = load_coefficients(repeated / "cases" / case.id / "forward_coefficients.npz")
        for original, recalculated in zip(first, second):
            np.testing.assert_allclose(list(original.values()), list(recalculated.values()),
                                       rtol=1e-12, atol=1e-12)


def test_exterior_targets_do_not_constrain_the_forward_solution(saved_surface):
    directory, configuration, _ = saved_surface
    _, evaluator, _ = experiment.load_parameter_evaluator(directory)
    original = experiment.solve_forward_response(
        configuration, evaluator, state_cases=experiment.STATE_CASES,
        n_max=1, n_theta=4, n_phi=8,
    )
    target_id = configuration.transformations[0].gstc.exterior.field_id
    changed = replace(
        configuration,
        fields={**configuration.fields, "different_target": replace(
            configuration.fields[target_id], effective_moment=(0.5, 2j, 0.3)
        )},
        transformations=(replace(configuration.transformations[0], gstc=replace(
            configuration.transformations[0].gstc, exterior=FieldReference("different_target")
        )), *configuration.transformations[1:]),
    )
    altered = experiment.solve_forward_response(
        changed, evaluator, state_cases=experiment.STATE_CASES, n_max=1, n_theta=4, n_phi=8,
    )
    for first, second in zip(original, altered):
        assert first["coefficients"] == second["coefficients"]
        np.testing.assert_array_equal(first["interior"], second["interior"])
        np.testing.assert_array_equal(first["exterior"], second["exterior"])
    assert original[0]["metrics"]["exterior_relative_field_error"] != altered[0]["metrics"][
        "exterior_relative_field_error"
    ]


@pytest.mark.parametrize("settings", [
    {"n_max": 0}, {"n_max": True}, {"n_theta": 1}, {"n_phi": 2}, {"n_max": 5},
])
def test_invalid_forward_resolution_fails_before_output(saved_surface, settings):
    directory, _, _ = saved_surface
    output = directory / "bad_resolution"
    with pytest.raises(ValueError):
        experiment.run_forward_validation(directory, output, **settings)
    assert not output.exists()


@pytest.mark.parametrize("mapping", [None, [], [["unknown", "electric_incidence"]], ["invalid"]])
def test_bad_saved_mapping_fails_before_output(saved_surface, mapping):
    directory, _, _ = saved_surface
    path = directory / "experiment.json"
    metadata = json.loads(path.read_text())
    metadata["state_cases"] = mapping
    path.write_text(json.dumps(metadata))
    output = directory / "bad_mapping"
    with pytest.raises(ValueError, match="STATE_CASES"):
        experiment.run_forward_validation(directory, output)
    assert not output.exists()


def test_solver_failure_records_failed_status(saved_surface, monkeypatch):
    directory, _, _ = saved_surface
    output = directory / "failed"

    def fail(**kwargs):
        raise ValueError("test forward failure")

    monkeypatch.setattr(experiment, "solve_collocation_general", fail)
    with pytest.raises(ValueError, match="test forward failure"):
        experiment.run_forward_validation(directory, output)
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "test forward failure"


def test_forward_result_reuses_plotting_and_protects_saved_outputs(saved_surface):
    directory, _, _ = saved_surface
    output = experiment.run_forward_validation(directory, directory / "forward_test")
    before = input_hashes(directory)
    saved_bytes = (output / "cases/electric_incidence/forward_coefficients.npz").read_bytes()
    with pytest.raises(FileExistsError, match="run directory"):
        experiment.run_forward_validation(directory, output)
    assert (
        output / "cases/electric_incidence/forward_coefficients.npz"
    ).read_bytes() == saved_bytes
    with pytest.raises(ValueError, match="no inverse candidate"):
        postprocess.prepare_plot_fields(output, solution="inverse")
    prepared = postprocess.prepare_plot_fields(output)
    paths = plotting.plot_fields(
        prepared, case_id="electric_incidence", reference_transformation="electric_z_to_x",
        line_components=("E_theta", "H_phi"), map_components=("E_theta", "H_phi"),
    )
    assert paths
    for path in paths:
        with Image.open(path) as picture:
            assert np.asarray(picture.convert("RGB")).std() > 5
    assert input_hashes(directory) == before


def test_forward_cli_from_other_working_directory(saved_surface, tmp_path):
    directory, _, _ = saved_surface
    output = directory / "cli_forward"
    result = subprocess.run(
        [sys.executable, str(Path(experiment.__file__).resolve()), "--forward-only",
         "--experiment-dir", str(directory), "--output-dir", str(output),
         "--n-max", "2", "--n-theta", "5", "--n-phi", "10"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "experiment.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["n_max"] == 2


@pytest.mark.parametrize("arguments", [
    ["--forward-only"], ["--n-max", "2"], ["--experiment-dir", "unused"],
    ["--forward-only", "--config", "unused"],
])
def test_cli_rejects_inconsistent_modes(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inverse_synthesis.py", *arguments])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
