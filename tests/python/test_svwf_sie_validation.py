"""Saved inverse materials enter the same mesh preparation and SIE batch workflow."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from applications.anisotropic_gstc_sphere import prepare_sie_inputs as preparation
from applications.anisotropic_gstc_sphere import sie_gstc_sphere as coordinator
from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    inverse_synthesis as inverse,
)
from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    run_sie_validation as validation,
)
from applications.anisotropic_gstc_sphere.workflow import (
    APPLICATION_DIR,
    load_workflow_configuration,
)


@pytest.fixture
def saved_surface(tmp_path):
    mesh = tmp_path / "sphere.obj"
    mesh.write_text(
        "v 2 0 0\nv -2 0 0\nv 0 2 0\nv 0 -2 0\nv 0 0 2\nv 0 0 -2\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n", encoding="ascii",
    )
    configuration = load_workflow_configuration(
        APPLICATION_DIR / "configurations" / "moment_rot" / "rot90_eps7.json"
    )
    configuration = replace(
        configuration,
        case=replace(configuration.case, relative_permittivity_interior=7.0),
        sie=replace(configuration.sie, mesh=replace(configuration.sie.mesh, path=str(mesh))),
    )
    directory = tmp_path / "experiment"
    directory.mkdir()
    (directory / "configuration.json").write_text(json.dumps(configuration.to_dict()))
    grid = inverse.surface_grid(4, 8)
    profiles, names = inverse.angular_basis(grid.theta, grid.phi, 1)
    coefficients = np.zeros((10, 4))
    coefficients[[0, 1, 3, 4], 0] = [1, 2, 3, 4]
    coefficients[0, 1] = 0.1
    coefficients[6, 2] = 0.05
    eta = configuration.case.eta_exterior
    parameters = inverse.surface_parameters(coefficients, profiles, eta)
    metadata = {
        "method": "joint_svwf_reciprocal_lossless", "status": "completed",
        "field_and_material_time_convention": "exp(+j omega t)",
        "angular_degree": 1, "reference_impedance_ohm": eta,
        "angular_profile_names": list(names),
        "material_parameter_names": list(inverse.PARAMETER_NAMES),
    }
    (directory / "experiment.json").write_text(json.dumps(metadata))
    np.savez(
        directory / "surface_parameters.npz", angular_coefficients=coefficients,
        theta=grid.theta, phi=grid.phi, Y=parameters.Y_ee, Chi=parameters.Y_em,
        Gamma=parameters.Y_me, Z=parameters.Y_mm,
    )
    return directory, configuration, coefficients


def source_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir() if path.is_file()
    }


def test_evaluator_uses_saved_settings_not_current_experiment_defaults(saved_surface, monkeypatch):
    directory, expected_configuration, coefficients = saved_surface
    before = source_hashes(directory)
    monkeypatch.setattr(inverse, "ANGULAR_DEGREE", 0)
    monkeypatch.setattr(inverse, "DEFAULT_CONFIG", Path("missing_configuration.json"))
    configuration, evaluator, source = validation.load_parameter_evaluator(directory)
    assert configuration == expected_configuration
    theta, phi = np.array([0.2, 1.1, 2.7]), np.array([-1.7, 0.4, 3.8])
    profiles, _ = inverse.angular_basis(theta, phi, 1)
    expected = inverse.surface_parameters(coefficients, profiles, source["reference_impedance_ohm"])
    np.testing.assert_array_equal(evaluator(theta, phi).as_matrix(), expected.as_matrix())
    assert source["sha256"] == before
    assert source["method"] == "svwf_inverse"
    assert source["angular_degree"] == 1
    assert source_hashes(directory) == before


def test_saved_material_uses_shared_preparation_once_without_any_solver(saved_surface, monkeypatch):
    directory, configuration, coefficients = saved_surface
    before = source_hashes(directory)
    loads = []
    original_load = preparation.load_triangle_obj

    def load_once(path):
        loads.append(path)
        return original_load(path)

    def forbidden(*args, **kwargs):
        pytest.fail("saved-material preparation must not extract, optimize, or solve")

    monkeypatch.setattr(preparation, "load_triangle_obj", load_once)
    monkeypatch.setattr(preparation, "prescribed_boundary_states", forbidden)
    monkeypatch.setattr(preparation, "extract_reciprocal_lossless_least_squares", forbidden)
    monkeypatch.setattr(inverse, "least_squares", forbidden)
    monkeypatch.setattr(inverse, "solve_collocation_general", forbidden)
    monkeypatch.setattr(coordinator.subprocess, "run", forbidden)
    path = validation.run_sie_validation(directory, prepare_only=True)
    assert loads == [configuration.mesh_path]
    inputs = json.loads((path.parent / "sie_inputs.json").read_text())
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "prepared"
    assert manifest["parameter_source"] == inputs["parameter_source"]
    assert manifest["parameter_source"]["sha256"] == before
    assert manifest["extraction_method"] is None
    assert manifest["solver"]["case_count"] == 2
    assert inputs["diagnostics"]["maximum_balanced_inverse_residual"] < 1e-12
    records = np.fromfile(path.parent / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    theta, phi = preparation.face_centroid_spherical_coordinates(
        original_load(configuration.mesh_path)
    )
    profiles, _ = inverse.angular_basis(theta, phi, 1)
    expected = inverse.surface_parameters(coefficients, profiles, configuration.case.eta_exterior)
    np.testing.assert_array_equal(
        records[:, :16].reshape(8, 4, 4), expected.as_matrix().transpose(2, 0, 1)
    )
    assert load_workflow_configuration(path.parent / "configuration.json") == configuration
    with pytest.raises(FileExistsError, match="run directory"):
        validation.run_sie_validation(directory, prepare_only=True)
    assert source_hashes(directory) == before


def test_saved_material_runs_unchanged_batch_command_and_output_checks(saved_surface, monkeypatch):
    directory, configuration, _ = saved_surface
    executable = directory / "fake_solver"
    executable.touch()
    destination = directory / "custom_sie_run"
    calls = []

    def fake_solver(command, *, check):
        calls.append(command)
        assert check
        assert command.count("--case") == 2
        assert command[command.index("--tensors") + 1] == str(destination / "face_tensors.bin")
        assert command[command.index("--epsilon-r-interior") + 1] == "7.0"
        running = json.loads((destination / "run.json").read_text())
        assert running["status"] == "running"
        assert running["parameter_source"]["method"] == "svwf_inverse"
        (destination / "batch_metrics.json").write_text("{}")
        for case in running["simulation"]["cases"]:
            for name, relative_path in case["files"].items():
                if name != "plots":
                    path = destination / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()

    monkeypatch.setattr(coordinator.subprocess, "run", fake_solver)
    path = validation.run_sie_validation(directory, run_dir=destination, executable=executable)
    manifest = json.loads(path.read_text())
    assert len(calls) == 1
    assert manifest["status"] == "completed"
    assert all(case["status"] == "completed" for case in manifest["simulation"]["cases"])
    assert manifest["simulation"]["cases"][0]["excitations"] == (
        configuration.to_dict()["simulation"]["cases"][0]["excitations"]
    )


@pytest.mark.parametrize("problem, message", [
    ("method", "reciprocal-lossless experiment"),
    ("status", "must be completed"),
    ("convention", "exp"),
    ("degree", "angular_degree"),
    ("eta", "impedance"),
    ("profiles", "profile order"),
    ("parameters", "parameter order"),
    ("missing_coefficients", "missing angular_coefficients"),
    ("shape", "shape"),
    ("complex", "real"),
    ("nonfinite", "finite"),
])
def test_invalid_saved_material_fails_before_any_run_is_created(saved_surface, problem, message):
    directory, _, coefficients = saved_surface
    metadata_path = directory / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    if problem in ("method", "status"):
        metadata[problem] = "unsupported"
    if problem == "convention":
        metadata["field_and_material_time_convention"] = "exp(-j omega t)"
    if problem == "degree":
        metadata["angular_degree"] = True
    if problem == "eta":
        metadata["reference_impedance_ohm"] = 0
    if problem == "profiles":
        metadata["angular_profile_names"].reverse()
    if problem == "parameters":
        metadata["material_parameter_names"].reverse()
    metadata_path.write_text(json.dumps(metadata))
    if problem == "missing_coefficients":
        np.savez(directory / "surface_parameters.npz", Y=np.zeros((2, 2, 1)))
    if problem == "shape":
        np.savez(directory / "surface_parameters.npz", angular_coefficients=coefficients[:, :-1])
    if problem == "complex":
        np.savez(directory / "surface_parameters.npz", angular_coefficients=coefficients + 1j)
    if problem == "nonfinite":
        coefficients[0, 0] = np.nan
        np.savez(directory / "surface_parameters.npz", angular_coefficients=coefficients)
    with pytest.raises(ValueError, match=message):
        validation.run_sie_validation(directory, prepare_only=True)
    assert not (directory / "sie_validation").exists()


def test_adapter_cli_from_another_working_directory(saved_surface, tmp_path):
    directory, _, _ = saved_surface
    result = subprocess.run(
        [sys.executable, str(Path(validation.__file__).resolve()),
         "--experiment-dir", str(directory), "--prepare-only"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Prepared SIE validation" in result.stdout
    assert (directory / "sie_validation" / "run.json").is_file()


def test_adapter_reuses_shared_meshing_when_configured_mesh_is_missing(saved_surface, monkeypatch):
    directory, configuration, _ = saved_surface
    mesh_path = directory / "new_mesh.obj"
    configuration = replace(
        configuration,
        sie=replace(configuration.sie, mesh=replace(configuration.sie.mesh, path=str(mesh_path))),
    )
    original = configuration.to_dict()
    (directory / "configuration.json").write_text(json.dumps(original))
    calls = []

    def fake_generate(radius, target_edge, output, *, repo_root):
        calls.append((radius, target_edge, output))
        output.write_bytes((directory.parent / "sphere.obj").read_bytes())

    monkeypatch.setattr(coordinator, "generate_mesh", fake_generate)
    path = validation.run_sie_validation(directory, prepare_only=True)
    assert calls == [(2.0, configuration.target_edge_length, mesh_path)]
    assert json.loads(path.read_text())["mesh"]["path"] == str(mesh_path)
    assert json.loads((directory / "configuration.json").read_text()) == original
