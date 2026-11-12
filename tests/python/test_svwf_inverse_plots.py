"""Saved-coefficient post-processing and plotting contracts for the SVWF experiment."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace

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
    PlaneWaveFieldDefinition,
    SamplingConfiguration,
    SimulationCaseConfiguration,
    SimulationExcitationConfiguration,
    load_workflow_configuration,
)
from spherical_expansion.coefficients import load_coefficients
from spherical_expansion.svwf_sphere import compute_fields


@pytest.fixture
def saved_experiment(tmp_path):
    configuration = load_workflow_configuration(experiment.DEFAULT_CONFIG)
    configuration = replace(configuration, sampling=SamplingConfiguration(
        zx_grid_size=4, radial_points=6
    ))
    directory = experiment.run_experiment(
        configuration, tmp_path / "experiment", n_max=1, n_theta=4, n_phi=8,
        angular_degree=1, max_nfev=1, verbose=0,
    )
    return directory, configuration


def input_hashes(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).parts[0] != "plot_data"
    }


def test_total_fields_keep_both_regional_incidents_and_complex_amplitudes(saved_experiment):
    _, configuration = saved_experiment
    configuration = replace(
        configuration,
        case=replace(configuration.case, relative_permittivity_interior=4),
        fields={
            **configuration.fields,
            "pi": PlaneWaveFieldDefinition((0, -0.3 + 0.5j, 0), (0, 0, 1)),
            "pe": PlaneWaveFieldDefinition((1 + 2j, 0, 0), (0, 0, 1)),
        },
    )
    case = SimulationCaseConfiguration("combined", (
        SimulationExcitationConfiguration(FieldReference("pi"), "interior"),
        SimulationExcitationConfiguration(FieldReference("pe"), "exterior"),
    ))
    points = np.array([[1, 0.3, 0.2], [3, 0.7, 0.6], [0, 0, 2]])
    coefficients = tuple(dict.fromkeys(((-1, 1), (0, 1), (1, 1)), 0j) for _ in range(4))
    regions, (electric, magnetic) = postprocess.total_fields(
        configuration, case, points, coefficients
    )
    np.testing.assert_array_equal(regions, [1, 2, 2])
    for index, region in enumerate(("interior", "exterior", "exterior")):
        amplitude = configuration.fields["pi" if region == "interior" else "pe"].electric_amplitude
        expected = np.array(amplitude) * np.exp(
            -1j * getattr(configuration.case, f"k_{region}") * points[index, 2]
        )
        np.testing.assert_allclose(electric[:, index], expected, atol=1e-14)
        np.testing.assert_allclose(
            magnetic[:, index], np.cross([0, 0, 1], expected)
            / getattr(configuration.case, f"eta_{region}"), atol=1e-14,
        )


def test_modal_fields_are_conjugated_and_added_to_incident_once(saved_experiment):
    directory, configuration = saved_experiment
    case = configuration.simulation.cases[0]
    coefficients = load_coefficients(directory / "cases" / case.id / "forward_coefficients.npz")
    points = np.array([[1, 0.3, 0.2], [3, 0.7, 0.6]])
    regions, fields = postprocess.total_fields(configuration, case, points, coefficients)
    radii = np.linalg.norm(points, axis=1)
    theta, phi = np.arccos(points[:, 2] / radii), np.arctan2(points[:, 1], points[:, 0])
    for index, region in enumerate(("interior", "exterior")):
        families = coefficients[2:] if region == "interior" else coefficients[:2]
        modal = compute_fields(
            getattr(configuration.case, f"k_{region}"), radii[index:index + 1],
            theta[index:index + 1], phi[index:index + 1], *families,
            kind=1 if region == "interior" else 3,
            eta=getattr(configuration.case, f"eta_{region}"),
        )
        incident = [np.zeros((3, 1), complex) for _ in "EH"]
        for excitation in case.excitations:
            if excitation.region == region:
                incident = postprocess.evaluate_prescribed_fields(
                    excitation.field, region, configuration, radius=radii[index:index + 1],
                    theta=theta[index:index + 1], phi=phi[index:index + 1],
                )
        for actual, expanded, prescribed in zip(fields, modal, incident):
            expected = postprocess.spherical_to_cartesian(
                np.asarray(expanded).conj() + prescribed,
                theta[index:index + 1], phi[index:index + 1],
            )
            np.testing.assert_allclose(actual[:, index], expected[:, 0], atol=1e-14)
    np.testing.assert_array_equal(regions, [1, 2])


def test_postprocessing_never_solves_and_preserves_original_artifacts(
    saved_experiment, monkeypatch
):
    directory, configuration = saved_experiment
    before = input_hashes(directory)

    def forbidden(*args, **kwargs):
        raise AssertionError("post-processing must not solve")

    monkeypatch.setattr(experiment, "least_squares", forbidden)
    monkeypatch.setattr(experiment, "solve_collocation_general", forbidden)
    prepared = [postprocess.prepare_plot_fields(directory, solution=solution)
                for solution in postprocess.SOLUTIONS]
    assert prepared[0] != prepared[1]
    for solution, destination in zip(postprocess.SOLUTIONS, prepared):
        manifest = json.loads((destination / "run.json").read_text())
        assert manifest["backend"] == "svwf"
        assert manifest["solution"] == solution
        assert manifest["status"] == "completed"
        assert manifest["post_processing"]["exterior_candidate_prescribed"] == (
            solution == "inverse"
        )
        for case in manifest["simulation"]["cases"]:
            assert case["status"] == "completed"
            data = plotting.load_field_csv(destination / case["files"]["computed_radial"],
                                           radius_m=configuration.case.radius_m)
            assert len(data["x"]) == 6
            zx = plotting.load_field_csv(destination / case["files"]["computed_zx"],
                                         radius_m=configuration.case.radius_m)
            plotting.zx_grid(zx, 4)
        postprocess.prepare_plot_fields(directory, solution=solution)
    assert input_hashes(directory) == before


def test_existing_plotter_accepts_and_labels_both_svwf_datasets(saved_experiment, monkeypatch):
    directory, _ = saved_experiment
    titles = []
    save_figure = plotting._save_figure

    def capture(figure, path):
        titles.append(figure._suptitle.get_text())
        save_figure(figure, path)

    monkeypatch.setattr(plotting, "_save_figure", capture)
    for solution in postprocess.SOLUTIONS:
        prepared = postprocess.prepare_plot_fields(directory, solution=solution)
        paths = plotting.plot_fields(
            prepared, case_id="electric_incidence", reference_transformation="electric_z_to_x",
            line_scale="db", map_scale="db", map_percentiles=(2, 95),
        )
        assert len(paths) == 10
        for path in paths:
            with Image.open(path) as picture:
                pixels = np.asarray(picture.convert("RGB"))
            assert pixels.std() > 5
            assert pixels.shape[0] > 500
    assert any("SVWF forward validation" in title for title in titles)
    assert any("SVWF inverse candidate" in title for title in titles)
    assert all("not enforced" in title for title in titles)


@pytest.mark.parametrize("problem", ["convention", "unfinished", "mapping", "modes", "missing"])
def test_bad_saved_inputs_fail_before_creating_output(saved_experiment, problem):
    directory, _ = saved_experiment
    path = directory / "experiment.json"
    manifest = json.loads(path.read_text())
    if problem == "convention":
        manifest["coefficient_time_convention"] = "exp(+j omega t)"
    elif problem == "unfinished":
        manifest["status"] = "failed"
    elif problem == "mapping":
        manifest["state_cases"][1][1] = "unknown"
    elif problem == "modes":
        manifest["n_max"] = 2
    elif problem == "missing":
        (directory / "cases/magnetic_incidence/forward_coefficients.npz").unlink()
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        postprocess.prepare_plot_fields(directory)
    assert not (directory / "plot_data").exists()


def test_failed_reconstruction_is_not_marked_completed(saved_experiment, monkeypatch):
    directory, _ = saved_experiment

    def broken(*args):
        raise ValueError("sample at incident source")

    monkeypatch.setattr(postprocess, "total_fields", broken)
    with pytest.raises(ValueError, match="incident source"):
        postprocess.prepare_plot_fields(directory)
    manifest = json.loads((directory / "plot_data/forward/run.json").read_text())
    assert manifest["status"] == "failed"
    assert all(case["status"] == "pending" for case in manifest["simulation"]["cases"])


def test_unknown_solution_is_rejected(saved_experiment):
    directory, _ = saved_experiment
    with pytest.raises(ValueError, match="solution"):
        postprocess.prepare_plot_fields(directory, solution="unknown")


@pytest.mark.parametrize("solution", postprocess.SOLUTIONS)
def test_cli_prepares_saved_coefficients(saved_experiment, solution):
    directory, _ = saved_experiment
    result = subprocess.run(
        [sys.executable, postprocess.__file__, "--experiment-dir", str(directory),
         "--solution", solution], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(directory / "plot_data" / solution)
