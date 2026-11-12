"""Field-evaluator contracts and opt-in GPU checks of regional reconstruction."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from applications.gstc_surface.evaluate_fields import DEFAULT_EXECUTABLE, field_command
from applications.gstc_surface.observations import read_observations
from applications.gstc_surface.workflow import load_workflow_configuration
from sie_mom.excitation import evaluate_dipole_fields_cartesian, evaluate_plane_wave_cartesian
from sie_mom.postprocessing.field_io import load_field_csv
from sie_mom.postprocessing.field_transforms import complex_vector


@pytest.fixture
def executable():
    if not DEFAULT_EXECUTABLE.is_file():
        pytest.skip("build the evaluate_surface_fields target")
    return DEFAULT_EXECUTABLE


def test_command_contains_physical_sources_and_no_prescribed_exterior_source(tmp_path):
    configuration = load_workflow_configuration()
    command = field_command(
        configuration, configuration.simulation.cases[0], executable=Path("evaluator.exe"),
        mesh=tmp_path / "saved mesh.obj", currents=tmp_path / "currents.bin",
        observations=tmp_path / "points.csv", output=tmp_path / "fields.csv",
    )
    assert command.count("--dipole") == 1
    assert command[command.index("--dipole") + 1:] == [
        "interior", "electric", "0.0", "0.0", "1.0", "0.1", "0.0", "0.0"
    ]
    assert str(tmp_path / "saved mesh.obj") in command
    assert "--tensors" not in command


@pytest.mark.parametrize("row", ["0,0,0,0,3", "1,0,0,0,1", "0,nan,0,0,1"])
def test_observation_reader_rejects_invalid_indices_regions_and_coordinates(tmp_path, row):
    path = tmp_path / "bad.csv"
    path.write_text("point_index,x,y,z,region\n" + row + "\n")
    with pytest.raises(ValueError):
        read_observations(path)


@pytest.mark.parametrize("option,value", [("--quadrature-rule", "21"), ("--frequency-hz", "nan")])
def test_evaluator_rejects_invalid_cli_before_loading_inputs(executable, tmp_path, option, value):
    result = subprocess.run([str(executable), option, value], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "output.csv").exists()


def setup_evaluator_files(tmp_path):
    mesh = tmp_path / "tetra.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    points = np.array([[.2, .2, .2], [.9, .9, .9], [.1, .2, .3], [-.3, .2, .1]])
    regions = np.array([1, 2, 1, 2])
    observations = tmp_path / "observations.csv"
    np.savetxt(
        observations, np.column_stack((np.arange(4), points, regions)), delimiter=",",
        header="point_index,x,y,z,region", comments="",
        fmt=["%d", "%.17g", "%.17g", "%.17g", "%d"],
    )
    currents = tmp_path / "currents.bin"
    np.zeros(24, dtype="<c16").tofile(currents)
    return mesh, observations, currents, points, regions


def evaluator_command(executable, mesh, observations, currents, output):
    return [
        str(executable), "--mesh", str(mesh), "--observations", str(observations),
        "--currents", str(currents), "--output", str(output),
        "--frequency-hz", "2e9", "--epsilon-r-interior", "5", "--mu-r-interior", "1",
        "--epsilon-r-exterior", "1", "--mu-r-exterior", "1",
    ]


def test_evaluator_rejects_noncontiguous_observations_without_gpu(executable, tmp_path):
    mesh, observations, currents, _, _ = setup_evaluator_files(tmp_path)
    observations.write_text("point_index,x,y,z,region\n7,.2,.2,.2,1\n")
    output = tmp_path / "output.csv"
    result = subprocess.run(
        evaluator_command(executable, mesh, observations, currents, output),
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and "contiguous" in result.stderr
    assert not output.exists()


cuda = pytest.mark.skipif(
    os.environ.get("SIE_MOM_TEST_CUDA_APP") != "1",
    reason="set SIE_MOM_TEST_CUDA_APP=1 for field-evaluator GPU integration checks",
)


@cuda
@pytest.mark.parametrize("dipole_type", ["electric", "magnetic"])
def test_zero_currents_recover_regional_incident_fields(executable, tmp_path, dipole_type):
    mesh, observations, currents, points, regions = setup_evaluator_files(tmp_path)
    output = tmp_path / "fields.csv"
    command = evaluator_command(executable, mesh, observations, currents, output) + [
        "--dipole", "interior", dipole_type, "0", "0", "1", ".1", ".1", ".1",
        "--plane-wave", "exterior", "1", "0", "0", "0", "0", "1",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    data = load_field_csv(output)
    epsilon, mu = 8.8541878188e-12, 1.25663706127e-6
    for region in (1, 2):
        mask = regions == region
        eps = epsilon * (5 if region == 1 else 1)
        k, eta = 2 * np.pi * 2e9 * np.sqrt(eps * mu), np.sqrt(mu / eps)
        if region == 1:
            expected = evaluate_dipole_fields_cartesian(
                k, points[mask], eta, effective_moment=(0, 0, 1),
                source_position=(.1, .1, .1), dipole_type=dipole_type,
            )
        else:
            expected = tuple(np.conjugate(field) for field in evaluate_plane_wave_cartesian(
                k, points[mask], eta, electric_field=(1, 0, 0), propagation_direction=(0, 0, 1)
            ))
        for kind, reference in zip("EH", expected):
            actual = complex_vector(data, kind)[mask]
            np.testing.assert_allclose(actual, reference.T, rtol=2e-8, atol=1e-12)


@cuda
def test_current_blocks_are_regional_and_fields_are_linear(executable, tmp_path):
    mesh, observations, currents, _, regions = setup_evaluator_files(tmp_path)
    coefficients = np.zeros(24, dtype="<c16")
    coefficients[:12] = np.arange(1, 13) * (1 + .3j)
    coefficients.tofile(currents)
    output = tmp_path / "fields1.csv"
    subprocess.run(
        evaluator_command(executable, mesh, observations, currents, output),
        check=True, capture_output=True, text=True,
    )
    first = load_field_csv(output)
    for kind in "EH":
        fields = complex_vector(first, kind)
        np.testing.assert_array_equal(fields[regions == 2], 0)
        assert np.linalg.norm(fields[regions == 1]) > 0
    (2 * coefficients).tofile(currents)
    output2 = tmp_path / "fields2.csv"
    subprocess.run(
        evaluator_command(executable, mesh, observations, currents, output2),
        check=True, capture_output=True, text=True,
    )
    second = load_field_csv(output2)
    for kind in "EH":
        np.testing.assert_allclose(
            complex_vector(second, kind), 2 * complex_vector(first, kind), rtol=1e-12, atol=1e-12
        )
