"""Independent feedback experiment: array contracts, iteration, and sampler smoke tests."""

import csv
import json
import os
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from applications.anisotropic_gstc_sphere import plot_fields as plotting
from applications.anisotropic_gstc_sphere.experiments import iterate_lossless as experiment
from applications.anisotropic_gstc_sphere.experiments import prepare_plot_fields as postprocess
from applications.anisotropic_gstc_sphere.workflow import (
    APPLICATION_DIR,
    DipoleFieldDefinition,
    FieldReference,
    GSTCFieldsConfiguration,
    GSTCTransformationConfiguration,
    PlaneWaveFieldDefinition,
    SamplingConfiguration,
    SimulationCaseConfiguration,
    SimulationConfiguration,
    SimulationExcitationConfiguration,
    load_workflow_configuration,
)


@pytest.fixture
def configuration(tmp_path):
    mesh = tmp_path / "sphere.obj"
    mesh.write_text(
        "v 2 0 0\nv -2 0 0\nv 0 2 0\nv 0 -2 0\nv 0 0 2\nv 0 0 -2\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n",
        encoding="ascii",
    )
    original = load_workflow_configuration(
        APPLICATION_DIR
        / "configurations/dipole_shift/refl_complex.json"
    )
    definitions = {
        "ez": DipoleFieldDefinition("electric", (0, 0, 1), (0, 0, 0)),
        "ex": DipoleFieldDefinition("electric", (1, 0, 0), (0, 0, 0)),
        "mz": DipoleFieldDefinition("magnetic", (0, 0, 1), (0, 0, 0)),
        "mx": DipoleFieldDefinition("magnetic", (1, 0, 0), (0, 0, 0)),
    }
    transformations = tuple(
        GSTCTransformationConfiguration(
            state, GSTCFieldsConfiguration(FieldReference(inside), FieldReference(outside))
        )
        for (state, _), inside, outside in zip(experiment.STATE_CASES, ("ez", "mz"), ("ex", "mx"))
    )
    cases = tuple(
        SimulationCaseConfiguration(
            case, (SimulationExcitationConfiguration(FieldReference(source), "interior"),)
        )
        for (_, case), source in zip(experiment.STATE_CASES, ("ez", "mz"))
    )
    return replace(
        original,
        fields=definitions,
        transformations=transformations,
        simulation=SimulationConfiguration(cases),
        extraction_method="reciprocal_lossless",
        sie=replace(original.sie, mesh=replace(original.sie.mesh, path=str(mesh))),
    )


def geometry_and_states(configuration):
    geometry = experiment.surface_geometry(configuration, 0.01)
    _, radii, theta, phi, _ = geometry
    return geometry, experiment.initialize_states(
        configuration, radii, theta, phi, experiment.STATE_CASES
    )


def test_initialization_uses_actual_centroids_not_ideal_sphere(configuration):
    (centroids, radii, theta, phi, areas), states = geometry_and_states(configuration)
    np.testing.assert_allclose(radii, 2 / np.sqrt(3))
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), radii)
    assert np.all(areas > 0)
    electric, magnetic = experiment.evaluate_prescribed_fields(
        FieldReference("ez"), "interior", configuration, radius=radii, theta=theta, phi=phi
    )
    np.testing.assert_array_equal(states[0].electric_interior, electric)
    np.testing.assert_array_equal(states[0].magnetic_interior, magnetic)
    assert all(state.electric_exterior.shape == (3, 8) for state in states)


@pytest.mark.parametrize("offset", [0, -0.01, float("nan"), float("inf"), 2.0])
def test_geometry_rejects_invalid_offsets(configuration, offset):
    with pytest.raises(ValueError, match="offset_m"):
        experiment.surface_geometry(configuration, offset)


def test_state_mapping_is_explicit_and_complete(configuration):
    _, radii, theta, phi, _ = experiment.surface_geometry(configuration, 0.01)
    for pairs in (
        experiment.STATE_CASES[:1],
        (("unknown", "electric_incidence"), experiment.STATE_CASES[1]),
    ):
        with pytest.raises(ValueError, match="every transformation"):
            experiment.initialize_states(configuration, radii, theta, phi, pairs)
    with pytest.raises(ValueError, match="every simulation case"):
        experiment.initialize_states(
            configuration,
            radii,
            theta,
            phi,
            (("electric_z_to_x", "unknown"), experiment.STATE_CASES[1]),
        )


def test_shared_map_export_is_lossless_reciprocal_and_invertible(configuration, tmp_path):
    _, states = geometry_and_states(configuration)
    fit = experiment.extract_and_export(states, configuration, tmp_path)
    records = np.fromfile(tmp_path / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    maps = records[:, :16].reshape(8, 4, 4)
    inverses = records[:, 16:].reshape(8, 4, 4)
    np.testing.assert_allclose(maps @ inverses, np.broadcast_to(np.eye(4), (8, 4, 4)), atol=1e-10)
    np.testing.assert_array_equal(maps, -maps.conj().transpose(0, 2, 1))
    np.testing.assert_array_equal(maps[:, 2:, :2], -maps[:, :2, 2:].transpose(0, 2, 1))
    np.testing.assert_array_equal(maps[:, :2, :2].real, 0)
    np.testing.assert_array_equal(maps[:, 2:, 2:].real, 0)
    np.testing.assert_array_equal(maps[:, :2, 2:].imag, 0)
    assert fit.state_relative_residual.shape == (2, 8)


def test_update_preserves_complex_fields_and_exterior_targets(configuration):
    _, states = geometry_and_states(configuration)
    interior = tuple(
        (1j * state.electric_interior, -1j * state.magnetic_interior) for state in states
    )
    updated = experiment.update_interior_states(states, interior, 0.5)
    for old, new in zip(states, updated):
        np.testing.assert_allclose(new.electric_interior, (0.5 + 0.5j) * old.electric_interior)
        np.testing.assert_allclose(new.magnetic_interior, (0.5 - 0.5j) * old.magnetic_interior)
        assert new.electric_exterior is old.electric_exterior
        assert new.magnetic_exterior is old.magnetic_exterior


@pytest.mark.parametrize("damping", [0, -1, 1.1, float("nan"), float("inf")])
def test_update_rejects_invalid_damping(configuration, damping):
    _, states = geometry_and_states(configuration)
    with pytest.raises(ValueError, match="damping"):
        experiment.update_interior_states(states, [], damping)


def test_complex_error_detects_a_phase_flip():
    target = (np.ones((3, 2), complex), np.ones((3, 2), complex))
    actual = tuple(-field for field in target)
    assert experiment.relative_field_error(actual, target, 377, np.ones(2)) == pytest.approx(2)


def test_solver_uses_original_incidents_and_disables_plot_grids(
    configuration, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(experiment, "run_logged", lambda command, log: calls.append(command))
    experiment.solve_currents(configuration, tmp_path, experiment.DEFAULT_EXECUTABLE)
    command = calls[0]
    assert "--no-fields" in command
    assert configuration.sie.evaluate_fields
    assert command.count("--case") == command.count("--dipole") == 2
    assert command[command.index("--tensors") + 1] == str(tmp_path / "face_tensors.bin")
    first = command.index("--dipole")
    assert command[first : first + 9] == [
        "--dipole",
        "interior",
        "electric",
        "0.0",
        "0.0",
        "1.0",
        "0.0",
        "0.0",
        "0.0",
    ]


def write_integrals(path, points, electric, magnetic, *, index_name="face_index"):
    names = [index_name, "x", "y", "z"]
    names += [
        f"Q{kind}{axis}_{part}" for kind in "EH" for axis in "xyz" for part in ("real", "imag")
    ]
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for face, point in enumerate(points):
            values = np.concatenate((electric[:, face], magnetic[:, face]))
            writer.writerow(
                [face, *point, *[part for value in values for part in (value.real, value.imag)]]
            )


def test_sampler_adds_incident_and_uses_the_correct_sign(configuration, tmp_path, monkeypatch):
    centroids, radii, theta, phi, _ = experiment.surface_geometry(configuration, 0.01)
    electric = np.full((3, 8), 1 + 2j)
    magnetic = np.full((3, 8), 3 - 4j)
    case = configuration.simulation.cases[0]
    case_directory = tmp_path / "cases" / case.id
    case_directory.mkdir(parents=True)

    def fake_sampler(command, log_path):
        output = case_directory / "surface_samples"
        output.mkdir()
        for region, signed_offset in (("interior", -0.01), ("exterior", 0.01)):
            points = centroids * (1 + signed_offset / radii[:, None])
            write_integrals(output / f"{region}_integrals.csv", points, electric, magnetic)

    monkeypatch.setattr(experiment, "run_logged", fake_sampler)
    interior, exterior = experiment.sample_total_fields(
        configuration, case, tmp_path, centroids, radii, theta, phi, 0.01, tmp_path / "sampler"
    )
    qe = experiment.cartesian_to_spherical(electric, theta, phi)
    qh = experiment.cartesian_to_spherical(magnetic, theta, phi)
    incident = experiment.incident_fields(configuration, case, "interior", radii - 0.01, theta, phi)
    np.testing.assert_allclose(interior[0], incident[0] - qe)
    np.testing.assert_allclose(interior[1], incident[1] - qh)
    np.testing.assert_allclose(exterior[0], -qe)
    np.testing.assert_allclose(exterior[1], -qh)


def test_reader_rejects_reordered_faces(tmp_path):
    points = np.array([[1, 0, 0], [0, 0, 1]])
    path = tmp_path / "integrals.csv"
    write_integrals(path, points[::-1], np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="coordinates"):
        experiment.read_integrals(path, points, np.array([np.pi / 2, 0]), np.zeros(2))


def test_fixed_count_loop_feeds_interior_only_and_keeps_incidence(
    configuration, tmp_path, monkeypatch
):
    geometry, initial = geometry_and_states(configuration)
    _, radii, theta, phi, _ = geometry
    commands = []

    def fake_solve(config, directory, executable):
        assert config is configuration
        commands.append(directory)

    def fake_sample(
        config,
        case,
        directory,
        centroids,
        sample_radii,
        sample_theta,
        sample_phi,
        offset,
        executable,
    ):
        state_index = [pair[1] for pair in experiment.STATE_CASES].index(case.id)
        factor = 1.01 + 0.01 * int(directory.name[-3:])
        state = initial[state_index]
        interior = (factor * state.electric_interior, factor * state.magnetic_interior)
        target = configuration.transformations[state_index].gstc.exterior
        exterior = experiment.evaluate_prescribed_fields(
            target, "exterior", config, radius=radii + offset, theta=theta, phi=phi
        )
        return interior, exterior

    monkeypatch.setattr(experiment, "solve_currents", fake_solve)
    monkeypatch.setattr(experiment, "sample_total_fields", fake_sample)
    executable = tmp_path / "placeholder"
    executable.touch()
    root = experiment.run_experiment(
        configuration,
        tmp_path / "experiment",
        iterations=2,
        solver_executable=executable,
        sampler_executable=executable,
    )
    assert len(commands) == 2
    first = np.load(root / "iteration_000/fields.npz")
    second = np.load(root / "iteration_001/fields.npz")
    np.testing.assert_array_equal(
        second["electric_interior_used"], first["electric_interior_computed"]
    )
    np.testing.assert_array_equal(
        second["magnetic_interior_used"], first["magnetic_interior_computed"]
    )
    np.testing.assert_array_equal(
        first["electric_exterior_check_target"], second["electric_exterior_check_target"]
    )
    targets = np.load(root / "geometry_and_targets.npz")
    np.testing.assert_array_equal(
        targets["electric_exterior_target"],
        np.stack([state.electric_exterior for state in initial]),
    )
    with (root / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 4
    assert all(float(row["exterior_relative_field_error"]) == 0 for row in history)
    with pytest.raises(FileExistsError):
        experiment.run_experiment(
            configuration, root, solver_executable=executable, sampler_executable=executable
        )


@pytest.mark.skipif(os.environ.get("SIE_MOM_TEST_CUDA_APP") != "1", reason="opt-in CUDA test")
def test_two_iteration_cuda_smoke(configuration, tmp_path):
    if not experiment.DEFAULT_EXECUTABLE.is_file() or not experiment.SAMPLER_EXECUTABLE.is_file():
        pytest.skip("build anisotropic_sphere_app and anisotropic_surface_sampler")
    root = experiment.run_experiment(configuration, tmp_path / "cuda_experiment", iterations=2)
    for iteration in range(2):
        directory = root / f"iteration_{iteration:03d}"
        fields = np.load(directory / "fields.npz")
        for name in fields.files:
            assert fields[name].shape == (2, 3, 8)
            assert np.isfinite(fields[name]).all()
        for _, case in experiment.STATE_CASES:
            assert (directory / "cases" / case / "surface_samples/interior_integrals.csv").is_file()
    previous = np.load(root / "iteration_000/fields.npz")
    current = np.load(root / "iteration_001/fields.npz")
    np.testing.assert_array_equal(
        current["electric_interior_used"], previous["electric_interior_computed"]
    )


@pytest.mark.parametrize("offset", ["0", "-0.01", "nan"])
def test_sampler_cli_rejects_bad_offsets_before_loading_mesh(tmp_path, offset):
    executable = experiment.SAMPLER_EXECUTABLE
    if not executable.is_file():
        pytest.skip("build anisotropic_surface_sampler")
    result = subprocess.run(
        [
            str(executable),
            "missing.obj",
            "missing.bin",
            str(tmp_path / "output"),
            "1e8",
            "1",
            "1",
            "1",
            "1",
            "17",
            offset,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "finite positive" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.fixture
def saved_iteration(configuration, tmp_path):
    configuration = replace(
        configuration, sampling=SamplingConfiguration(zx_grid_size=4, radial_points=5)
    )
    root = tmp_path / "saved_experiment"
    directory = root / "iteration_000"
    directory.mkdir(parents=True)
    experiment.write_json(root / "experiment.json", {"configuration": configuration.to_dict()})
    centroids, *_ = experiment.surface_geometry(configuration, 0.01)
    np.savez(root / "geometry_and_targets.npz", centroids=centroids)
    experiment.write_json(
        directory / "batch_metrics.json",
        {
            "schema_version": 2,
            "artifact": "anisotropic_sphere_app_batch",
            "case_ids": [case.id for case in configuration.simulation.cases],
            "mesh": {"vertex_count": 6, "face_count": 8, "edge_count": 12},
        },
    )
    for case in configuration.simulation.cases:
        case_dir = directory / "cases" / case.id
        case_dir.mkdir(parents=True)
        np.full(48, 1e-3 + 2e-3j, dtype="<c16").tofile(case_dir / "currents.bin")
        experiment.write_json(case_dir / "solver_metrics.json", {"saved": True})
        (case_dir / "surface_samples").mkdir()
        (case_dir / "surface_samples/interior_integrals.csv").write_text(
            "unchanged feedback", encoding="ascii"
        )
    (directory / "face_tensors.bin").write_bytes(b"unchanged tensors")
    np.savez(directory / "fields.npz", electric_interior_used=np.array([1 + 2j]))
    (root / "history.csv").write_text("unchanged history", encoding="ascii")
    return directory, configuration


def fake_plot_sampler(command, log):
    points = np.fromfile(command[-1], dtype="<f8").reshape(-1, 3)
    output = log.parent / "plot_samples"
    output.mkdir(exist_ok=True)
    for region, value in (("interior", 1 + 2j), ("exterior", -3 + 4j)):
        electric = np.full((3, len(points)), value)
        write_integrals(
            output / f"{region}_integrals.csv",
            points,
            electric,
            2 * electric,
            index_name="point_index",
        )


def test_plot_sampling_uses_saved_grids_and_one_ray(saved_iteration):
    _, configuration = saved_iteration
    points, zx_count = postprocess.observation_points(configuration)
    assert zx_count == 16 and points.shape == (21, 3)
    np.testing.assert_array_equal(points[:zx_count, 1], 0)
    axis = np.linspace(-6, 6, 4)
    np.testing.assert_array_equal(points[:zx_count, 0], np.tile(axis, 4))
    np.testing.assert_array_equal(points[:zx_count, 2], np.repeat(axis, 4))
    np.testing.assert_allclose(np.linalg.norm(points[zx_count:], axis=1), np.linspace(0.1, 4, 5))
    np.testing.assert_allclose(points[zx_count:, 0], points[zx_count:, 2])


def test_postprocessing_keeps_saved_inputs_and_supports_unchanged_plotter(
    saved_iteration, monkeypatch
):
    directory, configuration = saved_iteration
    originals = {path: path.read_bytes() for path in directory.parent.rglob("*") if path.is_file()}
    calls = []

    def sample(command, log):
        calls.append(command)
        fake_plot_sampler(command, log)

    def forbidden(*args, **kwargs):
        raise AssertionError("post-processing must not extract or solve")

    monkeypatch.setattr(experiment, "run_logged", sample)
    monkeypatch.setattr(experiment, "extract_and_export", forbidden)
    monkeypatch.setattr(experiment, "solve_currents", forbidden)
    # The only configuration the post-processor needs is the saved snapshot.
    executable = directory / "sampler_placeholder"
    executable.touch()
    manifest_path = postprocess.prepare_plot_fields(directory, sampler_executable=executable)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["artifact"] == "lossless_feedback_iteration_plot_data"
    assert len(calls) == 2 and all(command[-2] == "--points" for command in calls)
    snapshot = json.loads((directory / "configuration.json").read_text())
    assert snapshot == configuration.to_dict()
    for case in manifest["simulation"]["cases"]:
        assert case["status"] == "completed"
        for cut, count in (("zx", 16), ("radial", 5)):
            data = plotting.load_field_csv(directory / case["files"][f"computed_{cut}"], radius_m=2)
            assert len(data["point_index"]) == count
            np.testing.assert_array_equal(data["point_index"], np.arange(count))
            if cut == "radial":
                outside = data["region"] == 2
                np.testing.assert_allclose(plotting.complex_vector(data, "E")[outside], 3 - 4j)
    # Exercise the existing manifest reader, prescribed reference, and figure generation.
    paths = plotting.plot_fields(
        directory,
        case_id="electric_incidence",
        reference_transformation="electric_z_to_x",
        line_scale="db",
        map_scale="db",
        map_percentiles=(2, 95),
    )
    assert len(paths) == 10 and all(path.is_file() for path in paths)
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    # Repeat preparation without requiring a new solve or a new output directory.
    postprocess.prepare_plot_fields(directory, sampler_executable=executable)
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_postprocessing_total_fields_add_both_regional_incidents(configuration):
    configuration = replace(
        configuration,
        fields={**configuration.fields, "pw": PlaneWaveFieldDefinition((1, 0, 0), (0, 0, 1))},
    )
    case = SimulationCaseConfiguration(
        "both",
        (
            SimulationExcitationConfiguration(FieldReference("ez"), "interior"),
            SimulationExcitationConfiguration(FieldReference("pw"), "exterior"),
        ),
    )
    points = np.array([[1, 0, 0], [3, 0, 0], [0, 0, 2]])
    integrals = {
        "interior": (np.full((3, 3), 1 + 2j), np.full((3, 3), 2 + 3j)),
        "exterior": (np.full((3, 3), 4 + 5j), np.full((3, 3), 6 + 7j)),
    }
    regions, fields = postprocess.total_fields(configuration, case, points, integrals)
    np.testing.assert_array_equal(regions, [1, 2, 2])
    for index, region in enumerate(("interior", "exterior", "exterior")):
        radius = np.linalg.norm(points[index])
        theta = np.array([np.arccos(points[index, 2] / radius)])
        phi = np.array([0.0])
        source = experiment.incident_fields(
            configuration, case, region, np.array([radius]), theta, phi
        )
        for field, incident, integral in zip(fields, source, integrals[region]):
            expected = (
                postprocess.spherical_to_cartesian(incident, theta, phi)[:, 0] - integral[:, index]
            )
            np.testing.assert_allclose(field[:, index], expected)


@pytest.mark.parametrize("problem", ["current_size", "case_ids", "face_order", "missing_currents"])
def test_postprocessing_preflight_rejects_inconsistent_inputs(
    saved_iteration, problem, monkeypatch
):
    directory, configuration = saved_iteration
    if problem == "current_size":
        (directory / "cases/electric_incidence/currents.bin").write_bytes(b"bad")
    elif problem == "missing_currents":
        (directory / "cases/magnetic_incidence/currents.bin").unlink()
    elif problem == "face_order":
        path = directory.parent / "geometry_and_targets.npz"
        centroids = np.load(path)["centroids"]
        np.savez(path, centroids=centroids[::-1])
    else:
        path = directory / "batch_metrics.json"
        metrics = json.loads(path.read_text())
        metrics["case_ids"] = ["unexpected"]
        experiment.write_json(path, metrics)
    monkeypatch.setattr(
        experiment, "run_logged", lambda *_: pytest.fail("must fail before sampling")
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        postprocess.prepare_plot_fields(directory)
    assert not (directory / "run.json").exists()


def test_postprocessing_records_failed_sampling_without_claiming_completed_case(
    saved_iteration, monkeypatch
):
    directory, _ = saved_iteration
    executable = directory / "placeholder"
    executable.touch()

    def fail(*_):
        raise RuntimeError("test sampling failure")

    monkeypatch.setattr(experiment, "run_logged", fail)
    with pytest.raises(RuntimeError, match="sampling failure"):
        postprocess.prepare_plot_fields(directory, sampler_executable=executable)
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert all(case["status"] == "pending" for case in manifest["simulation"]["cases"])


@pytest.mark.parametrize("problem", ["empty", "incomplete", "nonfinite"])
def test_sampler_cli_rejects_bad_point_binary_before_loading_mesh(tmp_path, problem):
    executable = experiment.SAMPLER_EXECUTABLE
    if not executable.is_file():
        pytest.skip("build anisotropic_surface_sampler")
    path = tmp_path / "points.bin"
    values = {"empty": [], "incomplete": [1, 2], "nonfinite": [1, 2, np.nan]}[problem]
    np.asarray(values, dtype="<f8").tofile(path)
    result = subprocess.run(
        [
            str(executable),
            "missing.obj",
            "missing.bin",
            str(tmp_path / "output"),
            "1e8",
            "1",
            "1",
            "1",
            "1",
            "17",
            "--points",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "point input" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(os.environ.get("SIE_MOM_TEST_CUDA_APP") != "1", reason="opt-in CUDA test")
def test_postprocess_cuda_evaluates_saved_currents_without_solver(saved_iteration):
    directory, _ = saved_iteration
    if not experiment.SAMPLER_EXECUTABLE.is_file():
        pytest.skip("build anisotropic_surface_sampler")
    original = (directory / "cases/electric_incidence/currents.bin").read_bytes()
    postprocess.prepare_plot_fields(directory)
    for case in ("electric_incidence", "magnetic_incidence"):
        data = plotting.load_field_csv(
            directory / f"cases/{case}/fields/sie_physical_radial.csv", radius_m=2
        )
        assert len(data["point_index"]) == 5
    assert (directory / "cases/electric_incidence/currents.bin").read_bytes() == original
