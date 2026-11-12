"""App CLI contracts and opt-in CUDA workflow smoke tests on an eight-face sphere."""

import json
import os
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from applications.anisotropic_gstc_sphere import sie_gstc_sphere as coordinator
from applications.anisotropic_gstc_sphere.workflow import (
    APPLICATION_DIR,
    DipoleFieldDefinition,
    FieldReference,
    PlaneWaveFieldDefinition,
    SamplingConfiguration,
    SimulationCaseConfiguration,
    SimulationConfiguration,
    SimulationExcitationConfiguration,
    load_workflow_configuration,
)


@pytest.fixture
def executable():
    path = coordinator.DEFAULT_EXECUTABLE
    if not path.is_file():
        pytest.skip("anisotropic_sphere_app has not been built")
    return path


@pytest.mark.parametrize("case_arguments, error", [
    ([], "at least one --case"),
    (["--case", "empty"], "requires an excitation"),
    (["--case", "../escape"], "invalid case ID"),
    (["--case", "Uppercase"], "invalid case ID"),
    (["--plane-wave", "interior"], "requires a preceding --case"),
    (["--case", "test", "--dipole", "unknown"], "source region"),
    (["--case", "test", "--dipole", "interior", "invalid"], "dipole type"),
    (["--case", "test", "--plane-wave", "exterior", "1", "0"], "missing value"),
    (["--case", "test", "--dipole", "interior", "electric", "nan"], "invalid numeric"),
    (["--case", "test", "--dipole", "interior", "electric", "1e3000"], "stod"),
    (["--case", "test", "--plane-wave", "interior", "1", "0", "0", "0", "0", "1",
      "--case", "test"], "duplicate case ID"),
    (["--case", "test", "--plane-wave", "interior", "1", "0", "0", "0", "0", "1",
      "--dipole", "interior"], "duplicate source region"),
])
def test_cli_rejects_invalid_cases_before_loading_inputs(tmp_path, executable, case_arguments, error):
    command = [
        str(executable), "--mesh", str(tmp_path / "missing.obj"),
        "--tensors", str(tmp_path / "missing.bin"), "--output-dir", str(tmp_path / "run"),
        "--frequency-hz", "1e8", "--epsilon-r-interior", "1", "--mu-r-interior", "1",
        "--epsilon-r-exterior", "1", "--mu-r-exterior", "1", *case_arguments,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert error in result.stderr
    assert not (tmp_path / "run").exists()


def _incident(field, region):
    return SimulationExcitationConfiguration(FieldReference(field), region)


def _csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    components = [f"{kind}{axis}" for kind in "EH" for axis in "xyz"]
    fields = np.column_stack([
        data[f"{name}_real"] + 1j * data[f"{name}_imag"] for name in components
    ])
    return data, fields


def _assert_close(actual, expected):
    assert np.linalg.norm(actual - expected) <= 2e-8 * np.linalg.norm(expected) + 1e-12


@pytest.mark.skipif(
    os.environ.get("SIE_MOM_TEST_CUDA_APP") != "1",
    reason="set SIE_MOM_TEST_CUDA_APP=1 to run the CUDA app integration tests",
)
@pytest.mark.parametrize("diagnostics", [True, False])
def test_batch_workflow_outputs_and_two_region_superposition(tmp_path, executable, diagnostics):
    mesh = tmp_path / "sphere.obj"
    mesh.write_text(
        "v 2 0 0\nv -2 0 0\nv 0 2 0\nv 0 -2 0\nv 0 0 2\nv 0 0 -2\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n",
        encoding="ascii",
    )
    configuration = load_workflow_configuration(
        APPLICATION_DIR / "configurations" / "dipole_shift" / "refl_complex.json"
    )
    electric = _incident("electric_z_shifted", "interior")
    plane = _incident("plane_y", "exterior")
    configuration = replace(
        configuration,
        fields={
            **configuration.fields,
            "plane_y": PlaneWaveFieldDefinition((0, 1, 0), (2, 0, 2)),
            "plane_z": PlaneWaveFieldDefinition((0, 0, 1), (-1, 1, 0)),
            "magnetic_x_outside": DipoleFieldDefinition("magnetic", (1, 0, 0), (0, 0, 3.5)),
        },
        simulation=SimulationConfiguration((
            SimulationCaseConfiguration("electric", (electric,)),
            SimulationCaseConfiguration("magnetic", (_incident("magnetic_z_centered", "interior"),)),
            SimulationCaseConfiguration("plane", (plane,)),
            SimulationCaseConfiguration("combined", (plane, electric)),
            SimulationCaseConfiguration("external_dipole", (_incident("magnetic_x_outside", "exterior"),)),
            SimulationCaseConfiguration("internal_plane", (_incident("plane_z", "interior"),)),
        )),
        sampling=SamplingConfiguration(zx_grid_size=4, radial_points=7),
        sie=replace(
            configuration.sie,
            mesh=replace(configuration.sie.mesh, path=str(mesh)),
            quadrature_rule=13,
            compute_residual=diagnostics,
            evaluate_fields=diagnostics,
        ),
    )
    run_dir = tmp_path / "batch"
    manifest_path = coordinator.run_sie_workflow(
        configuration, run_dir=run_dir, executable=executable, use_existing_mesh=True
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert all(case["status"] == "completed" for case in manifest["simulation"]["cases"])
    assert load_workflow_configuration(run_dir / "configuration.json") == configuration
    batch = json.loads((run_dir / "batch_metrics.json").read_text())
    assert batch["case_ids"] == [case.id for case in configuration.simulation.cases]
    assert batch["mesh"] == {"vertex_count": 6, "face_count": 8, "edge_count": 12}
    solver = batch["solver"]
    assert solver["method"] == "dense_lu_batch" and solver["case_count"] == 6
    assert solver["matrix_dimension"] == 48
    assert solver["matrix_assembly_seconds"] > 0 and solver["factorization_seconds"] > 0
    assert solver["solve_seconds"] > 0
    assert (solver["residual_matrix_assembly_seconds"] > 0) == diagnostics
    currents = {}
    for case in manifest["simulation"]["cases"]:
        directory = run_dir / "cases" / case["id"]
        metrics = json.loads((directory / "solver_metrics.json").read_text())
        assert metrics["schema_version"] == 2 and metrics["case_id"] == case["id"]
        assert (directory / metrics["batch_metrics"]).resolve() == run_dir / "batch_metrics.json"
        assert "factorization_seconds" not in metrics["solver"]
        assert "solve_seconds" not in metrics["solver"]
        assert metrics["solver"]["residual_computed"] == diagnostics
        if diagnostics:
            assert metrics["solver"]["scaled_relative_residual"] < 1e-8
        else:
            assert "scaled_relative_residual" not in metrics["solver"]
            assert not (directory / "fields").exists()
        currents[case["id"]] = np.fromfile(directory / "currents.bin", dtype="<c16")
        assert currents[case["id"]].shape == (48,)
        assert np.isfinite(currents[case["id"]]).all()
        assert np.linalg.norm(currents[case["id"]]) > 0
    _assert_close(currents["combined"], currents["electric"] + currents["plane"])
    if diagnostics:
        for cut, count in (("zx", 16), ("radial", 7)):
            saved = {}
            for case in manifest["simulation"]["cases"]:
                saved[case["id"]] = {}
                for representation in ("physical", "interior_representation", "exterior_representation"):
                    data, fields = _csv(
                        run_dir / "cases" / case["id"] / "fields" / f"sie_{representation}_{cut}.csv"
                    )
                    assert data.shape == (count,) and np.isfinite(fields).all()
                    saved[case["id"]][representation] = fields
                physical = saved[case["id"]]["physical"]
                for region, representation in ((1, "interior_representation"), (2, "exterior_representation")):
                    mask = data["region"] == region
                    np.testing.assert_array_equal(physical[mask], saved[case["id"]][representation][mask])
            for representation in saved["combined"]:
                _assert_close(
                    saved["combined"][representation],
                    saved["electric"][representation] + saved["plane"][representation],
                )

    single_dir = tmp_path / "single"
    single_configuration = replace(
        configuration, simulation=SimulationConfiguration((configuration.simulation.cases[0],))
    )
    command = coordinator.solver_command(single_configuration, run_dir=single_dir, executable=executable)
    command[command.index("--tensors") + 1] = str(run_dir / "face_tensors.bin")
    subprocess.run(command, check=True)
    single_currents = np.fromfile(single_dir / "cases" / "electric" / "currents.bin", dtype="<c16")
    _assert_close(single_currents, currents["electric"])
