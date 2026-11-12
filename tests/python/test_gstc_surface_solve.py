"""Prepared-input provenance and solve-only batch orchestration; no GPU required."""

import hashlib
import json
import subprocess
from dataclasses import asdict

import numpy as np
import pytest

from applications.gstc_surface import sie_gstc_surface as coordinator
from applications.gstc_surface.workflow import DEFAULT_CONFIGURATION
from sie_mom.mesh.obj_mesh import TriangleMesh, load_triangle_obj, validate_triangle_mesh


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path):
    """Small synthetic preparation bundle for testing orchestration, not physics."""
    inputs = tmp_path / "prepared inputs"
    inputs.mkdir()
    mesh = tmp_path / "surface with spaces.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n",
        encoding="ascii",
    )
    raw = json.loads(DEFAULT_CONFIGURATION.read_text())
    raw["sie"]["mesh"]["path"] = str(mesh)
    write_json(inputs / "configuration.json", raw)
    report = validate_triangle_mesh(load_triangle_obj(mesh))
    report.require_valid(expected_euler_characteristic=2)
    np.tile(np.eye(4).ravel(), (report.face_count, 2)).astype("<c16").tofile(
        inputs / "face_tensors.bin"
    )
    manifest = {
        "schema_version": 1,
        "artifact": "gstc_surface_inputs",
        "status": "prepared_with_warnings",
        "basis": "face",
        "configuration": raw,
        "mesh": {**asdict(report), "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()},
        "matrix_dimension": 4 * report.edge_count,
        "dense_matrix_bytes": 16 * (4 * report.edge_count) ** 2,
        "warnings": ["Synthetic preparation warning"],
        "files": {"configuration": "configuration.json", "tensors": "face_tensors.bin"},
        "tensors": {
            "face_count": report.face_count,
            "path": "face_tensors.bin",
            "precision": "complex128",
            "byte_order": "little-endian",
            "ordering": "per face: row-major fields-to-currents map, then its inverse",
        },
    }
    write_json(inputs / "sie_inputs.json", manifest)
    return inputs, mesh, manifest


def fake_outputs(command, **kwargs):
    """Emit the existing C++ output contract, exercising real validation and I/O."""
    run_dir = coordinator.Path(command[command.index("--output-dir") + 1])
    manifest = json.loads((run_dir / "run.json").read_text())
    cases = manifest["simulation"]["cases"]
    expected = manifest["solver"]
    write_json(
        run_dir / "batch_metrics.json",
        {
            "case_ids": [case["id"] for case in cases],
            "mesh": manifest["mesh"],
            "solver": expected,
        },
    )
    for case in cases:
        directory = run_dir / "cases" / case["id"]
        directory.mkdir(parents=True)
        np.ones(expected["matrix_dimension"], dtype="<c16").tofile(directory / "currents.bin")
        write_json(
            directory / "solver_metrics.json",
            {
                "case_id": case["id"],
                "basis": "face",
                "current_data": {
                    "physical_currents": True,
                    "precision": "complex128",
                    "byte_order": "little-endian",
                    "ordering": "J1, M1, J2, M2; one block per RWG edge",
                    "complex_value_count": expected["matrix_dimension"],
                },
                "solver": {
                    "residual_computed": expected["compute_residual"],
                    "scaled_relative_residual": 2e-14,
                },
            },
        )
    kwargs["stdout"].write("Synthetic solver output\n")


def test_dry_run_is_read_only_and_preserves_two_rhs_and_source_regions(tmp_path, prepared):
    inputs, mesh, original = prepared
    run_dir = tmp_path / "new run"
    manifest = coordinator.run_sie_solve(
        inputs_dir=inputs, run_dir=run_dir, executable=tmp_path / "missing.exe", dry_run=True
    )
    assert manifest["status"] == "planned" and manifest["solver_executed"] is False
    assert not run_dir.exists()
    command = manifest["command"]
    threshold = manifest["solver"]["singularity_threshold"]
    assert threshold["effective_lambda0"] > threshold["configured_lambda0"]
    assert threshold["effective_uncovered_pair_count"] == 0
    assert (
        float(command[command.index("--singularity-threshold-lambda0") + 1])
        == (threshold["effective_lambda0"])
    )
    assert command.count("--case") == 2 and command.count("--dipole") == 2
    assert command[-1] == "--no-fields"
    assert command[command.index("--basis") + 1] == "face"
    assert command[command.index("--mesh") + 1] == str(run_dir / "surface.obj")
    assert "--zx-grid-size" not in command and "--radial-points" not in command
    for source in (i for i, arg in enumerate(command) if arg == "--dipole"):
        assert command[source + 1 : source + 3] == ["interior", "electric"]
        assert command[source + 6 : source + 9] == ["0.1", "0.0", "0.0"]
    assert manifest["warnings"] == original["warnings"]
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == original["mesh"]["sha256"]
    assert json.loads((inputs / "sie_inputs.json").read_text()) == original


def test_threshold_covers_vertex_only_pairs_and_is_invariant_in_wavelength_units():
    # Two perpendicular faces share only vertex 0, with known centroid distance.
    mesh = TriangleMesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 2], [0, 3, 4]]),
    )
    result = coordinator.choose_assembly_singularity_threshold(mesh, 2e9, 0.1)
    expected = np.sqrt(6) / 3
    assert result["touching_pair_count"] == 1
    assert result["maximum_touching_centroid_distance_m"] == pytest.approx(expected, rel=1e-6)
    assert result["effective_m"] == pytest.approx(1.05 * expected, rel=1e-6)
    assert result["configured_uncovered_pair_count"] == 1
    assert result["effective_uncovered_pair_count"] == 0
    for dtype in (np.float32, np.float64):
        vertices = mesh.vertices.astype(dtype)[mesh.faces]
        centroids = (vertices[:, 0] + vertices[:, 1] + vertices[:, 2]) / dtype(3)
        distance = np.float32(np.linalg.norm(centroids[0] - centroids[1]))
        assert distance < result["minimum_checked_float32_kernel_threshold_m"]
    scaled = TriangleMesh(mesh.vertices * 2, mesh.faces)
    moved = coordinator.choose_assembly_singularity_threshold(scaled, 1e9, 0.1)
    assert moved["effective_lambda0"] == pytest.approx(result["effective_lambda0"], rel=1e-6)
    floor = coordinator.choose_assembly_singularity_threshold(mesh, 2e9, 100)
    assert floor["effective_lambda0"] == 100


def test_threshold_deduplicates_edge_neighbors_and_covers_self_pairs():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    result = coordinator.choose_assembly_singularity_threshold(
        TriangleMesh(vertices, np.array([[0, 1, 2], [1, 3, 2]])), 2e9, 0.1
    )
    assert result["touching_pair_count"] == 1  # Shared edge is not counted twice.
    single = coordinator.choose_assembly_singularity_threshold(
        TriangleMesh(vertices[:3], np.array([[0, 1, 2]])), 2e9, 0.1
    )
    assert single["touching_pair_count"] == 0
    assert single["effective_lambda0"] == 0.1
    assert single["minimum_checked_float32_kernel_threshold_m"] > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"frequency_hz": 0},
        {"frequency_hz": np.nan},
        {"configured_lambda0": 0},
        {"safety_factor": 1},
        {"safety_factor": np.inf},
    ],
)
def test_threshold_rejects_invalid_controls(changes):
    mesh = TriangleMesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    args = {"frequency_hz": 2e9, "configured_lambda0": 0.1, **changes}
    with pytest.raises(ValueError):
        coordinator.choose_assembly_singularity_threshold(mesh, **args)


@pytest.mark.parametrize("corruption", ["mesh", "configuration", "tensors", "status"])
def test_rejects_changed_preparation_before_creating_run(tmp_path, prepared, corruption):
    inputs, mesh, original = prepared
    if corruption == "mesh":
        mesh.write_text(mesh.read_text() + "# changed identity\n")
    elif corruption == "configuration":
        raw = original["configuration"].copy()
        raw["case_id"] = "changed"
        write_json(inputs / "configuration.json", raw)
    elif corruption == "tensors":
        (inputs / "face_tensors.bin").write_bytes(b"truncated")
    else:
        original["status"] = "diagnostics_only"
        write_json(inputs / "sie_inputs.json", original)
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError):
        coordinator.run_sie_solve(inputs_dir=inputs, run_dir=run_dir, dry_run=True)
    assert not run_dir.exists()


def test_snapshots_inputs_invokes_one_batch_and_validates_outputs(tmp_path, prepared, monkeypatch):
    inputs, mesh, original = prepared
    executable = tmp_path / "solver.exe"
    executable.write_bytes(b"test executable identity")
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        fake_outputs(command, **kwargs)

    monkeypatch.setattr(coordinator.subprocess, "run", invoke)
    run_dir = tmp_path / "run with spaces"
    manifest = coordinator.run_sie_solve(inputs_dir=inputs, run_dir=run_dir, executable=executable)
    assert len(calls) == 1
    _, loaded, mesh_path = coordinator.load_solved_run(run_dir)
    assert loaded["status"] == "solved" and mesh_path == run_dir / "surface.obj"
    assert manifest["status"] == "solved" and manifest["solver_executed"] is True
    assert (run_dir / "surface.obj").read_bytes() == mesh.read_bytes()
    for name in ("configuration.json", "face_tensors.bin", "sie_inputs.json"):
        assert (run_dir / name).read_bytes() == (inputs / name).read_bytes()
    assert manifest["warnings"] == original["warnings"]
    for case in manifest["simulation"]["cases"]:
        assert case["status"] == "solved" and case["scaled_relative_residual"] == 2e-14
        assert not (run_dir / "cases" / case["id"] / "fields").exists()
    assert "Synthetic solver output" in (run_dir / "solver.log").read_text()
    with pytest.raises(FileExistsError):
        coordinator.run_sie_solve(inputs_dir=inputs, run_dir=run_dir, executable=executable)
    assert len(calls) == 1
    changed = json.loads((run_dir / "configuration.json").read_text())
    changed["physical"]["frequency_hz"] *= 2
    write_json(run_dir / "configuration.json", changed)
    with pytest.raises(ValueError, match="configuration differs"):
        coordinator.load_solved_run(run_dir)


@pytest.mark.parametrize("failure", ["process", "missing_outputs", "nonfinite_currents"])
def test_failure_retains_log_and_never_claims_a_successful_solve(
    tmp_path, prepared, monkeypatch, failure
):
    inputs, _, _ = prepared
    executable = tmp_path / "solver"
    executable.touch()

    def invoke(command, **kwargs):
        if failure == "process":
            kwargs["stdout"].write("Synthetic CUDA failure\n")
            raise subprocess.CalledProcessError(1, command)
        if failure == "nonfinite_currents":
            fake_outputs(command, **kwargs)
            run = coordinator.Path(command[command.index("--output-dir") + 1])
            path = next((run / "cases").glob("*/currents.bin"))
            data = np.fromfile(path, dtype="<c16")
            data[0] = np.nan
            data.tofile(path)

    monkeypatch.setattr(coordinator.subprocess, "run", invoke)
    run_dir = tmp_path / "failed"
    with pytest.raises((subprocess.CalledProcessError, FileNotFoundError, ValueError)):
        coordinator.run_sie_solve(inputs_dir=inputs, run_dir=run_dir, executable=executable)
    failed = json.loads((run_dir / "run.json").read_text())
    assert failed["status"] == "failed" and failed["solver_executed"] is False
    assert all(case["status"] == "failed" for case in failed["simulation"]["cases"])
    assert (run_dir / "solver.log").is_file()
    assert (run_dir / "surface.obj").is_file()
