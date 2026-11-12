"""Analytic geometry invariants and integration of the circular-ogive mesher."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from applications.gstc_surface.sie_gstc_surface import run_sie_solve
from applications.gstc_surface.workflow import load_workflow_configuration
from sie_mom.mesh.circular_ogive import CircularOgive
from sie_mom.mesh.obj_mesh import load_triangle_obj

CONFIG = Path(__file__).resolve().parents[2] / "applications/gstc_surface/configs/ogive/ppw7.json"


def test_sphere_limit_matches_independent_sphere_geometry():
    g = CircularOgive(0.7, 0.7, 0.7)
    points = np.random.default_rng(204).normal(size=(100, 3))
    np.testing.assert_array_equal(g.contains(points), np.linalg.norm(points, axis=1) < 0.7)
    np.testing.assert_allclose(
        g.outward_normals(points), points / np.linalg.norm(points, axis=1)[:, None], atol=1e-14
    )
    np.testing.assert_allclose(
        g.surface_residual(np.array([[0, 0.7, 0], [0.7, 0, 0], [-0.7, 0, 0]])), 0, atol=1e-14
    )
    assert g.volume == pytest.approx(4 * np.pi * 0.7**3 / 3, rel=1e-14)


def test_asymmetric_profile_normals_and_volume():
    g = CircularOgive(0.7, 0.84, 1.12)
    np.testing.assert_allclose(g.radius_profile([-0.84, 0, 1.12]), [0, 0.7, 0], atol=1e-14)
    np.testing.assert_array_equal(g.contains([[-0.84, 0, 0], [1.12, 0, 0], [0, 0.7, 0]]), False)
    assert g.contains([0, 0, 0])
    assert not g.contains([-1, 0, 0]) and g.contains([1, 0, 0])
    x = np.array([-0.8, -0.4, -0.001, 0.001, 0.5, 1.0])
    profile = g.radius_profile(x)
    points = np.column_stack((x, profile, np.zeros_like(x)))
    normals = g.outward_normals(points)
    # A finite-difference tangent is independent of the analytic normal formula.
    slope = (g.radius_profile(x + 1e-6) - g.radius_profile(x - 1e-6)) / 2e-6
    tangent = np.column_stack((np.ones_like(x), slope, np.zeros_like(x)))
    np.testing.assert_allclose(np.einsum("ij,ij->i", normals, tangent), 0, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-14)
    assert np.all(normals[:, 1] > 0)
    volume = quad(lambda t: np.pi * float(g.radius_profile(t)) ** 2, -0.84, 1.12, points=[0])[0]
    assert g.volume == pytest.approx(volume, rel=1e-12)
    np.testing.assert_allclose(g.outward_normals([0, 0.7, 0]), [0, 1, 0], atol=1e-14)


def test_rotation_translation_and_multidimensional_arrays():
    base = CircularOgive(0.1, 0.12, 0.16)
    moved = replace(base, center_m=(0.3, -0.2, 0.1), axis_direction=(0, 0, 1))
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    points = np.array([[0, 0.1, 0], [-0.12, 0, 0], [0.16, 0, 0], [0.05, 0.03, 0.02]])
    transformed = points @ rotation.T + moved.center_m
    np.testing.assert_array_equal(moved.contains(transformed), base.contains(points))
    np.testing.assert_allclose(
        moved.surface_residual(transformed), base.surface_residual(points), atol=1e-14
    )
    np.testing.assert_allclose(
        moved.outward_normals(transformed[[0, 3]]),
        base.outward_normals(points[[0, 3]]) @ rotation.T,
        atol=1e-14,
    )
    assert base.contains(points.reshape(2, 2, 3)).shape == (2, 2)


@pytest.mark.parametrize(
    "change",
    [
        {"max_radius_m": 0},
        {"max_radius_m": True},
        {"left_length_m": 0.09},
        {"right_length_m": np.nan},
        {"left_length_m": np.inf},
        {"center_m": (0, 0)},
        {"center_m": (0, 0, np.nan)},
        {"axis_direction": (0, 0, 0)},
        {"axis_direction": (2, 0, 0)},
    ],
)
def test_rejects_invalid_geometry(change):
    values = {"max_radius_m": 0.1, "left_length_m": 0.12, "right_length_m": 0.12, **change}
    with pytest.raises(ValueError):
        CircularOgive(**values)


def test_normal_at_sharp_tip_is_undefined_and_bad_points_are_rejected():
    g = CircularOgive(0.1, 0.12, 0.12)
    with pytest.raises(ValueError, match="undefined"):
        g.outward_normals([0.12, 0, 0])
    for points in ([0, 0], [0, 0, np.nan]):
        with pytest.raises(ValueError):
            g.contains(points)


def test_ogive_config_uses_centered_sources_and_strict_shape_schema(tmp_path):
    config = load_workflow_configuration(CONFIG)
    assert isinstance(config.geometry, CircularOgive)
    assert config.sie.basis == "face"
    assert config.geometry.left_length_m / config.geometry.max_radius_m == pytest.approx(1.2)
    for field in config.fields.values():
        assert field.source_position == (0, 0, 0)
        assert config.geometry.contains(field.source_position)
    raw = config.to_dict()
    raw["geometry"]["sphere_radius_m"] = 0.1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown geometry"):
        load_workflow_configuration(path)


def test_rotated_asymmetric_mesher_and_validation(tmp_path):
    pytest.importorskip("gmsh")
    from applications.gstc_surface.geometry import generate_mesh, validate_mesh

    config = load_workflow_configuration(CONFIG)
    geometry = CircularOgive(0.1, 0.12, 0.16, (0.3, -0.2, 0.1), (0, 0, 1))
    config = replace(
        config,
        geometry=geometry,
        sie=replace(
            config.sie,
            mesh=replace(
                config.sie.mesh, path=str(tmp_path / "rotated.obj"), points_per_wavelength=2.5
            ),
        ),
    )
    generate_mesh(config)
    mesh = load_triangle_obj(config.mesh_path)
    report = validate_mesh(mesh, geometry)
    assert report.euler_characteristic == 2
    assert abs(report.signed_volume / geometry.volume - 1) < 0.1
    with pytest.raises(ValueError, match="configured surface"):
        validate_mesh(mesh, replace(geometry, center_m=(0, 0, 0)))


def test_preparation_and_two_case_command_for_ogive(tmp_path):
    pytest.importorskip("gmsh")
    from applications.gstc_surface.prepare_sie_inputs import prepare_inputs

    raw = json.loads(CONFIG.read_text())
    raw["sie"]["mesh"].update(path=str(tmp_path / "ogive.obj"), points_per_wavelength=3)
    path = tmp_path / "configuration.json"
    path.write_text(json.dumps(raw))
    config = load_workflow_configuration(path)
    manifest_path = prepare_inputs(config, run_dir=tmp_path / "inputs")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["diagnostics"]["minimum_rank"] == 10
    assert manifest["diagnostics"]["maximum_balanced_inverse_residual"] < 1e-8
    assert manifest["mesh"]["sha256"] == hashlib.sha256(config.mesh_path.read_bytes()).hexdigest()
    records = np.fromfile(tmp_path / "inputs/face_tensors.bin", dtype="<c16").reshape(-1, 32)
    with np.load(tmp_path / "inputs/surface_samples.npz") as samples:
        np.testing.assert_array_equal(records[:, :16].reshape(-1, 4, 4), samples["surface_maps"])
    planned = run_sie_solve(inputs_dir=tmp_path / "inputs", run_dir=tmp_path / "sie", dry_run=True)
    command = planned["command"]
    assert command.count("--case") == 2 and command.count("--dipole") == 2
    assert "--no-fields" in command
    for i, value in enumerate(command):
        if value == "--dipole":
            assert command[i + 6 : i + 9] == ["0.0", "0.0", "0.0"]
    assert not (tmp_path / "sie").exists()
