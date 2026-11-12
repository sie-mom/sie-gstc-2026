"""Non-spherical geometry, physical configuration and tensor/face association."""

import hashlib
import json

import numpy as np
import pytest

from applications.gstc_surface.workflow import DEFAULT_CONFIGURATION, load_workflow_configuration
from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh
from sie_mom.mesh.tangent_cone_sphere import TangentConeSphere


def write_config(tmp_path, raw):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_tangent_geometry_is_invariant_under_rigid_rotation_and_translation():
    base = TangentConeSphere(0.0748538, 7)
    rotated = TangentConeSphere(0.0748538, 7, (0.2, -0.3, 0.4), (0, 0, 1))
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    points = np.array([[0.1, 0, 0], [-0.02, 0.01, 0.02], [0.7, 0, 0], [0.1, 0.2, 0]])
    # This rotation sends the original x axis to z.
    rotation = rotation.T
    transformed = points @ rotation.T + rotated.sphere_center_m
    np.testing.assert_array_equal(base.contains(points), [True, True, False, False])
    np.testing.assert_array_equal(rotated.contains(transformed), base.contains(points))
    junction = np.array([[base.junction, base.junction_radius, 0]])
    normal = base.outward_normals(junction)[0]
    expected = np.array([np.sin(base.half_angle), np.cos(base.half_angle), 0])
    np.testing.assert_allclose(normal, expected, atol=1e-14)
    np.testing.assert_allclose(
        rotated.outward_normals(junction @ rotation.T + rotated.sphere_center_m),
        base.outward_normals(junction) @ rotation.T,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "axis,angle", [((0, 0, 0), 7), ((2, 0, 0), 7), ((1, 0, 0), 0), ((1, 0, 0), 90)]
)
def test_geometry_rejects_degenerate_definitions(axis, angle):
    with pytest.raises(ValueError):
        TangentConeSphere(0.0748538, angle, axis_direction=axis)


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda raw: raw["sie"].update(basis="spherical"), "basis=face"),
        (
            lambda raw: raw["fields"]["electric_z_interior"].update(source_position=[1, 0, 0]),
            "source position",
        ),
        (
            lambda raw: raw["transformations"][0]["gstc"].update(interior="unknown"),
            "unknown transformation",
        ),
        (lambda raw: raw["physical"].update(radius_m=1), "unknown physical"),
    ],
)
def test_loader_rejects_incompatible_configuration(tmp_path, change, message):
    raw = json.loads(DEFAULT_CONFIGURATION.read_text())
    change(raw)
    with pytest.raises(ValueError, match=message):
        load_workflow_configuration(write_config(tmp_path, raw))


def test_preparation_exports_in_mesh_face_order_and_retains_failure_diagnostics(
    tmp_path, monkeypatch
):
    pytest.importorskip("gmsh")
    from applications.gstc_surface import prepare_sie_inputs as preparation

    raw = json.loads(DEFAULT_CONFIGURATION.read_text())
    raw["sie"]["mesh"]["path"] = str(tmp_path / "surface.obj")
    config = load_workflow_configuration(write_config(tmp_path, raw))
    run = tmp_path / "prep"
    manifest_path = preparation.prepare_inputs(config, run_dir=run)
    manifest = json.loads(manifest_path.read_text())
    report = validate_triangle_mesh(load_triangle_obj(config.mesh_path))
    report.require_valid(expected_euler_characteristic=2)
    assert 4 * report.edge_count == 6 * report.face_count
    assert manifest["mesh_relative_volume_error"] < 0.02
    mesh_hash = hashlib.sha256(config.mesh_path.read_bytes()).hexdigest()
    assert manifest["mesh"]["sha256"] == mesh_hash
    assert manifest["status"] == "prepared_with_warnings"
    assert manifest["solver_executed"] is False
    records = np.fromfile(run / "face_tensors.bin", dtype="<c16").reshape(-1, 32)
    with np.load(run / "surface_samples.npz") as samples:
        np.testing.assert_array_equal(records[:, :16].reshape(-1, 4, 4), samples["surface_maps"])
        assert np.isfinite(records).all()
        assert np.max(samples["smooth_normal_control_relative_residual"]) < 1e-7
        assert np.max(samples["relative_residual"]) > 1e-5
    for state in manifest["diagnostics"]["prescribed_normal_power"].values():
        assert state["area_weighted_relative_l2_defect"] < 1e-12
    with pytest.raises(FileExistsError):
        preparation.prepare_inputs(config, run_dir=run, use_existing_mesh=True)
    with pytest.raises(FileExistsError):
        preparation.prepare_inputs(config, run_dir=tmp_path / "another")
    assert hashlib.sha256(config.mesh_path.read_bytes()).hexdigest() == mesh_hash

    def unavailable_inverse(*args, **kwargs):
        raise ValueError("test: inverse unavailable")

    monkeypatch.setattr(preparation, "prepare_tensor_records", unavailable_inverse)
    failed_run = tmp_path / "diagnostics_only"
    with pytest.raises(ValueError, match="diagnostics retained"):
        preparation.prepare_inputs(config, run_dir=failed_run, use_existing_mesh=True)
    failed = json.loads((failed_run / "sie_inputs.json").read_text())
    assert failed["status"] == "diagnostics_only"
    assert (failed_run / "diagnostics.json").is_file()
    assert not (failed_run / "face_tensors.bin").exists()
