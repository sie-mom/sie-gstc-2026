"""Grading factors, configuration compatibility and actual graded mesh sizes."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from applications.gstc_surface.workflow import load_workflow_configuration
from sie_mom.mesh.circular_ogive import CircularOgive, OgiveTipGrading
from sie_mom.mesh.obj_mesh import load_triangle_obj

APP = Path(__file__).resolve().parents[2] / "applications/gstc_surface"


def test_grading_parameters_have_the_agreed_physical_meaning():
    geometry = CircularOgive(0.1, 0.12, 0.16)
    grading = OgiveTipGrading(size_factor=0.5, region_fraction=0.25)
    # Tip, halfway through the transition, inner boundary, middle, other tip.
    points = np.array([[-0.12, 0, 0], [-0.105, 0, 0], [-0.09, 0, 0], [0, 0.1, 0], [0.16, 0, 0]])
    np.testing.assert_allclose(
        grading.target_edge_length(geometry, points, 0.02),
        [0.01, 0.015, 0.02, 0.02, 0.01],
        atol=1e-15,
    )
    # The same fractional grading covers a different axial distance on the right.
    right_midpoint = [0.16 - 0.25 * 0.16 / 2, 0, 0]
    assert grading.target_edge_length(geometry, right_midpoint, 0.02) == pytest.approx(0.015)
    x = np.linspace(-0.12, 0, 100)
    sizes = grading.target_edge_length(geometry, np.column_stack((x, 0 * x, 0 * x)), 0.02)
    assert np.all(np.diff(sizes) >= -1e-15)


@pytest.mark.parametrize(
    "name,value",
    [
        ("size_factor", 0),
        ("size_factor", 1.1),
        ("size_factor", np.nan),
        ("region_fraction", 0),
        ("region_fraction", 1.1),
        ("region_fraction", True),
    ],
)
def test_invalid_grading_is_rejected(name, value):
    values = {"size_factor": 0.5, "region_fraction": 0.25, name: value}
    with pytest.raises(ValueError, match="tip_grading"):
        OgiveTipGrading(**values)


def test_existing_configs_remain_uniform_and_grading_roundtrips(tmp_path):
    for path in (APP / "configs/ogive/ppw7.json", APP / "configs/cone_sphere/ppw6.json"):
        assert load_workflow_configuration(path).sie.mesh.tip_grading is None
    config = load_workflow_configuration(APP / "configs/ogive/ppw4_grad.json")
    assert config.sie.mesh.tip_grading == OgiveTipGrading(0.5, 0.25)
    path = tmp_path / "roundtrip.json"
    path.write_text(json.dumps(config.to_dict()))
    assert load_workflow_configuration(path) == config
    raw = load_workflow_configuration(APP / "configs/cone_sphere/ppw6.json").to_dict()
    raw["sie"]["mesh"]["tip_grading"] = {"size_factor": 0.5, "region_fraction": 0.25}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires circular_ogive"):
        load_workflow_configuration(path)


@pytest.mark.parametrize(
    "change",
    [
        {"size_factor": 0.5},
        {"size_factor": 0.5, "region_fraction": 0.25, "unknown": 1},
        None,
    ],
)
def test_grading_schema_rejects_missing_unknown_or_null_settings(tmp_path, change):
    raw = json.loads((APP / "configs/ogive/ppw7.json").read_text())
    raw["sie"]["mesh"]["tip_grading"] = change
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_workflow_configuration(path)


@pytest.mark.parametrize(
    "geometry",
    [
        CircularOgive(0.1, 0.12, 0.12),
        CircularOgive(0.1, 0.12, 0.16, (0.3, -0.2, 0.1), (0, 0, 1)),
    ],
)
def test_actual_mesh_is_finer_at_both_tips_with_well_shaped_triangles(tmp_path, geometry):
    pytest.importorskip("gmsh")
    from applications.gstc_surface.geometry import generate_mesh, validate_mesh
    from sie_mom.mesh.face_frames import compute_face_frames

    config = load_workflow_configuration(APP / "configs/ogive/ppw4_grad.json")
    config = replace(
        config,
        geometry=geometry,
        sie=replace(config.sie, mesh=replace(config.sie.mesh, path=str(tmp_path / "graded.obj"))),
    )
    generate_mesh(config)
    mesh = load_triangle_obj(config.mesh_path)
    validate_mesh(mesh, geometry)
    edges = np.unique(
        np.sort(
            np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )
    midpoints = mesh.vertices[edges].mean(axis=1)
    axial, _ = geometry.axial_radial(midpoints)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]], axis=1)
    length = np.where(axial < 0, geometry.left_length_m, geometry.right_length_m)
    fraction = (length - abs(axial)) / length
    middle = np.median(lengths[fraction > 0.5])
    for mask in ((axial < 0) & (fraction < 0.1), (axial > 0) & (fraction < 0.1)):
        assert np.median(lengths[mask]) < 0.85 * middle
    assert middle == pytest.approx(config.target_edge_length, rel=0.2)
    frames = compute_face_frames(mesh)
    vertices = mesh.vertices[mesh.faces]
    squared_edges = np.sum((vertices - np.roll(vertices, 1, axis=1)) ** 2, axis=(1, 2))
    quality = 4 * np.sqrt(3) * frames.areas / squared_edges
    assert quality.min() > 0.4
