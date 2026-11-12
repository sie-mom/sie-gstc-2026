"""Reusable volume grids derived from the saved solve mesh."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from applications.gstc_surface.observations import prepare_observations, sampling_points
from applications.gstc_surface.workflow import DEFAULT_CONFIGURATION

pytest.importorskip("trimesh")
pytest.importorskip("rtree")


def test_grid_geometry_matches_metres_and_standard_spherical_ray():
    sampling = json.loads(DEFAULT_CONFIGURATION.read_text())["sampling"]
    grids = sampling_points(sampling)
    assert grids["zx"].shape == (40000, 3)
    np.testing.assert_array_equal(grids["zx"][[0, -1]], [[-.12, 0, -.12], [.66, 0, .12]])
    ray = grids["radial"]
    assert ray.shape == (300, 3)
    np.testing.assert_allclose(ray[:, 0], ray[:, 2])
    np.testing.assert_array_equal(ray[:, 1], 0)
    np.testing.assert_allclose(np.linalg.norm(ray[[0, -1]], axis=1), [.004, .15])


def test_preparation_uses_solve_snapshot_and_does_not_modify_currents(tmp_path):
    solve = tmp_path / "solve"
    solve.mkdir()
    root = Path(__file__).resolve().parents[2]
    mesh = root / "applications/gstc_surface/meshes/cone_sphere/surface.obj"
    shutil.copy2(mesh, solve / "surface.obj")
    raw = json.loads(DEFAULT_CONFIGURATION.read_text())
    raw["sampling"].update(zx_grid_size=8, radial_points=9)
    (solve / "configuration.json").write_text(json.dumps(raw))
    (solve / "sie_inputs.json").write_text(json.dumps({"configuration": raw}))
    manifest = {
        "application": "gstc_surface", "status": "solved",
        "mesh": {"sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()},
        "files": {"mesh": "surface.obj", "sie_inputs": "sie_inputs.json"},
        "input_preparation": {
            "manifest_sha256": hashlib.sha256((solve / "sie_inputs.json").read_bytes()).hexdigest()
        },
    }
    original = json.dumps(manifest)
    (solve / "run.json").write_text(original)
    output = tmp_path / "observations"
    path = prepare_observations(solve_dir=solve, output_dir=output)
    observations = json.loads(path.read_text())
    assert observations["mesh_sha256"] == manifest["mesh"]["sha256"]
    assert observations["backend"] == "numpy"
    for name, count in (("zx", 64), ("radial", 9)):
        data = np.genfromtxt(
            output / observations["grids"][name]["path"], delimiter=",", names=True
        )
        assert len(data) == count
        np.testing.assert_array_equal(data["point_index"], np.arange(count))
        assert np.isin(data["region"], [1, 2]).all()
    assert (solve / "run.json").read_text() == original
    with pytest.raises(FileExistsError):
        prepare_observations(solve_dir=solve, output_dir=output)
    (solve / "surface.obj").write_text("changed mesh")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_observations(solve_dir=solve, output_dir=tmp_path / "bad")
