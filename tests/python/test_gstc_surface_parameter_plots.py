"""Physical tensor indexing, mesh identity and saved facewise diagnostics."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from applications.gstc_surface import plot_surface_parameters as plotting
from sie_mom.mesh.face_frames import compute_face_frames
from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh


@pytest.fixture
def prepared_surface(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    mesh = root / "surface.obj"
    mesh.write_text(
        "v 1 0 0\nv -1 0 0\nv 0 1 0\nv 0 -1 0\nv 0 0 1\nv 0 0 -1\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n"
    )
    frames = compute_face_frames(load_triangle_obj(mesh))
    report = validate_triangle_mesh(load_triangle_obj(mesh))
    report.require_valid()
    coefficient = np.array(
        [
            [1j * 1e-3, 3j * 1e-3, 4, 5],
            [3j * 1e-3, 2j * 1e-3, 6, -4],
            [-4, -6, 7j * 100, 9j * 100],
            [-5, 4, 9j * 100, 8j * 100],
        ]
    )
    maps = np.arange(1, 9)[:, None, None] * coefficient
    np.concatenate((maps.reshape(8, 16), np.linalg.inv(maps).reshape(8, 16)), axis=1).astype(
        "<c16"
    ).tofile(root / "face_tensors.bin")
    np.savez(
        root / "surface_samples.npz",
        face_index=np.arange(8),
        surface_maps=maps,
        centroids=frames.centroids,
        t1=frames.t1,
        t2=frames.t2,
        normals=frames.normals,
        relative_residual=np.arange(8) / 1000,
    )
    # No analytic geometry is needed by the plotting application.
    configuration = {
        "extraction_method": "reciprocal_lossless",
        "sie": {"mesh": {"path": "meshes/unavailable.obj"}},
    }
    (root / "configuration.json").write_text(json.dumps(configuration))
    manifest = {
        "schema_version": 1,
        "artifact": "gstc_surface_inputs",
        "status": "prepared",
        "basis": "face",
        "configuration": configuration,
        "mesh": {
            "path": "old-machine/surface.obj",
            "sha256": plotting._sha256(mesh),
            "face_count": 8,
            "edge_count": report.edge_count,
            "vertex_count": 6,
        },
        "files": {
            "configuration": "configuration.json",
            "tensors": "face_tensors.bin",
            "samples": "surface_samples.npz",
        },
        "tensors": {
            "face_count": 8,
            "precision": "complex128",
            "byte_order": "little-endian",
            "ordering": "per face: row-major fields-to-currents map, then its inverse",
        },
    }
    (root / "sie_inputs.json").write_text(json.dumps(manifest))
    return root


def test_plots_keep_physical_units_and_face_values_and_do_not_modify_inputs(prepared_surface):
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared_surface.iterdir()}
    summary_path = plotting.plot_surface_parameters(prepared_surface, percentile=90, dpi=80)
    summary = json.loads(summary_path.read_text())
    assert summary["basis_rotation"] is False and summary["interpolation"] == "none"
    assert summary["lighting"] is False
    assert summary["notation"]["Z"] == "Y_mm"
    expected = {
        "electric": ("S", "imag", {"11": 0.001, "12": 0.003, "22": 0.002}),
        "magnetic": ("Ohm", "imag", {"11": 700, "12": 900, "22": 800}),
        "coupling": ("dimensionless", "real", {"11": 4, "12": 5, "21": 6, "22": -4}),
    }
    filenames = [summary["fit_residual"]["file"]]
    for block, (unit, part, coefficients) in expected.items():
        actual = summary["blocks"][block]
        assert actual["unit"] == unit and actual["part"] == part
        expected_symbol = {"electric": "Y", "magnetic": "Z", "coupling": r"\chi"}[block]
        assert actual["literature_symbol"] == expected_symbol
        filenames.append(actual["file"])
        for component, coefficient in coefficients.items():
            stats = actual["components"][component]
            assert stats["minimum"] == pytest.approx(min(coefficient, 8 * coefficient))
            assert stats["maximum"] == pytest.approx(max(coefficient, 8 * coefficient))
            assert stats["maximum_face_index"] == (7 if coefficient > 0 else 0)
    assert summary["blocks"]["electric"]["components"]["12"]["clipped_face_count"] > 0
    assert summary["fit_residual"]["maximum"] == pytest.approx(0.7)
    for filename in filenames:
        with Image.open(summary_path.parent / filename) as image:
            assert min(image.size) >= 300
            assert np.std(np.asarray(image.convert("RGB"))) > 20
    assert plotting.plt.get_fignums() == []
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    with pytest.raises(FileExistsError):
        plotting.plot_surface_parameters(prepared_surface)


@pytest.mark.parametrize("problem", ["mesh", "order", "tensor", "complex", "residual"])
def test_mismatched_or_unsupported_inputs_fail_before_output(prepared_surface, tmp_path, problem):
    if problem == "mesh":
        mesh = prepared_surface / "surface.obj"
        mesh.write_text(mesh.read_text() + "# changed identity\n")
        match = "SHA-256"
    elif problem in ("order", "residual"):
        path = prepared_surface / "surface_samples.npz"
        with np.load(path) as source:
            data = {key: source[key] for key in source.files}
        if problem == "order":
            data["face_index"] = data["face_index"][::-1]
        else:
            data["relative_residual"][0] = np.nan
        np.savez(path, **data)
        match = "face order" if problem == "order" else "residual"
    elif problem == "tensor":
        path = prepared_surface / "face_tensors.bin"
        data = np.fromfile(path, dtype="<c16")
        data[0] *= 2
        data.tofile(path)
        match = "face order or values"
    else:
        for name in ("configuration.json", "sie_inputs.json"):
            path = prepared_surface / name
            data = json.loads(path.read_text())
            config = data if name == "configuration.json" else data["configuration"]
            config["extraction_method"] = "reciprocal_complex"
            path.write_text(json.dumps(data))
        match = "reciprocal_lossless"
    output = tmp_path / "plots"
    with pytest.raises(ValueError, match=match):
        plotting.plot_surface_parameters(prepared_surface, output_dir=output)
    assert not output.exists()
