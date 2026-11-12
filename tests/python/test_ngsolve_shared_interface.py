"""Optional FEM checks for preserved facets and varying original-frame tensors."""

# ruff: noqa: E402

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ng = pytest.importorskip("ngsolve")

from sie_mom.mesh.face_frames import compute_face_frames
from sie_mom.mesh.obj_mesh import TriangleMesh
from sie_mom.validation.ngsolve import model as fem
from sie_mom.validation.ngsolve.tensor_transfer import (
    match_interface,
    shared_interface_coefficients,
)
from sie_mom.validation.ngsolve.volume_mesh import load_shared_mesh, make_shared_mesh


def test_original_experiment_keeps_its_configuration_and_solver_exports():
    path = (
        Path(__file__).resolve().parents[2]
        / "applications/anisotropic_gstc_sphere/experiments/ngsolve/model.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_ngsolve_adapter", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    config = legacy.load_workflow_configuration(
        legacy.APPLICATION / "configurations/dipole_pol_rot/dual_pol_eps7.json"
    )
    assert config.case.frequency_hz == 1e10
    assert legacy.sheet_admittance is fem.sheet_admittance
    assert callable(legacy.assemble) and callable(legacy.make_mesh)
    from applications.anisotropic_gstc_sphere import validate_ngsolve as sphere
    from applications.anisotropic_gstc_sphere.experiments.ngsolve import sample_zx as old_sampling

    assert old_sampling.sample_zx is sphere.sample_zx


@pytest.fixture(scope="module")
def problem():
    ng.SetNumThreads(4)
    ng.ngsglobals.msg_level = 0
    source = TriangleMesh(
        0.04 * np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]),
        np.array(
            [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
        ),
    )
    config = SimpleNamespace(
        case=SimpleNamespace(
            frequency_hz=2e9,
            relative_permittivity_interior=2.0,
            relative_permeability_interior=1.0,
            relative_permittivity_exterior=1.0,
            relative_permeability_exterior=1.0,
        )
    )
    mesh, diagnostics = make_shared_mesh(config, source, ppw=4, air_lambda=0.55, pml_lambda=0)
    chi = np.array([[0.3, -0.2], [0.1, 0.4]])
    sheet = np.block(
        [
            [1j * np.array([[1.3, 0.2], [0.2, 1.8]]), chi],
            [-chi.T, 1j * np.array([[0.8, -0.1], [-0.1, 1.1]])],
        ]
    )
    physical = (
        sheet
        * np.array([1 / fem.ETA0, 1 / fem.ETA0, 1, 1])[:, None]
        * np.array([1, 1, fem.ETA0, fem.ETA0])[None, :]
    )
    maps = np.stack([(1 + 0.1 * i) * physical for i in range(8)])
    return config, source, mesh, maps, diagnostics


def test_face_values_and_frames_are_constant_without_interpolation(problem, tmp_path):
    config, source, mesh, maps, diagnostics = problem
    k0 = fem.k0_of(config)
    c = shared_interface_coefficients(mesh, source, maps, k0)
    frames = compute_face_frames(source)
    assert diagnostics["all_interface_faces_separate_core_and_air"]
    assert diagnostics["core_volume_relative_error"] < 1e-12
    assert sorted(c.source_faces.tolist()) == list(range(8))
    rng = np.random.default_rng(32)
    electric = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
    for weights in ([1 / 3, 1 / 3, 1 / 3], [0.2, 0.3, 0.5]):
        xyz = np.einsum("ijk,j->ik", source.vertices[source.faces], weights) * k0
        points = mesh(*xyz.T, ng.BND)
        b = np.asarray(c.admittance(points)).reshape(-1, 4, 4)
        for actual, expected in zip(c.frame, (frames.t1, frames.t2)):
            np.testing.assert_allclose(np.asarray(actual(points)), expected, atol=1e-13)
        h = np.einsum("ijk,ik->ij", b, electric)
        average = np.hstack(((electric[:, :2] + electric[:, 2:]) / 2, (h[:, :2] + h[:, 2:]) / 2))
        currents = np.einsum("ijk,ik->ij", fem.normalized_map(maps), average)
        np.testing.assert_allclose(currents[:, :2], (h[:, 2:] - h[:, :2]) @ fem.R.T, atol=1e-12)
        np.testing.assert_allclose(
            currents[:, 2:], -(electric[:, 2:] - electric[:, :2]) @ fem.R.T, atol=1e-12
        )
    path = tmp_path / "mesh.vol.gz"
    mesh.ngmesh.Save(str(path))
    _, ids, _ = match_interface(ng.Mesh(str(path)), source, k0)
    np.testing.assert_array_equal(ids, c.source_faces)
    moved = source.vertices.copy()
    moved[0, 0] += 1e-4
    with pytest.raises(ValueError, match="moved|connectivity"):
        match_interface(mesh, TriangleMesh(moved, source.faces), k0)


def test_varying_face_gstc_manufactured_solution_converges(problem):
    config, source, mesh, maps, _ = problem
    coefficients = shared_interface_coefficients(mesh, source, maps, fem.k0_of(config))
    ei = ng.CF((1, 0.3j, 0)) * ng.exp(-1j * np.sqrt(2) * ng.z)
    hi = ng.Cross(ng.CF((0, 0, 1)), ei) * np.sqrt(2)
    eo = ng.CF((0.7, 0.1j, 0)) * ng.exp(-1j * ng.z)
    ho = ng.Cross(ng.CF((0, 0, 1)), eo)
    errors = []
    with ng.TaskManager():
        for order in (1, 2):
            system = fem.assemble(
                config,
                mesh,
                frame=coefficients.frame,
                admittance=coefficients.admittance,
                order=order,
            )
            hm, hp = system.fields_from_e(ei, eo)
            forcing = (fem.local(hi, system.frame) - hm, fem.local(ho, system.frame) - hp)
            zero = ng.CF((0, 0, 0))
            rhs = system.rhs(zero, zero, extra_h=forcing)
            solution = ng.GridFunction(system.space)
            solution.components[1].Set(eo, definedon=mesh.Boundaries("outer"))
            r = rhs.vec.CreateVector()
            r.data = rhs.vec - system.matrix.mat * solution.vec
            solution.vec.data += (
                system.matrix.mat.Inverse(system.space.FreeDofs(), inverse=fem.direct_solver()) * r
            )
            assert fem.residual(system, solution, rhs) < 1e-10
            values = []
            for numeric, exact, region in zip(solution.components, (ei, eo), ("core", "air")):

                def norm2(cf):
                    return float(
                        ng.Integrate(
                            (cf * ng.Conj(cf)).real, mesh, definedon=mesh.Materials(region), order=8
                        )
                    )

                values.append(np.sqrt(norm2(numeric - exact) / norm2(exact)))
            errors.append(values)
    print("varying-face manufactured E L2 errors:", errors)
    assert np.max(errors[1]) < 0.1
    assert np.all(np.asarray(errors[1]) < 0.7 * np.asarray(errors[0]))


def test_saved_mesh_restores_pml_and_rejects_wrong_shell_radius(problem, tmp_path):
    config, source, _, maps, _ = problem
    settings = dict(air_lambda=0.55, pml_lambda=0.15, pml_strength=3)
    original, diagnostics = make_shared_mesh(config, source, **settings)
    path = tmp_path / "mesh.vol.gz"
    original.ngmesh.Save(str(path))
    loaded, actual = load_shared_mesh(config, source, path, **settings)
    assert actual["tetrahedra"] == diagnostics["tetrahedra"]
    assert actual["all_interface_faces_separate_core_and_air"]

    def matrix(mesh):
        c = shared_interface_coefficients(mesh, source, maps, fem.k0_of(config))
        system = fem.assemble(config, mesh, frame=c.frame, admittance=c.admittance, order=1)
        return system.matrix.mat.AsVector().FV().NumPy().copy()

    with ng.TaskManager():
        expected = matrix(original)
        np.testing.assert_allclose(matrix(loaded), expected, atol=1e-12, rtol=1e-12)
        assert (
            np.linalg.norm(matrix(ng.Mesh(str(path))) - expected) / np.linalg.norm(expected) > 0.01
        )
    with pytest.raises(ValueError, match="shell radii"):
        load_shared_mesh(config, source, path, **{**settings, "air_lambda": 0.6})


def test_mesh_reuse_checks_previous_settings_and_saved_digest(problem, tmp_path):
    from applications.gstc_surface.validate_ngsolve import _reusable_mesh

    _, _, mesh, _, _ = problem
    mesh.ngmesh.Save(str(tmp_path / "mesh.vol.gz"))
    config = SimpleNamespace(to_dict=lambda: {"frequency_hz": 2e9})
    settings = SimpleNamespace(
        ppw=4, interior_ppw=None, air_lambda=0.55, pml_lambda=0, pml_strength=3
    )
    report = dict(
        backend="ngsolve",
        status="completed",
        mesh_sha256="source-interface",
        settings=vars(settings),
    )
    (tmp_path / "metrics.json").write_text(json.dumps(report))
    (tmp_path / "configuration.json").write_text(json.dumps(config.to_dict()))
    path, provenance = _reusable_mesh(tmp_path, config, "source-interface", settings)
    assert path == tmp_path / "mesh.vol.gz"
    report["volume_mesh_sha256"] = provenance["volume_mesh_sha256"]
    (tmp_path / "metrics.json").write_text(json.dumps(report))
    # Equivalent effective interior sizing is accepted; changing PML is not.
    settings.interior_ppw = 4
    _reusable_mesh(tmp_path, config, "source-interface", settings)
    settings.pml_strength = 4
    with pytest.raises(ValueError, match="pml_strength"):
        _reusable_mesh(tmp_path, config, "source-interface", settings)
    settings.pml_strength = 3
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        _reusable_mesh(tmp_path, config, "source-interface", settings)


def test_sie_comparison_preflight_reads_windows_paths_and_checks_hashes(tmp_path):
    from applications.gstc_surface.artifacts import sha256
    from applications.gstc_surface.validate_ngsolve import _comparison_paths

    path = tmp_path / "electric_z_incidence" / "sie_physical_zx.csv"
    path.parent.mkdir()
    path.write_text("saved reference CSV\n")
    manifest = {
        "cases": [
            {
                "id": "electric_z_incidence",
                "grids": {
                    "zx": {
                        "path": r"electric_z_incidence\sie_physical_zx.csv",
                        "sha256": sha256(path),
                    }
                },
            }
        ]
    }
    paths = _comparison_paths(tmp_path, manifest, ["electric_z_incidence"], ["zx"])
    assert paths["electric_z_incidence"]["zx"][0] == path
    with pytest.raises(ValueError, match="missing requested"):
        _comparison_paths(tmp_path, manifest, ["electric_z_incidence"], ["radial"])
    path.write_text("changed reference CSV\n")
    with pytest.raises(ValueError, match="SHA-256"):
        _comparison_paths(tmp_path, manifest, ["electric_z_incidence"], ["zx"])


def test_missing_comparison_csv_fails_before_backend_or_output_creation(tmp_path, monkeypatch):
    from applications.gstc_surface import validate_ngsolve as runner

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.zeros(32, dtype="<c16").tofile(inputs / "face_tensors.bin")
    config = SimpleNamespace(
        mesh_path=tmp_path / "source.obj",
        simulation=SimpleNamespace(cases=[SimpleNamespace(id="electric_z_incidence")]),
        to_dict=lambda: {"fixture": "matching configuration"},
    )
    references = tmp_path / "sie_fields"
    references.mkdir()
    (references / "configuration.json").write_text(json.dumps(config.to_dict()))
    manifest = {
        "status": "evaluated",
        "mesh_sha256": "mesh",
        "cases": [
            {
                "id": "electric_z_incidence",
                "grids": {"zx": {"path": r"electric_z_incidence\missing.csv", "sha256": "unused"}},
            }
        ],
    }
    (references / "fields.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runner, "_validated_inputs", lambda _: (config, {"mesh": {"sha256": "mesh"}}, None, [])
    )
    monkeypatch.setattr(runner, "load_triangle_obj", lambda _: None)
    monkeypatch.setattr(runner.fem, "incident_cf", lambda *args: None)
    monkeypatch.setattr(runner.fem, "k0_of", lambda _: 1)
    monkeypatch.setattr(runner, "_observations", lambda *args: ({"zx": None}, {}))

    def unexpected_backend(*args):
        pytest.fail("backend reached before checking the reference CSVs")

    monkeypatch.setattr(runner.fem, "direct_solver", unexpected_backend)
    args = SimpleNamespace(
        order=2,
        threads=4,
        inputs_dir=inputs,
        case_id=None,
        observations_dir=tmp_path / "observations",
        air_lambda=1.8,
        sie_fields_dir=references,
        output_dir=tmp_path / "new_run",
        inverse="auto",
    )
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        runner.run(args)
    assert not args.output_dir.exists()
