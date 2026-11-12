"""Algebra, source conventions, and a manufactured two-sided Maxwell problem."""

from dataclasses import replace

import model as m
import ngsolve as ng
import numpy as np
import pytest
from workflow import load_workflow_configuration

from sie_mom.excitation.fields import evaluate_dipole_fields_cartesian


@pytest.fixture(scope="module")
def config():
    return load_workflow_configuration(
        m.APPLICATION / "configurations/dipole_pol_rot/dual_pol_eps7.json"
    )


@pytest.fixture(scope="module")
def sheet():
    # Independent, invertible reciprocal-lossless test sheet, not fitted to a solution.
    y = 1j * np.array([[1.3, 0.2], [0.2, 1.8]])
    z = 1j * np.array([[0.8, -0.1], [-0.1, 1.1]])
    chi = np.array([[0.3, -0.2], [0.1, 0.4]])
    return np.block([[y, chi], [-chi.T, z]])


def test_admittance_satisfies_both_original_gstcs(sheet):
    rng = np.random.default_rng(31)
    e = rng.normal(size=(4, 20)) + 1j * rng.normal(size=(4, 20))
    h = m.sheet_admittance(sheet) @ e
    currents = sheet @ np.vstack(((e[:2] + e[2:]) / 2, (h[:2] + h[2:]) / 2))
    np.testing.assert_allclose(currents[:2], m.R @ (h[2:] - h[:2]), atol=1e-13)
    np.testing.assert_allclose(currents[2:], -m.R @ (e[2:] - e[:2]), atol=1e-13)
    bad = sheet.copy()
    bad[2:, 2:] = 0
    with pytest.raises(ValueError, match="singular"):
        m.sheet_admittance(bad)


def test_loader_preserves_complex_tensors_and_rejects_spatial_variation(config, sheet, tmp_path):
    nfaces = len(m.load_triangle_obj(config.mesh_path).faces)
    physical = (
        sheet
        * np.array([1 / m.ETA0, 1 / m.ETA0, 1, 1])[:, None]
        * np.array([1, 1, m.ETA0, m.ETA0])[None, :]
    )
    record = np.r_[physical.ravel(), np.linalg.inv(physical).ravel()]
    records = np.tile(record, (nfaces, 1)).astype("<c16")
    tensor_path = tmp_path / "face_tensors.bin"
    records.tofile(tensor_path)
    path = m.APPLICATION / "configurations/dipole_pol_rot/dual_pol_eps7.json"
    _, loaded, _ = m.load_benchmark(path, tensor_path)
    np.testing.assert_allclose(loaded, sheet, rtol=1e-14, atol=1e-14)
    records[-1, 0] += 0.01
    records.tofile(tensor_path)
    with pytest.raises(ValueError, match="constant"):
        m.load_benchmark(path, tensor_path)


def test_solver_selection_accepts_runtime_pardiso_without_compile_time_mkl(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(ng.config, "USE_MKL", False)
    monkeypatch.setattr(ng.config, "USE_PARDISO", False)
    monkeypatch.setattr(ng.config, "USE_UMFPACK", True)
    monkeypatch.setattr(ng, "_mkl_pardiso", SimpleNamespace(_pardiso=object()), raising=False)
    assert m.direct_solver("pardiso") == "pardiso"
    assert m.direct_solver() == "pardiso"
    monkeypatch.setattr(ng, "_mkl_pardiso", SimpleNamespace(_pardiso=None))
    assert m.direct_solver() == "umfpack"
    with pytest.raises(ValueError, match="unavailable"):
        m.direct_solver("pardiso")


@pytest.mark.parametrize("field_id", ["electric_z_interior", "magnetic_z_interior"])
def test_dipole_matches_repository_and_maxwell(config, field_id):
    mesh = m.make_mesh(config, ppw=1, pml_lambda=0)
    points = np.array([[0.1, 0.4, 0.8], [0.4, -0.3, 0.5], [-0.6, 0.2, -0.3]])
    e, h = m.dipole_cf(config, field_id, "interior")
    definition = config.fields[field_id]
    er, hr = evaluate_dipole_fields_cartesian(
        config.case.k_interior,
        points / m.k0_of(config),
        config.case.eta_interior,
        effective_moment=definition.effective_moment,
        dipole_type=definition.dipole_type,
    )
    np.testing.assert_allclose(m.evaluate(e, mesh, points), er.T / m.k0_of(config) ** 3, rtol=1e-12)
    np.testing.assert_allclose(
        m.evaluate(h, mesh, points), hr.T * m.ETA0 / m.k0_of(config) ** 3, rtol=1e-12
    )

    def curl(v):
        return ng.CF(
            (
                v[2].Diff(ng.y) - v[1].Diff(ng.z),
                v[0].Diff(ng.z) - v[2].Diff(ng.x),
                v[1].Diff(ng.x) - v[0].Diff(ng.y),
            )
        )

    eps, mu = m.medium(config, "interior")
    np.testing.assert_allclose(
        m.evaluate(curl(e), mesh, points),
        -1j * mu * m.evaluate(h, mesh, points),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        m.evaluate(curl(h), mesh, points),
        1j * eps * m.evaluate(e, mesh, points),
        rtol=1e-11,
        atol=1e-12,
    )


def test_two_trace_manufactured_solution_converges(config, sheet):
    # Modest electrical size keeps the regression fast. Geometry remains a sphere.
    # Exact Dirichlet data isolate volume/interface signs from PML truncation error.
    config = replace(config, case=replace(config.case, frequency_hz=2e9))
    mesh = m.make_mesh(config, ppw=4, air_lambda=0.23, pml_lambda=0, curve_order=4)
    k = np.sqrt(np.prod(m.medium(config, "interior")))
    eta_rel = np.sqrt(m.medium(config, "interior")[1] / m.medium(config, "interior")[0])
    ei = ng.CF((1, 0.3j, 0)) * ng.exp(-1j * k * ng.z)
    hi = ng.Cross(ng.CF((0, 0, 1)), ei) / eta_rel
    eo, ho = m.dipole_cf(config, "electric_z_interior", "exterior")
    errors = []
    for order in (1, 2):
        system = m.assemble(config, sheet, mesh, order=order)
        hm, hp = system.fields_from_e(ei, eo)
        forcing = (m.local(hi, system.frame) - hm, m.local(ho, system.frame) - hp)
        zero = ng.CF((0, 0, 0))
        rhs = system.rhs(zero, zero, extra_h=forcing)
        solution = ng.GridFunction(system.space)
        solution.components[1].Set(eo, definedon=mesh.Boundaries("outer"))
        r = rhs.vec.CreateVector()
        r.data = rhs.vec - system.matrix.mat * solution.vec
        solution.vec.data += (
            system.matrix.mat.Inverse(system.space.FreeDofs(), inverse=m.direct_solver()) * r
        )
        assert m.residual(system, solution, rhs) < 1e-10
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
    print("manufactured E relative L2 errors (interior, exterior):", errors)
    assert np.max(errors[1]) < 0.04
    assert np.all(np.asarray(errors[1]) < 0.6 * np.asarray(errors[0]))


@pytest.mark.parametrize("field_id", ["electric_z_interior", "magnetic_z_interior"])
def test_huygens_far_field_has_correct_amplitude_phase_and_polarization(config, sheet, field_id):
    from types import SimpleNamespace

    from run import diagnostics

    mesh = m.make_mesh(config, ppw=1, pml_lambda=0)
    e, h = m.dipole_cf(config, field_id, "exterior")
    _, far = diagnostics(SimpleNamespace(config=config, mesh=mesh), ((e, h), (e, h)), sheet, 0.95)
    n = far["direction"]
    expected = np.array([0, 0, 1]) - n * n[:, 2, None]
    if field_id.startswith("magnetic"):
        expected = -np.cross(n, np.array([0, 0, 1]))
    np.testing.assert_allclose(far["E_far_normalized"], expected, rtol=1e-11, atol=1e-12)


def test_zx_samples_use_total_fields_and_mask_pml_origin_and_interface(config):
    from types import SimpleNamespace

    from sample_zx import sample_zx

    mesh = m.make_mesh(config, ppw=1)
    fields = (
        m.dipole_cf(config, "electric_z_interior", "interior"),
        m.dipole_cf(config, "magnetic_z_exterior", "exterior"),
    )
    system = SimpleNamespace(config=config, mesh=mesh)
    # Grid includes origin and exact x/z intersections with the sheet.
    data = sample_zx(system, fields, config, 0.95, grid_size=25, span_radii=1.2)
    r = np.linalg.norm(data["xyz_m"], axis=1)
    valid = data["valid"]
    assert np.any(r == 0)
    assert not valid[r == 0].any()
    assert not valid[np.abs(r - config.case.radius_m) <= 1e-3 * config.case.radius_m].any()
    assert not valid[r >= data["physical_air_radius_m"]].any()
    assert np.isnan(data["E_cartesian"][~valid]).all()
    assert set(data["region"]) == {0, 1, 2}
    for region, index in (("interior", 0), ("exterior", 1)):
        mask = valid & (data["region"] == index + 1)
        field_id = "electric_z_interior" if index == 0 else "magnetic_z_exterior"
        definition = config.fields[field_id]
        e, h = evaluate_dipole_fields_cartesian(
            getattr(config.case, f"k_{region}"),
            data["xyz_m"][mask],
            getattr(config.case, f"eta_{region}"),
            effective_moment=definition.effective_moment,
            dipole_type=definition.dipole_type,
        )
        # Curved-mesh point inversion perturbs mathematically zero components
        # at roundoff level; scale the absolute tolerance to the full field.
        np.testing.assert_allclose(
            data["E_cartesian"][mask], e.T, rtol=1e-11, atol=1e-12 * np.max(np.abs(e))
        )
        np.testing.assert_allclose(
            data["H_cartesian"][mask], h.T, rtol=1e-11, atol=1e-12 * np.max(np.abs(h))
        )


def test_checkpoint_reload_preserves_zx_fields(config, tmp_path):
    from types import SimpleNamespace

    from sample_zx import sample_zx

    mesh = m.make_mesh(config, ppw=1)
    space = m.make_space(mesh, order=1)
    solution = ng.GridFunction(space)
    rng = np.random.default_rng(82)
    solution.vec.FV().NumPy()[:] = rng.normal(size=space.ndof) + 1j * rng.normal(size=space.ndof)
    incident = m.incident_cf(config, "electric_incidence")
    system = SimpleNamespace(config=config, mesh=mesh)
    before = sample_zx(
        system, m.total_fields(system, solution, *incident), config, 0.95, grid_size=25
    )
    mesh.ngmesh.Save(str(tmp_path / "mesh.vol.gz"))
    solution.Save(str(tmp_path / "solution.sol"))
    loaded_mesh = ng.Mesh(str(tmp_path / "mesh.vol.gz"))
    loaded = ng.GridFunction(m.make_space(loaded_mesh, order=1))
    loaded.Load(str(tmp_path / "solution.sol"))
    system = SimpleNamespace(config=config, mesh=loaded_mesh)
    after = sample_zx(system, m.total_fields(system, loaded, *incident), config, 0.95, grid_size=25)
    assert loaded_mesh.GetCurveOrder() == mesh.GetCurveOrder()
    np.testing.assert_array_equal(before["valid"], after["valid"])
    for field in ("E_cartesian", "H_cartesian"):
        valid = before["valid"]
        relative = np.linalg.norm(before[field][valid] - after[field][valid]) / np.linalg.norm(
            before[field][valid]
        )
        assert relative < 1e-7
