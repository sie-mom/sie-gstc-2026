import csv
import json
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from applications.anisotropic_gstc_sphere import plot_fields as plotting
from applications.anisotropic_gstc_sphere.sie_gstc_sphere import _run_manifest
from applications.anisotropic_gstc_sphere.workflow import (
    DEFAULT_CONFIGURATION,
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


def _field_data(points, radius=3):
    points = np.asarray(points, dtype=float)
    count = len(points)
    data = {"point_index": np.arange(count), **dict(zip("xyz", points.T))}
    data["region"] = np.where(np.linalg.norm(points, axis=1) < radius, 1, 2)
    for field, vector in (("E", (1 + 2j, -3 + 4j, 5 - 6j)), ("H", (-2 + 1j, 3j, 4 + 5j))):
        for axis, value in zip("xyz", vector):
            data[f"{field}{axis}_real"] = np.full(count, value.real)
            data[f"{field}{axis}_imag"] = np.full(count, value.imag)
    return data


def _write_csv(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(data)
        writer.writerows(zip(*data.values()))


@pytest.fixture
def saved_run(tmp_path):
    configuration = load_workflow_configuration(DEFAULT_CONFIGURATION)
    interior = SimulationExcitationConfiguration(FieldReference("offset_source"), "interior")
    exterior = SimulationExcitationConfiguration(FieldReference("plane_y"), "exterior")
    configuration = replace(
        configuration,
        case_id="arbitrary_fields",
        case=replace(configuration.case, radius_m=3),
        fields={
            **configuration.fields,
            "offset_source": DipoleFieldDefinition("electric", (1, 2, 3), (1, 0, 0.3)),
        },
        sampling=SamplingConfiguration(zx_grid_size=4, radial_points=4),
        simulation=SimulationConfiguration(
            (
                SimulationCaseConfiguration("alpha", (interior, exterior)),
                SimulationCaseConfiguration("beta", (exterior,)),
            )
        ),
    )
    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    (run_dir / "configuration.json").write_text(json.dumps(configuration.to_dict()))
    manifest = _run_manifest(configuration, None)
    manifest["status"] = "completed"
    radii = np.array([0.2, 0.8, 1.2, 1.8]) * 3
    radial = _field_data(radii[:, None] * np.array([1, 2, 2]) / 3)
    x, z = np.meshgrid(np.linspace(-6, 6, 4), np.linspace(-6, 6, 4))
    zx = _field_data(np.column_stack((x.ravel(), np.zeros(x.size), z.ravel())))
    for case in manifest["simulation"]["cases"]:
        case["status"] = "completed"
        for cut, data in (("radial", radial), ("zx", zx)):
            path = f"cases/{case['id']}/fields/custom_{cut}.csv"
            case["files"][f"computed_{cut}"] = path
            _write_csv(run_dir / path, data)
    (run_dir / "run.json").write_text(json.dumps(manifest))
    return run_dir, manifest, configuration


def _save_manifest(run_dir, manifest):
    (run_dir / "run.json").write_text(json.dumps(manifest))


def test_spherical_projection_preserves_complex_components_and_vector_norm():
    theta = np.array([0.2, 0.7, 1.4, 2.1])
    phi = np.array([-2.3, 0.4, 1.8, 2.7])
    r = np.array([0.2, 0.5, 1.5, 3])
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    points = np.column_stack((r * st * cp, r * st * sp, r * ct))
    radial_basis = points / r[:, None]
    theta_basis = np.column_stack((ct * cp, ct * sp, -st))
    phi_basis = np.column_stack((-sp, cp, np.zeros_like(phi)))
    expected = np.array([1 + 2j, -3 + 4j, 5 - 6j])
    cartesian = expected[0] * radial_basis + expected[1] * theta_basis + expected[2] * phi_basis
    data = _field_data(points)
    for field in "EH":
        for index, axis in enumerate("xyz"):
            data[f"{field}{axis}_real"] = cartesian[:, index].real
            data[f"{field}{axis}_imag"] = cartesian[:, index].imag
        actual = plotting.spherical_components(data, field)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-14)
        np.testing.assert_allclose(
            np.linalg.norm(actual, axis=1), np.linalg.norm(cartesian, axis=1)
        )


def test_spherical_projection_handles_axis_and_rejects_origin():
    data = _field_data([(0, 0, 1), (0, 0, -1)])
    x, y, z = plotting.complex_vector(data, "E").T
    actual = plotting.spherical_components(data, "E")
    np.testing.assert_allclose(actual[0], (z[0], x[0], y[0]))
    np.testing.assert_allclose(actual[1], (-z[1], -x[1], y[1]))
    with pytest.raises(ValueError, match="undefined"):
        plotting.spherical_components(_field_data([(0, 0, 0)]), "E")


@pytest.mark.parametrize("scale", ["linear", "log", "db"])
def test_single_case_writes_ten_nonblank_pngs_and_preserves_inputs(saved_run, scale):
    run_dir, manifest, _ = saved_run
    original = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    paths = plotting.plot_fields(run_dir, case_id="alpha", magnitude_scale=scale)
    assert {p.name for p in paths} == {
        "radial_magnitudes.png",
        "radial_electric_components.png",
        "radial_magnetic_components.png",
        "radial_spherical_E_r_H_r_magnitude.png",
        "radial_spherical_E_theta_H_theta_magnitude.png",
        "radial_spherical_E_phi_H_phi_magnitude.png",
        "zx_magnitudes.png",
        "zx_spherical_E_r_H_r_magnitude.png",
        "zx_spherical_E_theta_H_theta_magnitude.png",
        "zx_spherical_E_phi_H_phi_magnitude.png",
    }
    assert all(p.parent == run_dir / "cases/alpha/plots" for p in paths)
    assert not (run_dir / "cases/beta/plots").exists()
    for p in paths:
        with Image.open(p) as picture:
            pixels = np.asarray(picture.convert("RGB"))
        assert pixels.shape[0] > 500 and pixels.shape[1] > 500
        assert np.std(pixels) > 20
    assert all(p.read_bytes() == content for p, content in original.items())
    assert plt.get_fignums() == []


def test_all_cases_use_metadata_radius_sources_and_separate_regional_curves(saved_run, monkeypatch):
    run_dir, _, _ = saved_run
    figures = []

    def inspect_figure(figure, path):
        figures.append(path)
        if path.name == "radial_magnitudes.png":
            axis = figure.axes[0]
            interior, exterior, boundary = axis.lines
            np.testing.assert_allclose(interior.get_xdata(), (0.2, 0.8))
            np.testing.assert_allclose(exterior.get_xdata(), (1.2, 1.8))
            np.testing.assert_allclose(boundary.get_xdata(), (1, 1))
            assert "a = 3 m" in figure._suptitle.get_text()
        if path.name == "zx_magnitudes.png":
            for axis in figure.axes[:2]:
                assert axis.patches[0].radius == 1
                assert axis.get_aspect() == 1
                np.testing.assert_allclose(axis.get_xlim(), (-2, 2))
                if path.parent.parent.name == "alpha":
                    np.testing.assert_allclose(axis.collections[-1].get_offsets(), [[1 / 3, 0.1]])
                else:
                    assert len(axis.collections) == 1
        plt.close(figure)

    monkeypatch.setattr(plotting, "_save_figure", inspect_figure)
    paths = plotting.plot_fields(run_dir)
    assert paths == figures and len(paths) == 20
    assert {p.parent.parent.name for p in paths} == {"alpha", "beta"}


@pytest.mark.parametrize(
    "problem, error",
    [
        ("unknown_case", "unknown simulation case"),
        ("unfinished", "not completed"),
        ("no_fields", "no saved physical fields"),
        ("missing_file", "No such file"),
        ("schema", "schema_version 2"),
        ("backend", "SIE, SVWF, or COMSOL"),
        ("radius", "disagrees"),
        ("units", "unsupported saved field"),
        ("outside_path", "escapes"),
        ("absolute_path", "must be relative"),
        ("bad_later_case", "not completed"),
        ("excitations", "excitations disagree"),
    ],
)
def test_invalid_metadata_fails_before_creating_plots(saved_run, problem, error):
    run_dir, manifest, _ = saved_run
    first = manifest["simulation"]["cases"][0]
    case_id = "alpha"
    if problem == "unknown_case":
        case_id = "not_a_case"
    elif problem == "unfinished":
        first["status"] = "pending"
    elif problem == "no_fields":
        first["files"].pop("computed_radial")
    elif problem == "missing_file":
        (run_dir / first["files"]["computed_radial"]).unlink()
    elif problem == "schema":
        manifest["schema_version"] = 1
    elif problem == "backend":
        manifest["backend"] = "unknown"
    elif problem == "radius":
        manifest["radius_m"] = 2
    elif problem == "units":
        manifest["field_data"]["magnetic_field_unit"] = "mA/m"
    elif problem == "outside_path":
        first["files"]["plots"] = "../../plots"
    elif problem == "absolute_path":
        first["files"]["computed_zx"] = str(run_dir / first["files"]["computed_zx"])
    elif problem == "bad_later_case":
        manifest["simulation"]["cases"][1]["status"] = "failed"
        case_id = None
    elif problem == "excitations":
        first["excitations"] = [{"field": "plane_y", "region": "interior"}]
    _save_manifest(run_dir, manifest)
    with pytest.raises((OSError, ValueError), match=error):
        plotting.plot_fields(run_dir, case_id=case_id)
    assert not list(run_dir.rglob("plots"))


@pytest.mark.parametrize(
    "problem, error",
    [
        ("nonfinite", "non-finite"),
        ("indices", "indices"),
        ("region", "regions must"),
        ("wrong_region", "regions disagree"),
        ("missing_column", "columns"),
    ],
)
def test_field_csv_validation(tmp_path, problem, error):
    data = _field_data([(1, 0, 0), (4, 0, 0)])
    if problem == "nonfinite":
        data["Ex_real"][0] = np.nan
    elif problem == "indices":
        data["point_index"][1] = 0
    elif problem == "region":
        data["region"][0] = 4
    elif problem == "wrong_region":
        data["region"][0] = 2
    else:
        data.pop("Hx_imag")
    path = tmp_path / "fields.csv"
    _write_csv(path, data)
    with pytest.raises(ValueError, match=error):
        plotting.load_field_csv(path, radius_m=3)


def test_zx_grid_reorders_samples_and_rejects_duplicate_coordinates():
    x, z = np.meshgrid([-2, 2], [-2, 2])
    data = _field_data(np.column_stack((x.ravel(), np.zeros(x.size), z.ravel())))
    permutation = np.array([2, 0, 3, 1])
    shuffled = {name: values[permutation] for name, values in data.items()}
    actual_x, actual_z, order = plotting.zx_grid(shuffled, 2)
    np.testing.assert_array_equal(actual_x, [-2, 2])
    np.testing.assert_array_equal(actual_z, [-2, 2])
    np.testing.assert_array_equal(shuffled["point_index"][order], np.arange(4))
    shuffled["x"][2] = shuffled["x"][0]
    with pytest.raises(ValueError, match="duplicate or missing"):
        plotting.zx_grid(shuffled, 2)


@pytest.mark.parametrize(
    "problem, error",
    [
        ("count", "radial field count"),
        ("origin", "nonzero radii"),
        ("ray", "radial ray"),
    ],
)
def test_invalid_radial_coordinates_fail_before_creating_plots(saved_run, problem, error):
    run_dir, manifest, _ = saved_run
    path = run_dir / manifest["simulation"]["cases"][0]["files"]["computed_radial"]
    data = plotting.load_field_csv(path, radius_m=3)
    if problem == "count":
        data = {name: values[:-1] for name, values in data.items()}
    elif problem == "origin":
        for axis in "xyz":
            data[axis][0] = 0
    else:
        data["x"][0], data["y"][0], data["z"][0] = 0.6, 0, 0
    _write_csv(path, data)
    with pytest.raises(ValueError, match=error):
        plotting.plot_fields(run_dir, case_id="alpha")
    assert not list(run_dir.rglob("plots"))


@pytest.mark.parametrize("scale", ["log", "db"])
def test_zero_fields_can_be_plotted_with_requested_scale(saved_run, scale):
    run_dir, manifest, _ = saved_run
    files = manifest["simulation"]["cases"][0]["files"]
    for cut in ("radial", "zx"):
        path = run_dir / files[f"computed_{cut}"]
        data = plotting.load_field_csv(path, radius_m=3)
        for name, values in data.items():
            if name.startswith(("E", "H")):
                values[:] = 0
        _write_csv(path, data)
    paths = plotting.plot_fields(run_dir, case_id="alpha", magnitude_scale=scale)
    assert len(paths) == 10 and all(p.is_file() for p in paths)
    assert plt.get_fignums() == []


def test_db_display_uses_fixed_si_reference_and_preserves_amplitudes():
    magnitude = np.array([0, 0.5, 1, 10, 1e-100])
    original = magnitude.copy()
    display = plotting._display_magnitude(magnitude, "db")
    assert display.mask[0]
    np.testing.assert_allclose(display[1:], [-6.020599913279624, 0, 20, -2000])
    np.testing.assert_array_equal(magnitude, original)
    for scale in ("linear", "log"):
        np.testing.assert_array_equal(plotting._display_magnitude(magnitude, scale), magnitude)
    norm = plotting._magnitude_norm(display, "db")
    assert norm.vmin == -2000 and norm.vmax == 20


@pytest.mark.parametrize(
    "line_scale,map_scale", [("linear", "db"), ("db", "log"), ("log", "linear")]
)
def test_independent_display_scales_keep_original_field_amplitudes(
    saved_run, monkeypatch, line_scale, map_scale
):
    run_dir, manifest, _ = saved_run
    original = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    files = manifest["simulation"]["cases"][0]["files"]
    expected = {}
    for cut in ("radial", "zx"):
        data = plotting.load_field_csv(run_dir / files[f"computed_{cut}"], radius_m=3)
        expected[cut] = {
            field: np.linalg.norm(plotting.complex_vector(data, field), axis=1) for field in "EH"
        }

    def inspect_figure(figure, path):
        if path.name == "radial_magnitudes.png":
            for axis, field in zip(figure.axes, "EH"):
                values = np.concatenate([line.get_ydata() for line in axis.lines[:2]])
                target = expected["radial"][field]
                if line_scale == "db":
                    target = 20 * np.log10(target)
                np.testing.assert_allclose(values, target)
                assert axis.get_yscale() == ("log" if line_scale == "log" else "linear")
                assert axis.get_ylabel() == plotting._magnitude_label(field, line_scale)
        elif path.name == "zx_magnitudes.png":
            for axis, field in zip(figure.axes[:2], "EH"):
                artist = axis.collections[0]
                target = expected["zx"][field]
                if map_scale == "db":
                    target = 20 * np.log10(target)
                np.testing.assert_allclose(np.asarray(artist.get_array()).ravel(), target)
                assert isinstance(artist.norm, plotting.LogNorm) == (map_scale == "log")
                assert artist.colorbar.ax.get_ylabel() == plotting._magnitude_label(
                    field, map_scale
                )
        plt.close(figure)

    monkeypatch.setattr(plotting, "_save_figure", inspect_figure)
    plotting.plot_fields(run_dir, case_id="alpha", line_scale=line_scale, map_scale=map_scale)
    assert all(p.read_bytes() == content for p, content in original.items())


@pytest.mark.parametrize("option", ["magnitude_scale", "line_scale", "map_scale"])
def test_invalid_display_scale_fails_before_creating_plots(saved_run, option):
    run_dir, _, _ = saved_run
    with pytest.raises(ValueError, match="scale must be linear, log or db"):
        plotting.plot_fields(run_dir, **{option: "normalized"})
    assert not list(run_dir.rglob("plots"))


def test_cli_selects_case_and_independent_scales(saved_run, monkeypatch, capsys):
    run_dir, _, _ = saved_run
    paths = [run_dir / "cases/beta/plots/example.png"]

    def fake_plot(directory, *, case_id, magnitude_scale, line_scale, map_scale, **options):
        assert directory == run_dir and case_id == "beta" and magnitude_scale == "linear"
        assert line_scale == "log" and map_scale == "db"
        assert options == {
            "map_color_limits": "full",
            "map_percentiles": (2.0, 80.0),
            "line_basis": "cartesian",
            "map_basis": "spherical",
            "line_components": ("E_x", "H_y"),
            "map_components": ("E_theta", "H_phi"),
            "line_part": "magnitude",
            "map_part": "magnitude",
            "reference_transformation": "electric_displacement",
        }
        return paths

    monkeypatch.setattr(plotting, "plot_fields", fake_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_fields.py",
            "--run-dir",
            str(run_dir),
            "--case",
            "beta",
            "--magnitude-scale",
            "linear",
            "--line-scale",
            "log",
            "--map-scale",
            "db",
            "--map-color-limits",
            "full",
            "--map-percentiles",
            "2",
            "80",
            "--line-basis",
            "cartesian",
            "--map-basis",
            "spherical",
            "--line-components",
            "E_x",
            "H_y",
            "--map-components",
            "E_theta",
            "H_phi",
            "--reference-transformation",
            "electric_displacement",
        ],
    )
    assert plotting.main() == 0
    assert str(paths[0]) in capsys.readouterr().out


def test_cli_refreshes_default_comparisons_when_scale_changes(saved_run, monkeypatch, capsys):
    run_dir, manifest, configuration = saved_run
    reference_id = configuration.transformations[0].id
    data = plotting.load_field_csv(
        run_dir / manifest["simulation"]["cases"][0]["files"]["computed_radial"], radius_m=3
    )
    save_figure = plotting._save_figure
    filename = f"radial_spherical_E_theta_H_theta_magnitude_vs_{reference_id}.png"
    previous_image = None

    def inspect_and_save(figure, path):
        if path.name == filename:
            for axis, field in zip(figure.axes, "EH"):
                expected = np.abs(plotting._quantity(data, f"{field}_theta", "spherical"))
                if scale == "db":
                    expected = 20 * np.log10(expected)
                actual = np.ma.concatenate([axis.lines[0].get_ydata(), axis.lines[2].get_ydata()])
                np.testing.assert_allclose(actual, expected)
                assert axis.get_yscale() == ("log" if scale == "log" else "linear")
                assert ("dB re 1" in axis.get_ylabel()) == (scale == "db")
        save_figure(figure, path)

    monkeypatch.setattr(plotting, "_save_figure", inspect_and_save)
    for scale in ("linear", "log", "db"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "plot_fields.py",
                "--run-dir",
                str(run_dir),
                "--case",
                "alpha",
                "--reference-transformation",
                reference_id,
                "--line-scale",
                scale,
                "--map-scale",
                scale,
                "--map-color-limits",
                "percentile",
            ],
        )
        assert plotting.main() == 0
        output = capsys.readouterr().out.splitlines()
        path = run_dir / "cases/alpha/plots" / filename
        assert len(output) == 10 and str(path) in output
        current_image = path.read_bytes()
        assert current_image != previous_image
        previous_image = current_image
    assert plt.get_fignums() == []


@pytest.mark.parametrize("basis", ["spherical", "cartesian"])
def test_component_selection_keeps_complex_values_and_norms(basis):
    data = _field_data([(1, 2, 3), (-2, 1, 4)])
    for field in "EH":
        expected = (
            plotting.spherical_components(data, field)
            if basis == "spherical"
            else plotting.complex_vector(data, field)
        )
        for column, component in enumerate(plotting.BASES[basis]):
            np.testing.assert_allclose(
                plotting._quantity(data, f"{field}_{component}", basis), expected[:, column]
            )
        np.testing.assert_allclose(
            plotting._quantity(data, field, basis), np.linalg.norm(expected, axis=1)
        )


@pytest.mark.parametrize("part", ["magnitude", "real", "imag", "phase"])
def test_selected_component_display_parts(part):
    values = np.array([1 + 2j, -3 + 4j, 0j, np.nan + 1j])
    display = plotting._display_values(values, part, "linear")
    expected = {
        "magnitude": np.abs(values),
        "real": values.real,
        "imag": values.imag,
        "phase": np.angle(values, deg=True),
    }[part]
    np.testing.assert_allclose(display[:2], expected[:2])
    assert display.mask[-1]
    if part == "phase":
        assert display.mask[2]


def test_prescribed_samples_use_regional_media_complex_amplitudes_and_plane_wave_phase(saved_run):
    _, _, configuration = saved_run
    amplitude = np.array([1 + 2j, 0.5 - 3j, 0])
    configuration = replace(
        configuration,
        case=replace(
            configuration.case, relative_permittivity_interior=4, relative_permeability_exterior=2
        ),
        fields={
            **configuration.fields,
            "complex_plane": PlaneWaveFieldDefinition(tuple(amplitude), (0, 0, 1)),
        },
    )
    reference = GSTCTransformationConfiguration(
        "arbitrary_reference",
        GSTCFieldsConfiguration(FieldReference("complex_plane"), FieldReference("complex_plane")),
    )
    points = np.array([(0, 0, 0), (1, 0, 1), (4, 0, -2)])
    data = _field_data(points)
    target = plotting.prescribed_samples(data, configuration, reference)
    for region_id, name in ((1, "interior"), (2, "exterior")):
        mask = data["region"] == region_id
        electric = amplitude[None, :] * np.exp(
            -1j * getattr(configuration.case, f"k_{name}") * points[mask, 2, None]
        )
        magnetic = np.cross([0, 0, 1], electric) / getattr(configuration.case, f"eta_{name}")
        np.testing.assert_allclose(plotting.complex_vector(target, "E")[mask], electric, atol=1e-14)
        np.testing.assert_allclose(plotting.complex_vector(target, "H")[mask], magnetic, atol=1e-14)


def test_prescribed_samples_mask_only_regional_source_singularities_and_allow_null(saved_run):
    _, _, configuration = saved_run
    source = configuration.fields["offset_source"].source_position
    configuration = replace(
        configuration,
        fields={
            **configuration.fields,
            "outer_source": DipoleFieldDefinition("magnetic", (0, 0, 0.5), (0, 0, 0)),
        },
    )
    reference = GSTCTransformationConfiguration(
        "dipoles",
        GSTCFieldsConfiguration(FieldReference("offset_source"), FieldReference("outer_source")),
    )
    data = _field_data([source, (0, 0, 0), (4, 0, 1)])
    target = plotting.prescribed_samples(data, configuration, reference)
    for field in "EH":
        vectors = plotting.complex_vector(target, field)
        assert np.isnan(vectors[0]).all()
        assert np.isfinite(vectors[1:]).all()
        projected = plotting._field_components(target, field, "spherical")
        assert np.isnan(projected[1]).all()
    null = replace(reference, gstc=GSTCFieldsConfiguration(None, None))
    target = plotting.prescribed_samples(data, configuration, null)
    for field in "EH":
        np.testing.assert_array_equal(plotting.complex_vector(target, field), np.zeros((3, 3)))


def test_magnetic_target_keeps_half_amplitude_between_regions(saved_run):
    _, _, configuration = saved_run
    configuration = replace(
        configuration,
        fields={
            **configuration.fields,
            "full": DipoleFieldDefinition("magnetic", (0, 0, 1), (0, 0, 0)),
            "half": DipoleFieldDefinition("magnetic", (0, 0, 0.5), (0, 0, 0)),
        },
    )
    reference = GSTCTransformationConfiguration(
        "attenuation", GSTCFieldsConfiguration(FieldReference("full"), FieldReference("half"))
    )
    data = _field_data([(1, 0, 1), (1, 0, 1)])
    data["region"] = np.array([1, 2])
    target = plotting.prescribed_samples(data, configuration, reference)
    for field in "EH":
        vectors = plotting.complex_vector(target, field)
        np.testing.assert_allclose(vectors[1], 0.5 * vectors[0])


@pytest.mark.parametrize(
    "basis,selections", [("spherical", ("E_theta", "H_phi")), ("cartesian", ("E_x", "H_y"))]
)
@pytest.mark.parametrize("part", ["magnitude", "real", "imag", "phase"])
def test_selected_components_render_nonblank_comparison_figures(saved_run, basis, selections, part):
    run_dir, _, configuration = saved_run
    original = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    paths = plotting.plot_fields(
        run_dir,
        case_id="alpha",
        line_basis=basis,
        map_basis=basis,
        line_components=selections,
        map_components=selections,
        line_part=part,
        map_part=part,
        line_scale="linear",
        map_scale="db" if part == "magnitude" else "linear",
        reference_transformation=configuration.transformations[0].id,
    )
    assert len(paths) == 2
    for path in paths:
        assert basis in path.name and f"_{part}_vs_" in path.name
        with Image.open(path) as picture:
            pixels = np.asarray(picture.convert("RGB"))
        assert pixels.shape[0] > 500 and pixels.shape[1] > 500
        assert np.std(pixels) > 20
    assert all(p.read_bytes() == content for p, content in original.items())
    assert plt.get_fignums() == []


def test_reference_maps_share_limits_and_radial_curves_keep_separate_regions(
    saved_run, monkeypatch
):
    run_dir, _, configuration = saved_run
    reference_id = configuration.transformations[0].id

    def inspect_figure(figure, path):
        if path.name.startswith("zx_"):
            for computed, prescribed in (
                (figure.axes[0], figure.axes[1]),
                (figure.axes[2], figure.axes[3]),
            ):
                assert computed.collections[0].norm is prescribed.collections[0].norm
                assert computed.get_title().startswith("Computed")
                assert prescribed.get_title().startswith("Prescribed")
                assert "dB re 1" in computed.collections[0].colorbar.ax.get_ylabel()
        else:
            for axis in figure.axes:
                assert [line.get_label() for line in axis.lines[:4]] == [
                    "Interior computed",
                    "Interior prescribed",
                    "Exterior computed",
                    "Exterior prescribed",
                ]
                np.testing.assert_allclose(axis.lines[1].get_xdata(), [0.2, 0.8])
                np.testing.assert_allclose(axis.lines[3].get_xdata(), [1.2, 1.8])
                for computed, prescribed, marker in (
                    (axis.lines[0], axis.lines[1], "o"),
                    (axis.lines[2], axis.lines[3], "^"),
                ):
                    assert prescribed.get_color() == "black"
                    assert prescribed.get_linestyle() == "-"
                    assert computed.get_color() != prescribed.get_color()
                    assert computed.get_marker() == marker
                    assert computed.get_markerfacecolor() == "none"
                    assert computed.get_zorder() > prescribed.get_zorder()
        assert reference_id in figure._suptitle.get_text()
        plt.close(figure)

    monkeypatch.setattr(plotting, "_save_figure", inspect_figure)
    plotting.plot_fields(
        run_dir,
        case_id="alpha",
        line_components=("E_theta", "H_phi"),
        map_components=("E_theta", "H_phi"),
        line_scale="linear",
        map_scale="db",
        reference_transformation=reference_id,
    )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"line_basis": "local"}, "basis"),
        ({"map_basis": "local"}, "basis"),
        ({"line_components": ()}, "unique selections"),
        ({"map_components": ("E_theta", "E_theta")}, "unique selections"),
        ({"line_basis": "cartesian", "line_components": ("E_theta",)}, "unique selections"),
        ({"map_components": ("H_x",)}, "unique selections"),
        ({"line_components": ("E_theta",), "line_part": "real"}, "linear scale"),
        ({"map_components": ("H_phi",), "map_part": "imag", "map_scale": "db"}, "linear scale"),
        ({"line_part": "phase"}, "components must be selected"),
        ({"map_part": "power"}, "part must"),
        ({"map_components": ("E",), "map_part": "phase", "map_scale": "linear"}, "not E/H norms"),
        ({"reference_transformation": "not_a_transformation"}, "unknown reference"),
        ({"case_id": None, "reference_transformation": "any"}, "explicit simulation --case"),
    ],
)
def test_invalid_component_or_reference_selection_fails_before_plots(saved_run, options, error):
    run_dir, _, _ = saved_run
    with pytest.raises(ValueError, match=error):
        plotting.plot_fields(run_dir, **{"case_id": "alpha", **options})
    assert not list(run_dir.rglob("plots"))


def test_default_cartesian_components_and_reference_overlays(saved_run):
    run_dir, _, configuration = saved_run
    paths = plotting.plot_fields(
        run_dir,
        case_id="alpha",
        line_basis="cartesian",
        reference_transformation=configuration.transformations[0].id,
    )
    assert len(paths) == 10
    assert sum("components_cartesian" in path.name for path in paths) == 2
    assert all("_vs_" in path.name for path in paths)


def test_odd_zx_grid_masks_spherical_origin_and_single_component_layout(tmp_path):
    x, z = np.meshgrid([-6, 0, 6], [-6, 0, 6])
    data = _field_data(np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()]))
    assert np.isnan(plotting._quantity(data, "H_phi", "spherical")[4])
    assert plotting._display_values(
        plotting._quantity(data, "H_phi", "spherical"), "imag", "linear"
    ).mask[4]
    assert np.isfinite(plotting._quantity(data, "H_y", "cartesian")[4])
    for scale in ("linear", "log", "db"):
        path = plotting._plot_zx(
            data,
            radius_m=3,
            grid=plotting.zx_grid(data, 3),
            title="Origin masking",
            directory=tmp_path,
            magnitude_scale=scale,
            sources=(),
            selections=("H_phi",),
        )
        assert path.is_file()
    assert plt.get_fignums() == []


@pytest.mark.parametrize("scale", ["linear", "log", "db"])
@pytest.mark.parametrize("policy", ["percentile", "full"])
@pytest.mark.parametrize(
    "percentiles", [(2, 98), (2, 95), (2, 80), (2, 70), (0, 100), (12.5, 87.5)]
)
def test_map_color_limits_clip_display_only(scale, policy, percentiles):
    magnitude = np.r_[1e-8, np.linspace(1, 10, 100), 1e8]
    original = magnitude.copy()
    display = plotting._display_magnitude(magnitude, scale)
    norm = plotting._color_norm(display, "magnitude", scale, policy, percentiles)
    if policy == "percentile":
        np.testing.assert_allclose(
            [norm.vmin, norm.vmax], np.percentile(display.compressed(), percentiles)
        )
    else:
        assert norm.vmin == (0 if scale == "linear" else display.min())
        assert norm.vmax == display.max()
    np.testing.assert_array_equal(magnitude, original)


@pytest.mark.parametrize("scale", ["linear", "log", "db"])
@pytest.mark.parametrize("value", [0, 1, 1e-9])
@pytest.mark.parametrize("percentiles", [(2, 98), (2, 70)])
def test_percentile_limits_handle_zero_and_constant_plane_wave_magnitudes(
    scale, value, percentiles
):
    values = np.full(100, value)
    values[0] *= 1 + 1e-14
    display = plotting._display_magnitude(values, scale)
    norm = plotting._color_norm(display, "magnitude", scale, "percentile", percentiles)
    assert np.isfinite([norm.vmin, norm.vmax]).all()
    assert norm.vmax > norm.vmin
    if value:
        assert norm.vmin <= display.min() and norm.vmax >= display.max()


@pytest.mark.parametrize("policy", ["percentile", "full"])
@pytest.mark.parametrize("percentiles", [(2, 98), (2, 80)])
@pytest.mark.parametrize("scale", ["linear", "log", "db"])
def test_comparison_map_color_policy_is_shared_between_panels(
    saved_run, monkeypatch, policy, percentiles, scale
):
    run_dir, _, configuration = saved_run
    reference_id = configuration.transformations[0].id

    def inspect_figure(figure, path):
        if path.name.startswith("zx_"):
            for index, selection in enumerate(("E_theta", "H_phi")):
                first = figure.axes[index * 2].collections[0]
                second = figure.axes[index * 2 + 1].collections[0]
                shared = np.ma.concatenate(
                    [first.get_array().ravel(), second.get_array().ravel()]
                ).compressed()
                eligible = shared[shared > 0] if scale == "log" else shared
                expected = (
                    np.percentile(eligible, percentiles)
                    if policy == "percentile"
                    else [0 if scale == "linear" else eligible.min(), eligible.max()]
                )
                np.testing.assert_allclose([first.norm.vmin, first.norm.vmax], expected)
                assert first.norm is second.norm
                below = shared.min() < first.norm.vmin
                above = shared.max() > first.norm.vmax
                expected_extend = (
                    "both" if below and above else "min" if below else "max" if above else "neither"
                )
                assert first.colorbar.extend == expected_extend
        plt.close(figure)

    monkeypatch.setattr(plotting, "_save_figure", inspect_figure)
    options = {} if policy == "percentile" else {"map_color_limits": policy}
    plotting.plot_fields(
        run_dir,
        case_id="alpha",
        line_components=("E_theta", "H_phi"),
        map_components=("E_theta", "H_phi"),
        map_scale=scale,
        map_percentiles=percentiles,
        reference_transformation=reference_id,
        **options,
    )


def test_invalid_map_color_limits_fail_before_creating_plots(saved_run):
    run_dir, _, _ = saved_run
    with pytest.raises(ValueError, match="map color limits must"):
        plotting.plot_fields(run_dir, map_color_limits="normalized")
    assert not list(run_dir.rglob("plots"))


@pytest.mark.parametrize(
    "percentiles",
    [
        (-1, 95),
        (2, 101),
        (95, 2),
        (2, 2),
        (np.nan, 95),
        (2, np.inf),
        (),
        (2,),
        (2, 95, 98),
        ("low", 95),
        None,
    ],
)
def test_invalid_map_percentiles_fail_before_creating_plots(saved_run, percentiles):
    run_dir, _, _ = saved_run
    with pytest.raises(ValueError, match="map percentiles must satisfy"):
        plotting.plot_fields(run_dir, case_id="alpha", map_percentiles=percentiles)
    assert not list(run_dir.rglob("plots"))


@pytest.mark.parametrize("percentiles", [("-1", "95"), ("2", "101"), ("80", "70"), ("nan", "95")])
def test_cli_rejects_invalid_map_percentiles(saved_run, monkeypatch, capsys, percentiles):
    run_dir, _, _ = saved_run
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_fields.py", "--run-dir", str(run_dir), "--map-percentiles", *percentiles],
    )
    with pytest.raises(SystemExit) as error:
        plotting.main()
    assert error.value.code == 2
    assert "map percentiles must satisfy" in capsys.readouterr().err
    assert not list(run_dir.rglob("plots"))


@pytest.mark.parametrize("percentiles", [(2, 98), (10, 70)])
def test_signed_percentile_maps_remain_symmetric_and_phase_limits_unchanged(percentiles):
    values = np.r_[-1e8, np.linspace(-5, 10, 100), 1e8]
    bounds = np.percentile(values, percentiles)
    norm = plotting._color_norm(values, "real", "linear", "percentile", percentiles)
    np.testing.assert_allclose([norm.vmin, norm.vmax], [-max(abs(bounds)), max(abs(bounds))])
    for policy in ("percentile", "full"):
        phase = plotting._color_norm(values, "phase", "linear", policy, percentiles)
        assert phase.vmin == -180 and phase.vmax == 180


@pytest.mark.parametrize("basis", ["spherical", "cartesian"])
@pytest.mark.parametrize("scale", ["linear", "log", "db"])
@pytest.mark.parametrize("with_reference", [False, True])
def test_default_component_magnitudes_follow_basis_scale_and_reference(
    saved_run, monkeypatch, basis, scale, with_reference
):
    run_dir, manifest, configuration = saved_run
    files = manifest["simulation"]["cases"][0]["files"]
    samples = {
        cut: plotting.load_field_csv(run_dir / files[f"computed_{cut}"], radius_m=3)
        for cut in ("radial", "zx")
    }
    transformation = configuration.transformations[0]
    targets = {
        cut: plotting.prescribed_samples(data, configuration, transformation)
        if with_reference
        else None
        for cut, data in samples.items()
    }
    observed = set()

    def inspect_figure(figure, path):
        component = next(
            (
                name
                for name in plotting.BASES[basis]
                if f"_{basis}_E_{name}_H_{name}_magnitude" in path.name
            ),
            None,
        )
        cut = path.name.split("_")[0]
        if component is not None:
            observed.add((cut, component))
            assert ("_vs_" in path.name) == with_reference
            data, target = samples[cut], targets[cut]
            for index, field in enumerate("EH"):
                selection = f"{field}_{component}"
                values = np.ma.masked_invalid(np.abs(plotting._quantity(data, selection, basis)))
                expected = (
                    20 * np.ma.log10(np.ma.masked_less_equal(values, 0))
                    if scale == "db"
                    else values
                )
                prescribed = (
                    None
                    if target is None
                    else plotting._display_values(
                        plotting._quantity(target, selection, basis), "magnitude", scale
                    )
                )
                if cut == "radial":
                    assert len(figure.axes) == 2
                    axis = figure.axes[index]
                    interior, exterior = axis.lines[0], axis.lines[2 if with_reference else 1]
                    np.testing.assert_allclose(interior.get_ydata(), expected[:2])
                    np.testing.assert_allclose(exterior.get_ydata(), expected[2:])
                    np.testing.assert_allclose(interior.get_xdata(), [0.2, 0.8])
                    np.testing.assert_allclose(exterior.get_xdata(), [1.2, 1.8])
                    assert axis.get_ylabel() == plotting._magnitude_label(selection, scale)
                    assert axis.get_yscale() == ("log" if scale == "log" else "linear")
                    if target is not None:
                        np.testing.assert_allclose(axis.lines[1].get_ydata(), prescribed[:2])
                        np.testing.assert_allclose(axis.lines[3].get_ydata(), prescribed[2:])
                        if scale in ("log", "db"):
                            raw = plotting._quantity(target, selection, basis)
                            for section, label in ((raw[:2], "Interior"), (raw[2:], "Exterior")):
                                if np.all(section == 0):
                                    assert any(
                                        "Prescribed = 0" in text.get_text()
                                        and label in text.get_text()
                                        for text in axis.texts
                                    )
                else:
                    x, z, order = plotting.zx_grid(data, 4)
                    axis = figure.axes[index * 2 if with_reference else index]
                    artist = axis.collections[0]
                    np.testing.assert_allclose(artist.get_array().ravel(), expected[order].ravel())
                    assert artist.colorbar.ax.get_ylabel() == plotting._magnitude_label(
                        selection, scale
                    )
                    assert isinstance(artist.norm, plotting.LogNorm) == (scale == "log")
                    if prescribed is not None:
                        reference_artist = figure.axes[index * 2 + 1].collections[0]
                        assert artist.norm is reference_artist.norm
                        np.testing.assert_allclose(
                            reference_artist.get_array().ravel(), prescribed[order].ravel()
                        )
                        np.testing.assert_array_equal(
                            np.ma.getmaskarray(reference_artist.get_array()).ravel(),
                            np.ma.getmaskarray(prescribed[order]).ravel(),
                        )
        elif "_components" in path.name:
            assert all(axis.get_yscale() == "linear" for axis in figure.axes)
        plt.close(figure)

    monkeypatch.setattr(plotting, "_save_figure", inspect_figure)
    paths = plotting.plot_fields(
        run_dir,
        case_id="alpha",
        line_basis=basis,
        map_basis=basis,
        line_scale=scale,
        map_scale=scale,
        reference_transformation=transformation.id if with_reference else None,
    )
    assert len(paths) == 10
    assert observed == {
        (cut, component) for cut in ("radial", "zx") for component in plotting.BASES[basis]
    }
