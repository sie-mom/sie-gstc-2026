"""Saved-region targets, shared display references, and complex comparisons."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from applications.gstc_surface import plot_fields as plotting
from applications.gstc_surface.observations import sampling_points
from applications.gstc_surface.workflow import DEFAULT_CONFIGURATION, load_workflow_configuration
from sie_mom.postprocessing import complex_vector, write_field_csv


def field_data(points, regions, electric, magnetic):
    result = {
        "point_index": np.arange(len(points)),
        **dict(zip("xyz", np.asarray(points).T)),
        "region": np.asarray(regions),
    }
    for field, values in (("E", electric), ("H", magnetic)):
        for index, axis in enumerate("xyz"):
            result[f"{field}{axis}_real"] = np.asarray(values)[:, index].real
            result[f"{field}{axis}_imag"] = np.asarray(values)[:, index].imag
    return result


def test_matching_uses_excitation_references_instead_of_transformation_order():
    configuration = load_workflow_configuration()
    reversed_configuration = replace(
        configuration, transformations=tuple(reversed(configuration.transformations))
    )
    assert (
        plotting.reference_transformation(reversed_configuration, "electric_z_incidence").id
        == "electric_z_matching"
    )
    duplicated = replace(configuration.transformations[0], id="another_z_target")
    ambiguous = replace(configuration, transformations=(*configuration.transformations, duplicated))
    with pytest.raises(ValueError, match="one reference"):
        plotting.reference_transformation(ambiguous, "electric_z_incidence")
    assert (
        plotting.reference_transformation(
            ambiguous, "electric_z_incidence", "electric_z_matching"
        ).id
        == "electric_z_matching"
    )


def test_prescribed_samples_follow_saved_regions_without_geometry_reclassification():
    configuration = load_workflow_configuration()
    points = np.array([[0.2, 0, 0.001], [0.2, 0, 0.001]])
    assert configuration.geometry.contains(points).all()
    data = field_data(points, [1, 2], np.zeros((2, 3)), np.zeros((2, 3)))
    target = plotting.prescribed_samples(data, configuration, configuration.transformations[0])
    assert not np.allclose(complex_vector(target, "E")[0], complex_vector(target, "E")[1])
    for region_id, region in ((1, "interior"), (2, "exterior")):
        mask = data["region"] == region_id
        expected = plotting.evaluate_prescribed_fields_cartesian(
            getattr(configuration.transformations[0].gstc, region),
            region,
            configuration,
            points_xyz=points[mask],
        )
        for field, values in zip("EH", expected):
            np.testing.assert_array_equal(complex_vector(target, field)[mask], values.T)


def test_zero_components_keep_vector_reference_and_nan_projection_stays_undefined():
    target = field_data([[0, 0, 1]], [1], [[10, 0, 0]], [[0, 0, 0]])
    reference = plotting.display_reference(target, "E")
    assert reference == 10
    shown = plotting.display_magnitude(
        np.array([0.1, 0.0, np.nan]), scale="db", reference=reference, floor_db=80
    )
    np.testing.assert_allclose(shown[:2], [-40, -80])
    assert np.isnan(shown[2])
    assert plotting.display_reference(target, "H") == 1


def test_percentile_reference_does_not_normalize_each_component_or_clip_raw_db_values():
    vectors = np.array([[1000, 0, 0], [10, 0, 0], [10, 0, 0]])
    target = field_data([[0, 0, 1]] * 3, [1, 1, 1], vectors, vectors)
    reference = plotting.display_reference(target, "E", percentile=50)
    assert reference == 10
    shown = plotting.display_magnitude(vectors, scale="db", reference=reference, floor_db=80)
    assert shown[0, 0] == pytest.approx(40)
    assert shown[1, 0] == pytest.approx(0)
    np.testing.assert_array_equal(shown[:, 1:], -80)


def test_exterior_reference_excludes_source_peak_and_preserves_relative_amplitude():
    vectors = np.array([[1e9, 0, 0], [100, 0, 0], [10, 0, 0]])
    target = field_data([[0, 0, 1]] * 3, [1, 2, 2], vectors, vectors)
    reference = plotting.display_reference(target, "E", percentile=100, region="exterior")
    assert reference == 100
    shown = plotting.display_magnitude(vectors, scale="db", reference=reference, floor_db=80)
    np.testing.assert_allclose(shown[:, 0], [140, 0, -20])
    target["region"][:] = 1
    with pytest.raises(ValueError, match="no samples"):
        plotting.display_reference(target, "E", region="exterior")


def test_complex_error_keeps_phase_information_even_when_magnitudes_match():
    points = np.array([[0.0, 0, 1], [0.0, 0, 2]])
    vector = np.array([[1 + 2j, 3, 4j], [2, 1j, -3]])
    target = field_data(points, [1, 2], vector, 2 * vector)
    computed = field_data(points, [1, 2], 1j * vector, 2j * vector)
    errors = plotting.comparison_errors(computed, target)
    for region in ("interior", "exterior"):
        for field in "EH":
            assert errors[region][field]["relative_complex_l2"] == pytest.approx(np.sqrt(2))


@pytest.fixture
def saved_fields(tmp_path):
    directory = tmp_path / "fields"
    directory.mkdir()
    raw = json.loads(DEFAULT_CONFIGURATION.read_text())
    raw["sampling"].update(zx_grid_size=6, radial_points=9)
    (directory / "configuration.json").write_text(json.dumps(raw))
    configuration = load_workflow_configuration(directory / "configuration.json")
    solve = tmp_path / "solve"
    solve.mkdir()
    mesh = (
        Path(__file__).resolve().parents[2]
        / "applications/gstc_surface/meshes/cone_sphere/surface.obj"
    )
    shutil.copy2(mesh, solve / "surface.obj")
    case_id = "electric_z_incidence"
    case = {"id": case_id, "grids": {}}
    (directory / case_id).mkdir()
    for name, points in sampling_points(raw["sampling"]).items():
        regions = np.where(configuration.geometry.contains(points), 1, 2)
        data = field_data(points, regions, np.zeros_like(points), np.zeros_like(points))
        desired = plotting.prescribed_samples(data, configuration, configuration.transformations[0])
        path = directory / case_id / f"{name}.csv"
        write_field_csv(
            path,
            *points.T,
            regions,
            1.1 * complex_vector(desired, "E"),
            1.1 * complex_vector(desired, "H"),
        )
        case["grids"][name] = {
            "path": str(path.relative_to(directory)),
            "sha256": plotting._sha256(path),
        }
    manifest = {
        "application": "gstc_surface",
        "status": "evaluated",
        "cases": [case],
        "solve_directory": str(solve),
        "mesh_sha256": plotting._sha256(mesh),
        "sampling": raw["sampling"],
        "warnings": [],
    }
    (directory / "fields.json").write_text(json.dumps(manifest))
    return directory


@pytest.mark.parametrize("basis,scale", [("spherical", "db"), ("cartesian", "linear")])
def test_renders_components_and_reports_raw_complex_errors_without_changing_inputs(
    saved_fields, basis, scale
):
    original = {path: path.read_bytes() for path in saved_fields.rglob("*") if path.is_file()}
    options = (
        dict(
            vmin_db=-112, vmax_db=-58, map_reference_region="exterior", map_reference_percentile=100
        )
        if scale == "db"
        else {}
    )
    summary_path = plotting.plot_fields(saved_fields, basis=basis, scale=scale, **options)
    summary = json.loads(summary_path.read_text())
    assert summary["basis"] == basis and summary["origin_m"] == [0, 0, 0]
    if scale == "db":
        assert summary["map_limits_db"] == [-112, -58]
        assert summary["map_floor_db"] == 112
        assert summary["floor_db"] == 80  # radial display remains independent
        assert summary["map_reference_region"] == "exterior"
    case = summary["cases"][0]
    assert len(case["files"]) == 6
    for grid in case["grids"].values():
        for region in grid["errors"].values():
            for field in region.values():
                assert field["relative_complex_l2"] == pytest.approx(0.1)
    for name in case["files"]:
        with Image.open(summary_path.parent / name) as picture:
            assert min(picture.size) > 700
            assert np.std(np.asarray(picture.convert("RGB"))) > 20
    assert all(path.read_bytes() == content for path, content in original.items())
    assert plotting.plt.get_fignums() == []


def test_wrong_mesh_and_changed_csv_fail_before_creating_plots(saved_fields, tmp_path):
    output = tmp_path / "plots"
    wrong_mesh = tmp_path / "other.obj"
    wrong_mesh.write_text("different mesh identity")
    with pytest.raises(ValueError, match="mesh SHA-256"):
        plotting.plot_fields(saved_fields, output_dir=output, mesh_path=wrong_mesh)
    manifest = json.loads((saved_fields / "fields.json").read_text())
    csv = saved_fields / manifest["cases"][0]["grids"]["zx"]["path"]
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="CSV SHA-256"):
        plotting.plot_fields(saved_fields, output_dir=output)
    assert not output.exists()


def test_plot_legends_use_the_saved_solver_label(saved_fields, monkeypatch):
    path = saved_fields / "fields.json"
    manifest = json.loads(path.read_text())
    manifest["solver_label"] = "NGSolve"
    path.write_text(json.dumps(manifest))
    texts = []
    original_save = plotting._save_figure

    def inspect(figure, path):
        for axis in figure.axes:
            texts.append(axis.get_title())
            texts.extend(line.get_label() for line in axis.lines)
        original_save(figure, path)

    monkeypatch.setattr(plotting, "_save_figure", inspect)
    summary = json.loads(plotting.plot_fields(saved_fields).read_text())
    assert summary["solver_label"] == "NGSolve"
    assert any("NGSolve" in text for text in texts)
    assert not any("SIE" in text for text in texts)


def test_plots_accept_copied_windows_paths_without_changing_manifest(saved_fields):
    path = saved_fields / "fields.json"
    manifest = json.loads(path.read_text())
    for entry in manifest["cases"][0]["grids"].values():
        entry["path"] = entry["path"].replace("/", "\\")
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    summary = json.loads(plotting.plot_fields(saved_fields).read_text())
    assert len(summary["cases"][0]["files"]) == 6
    assert path.read_bytes() == original
