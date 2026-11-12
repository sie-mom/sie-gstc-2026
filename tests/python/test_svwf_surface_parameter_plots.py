"""Standalone material checks and signed parameter plots from saved arrays."""

import hashlib
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from applications.anisotropic_gstc_sphere.experiments.svwf_inverse import (
    plot_surface_parameters as plotting,
)


@pytest.fixture
def saved_parameters(tmp_path):
    rows, columns, eta = 4, 8, 377.0
    nodes, _ = np.polynomial.legendre.leggauss(rows)
    theta, phi = np.meshgrid(
        np.arccos(nodes[::-1]), 2 * np.pi * np.arange(columns) / columns, indexing="ij"
    )
    variation = (np.sin(theta) * np.cos(phi)).ravel()
    ones = np.ones(rows * columns)
    y = 1j / eta * np.array([
        [ones + 0.1 * variation, 0.2 * ones],
        [0.2 * ones, 1.5 * ones - 0.1 * variation],
    ])
    z = 1j * eta * np.array([
        [0.7 * ones, -0.1 * ones], [-0.1 * ones, 1.2 * ones],
    ])
    chi = np.array([
        [0.1 * variation, 0.05 * ones], [-0.03 * ones, 0.15 * variation],
    ], complex)
    directory = tmp_path / "saved"
    directory.mkdir()
    metadata = {
        "method": "joint_svwf_reciprocal_lossless",
        "n_theta": rows, "n_phi": columns, "angular_degree": 2,
        "reference_impedance_ohm": eta,
        "field_and_material_time_convention": "exp(+j omega t)",
    }
    (directory / "experiment.json").write_text(json.dumps(metadata))
    np.savez(directory / "surface_parameters.npz", theta=theta.ravel(), phi=phi.ravel(),
             Y=y, Chi=chi, Gamma=-chi.swapaxes(0, 1), Z=z)
    return directory


def test_saved_blocks_pass_material_checks(saved_parameters):
    metadata, parameters, theta, phi = plotting.load_saved_parameters(saved_parameters)
    checks, condition = plotting.material_checks(
        parameters, metadata["reference_impedance_ohm"], theta, phi
    )
    assert checks["sample_count"] == 32
    assert checks["all_tensors_finite"]
    assert checks["reciprocity_within_tolerance"]
    assert checks["losslessness_within_tolerance"]
    assert checks["relative_max_skew_hermitian_residual"] == 0
    assert checks["max_abs_real_eta_Y"] == 0
    assert checks["max_abs_real_Z_over_eta"] == 0
    assert checks["max_abs_imaginary_coupling"] == 0
    assert checks["numerically_singular_sample_count"] == 0
    assert condition.shape == (4, 8)
    np.testing.assert_allclose(
        checks["balanced_condition_number_min_median_p95_max"],
        np.percentile(condition, [0, 50, 95, 100]),
    )


@pytest.mark.parametrize("violation", ["reciprocity", "losslessness"])
def test_checks_detect_constraint_violations_without_projecting_data(saved_parameters, violation):
    metadata, parameters, theta, phi = plotting.load_saved_parameters(saved_parameters)
    if violation == "reciprocity":
        parameters.Y_me[0, 1] += 0.25
    else:
        parameters.Y_ee[0, 0] += 0.01 / metadata["reference_impedance_ohm"]
    before = parameters.as_matrix().copy()
    checks, _ = plotting.material_checks(
        parameters, metadata["reference_impedance_ohm"], theta, phi
    )
    assert not checks["losslessness_within_tolerance"]
    assert checks["reciprocity_within_tolerance"] == (violation == "losslessness")
    np.testing.assert_array_equal(before, parameters.as_matrix())


def test_singular_maps_are_distinct_from_material_constraint_failure(saved_parameters):
    metadata, parameters, theta, phi = plotting.load_saved_parameters(saved_parameters)
    parameters.Y_ee[:] = 0
    parameters.Y_em[:] = 0
    parameters.Y_me[:] = 0
    parameters.Y_mm[:] = 0
    checks, condition = plotting.material_checks(
        parameters, metadata["reference_impedance_ohm"], theta, phi
    )
    assert checks["reciprocity_within_tolerance"] and checks["losslessness_within_tolerance"]
    assert checks["numerically_singular_sample_count"] == 32
    assert checks["balanced_condition_number_min_median_p95_max"] == [None] * 4
    assert np.isinf(condition).all()
    json.dumps(checks, allow_nan=False)


def test_mixed_finite_and_singular_condition_percentiles_are_json_safe():
    result = plotting._condition_percentiles(np.array([1.0, 2.0, 3.0, np.inf]))
    assert result == [1.0, 2.5, None, None]
    json.dumps(result, allow_nan=False)


def test_periodic_display_closes_phi_seam_without_changing_signed_values():
    theta, phi = np.meshgrid([0.2, 1, 2], np.linspace(0, 2 * np.pi, 4, endpoint=False),
                            indexing="ij")
    values = np.arange(12).reshape(3, 4) - 6
    displayed_theta, displayed_phi, displayed_values = plotting._periodic_map(theta, phi, values)
    np.testing.assert_array_equal(displayed_values[:, :-1], values)
    np.testing.assert_array_equal(displayed_values[:, -1], values[:, 0])
    np.testing.assert_array_equal(displayed_phi[:, -1], 360)
    np.testing.assert_allclose(displayed_theta[:, -1], np.rad2deg(theta[:, 0]))


def test_plots_and_checks_preserve_saved_inputs_and_can_be_repeated(saved_parameters):
    inputs = [saved_parameters / name for name in ("experiment.json", "surface_parameters.npz")]
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
    for _ in range(2):
        paths = plotting.plot_surface_parameters(saved_parameters)
        assert {path.name for path in paths} == {
            "Y_imaginary.png", "Z_imaginary.png", "chi_real.png",
            "map_condition_number.png", "checks.json",
        }
        for path in paths:
            if path.suffix == ".png":
                with Image.open(path) as picture:
                    pixels = np.asarray(picture.convert("RGB"))
                assert pixels.shape[0] >= 500
                assert pixels.std() > 5
        assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs] == hashes
    checks = json.loads((saved_parameters / "parameter_checks/checks.json").read_text())
    assert checks["angular_degree"] == 2
    assert checks["reciprocity_within_tolerance"]
    assert not plt.get_fignums()


def test_all_singular_maps_can_still_be_plotted(saved_parameters):
    with np.load(saved_parameters / "surface_parameters.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    for name in ("Y", "Chi", "Gamma", "Z"):
        data[name] = np.zeros_like(data[name])
    np.savez(saved_parameters / "surface_parameters.npz", **data)
    paths = plotting.plot_surface_parameters(saved_parameters)
    checks = json.loads(paths[-1].read_text())
    assert checks["numerically_singular_sample_count"] == 32
    assert checks["balanced_condition_number_min_median_p95_max"] == [None] * 4
    assert paths[-2].is_file()


@pytest.mark.parametrize("problem", [
    "missing_block", "shape", "nonfinite_block", "nonfinite_angles", "grid_order",
    "rows", "impedance", "convention", "degree",
])
def test_invalid_saved_data_fail_before_writing_output(saved_parameters, problem):
    path = saved_parameters / "experiment.json"
    metadata = json.loads(path.read_text())
    with np.load(saved_parameters / "surface_parameters.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    if problem == "missing_block":
        data.pop("Gamma")
    elif problem == "shape":
        data["Y"] = data["Y"][:, :, :-1]
    elif problem == "nonfinite_block":
        data["Y"][0, 0, 0] = np.nan
    elif problem == "nonfinite_angles":
        data["theta"][0] = np.inf
    elif problem == "grid_order":
        data["phi"][[0, 1]] = data["phi"][[1, 0]]
    elif problem == "rows":
        metadata["n_theta"] = True
    elif problem == "impedance":
        metadata["reference_impedance_ohm"] = 0
    elif problem == "convention":
        metadata["field_and_material_time_convention"] = "exp(-j omega t)"
    elif problem == "degree":
        metadata["angular_degree"] = -1
    path.write_text(json.dumps(metadata))
    np.savez(saved_parameters / "surface_parameters.npz", **data)
    with pytest.raises(ValueError):
        plotting.plot_surface_parameters(saved_parameters)
    assert not (saved_parameters / "parameter_checks").exists()


def test_cli_needs_only_saved_metadata_and_tensors(saved_parameters, tmp_path):
    result = subprocess.run(
        [sys.executable, plotting.__file__, "--experiment-dir", str(saved_parameters)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 5
    assert (saved_parameters / "parameter_checks/checks.json").is_file()
