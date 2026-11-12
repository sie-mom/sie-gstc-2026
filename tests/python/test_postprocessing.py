"""Tests for sie_mom.postprocessing shared utilities."""

import numpy as np
import pytest
from sie_mom.postprocessing import (
    budhu_moment_scale,
    cartesian_to_spherical,
    complex_vector,
    dominant_tangential_components,
    load_field_csv,
    spherical_components,
    spherical_coordinates_from_cartesian,
    spherical_to_cartesian,
    to_decibels,
    write_field_csv,
    zx_grid,
)


def test_spherical_cartesian_roundtrip():
    rng = np.random.default_rng(42)
    theta = rng.uniform(0.1, np.pi - 0.1, size=(50,))
    phi = rng.uniform(-np.pi, np.pi, size=(50,))

    # Random complex Cartesian vector (3, 50)
    fx = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    fy = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    fz = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    cart = np.stack([fx, fy, fz], axis=0)

    sph = cartesian_to_spherical(cart, theta, phi)
    recovered = spherical_to_cartesian(sph, theta, phi)

    np.testing.assert_allclose(cart, recovered, rtol=1e-14, atol=1e-14)


def test_complex_vector_both_formats():
    data_standard = {
        "Ex_real": np.array([1.0, 2.0]),
        "Ex_imag": np.array([0.5, -0.5]),
        "Ey_real": np.array([3.0, 4.0]),
        "Ey_imag": np.array([1.5, -1.5]),
        "Ez_real": np.array([5.0, 6.0]),
        "Ez_imag": np.array([2.5, -2.5]),
    }
    vec_std = complex_vector(data_standard, "E")
    expected = np.array([
        [1.0 + 0.5j, 3.0 + 1.5j, 5.0 + 2.5j],
        [2.0 - 0.5j, 4.0 - 1.5j, 6.0 - 2.5j],
    ])
    np.testing.assert_allclose(vec_std, expected)

    data_alias = {
        "E_x_re": np.array([1.0, 2.0]),
        "E_x_im": np.array([0.5, -0.5]),
        "E_y_re": np.array([3.0, 4.0]),
        "E_y_im": np.array([1.5, -1.5]),
        "E_z_re": np.array([5.0, 6.0]),
        "E_z_im": np.array([2.5, -2.5]),
    }
    vec_alias = complex_vector(data_alias, "E")
    np.testing.assert_allclose(vec_alias, expected)


def test_load_and_write_field_csv(tmp_path):
    csv_file = tmp_path / "fields.csv"
    n = 10
    x = np.linspace(-1, 1, n)
    y = np.zeros(n)
    z = np.linspace(-1, 1, n)
    region = np.where(np.sqrt(x**2 + z**2) < 0.5, 1, 2)
    e = np.ones((n, 3), dtype=complex) * (1 + 2j)
    h = np.ones((n, 3), dtype=complex) * (3 + 4j)

    write_field_csv(csv_file, x, y, z, region, e, h)
    loaded = load_field_csv(csv_file, radius_m=0.5)

    assert len(loaded["x"]) == n
    np.testing.assert_allclose(loaded["x"], x)
    np.testing.assert_allclose(loaded["region"], region)
    np.testing.assert_allclose(loaded["Ex_real"], 1.0)
    np.testing.assert_allclose(loaded["Ex_imag"], 2.0)
    np.testing.assert_allclose(loaded["Hx_real"], 3.0)
    np.testing.assert_allclose(loaded["Hx_imag"], 4.0)


def test_to_decibels():
    values = np.array([1.0, 0.1, 0.01, 1e-6])
    db = to_decibels(values, floor_relative_db=60.0)
    np.testing.assert_allclose(db[0], 0.0)
    np.testing.assert_allclose(db[1], -20.0)
    np.testing.assert_allclose(db[2], -40.0)
    np.testing.assert_allclose(db[3], -60.0)


def test_zx_grid():
    n = 4
    x = np.linspace(-1, 1, n)
    z = np.linspace(0, 2, n)
    xx, zz = np.meshgrid(x, z)
    data = {
        "x": xx.ravel(),
        "z": zz.ravel(),
    }
    gx, gz, order = zx_grid(data, grid_size=n)
    np.testing.assert_allclose(gx, x)
    np.testing.assert_allclose(gz, z)
    assert len(order) == n * n

