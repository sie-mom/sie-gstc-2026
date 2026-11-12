from pathlib import Path

import numpy as np
import pytest

from spherical_expansion.coefficients import load_coefficients, save_coefficients


def coefficient_families():
    modes = [(m, n) for n in range(1, 4) for m in range(-n, n + 1)]
    return tuple(
        {
            (m, n): complex(family + index / 10, n - m / 7)
            for index, (m, n) in enumerate(modes)
        }
        for family in range(4)
    )


def test_coefficient_archive_round_trip(tmp_path: Path):
    expected = coefficient_families()
    path = tmp_path / "coefficients.npz"
    save_coefficients(path, *expected)
    actual = load_coefficients(path)
    assert actual == expected

    with np.load(path, allow_pickle=False) as archive:
        assert int(archive["schema_version"].item()) == 1
        assert archive["mode_n"].tolist() == sorted(
            archive["mode_n"].tolist()
        )


def test_coefficient_archive_rejects_different_mode_sets(tmp_path: Path):
    families = list(coefficient_families())
    families[-1] = dict(families[-1])
    families[-1].pop(next(iter(families[-1])))
    with pytest.raises(ValueError, match="identical modes"):
        save_coefficients(tmp_path / "coefficients.npz", *families)


def test_coefficient_archive_rejects_nonfinite_values(tmp_path: Path):
    families = list(coefficient_families())
    families[0] = dict(families[0])
    families[0][next(iter(families[0]))] = np.nan + 0j
    with pytest.raises(ValueError, match="finite"):
        save_coefficients(tmp_path / "coefficients.npz", *families)


def test_coefficient_loader_rejects_unknown_schema(tmp_path: Path):
    path = tmp_path / "coefficients.npz"
    np.savez(
        path,
        schema_version=np.asarray(99),
        mode_m=np.asarray([0]),
        mode_n=np.asarray([1]),
        exterior_m=np.asarray([1j]),
        exterior_n=np.asarray([1j]),
        interior_m=np.asarray([1j]),
        interior_n=np.asarray([1j]),
    )
    with pytest.raises(ValueError, match="schema version"):
        load_coefficients(path)


def test_coefficient_loader_rejects_duplicate_modes(tmp_path: Path):
    path = tmp_path / "coefficients.npz"
    arrays = {
        "schema_version": np.asarray(1),
        "mode_m": np.asarray([0, 0]),
        "mode_n": np.asarray([1, 1]),
        "exterior_m": np.asarray([1j, 2j]),
        "exterior_n": np.asarray([1j, 2j]),
        "interior_m": np.asarray([1j, 2j]),
        "interior_n": np.asarray([1j, 2j]),
    }
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="duplicate modes"):
        load_coefficients(path)
