from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c

from applications.anisotropic_gstc_sphere.comsol.bianisotropic_sheet_validation import (
    _decompose_linear_moment,
    load_config_defaults,
    resolve_excitation_cases,
)
from applications.anisotropic_gstc_sphere.comsol.comsol_helpers import (
    compute_dipole_normalization,
)
from applications.anisotropic_gstc_sphere.comsol.interpolation import ETA_0


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_CONFIG = (
    REPO_ROOT
    / "applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot"
    / "e2m_eps7.json"
)
DUAL_CONFIG = (
    REPO_ROOT
    / "applications/anisotropic_gstc_sphere/configurations/dipole_pol_rot"
    / "dual_pol_eps7.json"
)


def _dummy_args(
    config: Path | None,
    dipole_type: str | None = None,
    dipole_direction: list[float] | None = None,
    case_id: str | None = None,
    cases: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        dipole_type=dipole_type,
        dipole_direction=dipole_direction,
        case_id=case_id,
        cases=cases,
    )


def test_decompose_linear_moment_real_and_complex_phase() -> None:
    unit_dir, amp = _decompose_linear_moment({"real": [0.0, 0.0, 2.5], "imag": [0.0, 0.0, 0.0]})
    assert unit_dir == pytest.approx((0.0, 0.0, 1.0))
    assert amp == pytest.approx(2.5 + 0.0j)

    # Global complex phase factor along z-axis remains linearly polarized
    unit_dir_c, amp_c = _decompose_linear_moment({"real": [0.0, 0.0, 0.0], "imag": [0.0, 0.0, 3.0]})
    assert unit_dir_c == pytest.approx((0.0, 0.0, 1.0))
    assert amp_c == pytest.approx(0.0 + 3.0j)


def test_decompose_linear_moment_rejects_elliptical_and_zero() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _decompose_linear_moment([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Elliptically polarized"):
        _decompose_linear_moment({"real": [1.0, 0.0, 0.0], "imag": [0.0, 1.0, 0.0]})


def test_resolve_excitation_cases_single_and_dual_configs() -> None:
    # Single-case config -> 1 case (100% backward compatible)
    args_single = _dummy_args(SINGLE_CONFIG)
    defaults_single = load_config_defaults(args_single)
    cases_single = resolve_excitation_cases(args_single, defaults_single)
    assert len(cases_single) == 1
    assert cases_single[0].case_id == "electric_incidence"
    assert cases_single[0].dipole_type == "electric"
    assert cases_single[0].dipole_direction == pytest.approx((0.0, 0.0, 1.0))
    assert cases_single[0].effective_moment_amplitude == pytest.approx(1.0 + 0.0j)

    # Dual-case config -> 2 cases (electric_incidence + magnetic_incidence)
    args_dual = _dummy_args(DUAL_CONFIG)
    defaults_dual = load_config_defaults(args_dual)
    cases_dual = resolve_excitation_cases(args_dual, defaults_dual)
    assert [c_item.case_id for c_item in cases_dual] == [
        "electric_incidence",
        "magnetic_incidence",
    ]
    assert [c_item.dipole_type for c_item in cases_dual] == ["electric", "magnetic"]
    for c_item in cases_dual:
        assert c_item.dipole_direction == pytest.approx((0.0, 0.0, 1.0))
        assert c_item.effective_moment_amplitude == pytest.approx(1.0 + 0.0j)

    # Explicit CLI override forces single-case execution even when config has 2 cases
    args_override = _dummy_args(DUAL_CONFIG, dipole_type="magnetic", dipole_direction=[1.0, 0.0, 0.0])
    cases_override = resolve_excitation_cases(args_override, defaults_dual)
    assert len(cases_override) == 1
    assert cases_override[0].dipole_type == "magnetic"
    assert cases_override[0].dipole_direction == pytest.approx((1.0, 0.0, 0.0))

    # Schema v2 canonical case filtering via --case-id
    args_case_e = _dummy_args(DUAL_CONFIG, case_id="electric_incidence")
    cases_e = resolve_excitation_cases(args_case_e, defaults_dual)
    assert len(cases_e) == 1
    assert cases_e[0].case_id == "electric_incidence"
    assert cases_e[0].dipole_type == "electric"

    args_case_m = _dummy_args(DUAL_CONFIG, case_id="magnetic_incidence")
    cases_m = resolve_excitation_cases(args_case_m, defaults_dual)
    assert len(cases_m) == 1
    assert cases_m[0].case_id == "magnetic_incidence"
    assert cases_m[0].dipole_type == "magnetic"

    # Invalid case_id raises clear error
    args_invalid = _dummy_args(DUAL_CONFIG, case_id="nonexistent_case")
    with pytest.raises(ValueError, match="Selected case 'nonexistent_case' not found"):
        resolve_excitation_cases(args_invalid, defaults_dual)


def test_compute_dipole_normalization_electric_and_magnetic() -> None:
    f0 = 10.0e9
    er_int = 7.0
    mur_int = 1.0
    omega = 2.0 * math.pi * f0
    k_int = omega * math.sqrt(er_int * mur_int) / c
    eta_int = ETA_0 * math.sqrt(mur_int / er_int)

    s_e = compute_dipole_normalization(f0, er_interior=er_int, mur_interior=mur_int, dipole_type="electric")
    assert s_e == pytest.approx(-1j * eta_int / (4.0 * math.pi * k_int))

    s_m = compute_dipole_normalization(f0, er_interior=er_int, mur_interior=mur_int, dipole_type="magnetic")
    assert s_m == pytest.approx(eta_int / (4.0 * math.pi) + 0.0j)
    assert s_m.real > 0.0

    # Scaling by effective_moment_amplitude divides unit_scale by amplitude
    s_m_scaled = compute_dipole_normalization(
        f0,
        er_interior=er_int,
        mur_interior=mur_int,
        dipole_type="magnetic",
        effective_moment_amplitude=2.0 + 0.0j,
    )
    assert s_m_scaled == pytest.approx(0.5 * s_m)
