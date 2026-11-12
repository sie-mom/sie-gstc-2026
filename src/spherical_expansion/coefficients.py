"""Persistent storage for solved spherical-vector-wave coefficients."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np

Mode: TypeAlias = tuple[int, int]
CoefficientMap: TypeAlias = dict[Mode, complex]
CoefficientSet: TypeAlias = tuple[
    CoefficientMap,
    CoefficientMap,
    CoefficientMap,
    CoefficientMap,
]

SCHEMA_VERSION = 1


def _ordered_modes(*coefficient_maps: CoefficientMap) -> list[Mode]:
    if not coefficient_maps:
        raise ValueError("at least one coefficient map is required")
    expected = set(coefficient_maps[0])
    if any(set(values) != expected for values in coefficient_maps[1:]):
        raise ValueError("all coefficient maps must contain identical modes")
    if not expected:
        raise ValueError("coefficient maps cannot be empty")
    modes = sorted(expected, key=lambda mode: (mode[1], mode[0]))
    for m, n in modes:
        if n < 1 or abs(m) > n:
            raise ValueError(f"invalid spherical mode (m={m}, n={n})")
    return modes


def save_coefficients(
    path: Path,
    exterior_m: CoefficientMap,
    exterior_n: CoefficientMap,
    interior_m: CoefficientMap,
    interior_n: CoefficientMap,
) -> None:
    """Save four coefficient families in a portable, self-describing archive."""

    modes = _ordered_modes(exterior_m, exterior_n, interior_m, interior_n)
    mode_m = np.asarray([mode[0] for mode in modes], dtype=np.int32)
    mode_n = np.asarray([mode[1] for mode in modes], dtype=np.int32)

    def values(coefficients: CoefficientMap) -> np.ndarray:
        result = np.asarray([coefficients[mode] for mode in modes], dtype=np.complex128)
        if not np.all(np.isfinite(result)):
            raise ValueError("SVWF coefficients must be finite")
        return result

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        mode_m=mode_m,
        mode_n=mode_n,
        exterior_m=values(exterior_m),
        exterior_n=values(exterior_n),
        interior_m=values(interior_m),
        interior_n=values(interior_n),
    )


def load_coefficients(path: Path) -> CoefficientSet:
    """Load and validate coefficient maps written by :func:`save_coefficients`."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema_version",
            "mode_m",
            "mode_n",
            "exterior_m",
            "exterior_n",
            "interior_m",
            "interior_n",
        }
        if set(archive.files) != required:
            raise ValueError(f"invalid coefficient archive fields: {archive.files}")
        schema = archive["schema_version"]
        if schema.shape != ():
            raise ValueError("invalid coefficient schema version field")
        schema_version = int(schema.item())
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported coefficient schema version {schema_version}")
        mode_m = np.asarray(archive["mode_m"], dtype=np.int32)
        mode_n = np.asarray(archive["mode_n"], dtype=np.int32)
        if mode_m.ndim != 1 or mode_n.shape != mode_m.shape or mode_m.size == 0:
            raise ValueError("invalid coefficient mode arrays")
        modes = [(int(m), int(n)) for m, n in zip(mode_m, mode_n)]
        if len(set(modes)) != len(modes):
            raise ValueError("coefficient archive contains duplicate modes")
        _ordered_modes(dict.fromkeys(modes, 0j))

        output: list[CoefficientMap] = []
        for name in ("exterior_m", "exterior_n", "interior_m", "interior_n"):
            values = np.asarray(archive[name], dtype=np.complex128)
            if values.shape != mode_m.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid coefficient array {name}")
            output.append(dict(zip(modes, values)))
    return output[0], output[1], output[2], output[3]
