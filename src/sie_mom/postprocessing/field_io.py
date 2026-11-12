"""Standardized CSV readers and writers for electromagnetic field tables."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

STANDARD_FIELD_COLUMNS = (
    "point_index",
    "x",
    "y",
    "z",
    "region",
    "Ex_real",
    "Ex_imag",
    "Ey_real",
    "Ey_imag",
    "Ez_real",
    "Ez_imag",
    "Hx_real",
    "Hx_imag",
    "Hy_real",
    "Hy_imag",
    "Hz_real",
    "Hz_imag",
)

COLUMN_ALIASES = {
    "E_x_re": "Ex_real",
    "E_x_im": "Ex_imag",
    "E_y_re": "Ey_real",
    "E_y_im": "Ey_imag",
    "E_z_re": "Ez_real",
    "E_z_im": "Ez_imag",
    "H_x_re": "Hx_real",
    "H_x_im": "Hx_imag",
    "H_y_re": "Hy_real",
    "H_y_im": "Hy_imag",
    "H_z_re": "Hz_real",
    "H_z_im": "Hz_imag",
}


def load_field_csv(
    path: Path | str,
    *,
    radius_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Load Cartesian complex electromagnetic field CSV data into structured arrays.

    Supports both 'Ex_real'/'Ex_imag' and 'E_x_re'/'E_x_im' column header styles.
    Validates finite values, unique contiguous indices, and verifies region assignments
    against sphere radius if provided.
    """
    csv_path = Path(path)
    required = {"point_index", "x", "y", "z", "region"}
    required.update(
        f"{f}{axis}_{part}"
        for f in "EH"
        for axis in "xyz"
        for part in ("real", "imag")
    )

    with csv_path.open(encoding="utf-8") as stream:
        raw_header = next(csv.reader(stream), [])
        if len(set(raw_header)) != len(raw_header):
            raise ValueError(f"missing or duplicate field CSV columns: {csv_path}")
        header = [COLUMN_ALIASES.get(col, col) for col in raw_header]
        if not required.issubset(header) and not ("point_index" not in header and (required - {"point_index"}).issubset(header)):
            raise ValueError(f"missing or duplicate field CSV columns: {csv_path}")
        stream.seek(0)
        values = np.genfromtxt(stream, delimiter=",", names=True, ndmin=1)

    if values.size == 0:
        raise ValueError(f"empty field CSV: {csv_path}")

    data: dict[str, np.ndarray] = {}
    for raw_name in values.dtype.names:
        standard_name = COLUMN_ALIASES.get(raw_name, raw_name)
        data[standard_name] = np.asarray(values[raw_name])

    if "point_index" not in data:
        data["point_index"] = np.arange(values.size, dtype=np.int64)

    if not all(np.isfinite(array).all() for array in data.values()):
        raise ValueError(f"non-finite field CSV data: {csv_path}")

    if not np.array_equal(np.sort(data["point_index"]), np.arange(values.size)):
        raise ValueError(f"field CSV point indices must be unique and contiguous: {csv_path}")

    if radius_m is not None:
        if not np.isin(data["region"], (1, 2, 3)).all():
            raise ValueError(f"field CSV regions must be 1, 2, or 3: {csv_path}")
        radius = float(radius_m)
        radii = np.sqrt(data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2)
        inside = (radii < radius * (1 - 1e-6)) & (data["region"] != 3)
        outside = (radii > radius * (1 + 1e-6)) & (data["region"] != 3)
        if np.any(data["region"][inside] != 1) or np.any(data["region"][outside] != 2):
            raise ValueError(
                f"field CSV regions disagree with the saved sphere radius: {csv_path}"
            )

    return data


def write_field_csv(
    path: Path | str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    region: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    *,
    point_index: np.ndarray | None = None,
) -> None:
    """Write standardized Cartesian field table to CSV."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(x)
    indices = np.arange(n_points) if point_index is None else point_index

    e = np.asarray(electric, dtype=complex)
    h = np.asarray(magnetic, dtype=complex)
    if e.shape[0] == 3 and e.shape[-1] != 3:
        e = e.T
    if h.shape[0] == 3 and h.shape[-1] != 3:
        h = h.T

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(STANDARD_FIELD_COLUMNS)
        for i in range(n_points):
            writer.writerow([
                int(indices[i]),
                f"{x[i]:.17e}",
                f"{y[i]:.17e}",
                f"{z[i]:.17e}",
                int(region[i]),
                f"{e[i, 0].real:.17e}",
                f"{e[i, 0].imag:.17e}",
                f"{e[i, 1].real:.17e}",
                f"{e[i, 1].imag:.17e}",
                f"{e[i, 2].real:.17e}",
                f"{e[i, 2].imag:.17e}",
                f"{h[i, 0].real:.17e}",
                f"{h[i, 0].imag:.17e}",
                f"{h[i, 1].real:.17e}",
                f"{h[i, 1].imag:.17e}",
                f"{h[i, 2].real:.17e}",
                f"{h[i, 2].imag:.17e}",
            ])
