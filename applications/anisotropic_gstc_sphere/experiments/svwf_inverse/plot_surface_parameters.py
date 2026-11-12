"""Check and plot saved reciprocal-lossless surface parameters on a theta/phi grid.

Read experiment.json and surface_parameters.npz only. Save signed imaginary Y/Z,
real chi, impedance-balanced map conditioning, and material-constraint diagnostics
under parameter_checks/. No synthesis, field evaluation, or solver is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from sie_mom.gstc.surface_parameters import SurfaceParameters

CONSTRAINT_TOLERANCE = 1e-10  # Absolute tolerance after impedance balancing.
DPI = 160


def load_saved_parameters(experiment_dir):
    """Validate physical tensor blocks and their saved rectangular angular grid."""
    with (experiment_dir / "experiment.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("method") != "joint_svwf_reciprocal_lossless":
        raise ValueError("expected a joint SVWF reciprocal-lossless experiment")
    if metadata.get("field_and_material_time_convention") != "exp(+j omega t)":
        raise ValueError("surface parameters must use exp(+j omega t)")
    rows, columns = metadata.get("n_theta"), metadata.get("n_phi")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 2
           for value in (rows, columns)):
        raise ValueError("saved n_theta and n_phi must be integers >= 2")
    degree = metadata.get("angular_degree")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("saved angular_degree must be a nonnegative integer")
    eta = metadata.get("reference_impedance_ohm")
    if (isinstance(eta, bool) or not isinstance(eta, (int, float))
            or not np.isfinite(eta) or eta <= 0):
        raise ValueError("saved reference impedance must be finite and positive")
    count = rows * columns
    with np.load(experiment_dir / "surface_parameters.npz", allow_pickle=False) as archive:
        required = {"Y", "Chi", "Gamma", "Z", "theta", "phi"}
        if not required.issubset(archive.files):
            raise ValueError("surface archive is missing tensor blocks or angles")
        blocks = []
        for name in ("Y", "Chi", "Gamma", "Z"):
            value = np.asarray(archive[name], dtype=complex)
            if value.shape != (2, 2, count) or not np.isfinite(value).all():
                raise ValueError(f"surface block {name} must be finite with shape (2, 2, {count})")
            blocks.append(value)
        angles = []
        for name in ("theta", "phi"):
            value = np.asarray(archive[name])
            if (np.iscomplexobj(value) or value.shape != (count,)
                    or not np.isfinite(value).all()):
                raise ValueError(f"saved {name} must be finite and real with shape ({count},)")
            angles.append(value.reshape(rows, columns))
    theta, phi = angles
    if (not np.allclose(theta, theta[:, :1], rtol=0, atol=1e-12)
            or not np.allclose(phi, phi[:1, :], rtol=0, atol=1e-12)
            or not np.all(np.diff(theta[:, 0]) > 0)
            or not np.all((theta >= 0) & (theta <= np.pi))
            or not np.allclose(phi[0], 2 * np.pi * np.arange(columns) / columns,
                               rtol=0, atol=1e-12)):
        raise ValueError("saved angles must form an ordered theta/uniform-phi grid")
    return metadata, SurfaceParameters(*blocks), theta, phi


def _condition_percentiles(condition):
    """Use linear percentiles, writing JSON null where the result is infinite."""
    ordered = np.sort(condition.ravel())
    output = []
    for percentile in (0, 50, 95, 100):
        index = (len(ordered) - 1) * percentile / 100
        lower, upper = ordered[int(np.floor(index))], ordered[int(np.ceil(index))]
        value = lower if lower == upper else lower + (upper - lower) * (index % 1)
        output.append(float(value) if np.isfinite(value) else None)
    return output


def material_checks(parameters, reference_impedance, theta, phi):
    """Measure reciprocity, losslessness, and conditioning without projecting the data."""
    y, chi, gamma, z = (
        parameters.Y_ee, parameters.Y_em, parameters.Y_me, parameters.Y_mm
    )
    eta = reference_impedance
    balanced = parameters.as_matrix().transpose(2, 0, 1).copy()
    balanced[:, :2, :2] *= eta
    balanced[:, 2:, 2:] /= eta
    if not np.isfinite(balanced).all():
        raise ValueError("impedance-balanced surface map must be finite")
    singular_values = np.linalg.svd(balanced, compute_uv=False)
    # Report numerically singular maps separately from large, finite condition numbers.
    singular = singular_values[:, -1] <= 4 * np.finfo(float).eps * singular_values[:, 0]
    condition = np.divide(
        singular_values[:, 0], singular_values[:, -1],
        out=np.full(len(balanced), np.inf), where=~singular,
    )
    hermitian_part = balanced + balanced.conj().swapaxes(1, 2)
    hermitian_defect = float(np.max(np.abs(hermitian_part)))
    scale = max(float(np.max(np.abs(balanced))), np.finfo(float).tiny)
    reciprocity_residuals = (
        eta * (y - y.swapaxes(0, 1)),
        (z - z.swapaxes(0, 1)) / eta,
        gamma + chi.swapaxes(0, 1),
    )
    defects = tuple(float(np.max(np.abs(value))) for value in reciprocity_residuals)
    worst = int(np.argmax(condition))
    checks = {
        "sample_count": int(theta.size),
        "all_tensors_finite": True,
        "constraint_tolerance": CONSTRAINT_TOLERANCE,
        "max_abs_eta_Y_reciprocity_residual": defects[0],
        "max_abs_Z_over_eta_reciprocity_residual": defects[1],
        "max_abs_coupling_reciprocity_residual": defects[2],
        "max_abs_real_eta_Y": float(np.max(np.abs(eta * y.real))),
        "max_abs_real_Z_over_eta": float(np.max(np.abs(z.real / eta))),
        "max_abs_imaginary_coupling": float(max(np.max(np.abs(chi.imag)),
                                                np.max(np.abs(gamma.imag)))),
        "max_abs_skew_hermitian_residual": hermitian_defect,
        "relative_max_skew_hermitian_residual": hermitian_defect / scale,
        "reciprocity_within_tolerance": bool(max(defects) <= CONSTRAINT_TOLERANCE),
        "losslessness_within_tolerance": bool(hermitian_defect <= CONSTRAINT_TOLERANCE),
        "numerically_singular_sample_count": int(np.count_nonzero(singular)),
        "balanced_condition_number_min_median_p95_max": _condition_percentiles(condition),
        "maximum_condition_location_degrees": {
            "theta": float(np.rad2deg(theta.ravel()[worst])),
            "phi": float(np.rad2deg(phi.ravel()[worst])),
        },
        "interpretation": (
            "Material constraints only; no claim of exact GSTCs, field matching, or "
            "refinement convergence. Conditioning refers to the impedance-balanced "
            "local map, not the SVWF solver. Infinite condition percentiles use null."
        ),
    }
    return checks, condition.reshape(theta.shape)


def _periodic_map(theta, phi, values):
    """Close the phi seam for display without adding any independent samples."""
    return (
        np.rad2deg(np.column_stack((theta, theta[:, 0]))),
        np.column_stack((np.rad2deg(phi), np.full(len(phi), 360.0))),
        np.ma.column_stack((values, values[:, 0])),
    )


def _save_figure(figure, path):
    try:
        figure.savefig(path, dpi=DPI)
    finally:
        plt.close(figure)


def _plot_entries(theta, phi, tensor, entries, symbol, part, unit, degree, path):
    layout = (2, 2) if len(entries) == 4 else (1, 3)
    figure, axes = plt.subplots(
        *layout, figsize=(10, 8) if len(entries) == 4 else (14, 4.8),
        squeeze=False, constrained_layout=True,
    )
    labels = (r"\theta", r"\phi")
    limit = float(np.max(np.abs(tensor))) or 1.0
    for axis, (row, column) in zip(axes.ravel(), entries):
        theta_degrees, phi_degrees, values = _periodic_map(
            theta, phi, tensor[row, column].reshape(theta.shape)
        )
        artist = axis.pcolormesh(
            phi_degrees, theta_degrees, values, shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit,
        )
        axis.set_title(rf"$\operatorname{{{part}}}{symbol}_{{{labels[row]}{labels[column]}}}$")
        axis.set_xlabel(r"$\phi$ [deg]")
        axis.set_ylabel(r"$\theta$ [deg]")
        axis.set_xlim(0, 360)
        axis.set_ylim(180, 0)
    figure.colorbar(
        artist, ax=list(axes.ravel()), shrink=0.85,
        label=rf"$\operatorname{{{part}}}{symbol}$ [{unit}]",
    )
    figure.suptitle(f"Surface parameters | angular degree {degree}")
    _save_figure(figure, path)


def _plot_conditioning(theta, phi, condition, path):
    finite = condition[np.isfinite(condition)]
    lower = max(1.0, float(finite.min())) if finite.size else 1.0
    upper = float(finite.max()) if finite.size else 10.0
    if upper <= lower:
        upper = 10 * lower
    theta_degrees, phi_degrees, values = _periodic_map(
        theta, phi, np.ma.masked_invalid(condition)
    )
    cmap = plt.get_cmap("viridis").with_extremes(bad="#f5c2c2")
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    artist = axis.pcolormesh(
        phi_degrees, theta_degrees, values, shading="auto", cmap=cmap,
        norm=LogNorm(vmin=lower, vmax=upper),
    )
    axis.set_xlabel(r"$\phi$ [deg]")
    axis.set_ylabel(r"$\theta$ [deg]")
    axis.set_xlim(0, 360)
    axis.set_ylim(180, 0)
    title = "Impedance-balanced surface map condition number"
    count = int(np.count_nonzero(~np.isfinite(condition)))
    if count:
        title += f"\nNumerically singular samples: {count} (pink)"
    axis.set_title(title)
    figure.colorbar(artist, ax=axis, label="Condition number (log scale)")
    _save_figure(figure, path)


def plot_surface_parameters(experiment_dir):
    """Write four inspection figures and checks.json; leave all saved inputs unchanged."""
    experiment_dir = Path(experiment_dir).resolve()
    metadata, parameters, theta, phi = load_saved_parameters(experiment_dir)
    checks, condition = material_checks(
        parameters, metadata["reference_impedance_ohm"], theta, phi
    )
    checks["angular_degree"] = metadata["angular_degree"]
    output = experiment_dir / "parameter_checks"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, tensor, entries, symbol, part, unit in (
        ("Y_imaginary", parameters.Y_ee.imag, ((0, 0), (0, 1), (1, 1)), "Y", "Im", "S"),
        ("Z_imaginary", parameters.Y_mm.imag, ((0, 0), (0, 1), (1, 1)), "Z", "Im", r"$\Omega$"),
        ("chi_real", parameters.Y_em.real, ((0, 0), (0, 1), (1, 0), (1, 1)),
         r"\chi", "Re", "dimensionless"),
    ):
        path = output / f"{name}.png"
        _plot_entries(theta, phi, tensor, entries, symbol, part, unit,
                      metadata["angular_degree"], path)
        paths.append(path)
    path = output / "map_condition_number.png"
    _plot_conditioning(theta, phi, condition, path)
    paths.append(path)
    path = output / "checks.json"
    path.write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = plot_surface_parameters(args.experiment_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
