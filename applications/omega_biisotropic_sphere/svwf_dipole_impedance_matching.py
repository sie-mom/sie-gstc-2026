#!/usr/bin/env python3
"""Validate a spherical omega matching layer with the SVWF solver.

The physical case follows the centered, z-directed electric-dipole example:
the interior has ``eps_r = 7`` and ``mu_r = 1``, the exterior is free space,
``f = 10 GHz``, and the sheet radius is ``0.7 lambda_0``.

The analytical dipole evaluator uses the ``exp(+j omega t)`` convention,
whereas the current SVWF implementation uses ``exp(-j omega t)``.  This
script conjugates the prescribed dipole phasors before passing them to the
SVWF solver and extracts the GSTC parameters from those same conjugated
fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

try:
    from .dipole_impedance_matching import (
        apply_cap_perturbation,
        extract_case_parameters,
        prescribed_dipole_fields,
        prescribed_total_fields,
        spherical_to_cartesian,
    )
    from .workflow import (
        DipoleMatchingCase,
        ExtractionMethod,
        SamplingConfiguration,
        SVWFConfiguration,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
        write_field_csv,
    )
except ImportError:  # Support direct execution from the repository root.
    from dipole_impedance_matching import (  # type: ignore[no-redef]
        apply_cap_perturbation,
        extract_case_parameters,
        prescribed_dipole_fields,
        prescribed_total_fields,
        spherical_to_cartesian,
    )
    from workflow import (  # type: ignore[no-redef]
        DipoleMatchingCase,
        ExtractionMethod,
        SamplingConfiguration,
        SVWFConfiguration,
        WorkflowConfiguration,
        default_run_directory,
        load_workflow_configuration,
        write_field_csv,
    )

from sie_mom.gstc.omega_biisotropic import (
    normal_power_density,
    omega_tensor_blocks,
)
from spherical_expansion.coefficients import load_coefficients, save_coefficients
from spherical_expansion.solver import solve_collocation_general
from spherical_expansion.svwf_sphere import compute_fields, compute_fields_chunked

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None


def _peak_resident_memory_bytes() -> int | None:
    """Return process peak RSS in bytes on supported operating systems."""

    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the target HPC systems report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _peak_child_resident_memory_bytes() -> int | None:
    """Return peak RSS reported for completed child processes, in bytes."""

    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _relative_field_error(
    actual_electric: np.ndarray,
    actual_magnetic: np.ndarray,
    expected_electric: np.ndarray,
    expected_magnetic: np.ndarray,
) -> float:
    difference = np.concatenate(
        (
            (actual_electric - expected_electric).ravel(),
            (actual_magnetic - expected_magnetic).ravel(),
        )
    )
    reference = np.concatenate(
        (expected_electric.ravel(), expected_magnetic.ravel())
    )
    return float(np.linalg.norm(difference) / np.linalg.norm(reference))


def _combined_field_norm(electric: np.ndarray, magnetic: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.concatenate((electric.ravel(), magnetic.ravel()))
        )
    )


def _evaluate_svwf_total_fields(
    radius: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    case: DipoleMatchingCase,
    exterior_m: dict,
    exterior_n: dict,
    interior_m: dict,
    interior_n: dict,
    field_workers: int = 1,
    field_chunk_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the SVWF solution and return canonical ``exp(+j omega t)`` fields."""

    electric = np.full((3, radius.size), np.nan + 0j, dtype=complex)
    magnetic = np.full((3, radius.size), np.nan + 0j, dtype=complex)
    valid = radius > 0.0
    interior = valid & (radius < case.radius)
    exterior = valid & ~interior

    if np.any(interior):
        incident_electric, incident_magnetic, _, _ = prescribed_dipole_fields(
            theta[interior],
            phi[interior],
            case,
            radius=radius[interior],
            svwf_convention=True,
        )
        reflected_electric, reflected_magnetic = compute_fields_chunked(
            case.k_interior,
            radius[interior],
            theta[interior],
            phi[interior],
            interior_m,
            interior_n,
            kind=1,
            eta=case.eta_interior,
            workers=field_workers,
            chunk_size=field_chunk_size,
        )
        electric[:, interior] = np.conjugate(
            incident_electric + np.asarray(reflected_electric)
        )
        magnetic[:, interior] = np.conjugate(
            incident_magnetic + np.asarray(reflected_magnetic)
        )

    if np.any(exterior):
        transmitted_electric, transmitted_magnetic = compute_fields_chunked(
            case.k_exterior,
            radius[exterior],
            theta[exterior],
            phi[exterior],
            exterior_m,
            exterior_n,
            kind=3,
            eta=case.eta_exterior,
            workers=field_workers,
            chunk_size=field_chunk_size,
        )
        electric[:, exterior] = np.conjugate(np.asarray(transmitted_electric))
        magnetic[:, exterior] = np.conjugate(np.asarray(transmitted_magnetic))

    return electric, magnetic


def _write_workflow_results(
    *,
    run_dir: Path,
    configuration: WorkflowConfiguration,
    metrics: dict[str, float],
    exterior_m: dict,
    exterior_n: dict,
    interior_m: dict,
    interior_n: dict,
    coefficient_source: Path | None = None,
    field_workers: int = 1,
    field_chunk_size: int | None = None,
) -> None:
    output_started = perf_counter()
    case = configuration.case
    extraction_method = configuration.extraction_method
    n_max = configuration.svwf.n_max
    n_theta = configuration.svwf.n_theta
    n_phi = configuration.svwf.n_phi
    zx_grid_size = configuration.sampling.zx_grid_size
    zx_span_radii = configuration.sampling.zx_span_radii
    radial_points = configuration.sampling.radial_points
    radial_theta_degrees = configuration.sampling.radial_theta_degrees
    radial_phi_degrees = configuration.sampling.radial_phi_degrees
    radial_min_radii = configuration.sampling.radial_min_radii
    radial_max_radii = configuration.sampling.radial_max_radii

    if zx_grid_size < 2 or zx_grid_size % 2 != 0:
        raise ValueError("zx_grid_size must be an even integer greater than one")
    if radial_points < 2:
        raise ValueError("radial_points must be at least two")
    if zx_span_radii <= 1.0:
        raise ValueError("zx_span_radii must be greater than one")
    if not 0.0 < radial_min_radii < 1.0:
        raise ValueError("radial_min_radii must lie strictly between zero and one")
    if radial_max_radii <= 1.0:
        raise ValueError("radial_max_radii must be greater than one")
    if field_workers < 1:
        raise ValueError("field_workers must be at least one")
    if field_chunk_size is not None and field_chunk_size < 1:
        raise ValueError("field_chunk_size must be at least one")

    metrics["field_workers"] = field_workers
    if field_chunk_size is not None:
        metrics["field_chunk_size"] = field_chunk_size

    fields_dir = run_dir / "fields"
    plots_dir = run_dir / "plots"
    fields_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    coefficient_path = run_dir / "svwf_coefficients.npz"
    coefficient_started = perf_counter()
    save_coefficients(
        coefficient_path,
        exterior_m,
        exterior_n,
        interior_m,
        interior_n,
    )
    metrics["coefficient_export_seconds"] = perf_counter() - coefficient_started

    span = zx_span_radii * case.radius
    x_axis = np.linspace(-span, span, zx_grid_size)
    z_axis = np.linspace(-span, span, zx_grid_size)
    x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="xy")
    x = x_grid.ravel()
    y = np.zeros_like(x)
    z = z_grid.ravel()
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / radius, -1.0, 1.0))
    phi = np.arctan2(y, x)
    region = np.where(radius < case.radius, 1, 2)

    zx_evaluation_started = perf_counter()
    prescribed_electric, prescribed_magnetic = prescribed_total_fields(
        radius, theta, phi, case
    )
    svwf_electric, svwf_magnetic = _evaluate_svwf_total_fields(
        radius,
        theta,
        phi,
        case,
        exterior_m,
        exterior_n,
        interior_m,
        interior_n,
        field_workers,
        field_chunk_size,
    )
    prescribed_electric_cartesian = spherical_to_cartesian(
        prescribed_electric, theta, phi
    )
    prescribed_magnetic_cartesian = spherical_to_cartesian(
        prescribed_magnetic, theta, phi
    )
    svwf_electric_cartesian = spherical_to_cartesian(svwf_electric, theta, phi)
    svwf_magnetic_cartesian = spherical_to_cartesian(svwf_magnetic, theta, phi)
    metrics["zx_field_evaluation_seconds"] = (
        perf_counter() - zx_evaluation_started
    )

    prescribed_zx = fields_dir / "prescribed_zx.csv"
    svwf_zx = fields_dir / "svwf_zx.csv"
    zx_export_started = perf_counter()
    write_field_csv(
        prescribed_zx,
        x,
        y,
        z,
        region,
        prescribed_electric_cartesian,
        prescribed_magnetic_cartesian,
    )
    write_field_csv(
        svwf_zx,
        x,
        y,
        z,
        region,
        svwf_electric_cartesian,
        svwf_magnetic_cartesian,
    )
    metrics["zx_field_export_seconds"] = perf_counter() - zx_export_started

    radial_radius = np.linspace(
        radial_min_radii * case.radius,
        radial_max_radii * case.radius,
        radial_points,
    )
    radial_theta = np.full(radial_points, np.deg2rad(radial_theta_degrees))
    radial_phi = np.full(radial_points, np.deg2rad(radial_phi_degrees))
    radial_x = radial_radius * np.sin(radial_theta) * np.cos(radial_phi)
    radial_y = radial_radius * np.sin(radial_theta) * np.sin(radial_phi)
    radial_z = radial_radius * np.cos(radial_theta)
    radial_region = np.where(radial_radius < case.radius, 1, 2)

    radial_evaluation_started = perf_counter()
    prescribed_electric, prescribed_magnetic = prescribed_total_fields(
        radial_radius, radial_theta, radial_phi, case
    )
    svwf_electric, svwf_magnetic = _evaluate_svwf_total_fields(
        radial_radius,
        radial_theta,
        radial_phi,
        case,
        exterior_m,
        exterior_n,
        interior_m,
        interior_n,
        field_workers,
        field_chunk_size,
    )
    prescribed_electric_cartesian = spherical_to_cartesian(
        prescribed_electric, radial_theta, radial_phi
    )
    prescribed_magnetic_cartesian = spherical_to_cartesian(
        prescribed_magnetic, radial_theta, radial_phi
    )
    svwf_electric_cartesian = spherical_to_cartesian(
        svwf_electric, radial_theta, radial_phi
    )
    svwf_magnetic_cartesian = spherical_to_cartesian(
        svwf_magnetic, radial_theta, radial_phi
    )
    metrics["radial_field_evaluation_seconds"] = (
        perf_counter() - radial_evaluation_started
    )

    prescribed_radial = fields_dir / "prescribed_radial.csv"
    svwf_radial = fields_dir / "svwf_radial.csv"
    radial_export_started = perf_counter()
    write_field_csv(
        prescribed_radial,
        radial_x,
        radial_y,
        radial_z,
        radial_region,
        prescribed_electric_cartesian,
        prescribed_magnetic_cartesian,
    )
    write_field_csv(
        svwf_radial,
        radial_x,
        radial_y,
        radial_z,
        radial_region,
        svwf_electric_cartesian,
        svwf_magnetic_cartesian,
    )
    metrics["radial_field_export_seconds"] = (
        perf_counter() - radial_export_started
    )

    configuration_path = run_dir / "configuration.json"
    with configuration_path.open("w", encoding="utf-8") as stream:
        json.dump(configuration.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    metrics["output_workflow_seconds"] = perf_counter() - output_started
    peak_resident_memory = _peak_resident_memory_bytes()
    if peak_resident_memory is not None:
        metrics["peak_resident_memory_bytes"] = peak_resident_memory
        metrics["peak_child_resident_memory_bytes"] = (
            _peak_child_resident_memory_bytes()
        )

    manifest = {
        "schema_version": 1,
        "case": "dipole_impedance_matching",
        "case_id": configuration.case_id,
        "backend": "svwf",
        "workflow_mode": (
            "field_evaluation" if coefficient_source else "solve_and_evaluate"
        ),
        "phasor_convention": "exp(+j omega t)",
        "field_data": {
            "coordinate_system": "Cartesian",
            "coordinate_unit": "m",
            "electric_field_unit": "V/m",
            "magnetic_field_unit": "A/m",
            "regions": {"1": "interior", "2": "exterior"},
        },
        "frequency_hz": case.frequency,
        "radius_m": case.radius,
        "radius_lambda0": case.radius_in_free_space_wavelengths,
        "interior": {
            "epsilon_r": case.relative_permittivity_interior,
            "mu_r": case.relative_permeability_interior,
        },
        "exterior": {
            "epsilon_r": case.relative_permittivity_exterior,
            "mu_r": case.relative_permeability_exterior,
        },
        "source": {
            "type": "electric_dipole",
            "position_m": [0.0, 0.0, 0.0],
            "direction": [0.0, 0.0, 1.0],
            "effective_moment_interior": case.effective_moment_interior,
            "effective_moment_exterior": case.effective_moment_exterior,
        },
        "extraction_method": extraction_method,
        "cap_perturbation": configuration.to_dict().get("cap_perturbation"),
        "reference_label": configuration.reference_label,
        "solver": {
            "backend": "svwf",
            "settings": {"n_max": n_max, "n_theta": n_theta, "n_phi": n_phi},
        },
        "field_evaluation": {
            "workers": field_workers,
            "chunk_size": field_chunk_size,
            "default_chunk_policy": "ceil(points / (4 * workers))",
        },
        "svwf": {"n_max": n_max, "n_theta": n_theta, "n_phi": n_phi},
        "sampling": {
            "zx": {"shape": [zx_grid_size, zx_grid_size], "span_radii": zx_span_radii},
            "radial": {
                "points": radial_points,
                "theta_degrees": radial_theta_degrees,
                "phi_degrees": radial_phi_degrees,
                "minimum_radius_over_a": radial_min_radii,
                "maximum_radius_over_a": radial_max_radii,
            },
        },
        "performance_measurements": {
            "duration_unit": "seconds",
            "memory_unit": "bytes",
            "peak_resident_memory_bytes": (
                "measured process maximum resident set size"
            ),
            "peak_child_resident_memory_bytes": (
                "maximum resident set size reported for completed workers"
            ),
            "solver_matrix_bytes": "allocated collocation matrix Z only",
        },
        "metrics": metrics,
        "files": {
            "configuration": str(configuration_path.relative_to(run_dir)),
            "coefficients": str(coefficient_path.relative_to(run_dir)),
            "reference_zx": str(prescribed_zx.relative_to(run_dir)),
            "computed_zx": str(svwf_zx.relative_to(run_dir)),
            "reference_radial": str(prescribed_radial.relative_to(run_dir)),
            "computed_radial": str(svwf_radial.relative_to(run_dir)),
            "prescribed_zx": str(prescribed_zx.relative_to(run_dir)),
            "svwf_zx": str(svwf_zx.relative_to(run_dir)),
            "prescribed_radial": str(prescribed_radial.relative_to(run_dir)),
            "svwf_radial": str(svwf_radial.relative_to(run_dir)),
        },
    }
    if coefficient_source is not None:
        manifest["coefficient_source"] = str(coefficient_source.resolve())
    with (run_dir / "run.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_svwf_field_evaluation(
    *,
    source_run_dir: Path,
    run_dir: Path,
    workflow_configuration: WorkflowConfiguration,
    field_workers: int = 1,
    field_chunk_size: int | None = None,
) -> dict[str, float]:
    """Regenerate field samples from a previously solved coefficient archive."""

    if source_run_dir.resolve() == run_dir.resolve():
        raise ValueError("field-only output must use a different run directory")
    coefficient_path = source_run_dir / "svwf_coefficients.npz"
    load_started = perf_counter()
    exterior_m, exterior_n, interior_m, interior_n = load_coefficients(
        coefficient_path
    )
    load_seconds = perf_counter() - load_started

    expected_modes = {
        (m, n)
        for n in range(1, workflow_configuration.svwf.n_max + 1)
        for m in range(-n, n + 1)
    }
    if set(exterior_m) != expected_modes:
        raise ValueError(
            "coefficient modes do not match configuration svwf.n_max="
            f"{workflow_configuration.svwf.n_max}"
        )

    metrics = {
        "coefficient_load_seconds": load_seconds,
        "coefficient_mode_count": len(expected_modes),
    }
    _write_workflow_results(
        run_dir=run_dir,
        configuration=workflow_configuration,
        metrics=metrics,
        exterior_m=exterior_m,
        exterior_n=exterior_n,
        interior_m=interior_m,
        interior_n=interior_n,
        coefficient_source=coefficient_path,
        field_workers=field_workers,
        field_chunk_size=field_chunk_size,
    )
    return metrics


def run_svwf_validation(
    *,
    case: DipoleMatchingCase,
    n_max: int = 4,
    n_theta: int = 18,
    n_phi: int = 12,
    extraction_method: ExtractionMethod = "least_squares",
    run_dir: Path | None = None,
    zx_grid_size: int = 200,
    zx_span_radii: float = 2.0,
    radial_points: int = 300,
    radial_theta_degrees: float = 45.0,
    radial_phi_degrees: float = 0.0,
    radial_min_radii: float = 0.05,
    radial_max_radii: float = 2.0,
    workflow_configuration: WorkflowConfiguration | None = None,
    field_workers: int = 1,
    field_chunk_size: int | None = None,
) -> dict[str, float]:
    """Run the end-to-end extraction and SVWF boundary-field validation."""

    cap = workflow_configuration.cap_perturbation if workflow_configuration else None
    cache: dict[str, object] = {
        "theta": None,
        "phi": None,
        "blocks": None,
    }

    def tensor_blocks(theta: np.ndarray, phi: np.ndarray) -> dict[str, np.ndarray]:
        if theta is cache["theta"] and phi is cache["phi"]:
            return cast(dict[str, np.ndarray], cache["blocks"])
        extraction = extract_case_parameters(
            theta,
            phi,
            case,
            method=extraction_method,
            svwf_convention=True,
        )
        if not np.all(extraction.valid):
            raise RuntimeError("omega extraction is degenerate on the SVWF grid")
        parameters, cap_mask = apply_cap_perturbation(
            extraction.parameters, theta, phi, cap
        )
        if cap is not None:
            if cap.parameter_scale != 1.0 and not np.any(cap_mask):
                raise ValueError(
                    "no SVWF collocation points sample the cap; increase n_theta/n_phi"
                )
            print(
                f"  cap collocation points: {np.count_nonzero(cap_mask)} / {theta.size}",
                flush=True,
            )
        blocks = omega_tensor_blocks(parameters)
        cache.update(
            theta=theta, phi=phi, blocks=blocks,
            cap_point_count=int(np.count_nonzero(cap_mask)),
        )
        return blocks

    gstc_parameters = {
        name: (lambda theta, phi, block=name: tensor_blocks(theta, phi)[block])
        for name in ("Y", "Chi", "Gamma", "Z")
    }

    def incident_field(theta: np.ndarray, phi: np.ndarray):
        electric, magnetic, _, _ = prescribed_dipole_fields(
            theta, phi, case, svwf_convention=True
        )
        return electric[1], electric[2], magnetic[1], magnetic[2]

    result = solve_collocation_general(
        n_max=n_max,
        k0=case.k_exterior,
        eta0=case.eta_exterior,
        k1=case.k_interior,
        eta1=case.eta_interior,
        radius=case.radius,
        gstc_params=gstc_parameters,
        n_theta=n_theta,
        n_phi=n_phi,
        return_metrics=True,
        incident_field_func=incident_field,
        is_interior_source=True,
    )
    exterior_m, exterior_n, interior_m, interior_n, solver_metrics = result

    theta_validation = np.linspace(0.02, np.pi - 0.02, 121)
    phi_validation = np.zeros_like(theta_validation)
    radius_validation = np.full_like(theta_validation, case.radius)
    (
        expected_electric_interior,
        expected_magnetic_interior,
        expected_electric_exterior,
        expected_magnetic_exterior,
    ) = prescribed_dipole_fields(
        theta_validation, phi_validation, case, svwf_convention=True
    )

    reflected_electric, reflected_magnetic = compute_fields(
        case.k_interior,
        radius_validation,
        theta_validation,
        phi_validation,
        interior_m,
        interior_n,
        kind=1,
        eta=case.eta_interior,
        grid=False,
    )
    transmitted_electric, transmitted_magnetic = compute_fields(
        case.k_exterior,
        radius_validation,
        theta_validation,
        phi_validation,
        exterior_m,
        exterior_n,
        kind=3,
        eta=case.eta_exterior,
        grid=False,
    )

    reflected_electric_array = np.asarray(reflected_electric)
    reflected_magnetic_array = np.asarray(reflected_magnetic)
    transmitted_electric_array = np.asarray(transmitted_electric)
    transmitted_magnetic_array = np.asarray(transmitted_magnetic)
    power_interior = normal_power_density(
        expected_electric_interior[1:], expected_magnetic_interior[1:]
    )
    power_exterior = normal_power_density(
        expected_electric_exterior[1:], expected_magnetic_exterior[1:]
    )
    power_error = np.linalg.norm(power_exterior - power_interior) / np.linalg.norm(
        power_interior
    )
    physical_extraction = extract_case_parameters(
        theta_validation,
        phi_validation,
        case,
        method=extraction_method,
        svwf_convention=False,
    )
    if not np.all(physical_extraction.valid):
        raise RuntimeError("omega extraction is degenerate on the validation grid")
    B = physical_extraction.parameters.B
    X = physical_extraction.parameters.X
    chi = physical_extraction.parameters.chi

    def relative_spread(values: np.ndarray) -> float:
        mean = np.mean(values)
        return float(np.max(np.abs(values - mean)) / np.abs(mean))

    metrics = {
        "effective_moment_ratio": case.effective_moment_ratio,
        "B_exp_plus_jwt": float(np.mean(B)),
        "X_exp_plus_jwt": float(np.mean(X)),
        "chi": float(np.mean(chi)),
        "parameter_uniformity_error": max(
            relative_spread(B), relative_spread(X), relative_spread(chi)
        ),
        "prescribed_power_error": float(power_error),
        "solver_residual": float(solver_metrics["residual_rel"]),
        "interior_reflection_error": _combined_field_norm(
            reflected_electric_array, reflected_magnetic_array
        )
        / _combined_field_norm(
            expected_electric_interior, expected_magnetic_interior
        ),
        "exterior_transmission_error": _relative_field_error(
            transmitted_electric_array,
            transmitted_magnetic_array,
            expected_electric_exterior,
            expected_magnetic_exterior,
        ),
    }
    for name in (
        "setup_seconds",
        "basis_seconds",
        "matrix_assembly_seconds",
        "scaling_seconds",
        "least_squares_seconds",
        "residual_seconds",
        "total_seconds",
        "equation_count",
        "unknown_count",
        "matrix_bytes",
        "full_size_matrix_arrays",
    ):
        metrics[f"solver_{name}"] = solver_metrics[name]
    if cap is not None:
        metrics["cap_collocation_point_count"] = int(cache["cap_point_count"])
        # The original analytical fields characterize the unperturbed sheet.
        # Departures from that reference are expected when the sheet is changed.
        if cap.parameter_scale != 1.0:
            for name in (
                "B_exp_plus_jwt", "X_exp_plus_jwt", "chi", "parameter_uniformity_error"
            ):
                metrics[f"baseline_{name}"] = metrics.pop(name)
            metrics["interior_reflected_relative_norm"] = metrics.pop(
                "interior_reflection_error"
            )
            metrics["exterior_change_from_reference"] = metrics.pop(
                "exterior_transmission_error"
            )
    if run_dir is not None:
        configuration = replace(
            workflow_configuration or WorkflowConfiguration(),
            case=case,
            extraction_method=extraction_method,
            svwf=SVWFConfiguration(
                n_max=n_max,
                n_theta=n_theta,
                n_phi=n_phi,
            ),
            sampling=SamplingConfiguration(
                zx_grid_size=zx_grid_size,
                zx_span_radii=zx_span_radii,
                radial_points=radial_points,
                radial_theta_degrees=radial_theta_degrees,
                radial_phi_degrees=radial_phi_degrees,
                radial_min_radii=radial_min_radii,
                radial_max_radii=radial_max_radii,
            ),
        )
        _write_workflow_results(
            run_dir=run_dir,
            configuration=configuration,
            metrics=metrics,
            exterior_m=exterior_m,
            exterior_n=exterior_n,
            interior_m=interior_m,
            interior_n=interior_n,
            field_workers=field_workers,
            field_chunk_size=field_chunk_size,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--evaluate-from-run",
        type=Path,
        help=(
            "skip the solve and load configuration.json plus "
            "svwf_coefficients.npz from this completed run"
        ),
    )
    parser.add_argument("--frequency-hz", type=float)
    parser.add_argument("--radius-lambda0", type=float)
    parser.add_argument("--epsilon-r-interior", type=float)
    parser.add_argument("--mu-r-interior", type=float)
    parser.add_argument("--epsilon-r-exterior", type=float)
    parser.add_argument("--mu-r-exterior", type=float)
    parser.add_argument("--effective-moment-interior", type=float)
    parser.add_argument("--n-max", type=int)
    parser.add_argument("--n-theta", type=int)
    parser.add_argument("--n-phi", type=int)
    parser.add_argument(
        "--extraction-method",
        choices=("projected", "least_squares"),
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--zx-grid-size", type=int)
    parser.add_argument("--zx-span-radii", type=float)
    parser.add_argument("--radial-points", type=int)
    parser.add_argument("--radial-theta", type=float)
    parser.add_argument("--radial-phi", type=float)
    parser.add_argument("--radial-min-radii", type=float)
    parser.add_argument("--radial-max-radii", type=float)
    parser.add_argument("--field-workers", type=int, default=1)
    parser.add_argument("--field-chunk-size", type=int)
    arguments = parser.parse_args()

    if arguments.evaluate_from_run is not None:
        forbidden = {
            "--config": arguments.config,
            "--frequency-hz": arguments.frequency_hz,
            "--radius-lambda0": arguments.radius_lambda0,
            "--epsilon-r-interior": arguments.epsilon_r_interior,
            "--mu-r-interior": arguments.mu_r_interior,
            "--epsilon-r-exterior": arguments.epsilon_r_exterior,
            "--mu-r-exterior": arguments.mu_r_exterior,
            "--effective-moment-interior": arguments.effective_moment_interior,
            "--n-max": arguments.n_max,
            "--n-theta": arguments.n_theta,
            "--n-phi": arguments.n_phi,
            "--extraction-method": arguments.extraction_method,
        }
        conflicts = [name for name, value in forbidden.items() if value is not None]
        if conflicts:
            parser.error(
                "--evaluate-from-run cannot be combined with "
                + ", ".join(conflicts)
            )
        if arguments.run_dir is None:
            parser.error("--evaluate-from-run requires a distinct --run-dir")
        configuration = load_workflow_configuration(
            arguments.evaluate_from_run / "configuration.json"
        )
    else:
        configuration = (
            load_workflow_configuration(arguments.config)
            if arguments.config is not None
            else WorkflowConfiguration()
        )
    case_overrides = {
        "frequency": arguments.frequency_hz,
        "radius_in_free_space_wavelengths": arguments.radius_lambda0,
        "relative_permittivity_interior": arguments.epsilon_r_interior,
        "relative_permeability_interior": arguments.mu_r_interior,
        "relative_permittivity_exterior": arguments.epsilon_r_exterior,
        "relative_permeability_exterior": arguments.mu_r_exterior,
        "effective_moment_interior": arguments.effective_moment_interior,
    }
    case = replace(
        configuration.case,
        **{name: value for name, value in case_overrides.items() if value is not None},
    )
    extraction_method = (
        arguments.extraction_method
        if arguments.extraction_method is not None
        else configuration.extraction_method
    )
    svwf_overrides = {
        "n_max": arguments.n_max,
        "n_theta": arguments.n_theta,
        "n_phi": arguments.n_phi,
    }
    sampling_overrides = {
        "zx_grid_size": arguments.zx_grid_size,
        "zx_span_radii": arguments.zx_span_radii,
        "radial_points": arguments.radial_points,
        "radial_theta_degrees": arguments.radial_theta,
        "radial_phi_degrees": arguments.radial_phi,
        "radial_min_radii": arguments.radial_min_radii,
        "radial_max_radii": arguments.radial_max_radii,
    }
    configuration = replace(
        configuration,
        case=case,
        extraction_method=extraction_method,
        svwf=replace(
            configuration.svwf,
            **{
                name: value
                for name, value in svwf_overrides.items()
                if value is not None
            },
        ),
        sampling=replace(
            configuration.sampling,
            **{
                name: value
                for name, value in sampling_overrides.items()
                if value is not None
            },
        ),
    )
    run_dir = arguments.run_dir or default_run_directory(
        configuration.case_id, "svwf", configuration.extraction_method
    )
    if arguments.evaluate_from_run is not None:
        metrics = run_svwf_field_evaluation(
            source_run_dir=arguments.evaluate_from_run,
            run_dir=run_dir,
            workflow_configuration=configuration,
            field_workers=arguments.field_workers,
            field_chunk_size=arguments.field_chunk_size,
        )
    else:
        metrics = run_svwf_validation(
            case=configuration.case,
            n_max=configuration.svwf.n_max,
            n_theta=configuration.svwf.n_theta,
            n_phi=configuration.svwf.n_phi,
            extraction_method=configuration.extraction_method,
            run_dir=run_dir,
            zx_grid_size=configuration.sampling.zx_grid_size,
            zx_span_radii=configuration.sampling.zx_span_radii,
            radial_points=configuration.sampling.radial_points,
            radial_theta_degrees=configuration.sampling.radial_theta_degrees,
            radial_phi_degrees=configuration.sampling.radial_phi_degrees,
            radial_min_radii=configuration.sampling.radial_min_radii,
            radial_max_radii=configuration.sampling.radial_max_radii,
            workflow_configuration=configuration,
            field_workers=arguments.field_workers,
            field_chunk_size=arguments.field_chunk_size,
        )

    print("Spherical omega-biisotropic dipole matching")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6e}")
    print(f"  run_dir: {run_dir}")


if __name__ == "__main__":
    main()
