"""Experimental interior-field feedback for reciprocal-lossless GSTC synthesis.

Fixed exterior targets and initial interior fields are evaluated at actual
flat-face centroids, unlike the production workflow's ideal-sphere samples.
SIE feedback is sampled at c - d*c/|c| and treated as an approximate interior
boundary trace. All tensors retain the spherical theta/phi convention.
This is a fixed-count experiment, not a convergence-guaranteed extraction.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sie_mom.gstc import extract_reciprocal_lossless_least_squares, form_boundary_quantities
from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh
from sie_mom.postprocessing import cartesian_to_spherical

if __package__:
    from ..prescribed_fields import PrescribedFieldState, evaluate_prescribed_fields
    from ..sie_gstc_sphere import DEFAULT_EXECUTABLE, solver_command
    from ..workflow import (
        APPLICATION_DIR,
        REPO_ROOT,
        create_run_directory,
        load_workflow_configuration,
        write_json,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prescribed_fields import PrescribedFieldState, evaluate_prescribed_fields
    from sie_gstc_sphere import DEFAULT_EXECUTABLE, solver_command
    from workflow import (
        APPLICATION_DIR,
        REPO_ROOT,
        create_run_directory,
        load_workflow_configuration,
        write_json,
    )


# Edit these settings directly. Physical setup, fields, and excitations stay in JSON.
DEFAULT_CONFIG = (
    APPLICATION_DIR
    / "configurations/moment_rot/rot90_eps7.json"
)
ITERATIONS = 5
OFFSET_M = 0.01
DAMPING = 1.0  # Direct replacement; try 0.5 if the iteration oscillates.
STATE_CASES = (
    ("electric_z_to_x", "electric_incidence"),
    ("magnetic_z_to_x", "magnetic_incidence"),
)
SAMPLER_EXECUTABLE = (
    REPO_ROOT / "build/applications/anisotropic_gstc_sphere/anisotropic_surface_sampler"
)


def surface_geometry(configuration, offset_m):
    """Return centroids, spherical angles, and areas in unchanged OBJ face order."""
    mesh = load_triangle_obj(configuration.mesh_path)
    validate_triangle_mesh(mesh).require_valid(expected_euler_characteristic=2)
    if not np.allclose(
        np.linalg.norm(mesh.vertices, axis=1), configuration.case.radius_m, rtol=1e-4, atol=0
    ):
        raise ValueError("mesh vertices must lie on the configured sphere")
    triangles = mesh.vertices[mesh.faces]
    centroids = triangles.mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)
    if not np.isfinite(offset_m) or not 0 < offset_m < radii.min():
        raise ValueError("offset_m must be positive and smaller than every centroid radius")
    theta = np.arccos(np.clip(centroids[:, 2] / radii, -1, 1))
    phi = np.arctan2(centroids[:, 1], centroids[:, 0])
    areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )
    return centroids, radii, theta, phi, areas


def initialize_states(configuration, radii, theta, phi, state_cases):
    """Evaluate both sides of each selected transformation at the face centroids."""
    transformations = {state.id: state for state in configuration.transformations}
    cases = {case.id for case in configuration.simulation.cases}
    state_ids = [state_id for state_id, _ in state_cases]
    case_ids = [case_id for _, case_id in state_cases]
    if set(state_ids) != set(transformations) or len(state_ids) != len(set(state_ids)):
        raise ValueError("STATE_CASES must list every transformation exactly once")
    if set(case_ids) != cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("STATE_CASES must list every simulation case exactly once")
    states = []
    for state_id, _ in state_cases:
        transformation = transformations[state_id]
        interior = evaluate_prescribed_fields(
            transformation.gstc.interior,
            "interior",
            configuration,
            radius=radii,
            theta=theta,
            phi=phi,
        )
        exterior = evaluate_prescribed_fields(
            transformation.gstc.exterior,
            "exterior",
            configuration,
            radius=radii,
            theta=theta,
            phi=phi,
        )
        states.append(PrescribedFieldState(state_id, *interior, *exterior))
    return tuple(states)


def extract_and_export(states, configuration, directory):
    """Fit one shared lossless map to all states, then export map and inverse."""
    quantities = [
        form_boundary_quantities(
            state.electric_interior[1:],
            state.magnetic_interior[1:],
            state.electric_exterior[1:],
            state.magnetic_exterior[1:],
        )
        for state in states
    ]
    extraction = extract_reciprocal_lossless_least_squares(
        quantities,
        reference_impedance=configuration.case.eta_exterior,
        structure=configuration.tensor_structure,
    )
    maps = np.moveaxis(extraction.parameters.as_matrix(), -1, 0)
    if not np.isfinite(maps).all():
        raise ValueError("extraction produced non-finite surface maps")
    try:
        inverses = np.linalg.inv(maps)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "surface map is singular; the current SIE requires its full inverse"
        ) from error
    if not np.isfinite(inverses).all():
        raise ValueError("surface map inverse is non-finite")
    records = np.concatenate((maps.reshape(-1, 16), inverses.reshape(-1, 16)), axis=1)
    records.astype("<c16").tofile(directory / "face_tensors.bin")
    np.savez(
        directory / "extraction.npz",
        rank=extraction.rank,
        relative_residual=extraction.relative_residual,
        state_relative_residual=extraction.state_relative_residual,
        condition_number=extraction.condition_number,
        inverse_residual=np.linalg.norm(maps @ inverses - np.eye(4), axis=(1, 2)),
    )
    return extraction


def run_logged(command, log_path):
    """Keep the exact command and subprocess output beside its numerical data."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"subprocess failed; see {log_path}") from error


def solve_currents(configuration, directory, executable):
    """Solve all unchanged incident cases together; skip production plot grids."""
    numerical = replace(configuration, sie=replace(configuration.sie, evaluate_fields=False))
    command = solver_command(numerical, run_dir=directory, executable=executable)
    run_logged(command, directory / "solver.log")


def read_integrals(path, expected_points, theta, phi):
    """Read Q, verify face IDs/coordinates, and return spherical E/H integrals."""
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if len(data) != len(expected_points) or not np.array_equal(
        data["face_index"], np.arange(len(expected_points))
    ):
        raise ValueError("sampled integrals do not preserve OBJ face order")
    points = np.column_stack([data[axis] for axis in "xyz"])
    if not np.allclose(points, expected_points, rtol=1e-10, atol=1e-12):
        raise ValueError("sampled coordinates do not match the requested centroid offsets")
    fields = []
    for kind in "EH":
        cartesian = np.stack(
            [data[f"Q{kind}{axis}_real"] + 1j * data[f"Q{kind}{axis}_imag"] for axis in "xyz"]
        )
        if not np.isfinite(cartesian).all():
            raise ValueError("sampled integral contributions must be finite")
        fields.append(cartesian_to_spherical(cartesian, theta, phi))
    return tuple(fields)


def incident_fields(configuration, simulation_case, region, radii, theta, phi):
    """Evaluate only that region's unchanged configured incident excitation."""
    reference = next(
        (
            excitation.field
            for excitation in simulation_case.excitations
            if excitation.region == region
        ),
        None,
    )
    return evaluate_prescribed_fields(
        reference, region, configuration, radius=radii, theta=theta, phi=phi
    )


def sample_total_fields(
    configuration, simulation_case, directory, centroids, radii, theta, phi, offset_m, executable
):
    """Sample both regional Q representations and form total fields as incident - Q."""
    case_directory = directory / "cases" / simulation_case.id
    output = case_directory / "surface_samples"
    physical = configuration.case
    command = [
        str(executable),
        str(configuration.mesh_path.resolve()),
        str(case_directory / "currents.bin"),
        str(output),
        str(physical.frequency_hz),
        str(physical.relative_permittivity_interior),
        str(physical.relative_permeability_interior),
        str(physical.relative_permittivity_exterior),
        str(physical.relative_permeability_exterior),
        str(configuration.sie.quadrature_rule),
        str(offset_m),
    ]
    run_logged(command, case_directory / "sampling.log")
    totals = []
    for region, signed_offset in (("interior", -offset_m), ("exterior", offset_m)):
        points = centroids * (1 + signed_offset / radii[:, None])
        integrals = read_integrals(output / f"{region}_integrals.csv", points, theta, phi)
        incident = incident_fields(
            configuration, simulation_case, region, radii + signed_offset, theta, phi
        )
        totals.append(tuple(source - integral for source, integral in zip(incident, integrals)))
    return tuple(totals)


def update_interior_states(states, interior_fields, damping):
    """Replace interior total E/H together; leave every exterior target unchanged."""
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    if len(states) != len(interior_fields):
        raise ValueError("one computed interior field pair is required per state")
    updated = []
    for state, (electric, magnetic) in zip(states, interior_fields):
        if (
            electric.shape != state.electric_interior.shape
            or magnetic.shape != state.magnetic_interior.shape
        ):
            raise ValueError("computed interior fields have the wrong shape")
        if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
            raise ValueError("computed interior fields must be finite")
        updated.append(
            replace(
                state,
                electric_interior=(1 - damping) * state.electric_interior + damping * electric,
                magnetic_interior=(1 - damping) * state.magnetic_interior + damping * magnetic,
            )
        )
    return tuple(updated)


def relative_field_error(actual, target, eta, areas):
    """Area-weighted complex E/H error; eta converts magnetic fields to V/m."""
    difference = np.concatenate((actual[0] - target[0], eta * (actual[1] - target[1])))
    reference = np.concatenate((target[0], eta * target[1]))

    def norm(field):
        return np.sqrt(np.sum(np.abs(field) ** 2 * areas[None, :]))

    return float(norm(difference) / max(norm(reference), np.finfo(float).tiny))


def run_experiment(
    configuration,
    output_dir=None,
    *,
    iterations=ITERATIONS,
    offset_m=OFFSET_M,
    damping=DAMPING,
    state_cases=STATE_CASES,
    solver_executable=DEFAULT_EXECUTABLE,
    sampler_executable=SAMPLER_EXECUTABLE,
):
    """Run the fixed-count extraction -> batch solve -> interior feedback experiment."""
    if configuration.extraction_method != "reciprocal_lossless":
        raise ValueError("this experiment requires extraction_method=reciprocal_lossless")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    for executable in (solver_executable, sampler_executable):
        if not executable.is_file():
            raise FileNotFoundError(
                f"missing executable: {executable}; "
                "build anisotropic_sphere_app and anisotropic_surface_sampler"
            )
    centroids, radii, theta, phi, areas = surface_geometry(configuration, offset_m)

    # Initialization: ideal transformations, with both regional targets at centroids.
    states = initialize_states(configuration, radii, theta, phi, state_cases)
    transformations = {state.id: state for state in configuration.transformations}
    cases = {case.id: case for case in configuration.simulation.cases}
    exterior_checks = tuple(
        evaluate_prescribed_fields(
            transformations[state_id].gstc.exterior,
            "exterior",
            configuration,
            radius=radii + offset_m,
            theta=theta,
            phi=phi,
        )
        for state_id, _ in state_cases
    )
    # Preflight the existing real-amplitude C++ source contract before creating output.
    solver_command(configuration, run_dir=Path("unused"), executable=solver_executable)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = (
            APPLICATION_DIR
            / "results"
            / configuration.case_id
            / "experiments/lossless_feedback"
            / stamp
        )
    output_dir = output_dir.resolve()
    create_run_directory(output_dir)
    write_json(
        output_dir / "experiment.json",
        {
            "configuration": configuration.to_dict(),
            "iterations": iterations,
            "offset_m": offset_m,
            "damping": damping,
            "state_cases": list(state_cases),
            "boundary_points": "actual face centroids",
            "sampling": "c +/- d*c/|c|; approximate boundary traces",
            "tangent_basis": "theta, phi; outward radial normal",
        },
    )
    np.savez(
        output_dir / "geometry_and_targets.npz",
        centroids=centroids,
        theta=theta,
        phi=phi,
        areas=areas,
        electric_interior_initial=np.stack([state.electric_interior for state in states]),
        magnetic_interior_initial=np.stack([state.magnetic_interior for state in states]),
        electric_exterior_target=np.stack([state.electric_exterior for state in states]),
        magnetic_exterior_target=np.stack([state.magnetic_exterior for state in states]),
    )
    print(f"Experiment: {output_dir}", flush=True)

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(
            history,
            fieldnames=(
                "iteration",
                "transformation",
                "case",
                "extraction_max_relative_residual",
                "exterior_relative_field_error",
                "interior_relative_update",
            ),
        )
        writer.writeheader()
        for iteration in range(iterations):
            directory = output_dir / f"iteration_{iteration:03d}"
            directory.mkdir()

            # 1. Extract a shared material from the two current interior states.
            extraction = extract_and_export(states, configuration, directory)
            # 2. Solve the original incident excitations; neither RHS is updated.
            solve_currents(configuration, directory, solver_executable)
            # 3. Sample physical total fields for both excitation cases.
            interior_fields, exterior_fields = [], []
            for state_index, (state_id, case_id) in enumerate(state_cases):
                interior, exterior = sample_total_fields(
                    configuration,
                    cases[case_id],
                    directory,
                    centroids,
                    radii,
                    theta,
                    phi,
                    offset_m,
                    sampler_executable,
                )
                interior_fields.append(interior)
                exterior_fields.append(exterior)
                exterior_error = relative_field_error(
                    exterior, exterior_checks[state_index], configuration.case.eta_exterior, areas
                )
                interior_update = relative_field_error(
                    interior,
                    (states[state_index].electric_interior, states[state_index].magnetic_interior),
                    configuration.case.eta_interior,
                    areas,
                )
                writer.writerow(
                    {
                        "iteration": iteration,
                        "transformation": state_id,
                        "case": case_id,
                        "extraction_max_relative_residual": float(
                            extraction.state_relative_residual[state_index].max()
                        ),
                        "exterior_relative_field_error": exterior_error,
                        "interior_relative_update": interior_update,
                    }
                )
                print(
                    f"Iteration {iteration}: {case_id}: exterior error={exterior_error:.3e}; "
                    f"interior update={interior_update:.3e}",
                    flush=True,
                )
            history.flush()
            np.savez(
                directory / "fields.npz",
                electric_interior_used=np.stack([state.electric_interior for state in states]),
                magnetic_interior_used=np.stack([state.magnetic_interior for state in states]),
                electric_interior_computed=np.stack([field[0] for field in interior_fields]),
                magnetic_interior_computed=np.stack([field[1] for field in interior_fields]),
                electric_exterior_computed=np.stack([field[0] for field in exterior_fields]),
                magnetic_exterior_computed=np.stack([field[1] for field in exterior_fields]),
                electric_exterior_check_target=np.stack([field[0] for field in exterior_checks]),
                magnetic_exterior_check_target=np.stack([field[1] for field in exterior_checks]),
            )
            # 4. Feed back interior total fields only, without rescaling the targets.
            states = update_interior_states(states, interior_fields, damping)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="New or empty experimental directory")
    args = parser.parse_args()
    output = run_experiment(load_workflow_configuration(args.config), args.output_dir)
    print(f"Finished fixed-count experiment. History: {output / 'history.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
