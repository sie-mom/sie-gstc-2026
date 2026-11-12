"""Prepare radial/ZX total-field CSVs and plotting metadata from saved currents.

Read the parent experiment's configuration snapshot and a selected iteration's
physical currents. Evaluate surface integrals and add the unchanged regional
incident fields, without extracting parameters or running a dense solve.
Outputs follow the existing plot_fields.py contract; feedback data stay intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from sie_mom.mesh.obj_mesh import load_triangle_obj, validate_triangle_mesh

if __package__:
    from ..workflow import load_workflow_configuration, write_json
    from . import iterate_lossless as feedback
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import iterate_lossless as feedback
    from workflow import load_workflow_configuration, write_json


def observation_points(configuration):
    """Return ZX-then-radial Cartesian points using the saved sampling settings."""
    sampling = configuration.sampling
    radius = configuration.case.radius_m
    axis = np.linspace(
        -sampling.zx_span_radii * radius, sampling.zx_span_radii * radius, sampling.zx_grid_size
    )
    x, z = np.meshgrid(axis, axis)
    zx = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    theta, phi = np.deg2rad([sampling.radial_theta_degrees, sampling.radial_phi_degrees])
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    radii = radius * np.linspace(
        sampling.radial_min_radii, sampling.radial_max_radii, sampling.radial_points
    )
    return np.concatenate((zx, radii[:, None] * direction)), len(zx)


def spherical_to_cartesian(field, theta, phi):
    """Convert component-first complex (r, theta, phi) fields to XYZ."""
    r, t, p = field
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    return np.stack(
        (r * st * cp + t * ct * cp - p * sp, r * st * sp + t * ct * sp + p * cp, r * ct - t * st)
    )


def read_point_integrals(path, points):
    """Read ordered finite Cartesian Q_E/Q_H at the requested observation points."""
    names = ["point_index", "x", "y", "z"] + [
        f"Q{kind}{axis}_{part}" for kind in "EH" for axis in "xyz" for part in ("real", "imag")
    ]
    with path.open(encoding="utf-8") as stream:
        header = next(csv.reader(stream), [])
        if len(set(header)) != len(header) or not set(names).issubset(header):
            raise ValueError(f"missing or duplicate integral CSV columns: {path}")
        stream.seek(0)
        data = np.genfromtxt(stream, delimiter=",", names=True, ndmin=1)
    if len(data) != len(points) or not np.array_equal(data["point_index"], np.arange(len(points))):
        raise ValueError("sampled integrals do not preserve observation point order")
    if not all(np.isfinite(data[name]).all() for name in names):
        raise ValueError("sampled integral contributions must be finite")
    if not np.allclose(
        np.column_stack([data[axis] for axis in "xyz"]), points, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("sampled coordinates do not match the requested plot points")
    return tuple(
        np.stack([data[f"Q{kind}{axis}_real"] + 1j * data[f"Q{kind}{axis}_imag"] for axis in "xyz"])
        for kind in "EH"
    )


def total_fields(configuration, simulation_case, points, integrals):
    """Select each physical region and form total E/H = incident - Q, without scaling."""
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii == 0):
        raise ValueError("plot sampling must not include the sphere center")
    theta = np.arccos(np.clip(points[:, 2] / radii, -1, 1))
    phi = np.arctan2(points[:, 1], points[:, 0])
    regions = np.where(radii < configuration.case.radius_m, 1, 2)
    totals = [np.empty((3, len(points)), dtype=complex) for _ in "EH"]
    for region_id, region in ((1, "interior"), (2, "exterior")):
        indices = np.flatnonzero(regions == region_id)
        if not len(indices):
            continue
        incident = feedback.incident_fields(
            configuration, simulation_case, region, radii[indices], theta[indices], phi[indices]
        )
        for target, source, integral in zip(totals, incident, integrals[region]):
            target[:, indices] = (
                spherical_to_cartesian(source, theta[indices], phi[indices]) - integral[:, indices]
            )
    if not all(np.isfinite(field).all() for field in totals):
        raise ValueError(
            "total fields are non-finite; check for samples at incident dipole sources"
        )
    return regions, tuple(totals)


def write_field_csv(path, points, regions, fields):
    """Save Cartesian complex E/H in the existing plotter's physical-field format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["point_index", "x", "y", "z", "region"]
            + [
                f"{kind}{axis}_{part}"
                for kind in "EH"
                for axis in "xyz"
                for part in ("real", "imag")
            ]
        )
        for index, point in enumerate(points):
            values = np.concatenate([field[:, index] for field in fields])
            writer.writerow(
                [
                    index,
                    *point,
                    int(regions[index]),
                    *[part for value in values for part in (value.real, value.imag)],
                ]
            )


def validate_saved_inputs(configuration, iteration_dir):
    """Check the original mesh/face ordering and four-block current files before sampling."""
    with (iteration_dir / "batch_metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if (
        metrics.get("artifact") != "anisotropic_sphere_app_batch"
        or metrics.get("schema_version") != 2
    ):
        raise ValueError("expected saved anisotropic batch solver metrics")
    if metrics["case_ids"] != [case.id for case in configuration.simulation.cases]:
        raise ValueError("saved solver cases disagree with the experiment configuration")
    mesh = load_triangle_obj(configuration.mesh_path)
    report = validate_triangle_mesh(mesh)
    report.require_valid(expected_euler_characteristic=2)
    for name in ("vertex_count", "face_count", "edge_count"):
        if getattr(report, name) != metrics["mesh"][name]:
            raise ValueError(f"mesh {name} differs from the saved solver mesh")
    with np.load(iteration_dir.parent / "geometry_and_targets.npz") as geometry:
        centroids = mesh.vertices[mesh.faces].mean(axis=1)
        if centroids.shape != geometry["centroids"].shape or not np.allclose(
            centroids, geometry["centroids"], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("mesh centroids or face order differ from the saved experiment")
    for case in configuration.simulation.cases:
        currents = iteration_dir / "cases" / case.id / "currents.bin"
        if currents.stat().st_size != 4 * report.edge_count * np.dtype("<c16").itemsize:
            raise ValueError(f"{currents} must contain four complex128 RWG blocks")


def plot_manifest(configuration, iteration_dir):
    """Describe post-processed iteration fields without claiming another solver run."""
    resolved = configuration.to_dict()
    for case in resolved["simulation"]["cases"]:
        prefix = f"cases/{case['id']}"
        case.update(
            status="pending",
            files={
                "currents": f"{prefix}/currents.bin",
                "solver_metrics": f"{prefix}/solver_metrics.json",
                "computed_radial": f"{prefix}/fields/sie_physical_radial.csv",
                "computed_zx": f"{prefix}/fields/sie_physical_zx.csv",
                "plots": f"{prefix}/plots",
            },
        )
    return {
        "schema_version": 2,
        "application": "anisotropic_gstc_sphere",
        "artifact": "lossless_feedback_iteration_plot_data",
        "backend": "sie",
        "case_id": configuration.case_id,
        "iteration": iteration_dir.name,
        "status": "preparing",
        "phasor_convention": "exp(+j omega t)",
        "frequency_hz": configuration.case.frequency_hz,
        "radius_m": configuration.case.radius_m,
        "simulation": resolved["simulation"],
        "sampling": asdict(configuration.sampling),
        "field_data": {
            "coordinate_system": "Cartesian",
            "coordinate_unit": "m",
            "electric_field_unit": "V/m",
            "magnetic_field_unit": "A/m",
            "regions": {"1": "interior", "2": "exterior"},
            "physical_construction": "incident - Q; interior for r<a, exterior for r>=a",
        },
        "post_processing": {
            "source": "saved physical currents; no extraction or dense solve",
            "reference_fields": "original configured transformations, not updated feedback states",
        },
        "files": {"configuration": "configuration.json", "batch_metrics": "batch_metrics.json"},
    }


def prepare_plot_fields(iteration_dir, *, sampler_executable=feedback.SAMPLER_EXECUTABLE):
    """Evaluate saved currents for every case and make this iteration readable by plot_fields.py."""
    iteration_dir = Path(iteration_dir).resolve()
    sampler_executable = Path(sampler_executable).resolve()
    with (iteration_dir.parent / "experiment.json").open(encoding="utf-8") as stream:
        snapshot = json.load(stream)["configuration"]
    # Reuse the existing validator without consulting the original mutable JSON.
    with TemporaryDirectory(prefix="sie-mom-plot-config-") as temporary:
        config_path = Path(temporary) / "configuration.json"
        write_json(config_path, snapshot)
        configuration = load_workflow_configuration(config_path)
    validate_saved_inputs(configuration, iteration_dir)
    if not sampler_executable.is_file():
        raise FileNotFoundError(
            "build the sampler: cmake --build build --target anisotropic_surface_sampler -j2"
        )

    # 1. Recreate the saved plotting grids, independent of the feedback samples.
    points, zx_count = observation_points(configuration)
    points_path = iteration_dir / "plot_points.bin"
    points.astype("<f8").tofile(points_path)
    write_json(iteration_dir / "configuration.json", configuration.to_dict())
    manifest = plot_manifest(configuration, iteration_dir)
    manifest_path = iteration_dir / "run.json"
    write_json(manifest_path, manifest)
    physical = configuration.case
    try:
        for case, saved_case in zip(
            configuration.simulation.cases, manifest["simulation"]["cases"]
        ):
            case_dir = iteration_dir / "cases" / case.id
            output = case_dir / "plot_samples"
            command = [
                str(sampler_executable),
                str(configuration.mesh_path.resolve()),
                str(case_dir / "currents.bin"),
                str(output),
                str(physical.frequency_hz),
                str(physical.relative_permittivity_interior),
                str(physical.relative_permeability_interior),
                str(physical.relative_permittivity_exterior),
                str(physical.relative_permeability_exterior),
                str(configuration.sie.quadrature_rule),
                "--points",
                str(points_path),
            ]
            print(f"Sampling saved currents: {case.id} ({len(points)} points)", flush=True)
            # 2. Evaluate Q only, then construct the selected regional total fields.
            feedback.run_logged(command, case_dir / "plot_sampling.log")
            integrals = {
                region: read_point_integrals(output / f"{region}_integrals.csv", points)
                for region in ("interior", "exterior")
            }
            regions, fields = total_fields(configuration, case, points, integrals)
            for cut, selected in (("zx", slice(0, zx_count)), ("radial", slice(zx_count, None))):
                write_field_csv(
                    iteration_dir / saved_case["files"][f"computed_{cut}"],
                    points[selected],
                    regions[selected],
                    tuple(field[:, selected] for field in fields),
                )
            saved_case.update(status="completed", sampling_command=command)
            write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        write_json(manifest_path, manifest)
        raise
    # 3. The unchanged plotter can now read this iteration as its run directory.
    manifest["status"] = "completed"
    write_json(manifest_path, manifest)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = prepare_plot_fields(arguments.iteration_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.error(str(error))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
