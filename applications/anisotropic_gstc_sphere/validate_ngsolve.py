#!/usr/bin/env python3
"""Validate the saved dual-polarization sphere with the shared NGSolve backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ngsolve as ng
import numpy as np
import scipy

from applications.anisotropic_gstc_sphere.workflow import load_workflow_configuration
from sie_mom.mesh.obj_mesh import load_triangle_obj
from sie_mom.validation.ngsolve import model as m
from sie_mom.validation.ngsolve.tensor_transfer import spherical_tangent_frame
from sie_mom.validation.ngsolve.volume_mesh import make_sphere_mesh

APPLICATION = Path(__file__).resolve().parent
REPOSITORY = APPLICATION.parents[1]
DEFAULT_INPUT = APPLICATION / "results/dipole_pol_rot/dual_pol_eps7"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(config_path, tensor_path):
    config = load_workflow_configuration(Path(config_path))
    if config.sie.basis != "spherical":
        raise ValueError("this example requires tensors in the spherical theta/phi basis")
    mesh = load_triangle_obj(config.mesh_path)
    if not np.allclose(
        np.linalg.norm(mesh.vertices, axis=1), config.case.radius_m, rtol=1e-4, atol=0
    ):
        raise ValueError("source OBJ vertices must lie on the configured sphere")
    nfaces = len(mesh.faces)
    size = Path(tensor_path).stat().st_size
    if size == nfaces * 32 * 16:
        dtype = "<c16"
    elif size == nfaces * 32 * 8:
        dtype = "<c8"
    else:
        raise ValueError("expected 32 complex numbers per OBJ face in face_tensors.bin")
    maps = np.fromfile(tensor_path, dtype=dtype).reshape(nfaces, 32)[:, :16].reshape(-1, 4, 4)
    maps = np.array([m.normalized_map(face_map) for face_map in maps])
    spread = float(np.max(np.abs(maps - maps[0])))
    if not np.isfinite(maps).all() or spread > 1e-9 * max(1, np.max(np.abs(maps))):
        raise ValueError("this first example supports spatially constant local tensors only")
    metadata = {
        "configuration": str(Path(config_path).resolve()),
        "configuration_sha256": sha256(config_path),
        "tensors": str(Path(tensor_path).resolve()),
        "tensors_sha256": sha256(tensor_path),
        "source_mesh": str(config.mesh_path),
        "source_mesh_sha256": sha256(config.mesh_path),
        "source_faces": nfaces,
        "normalized_tensor_max_spread": spread,
        "Z_condition_number": float(np.linalg.cond(maps[0, 2:, 2:])),
    }
    return config, maps[0], metadata


def incident_cf_sphere(config, case_id):
    cases = {case.id: case for case in config.simulation.cases}
    if case_id not in cases:
        raise ValueError(f"unknown case {case_id!r}; choose from {list(cases)}")
    e, h = ng.CF((0j, 0j, 0j)), ng.CF((0j, 0j, 0j))
    for excitation in cases[case_id].excitations:
        if excitation.region != "interior":
            raise ValueError("this example supports interior excitations only")
        definition = config.fields[excitation.field.field_id]
        if (
            getattr(definition, "dipole_type", None) not in ("electric", "magnetic")
            or np.linalg.norm(definition.source_position) != 0
        ):
            raise ValueError("this example requires centered dipoles")
        ei, hi = m.dipole_cf(config, excitation.field.field_id, "interior")
        e, h = e + ei, h + hi
    return e, h


def assemble_sphere(config, surface_map, mesh, order=2):
    admittance = ng.CF(tuple(m.sheet_admittance(surface_map).ravel()), dims=(4, 4))
    return m.assemble(
        config, mesh, frame=spherical_tangent_frame(mesh), admittance=admittance, order=order
    )


def sphere_quadrature(n=20):
    z, weights = np.polynomial.legendre.leggauss(n)
    phi = 2 * np.pi * (np.arange(2 * n) + 0.5) / (2 * n)
    z, phi = np.meshgrid(z, phi, indexing="ij")
    sin_theta = np.sqrt(1 - z * z)
    normal = np.stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), z), axis=-1).reshape(-1, 3)
    theta = np.stack((z * np.cos(phi), z * np.sin(phi), -sin_theta), axis=-1).reshape(-1, 3)
    azimuth = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1).reshape(-1, 3)
    weights = np.repeat(weights, 2 * n) * 2 * np.pi / (2 * n)
    return normal, theta, azimuth, weights


def relative_difference(value, reference):
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def diagnostics(system, fields, surface_map, air_lambda):
    """Independent volume-curl H samples; never substitute the enforced sheet law."""
    n, theta, phi, weights = sphere_quadrature()
    a = m.k0_of(system.config) * system.config.case.radius_m

    def sample(region, radius):
        return tuple(m.evaluate(cf, system.mesh, radius * n) for cf in fields[region])

    def tangential(vector):
        return np.stack((np.sum(vector * theta, axis=1), np.sum(vector * phi, axis=1)), axis=1)

    sheet_errors = []
    for offset in (0.02, 0.01, 0.005):
        em, hm = (tangential(v) for v in sample(0, a * (1 - offset)))
        ep, hp = (tangential(v) for v in sample(1, a * (1 + offset)))
        currents = np.hstack(((em + ep) / 2, (hm + hp) / 2)) @ surface_map.T
        j = (hp - hm) @ m.R.T
        magnetic = -(ep - em) @ m.R.T

        def weighted_norm(values):
            return float(np.sqrt(np.sum(weights[:, None] * np.abs(values) ** 2)))

        sheet_errors.append(
            {
                "offset_over_a": offset,
                "electric_jump_relative_residual": weighted_norm(j - currents[:, :2])
                / max(weighted_norm(j), weighted_norm(currents[:, :2]), 1e-30),
                "magnetic_jump_relative_residual": weighted_norm(magnetic - currents[:, 2:])
                / max(weighted_norm(magnetic), weighted_norm(currents[:, 2:]), 1e-30),
            }
        )
    powers = []
    for region, radius in ((0, 0.9 * a), (1, min(a + 0.2 * np.pi, 2 * np.pi * air_lambda - 0.1))):
        e, h = sample(region, radius)
        powers.append(
            float(
                0.5 * radius**2 * np.sum(weights * np.sum(np.cross(e, h.conj()).real * n, axis=1))
            )
        )
    # Near-to-far integral, with outward Love currents J=n x h and M=-n x e.
    radius = min(a + 0.2 * np.pi, 2 * np.pi * air_lambda - 0.1)
    e, h = sample(1, radius)
    j, magnetic = np.cross(n, h), -np.cross(n, e)
    observation_theta = np.linspace(0, np.pi, 181)
    directions = np.stack(
        (np.sin(observation_theta), np.zeros(181), np.cos(observation_theta)), axis=1
    )
    eps, mu = m.medium(system.config, "exterior")
    k, eta = np.sqrt(eps * mu), np.sqrt(mu / eps)
    phase_weights = np.exp(1j * k * radius * directions @ n.T) * (radius**2 * weights)[None, :]
    jf, mf = phase_weights @ j, phase_weights @ magnetic
    far = (
        -1j
        * k
        / (4 * np.pi)
        * (
            eta * (jf - directions * np.sum(directions * jf, axis=1)[:, None])
            - np.cross(directions, mf)
        )
    )
    return {
        "gstc_offset_probes": sheet_errors,
        "power_inner_normalized": powers[0],
        "power_outer_normalized": powers[1],
        "power_relative_imbalance": abs(powers[1] - powers[0])
        / max(abs(powers[0]), abs(powers[1]), 1e-30),
        "probe_note": "Finite-offset samples of E and volume-curl H; not exact boundary traces.",
    }, {
        "theta_rad": observation_theta,
        "direction": directions,
        "E_far_normalized": far,
        "huygens_radius_m": radius / m.k0_of(system.config),
    }


def sample_zx(system, fields, config, air_lambda, grid_size=None, span_radii=None):
    """Sample total E/H on COMSOL's square ZX grid, masking unavailable air points.

    The half-span is min(configured span*a, air radius), as in COMSOL. A radial
    PML leaves square corners outside physical air; these remain NaN, never zero.
    """
    n = config.sampling.zx_grid_size if grid_size is None else grid_size
    span = config.sampling.zx_span_radii if span_radii is None else span_radii
    if n < 2 or not np.isfinite(span) or span <= 0:
        raise ValueError("ZX grid needs at least two points and a positive finite half-span")
    k0 = m.k0_of(config)
    a = config.case.radius_m
    air_radius = 2 * np.pi * air_lambda / k0
    extent = min(span * a, air_radius)
    axis = np.linspace(-extent, extent, n)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    xyz = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    r = np.linalg.norm(xyz, axis=1)
    valid = (r < air_radius * (1 - 1e-8)) & (r > 1e-10 * a) & (np.abs(r - a) > 1e-3 * a)
    e = np.full((len(r), 3), np.nan + 1j * np.nan)
    h = np.full_like(e, np.nan + 1j * np.nan)
    interior = r < a
    for region, mask in ((0, valid & interior), (1, valid & ~interior)):
        if mask.any():
            e[mask], h[mask] = (
                m.evaluate(cf, system.mesh, xyz[mask] * k0) for cf in fields[region]
            )
    return {
        "xyz_m": xyz,
        "region": np.where(valid, np.where(interior, 1, 2), 0),
        "valid": valid,
        "E_cartesian": e * k0**3,
        "H_cartesian": h * k0**3 / m.ETA0,
        "grid_size": n,
        "extent_m": extent,
        "physical_air_radius_m": air_radius,
        "interface_exclusion_m": 1e-3 * a,
    }


def sample_radial(system, fields, config, case_id, air_lambda):
    sampling = config.to_dict()["sampling"]
    a = config.case.radius_m
    wavelength = 2 * np.pi / m.k0_of(config)
    radius = np.linspace(
        sampling["radial_min_radii"] * a,
        min(sampling["radial_max_radii"] * a, (air_lambda - 0.02) * wavelength),
        sampling["radial_points"],
    )
    radius = radius[np.abs(radius - a) > 1e-3 * a]
    theta, phi = np.deg2rad([sampling["radial_theta_degrees"], sampling["radial_phi_degrees"]])
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    xyz = radius[:, None] * direction
    interior = radius < a
    e, h = np.empty((len(radius), 3), complex), np.empty((len(radius), 3), complex)
    for index, mask in ((0, interior), (1, ~interior)):
        e[mask], h[mask] = (
            m.evaluate(cf, system.mesh, xyz[mask] * m.k0_of(config)) for cf in fields[index]
        )
    scale = m.k0_of(config) ** 3
    data = {
        "radius_m": radius,
        "xyz_m": xyz,
        "interior": interior,
        "E_cartesian": scale * e,
        "H_cartesian": scale / m.ETA0 * h,
    }
    metrics = {}
    case = next(c for c in config.simulation.cases if c.id == case_id)
    targets = [
        t
        for t in config.transformations
        if t.gstc.interior is not None
        and len(case.excitations) == 1
        and t.gstc.interior.field_id == case.excitations[0].field.field_id
    ]
    if len(targets) == 1:
        target = targets[0]
        et, ht = np.zeros_like(e), np.zeros_like(h)
        for region, mask in (("interior", interior), ("exterior", ~interior)):
            reference = getattr(target.gstc, region)
            if reference is not None:
                et[mask], ht[mask] = (
                    m.evaluate(cf, system.mesh, xyz[mask] * m.k0_of(config))
                    for cf in m.dipole_cf(config, reference.field_id, region)
                )
            metrics[f"target_{region}_E_radial_relative_difference"] = relative_difference(
                e[mask], et[mask]
            )
            metrics[f"target_{region}_H_radial_relative_difference"] = relative_difference(
                h[mask], ht[mask]
            )
        data.update(target_E_cartesian=et * scale, target_H_cartesian=ht * scale / m.ETA0)
        metrics["target_transformation"] = target.id
        metrics["target_note"] = (
            "Prescribed design fields, not an independently established exact solution."
        )
    return data, metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_INPUT / "configuration.json")
    p.add_argument("--tensors", type=Path, default=DEFAULT_INPUT / "face_tensors.bin")
    p.add_argument("--case-id", default="all", help="all or an ID from simulation.cases")
    p.add_argument("--order", type=int, default=3)
    p.add_argument(
        "--ppw", type=float, default=3, help="maximum volume edge size: regional wavelength / ppw"
    )
    p.add_argument(
        "--interior-ppw",
        type=float,
        help="optional core/sheet resolution override, keeping exterior ppw unchanged",
    )
    p.add_argument("--curve-order", type=int, default=3)
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="NGSolve task threads and runtime MKL PARDISO thread limit; UMFPACK differs",
    )
    p.add_argument("--inverse", choices=("auto", "umfpack", "pardiso"), default="auto")
    p.add_argument(
        "--pardiso-ordering",
        choices=("minimum_degree", "metis", "parallel_metis"),
        default="metis",
        help="PARDISO sparse ordering (ignored by UMFPACK)",
    )
    p.add_argument("--air-lambda", type=float, default=0.95)
    p.add_argument("--pml-lambda", type=float, default=0.25)
    p.add_argument("--pml-strength", type=float, default=3)
    p.add_argument(
        "--output-dir",
        type=Path,
        help="new or empty directory; defaults to a timestamped experiment result",
    )
    p.add_argument(
        "--vtk", action="store_true", help="also export FEM correction/exterior E for ParaView"
    )
    return p


def run(args):
    if args.order < 1 or args.threads < 1 or args.pml_lambda <= 0:
        raise ValueError("order, threads and PML thickness must be positive")
    config, sheet, metadata = load_benchmark(args.config, args.tensors)
    if args.air_lambda <= m.k0_of(config) * config.case.radius_m / (2 * np.pi) + 0.02:
        raise ValueError("air extent must leave more than 0.02 lambda0 beyond the sheet for probes")
    case_ids = [c.id for c in config.simulation.cases] if args.case_id == "all" else [args.case_id]
    sources = {case_id: incident_cf_sphere(config, case_id) for case_id in case_ids}
    solver = m.direct_solver(args.inverse)
    output = args.output_dir or (
        DEFAULT_INPUT
        / "experiments/ngsolve"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ng.SetNumThreads(args.threads)
    ng.ngsglobals.msg_level = 0
    report = {
        "inputs": metadata,
        "physical": config.to_dict()["physical"],
        "ngsolve": ng.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "command": sys.argv,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "inverse": solver,
        "inverse_options": {"ordering": args.pardiso_ordering} if solver == "pardiso" else {},
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "MKL_DYNAMIC": os.environ.get("MKL_DYNAMIC"),
        "status": "running",
        "cases": {},
    }

    def save():
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    (output / "configuration.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    np.savez(
        output / "sheet.npz", normalized_surface_map=sheet, admittance=m.sheet_admittance(sheet)
    )
    print(f"Output: {output}", flush=True)
    try:
        start = time.perf_counter()
        with ng.TaskManager():
            mesh = make_sphere_mesh(
                config,
                ppw=args.ppw,
                curve_order=args.curve_order,
                air_lambda=args.air_lambda,
                pml_lambda=args.pml_lambda,
                pml_strength=args.pml_strength,
                interior_ppw=args.interior_ppw,
            )
            report["mesh_seconds"] = time.perf_counter() - start
            mesh.ngmesh.Save(str(output / "mesh.vol.gz"))
            print(f"Mesh: {mesh.ne} tetrahedra, {report['mesh_seconds']:.2f} s", flush=True)
            system = assemble_sphere(config, sheet, mesh, order=args.order)
            report.update(
                tetrahedra=mesh.ne,
                dofs=system.space.ndof,
                free_dofs=sum(system.space.FreeDofs()),
                assembly_seconds=system.assembly_seconds,
            )
            print(
                f"Assembly: {system.space.ndof} DOFs, {system.assembly_seconds:.2f} s", flush=True
            )
            save()
            start = time.perf_counter()
            cpu_start = time.process_time()
            inverse = system.matrix.mat.Inverse(
                system.space.FreeDofs(), inverse=solver, flags=report["inverse_options"]
            )
            report["factorization_seconds"] = time.perf_counter() - start
            report["factorization_cpu_seconds"] = time.process_time() - cpu_start
            report["factorization_average_cpu_cores"] = (
                report["factorization_cpu_seconds"] / report["factorization_seconds"]
            )
            try:
                from threadpoolctl import threadpool_info

                report["threadpools"] = threadpool_info()
            except ImportError:
                pass
            print(
                f"Factorization ({solver}): {report['factorization_seconds']:.2f} s; "
                f"shared by {len(case_ids)} RHS",
                flush=True,
            )
            save()
            for case_id, (ei, hi) in sources.items():
                start = time.perf_counter()
                rhs = system.rhs(ei, hi)
                report["cases"][case_id] = result = {
                    "rhs_assembly_seconds": time.perf_counter() - start
                }
                solution = ng.GridFunction(system.space)
                start = time.perf_counter()
                solution.vec.data = inverse * rhs.vec
                result["backsolve_seconds"] = time.perf_counter() - start
                result["linear_relative_residual"] = m.residual(system, solution, rhs)
                if (
                    not np.isfinite(result["linear_relative_residual"])
                    or result["linear_relative_residual"] > 1e-8
                ):
                    raise RuntimeError(f"{case_id}: unacceptable linear residual")
                fields = m.total_fields(system, solution, ei, hi)
                diag, far = diagnostics(system, fields, sheet, args.air_lambda)
                radial, target = sample_radial(system, fields, config, case_id, args.air_lambda)
                result.update(diag, **target)
                directory = output / case_id
                directory.mkdir()
                solution.Save(str(directory / "solution.sol"))
                np.savez(directory / "radial_fields.npz", **radial)
                np.savez(directory / "far_field_phi0.npz", **far)
                sample_start = time.perf_counter()
                zx = sample_zx(system, fields, config, args.air_lambda)
                np.savez(directory / "zx_fields.npz", **zx)
                result["zx_sampling_seconds"] = time.perf_counter() - sample_start
                result["zx_valid_points"] = int(zx["valid"].sum())
                result["zx_grid_size"] = int(zx["grid_size"])
                if args.vtk:
                    ng.VTKOutput(
                        mesh,
                        coefs=[
                            solution.components[0].real,
                            solution.components[0].imag,
                            solution.components[1].real,
                            solution.components[1].imag,
                        ],
                        names=[
                            "E_correction_real",
                            "E_correction_imag",
                            "E_exterior_real",
                            "E_exterior_imag",
                        ],
                        filename=str(directory / "fields"),
                        subdivision=1,
                    ).Do()
                print(
                    f"{case_id}: linear residual {result['linear_relative_residual']:.2e}; "
                    f"offset GSTC residuals {diag['gstc_offset_probes'][-1]}",
                    flush=True,
                )
                save()
        report["status"] = "completed; spatial/PML convergence must be assessed separately"
        save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    print(
        "Completed. Inspect metrics.json and compare mesh/PML refinements "
        "before claiming accuracy.",
        flush=True,
    )
    return output


if __name__ == "__main__":
    run(parser().parse_args())
