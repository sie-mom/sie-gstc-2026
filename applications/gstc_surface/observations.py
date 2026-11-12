"""Prepare mesh-classified volume observation grids without solving currents."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sie_mom.mesh.obj_mesh import load_triangle_obj
from sie_mom.mesh.point_classification import MeshPointClassifier

from .artifacts import read_observations as read_observations
from .sie_gstc_surface import _sha256, _write_json, load_solved_run
from .workflow import load_workflow_configuration


def sampling_points(sampling):
    """Cartesian ZX grid and origin-based radial ray, with distances in metres."""
    x = np.linspace(*sampling["zx_x_limits_m"], sampling["zx_grid_size"])
    z = np.linspace(*sampling["zx_z_limits_m"], sampling["zx_grid_size"])
    xx, zz = np.meshgrid(x, z)
    theta, phi = np.deg2rad([sampling["radial_theta_degrees"], sampling["radial_phi_degrees"]])
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    radial = np.linspace(
        sampling["radial_min_m"], sampling["radial_max_m"], sampling["radial_points"]
    )
    return {
        "zx": np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel())),
        "radial": radial[:, None] * direction,
    }


def prepare_observations(*, solve_dir, output_dir=None, sampling_config=None, backend="numpy"):
    """Read the solve's mesh snapshot and save reusable point/region CSVs."""
    solve_dir = Path(solve_dir).resolve()
    configuration, solve, mesh_path = load_solved_run(solve_dir)
    sampling = load_workflow_configuration(sampling_config or solve_dir / "configuration.json").raw[
        "sampling"
    ]
    grids = sampling_points(sampling)
    classifier = MeshPointClassifier(load_triangle_obj(mesh_path), backend=backend)
    output_dir = Path(output_dir or solve_dir / "observations/run01").resolve()
    if output_dir.exists():
        raise FileExistsError(f"observation directory already exists: {output_dir}")
    # Complete classification before creating an output directory.
    classified = {}
    for name, points in grids.items():
        start = time.perf_counter()
        regions = classifier.classify(points)
        classified[name] = (regions, time.perf_counter() - start)
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "artifact": "gstc_surface_observations",
        "solve_directory": str(solve_dir),
        "mesh_sha256": solve["mesh"]["sha256"],
        "backend": backend,
        "regions": {"1": "interior", "2": "exterior"},
        "coordinate_unit": "m",
        "sampling": sampling,
        "grids": {},
    }
    import trimesh

    manifest["trimesh_version"] = trimesh.__version__
    for name, points in grids.items():
        regions, seconds = classified[name]
        path = output_dir / f"observations_{name}.csv"
        np.savetxt(
            path,
            np.column_stack((np.arange(len(points)), points, regions)),
            delimiter=",",
            header="point_index,x,y,z,region",
            comments="",
            fmt=["%d", "%.17g", "%.17g", "%.17g", "%d"],
        )
        analytic = configuration.geometry.contains(points)
        manifest["grids"][name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "point_count": len(points),
            "interior_count": int(np.count_nonzero(regions == 1)),
            "exterior_count": int(np.count_nonzero(regions == 2)),
            "classification_seconds": seconds,
            "analytic_mesh_disagreement_count": int(np.count_nonzero(analytic != (regions == 1))),
        }
        print(f"{name}: {len(points)} points, classified in {seconds:.3f} s", flush=True)
    _write_json(output_dir / "observations.json", manifest)
    return output_dir / "observations.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--sampling-config",
        type=Path,
        help="Use only this configuration's sampling section; currents are unchanged",
    )
    parser.add_argument("--backend", choices=("numpy", "embree"), default="numpy")
    args = parser.parse_args()
    print(prepare_observations(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
