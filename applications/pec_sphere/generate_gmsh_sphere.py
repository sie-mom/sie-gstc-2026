#!/usr/bin/env python3
"""Generate sphere mesh for dielectric_sphere using in-script parameters.

This script is intentionally NOT a CLI.
Edit the physical constants below (freq, radius factor, ppw, output path)
and run it directly.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sie_mom.mesh.mesh_generators import CommonMeshOptions, SphereGenerator, SphereParams


def main() -> int:
    # --- Physical Parameters (edit these for this application) ---
    c0 = 299792458.0
    freq = 1e9  # Hz
    lambda0 = c0 / freq

    # Sphere size for this run
    radius = 2.0 * lambda0

    # Mesh resolution helper: h = lambda0 / ppw (typical CEM choice)
    ppw = 10.0
    mesh_size = lambda0 / ppw

    # Output configuration for this application
    output_obj = REPO_ROOT / "applications/pec_sphere/data/sphere_gmsh.obj"
    output_obj.parent.mkdir(parents=True, exist_ok=True)

    # Set True only when you want to inspect the mesh interactively
    open_gui = False

    # Optional: run mesh health checks immediately after generation.
    # Keep this True for day-to-day workflow safety.
    run_health_check = True

    print("--- dielectric_sphere mesh generation ---")
    print(f"freq      : {freq:.6e} Hz")
    print(f"lambda0   : {lambda0:.6e} m")
    print(f"radius    : {radius:.6e} m")
    print(f"mesh_size : {mesh_size:.6e} m (ppw={ppw:g})")
    print(f"output    : {output_obj}")

    options = CommonMeshOptions(
        output_obj=str(output_obj),
        mesh_size_min=0.8 * mesh_size,
        mesh_size_max=mesh_size,
        headless=not open_gui,
        model_name="dielectric_sphere",
    )

    SphereGenerator(SphereParams(radius=radius), options).generate()

    if run_health_check:
        tool = REPO_ROOT / "build/applications/mesh_apps/mesh_obj_loader_tool"
        if tool.exists():
            print("Running mesh health check...")
            result = subprocess.run([str(tool), str(output_obj)], check=False)
            if result.returncode != 0:
                print("Mesh health check failed.")
                return result.returncode
        else:
            print(
                "mesh_obj_loader_tool not found. Build it with: "
                "cmake --build build --target mesh_obj_loader_tool"
            )

    print("Mesh generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
