"""Centralized sphere mesh generation."""

import argparse
import subprocess
from pathlib import Path
import numpy as np

from sie_mom.mesh.obj_mesh import (
    MeshValidationReport,
    load_triangle_obj,
    validate_triangle_mesh,
)
from sie_mom.mesh.mesh_generators import (
    CommonMeshOptions,
    SphereGenerator,
    SphereParams,
)

def validate_sie_mesh(mesh_path: Path) -> MeshValidationReport:
    """Validate the unmodified OBJ used by tensor generation and the SIE."""
    report = validate_triangle_mesh(load_triangle_obj(mesh_path))
    report.require_valid(expected_euler_characteristic=2)
    print("Python mesh validation:")
    print(f"  vertices: {report.vertex_count}")
    print(f"  faces: {report.face_count}")
    print(f"  edges: {report.edge_count}")
    print(f"  orphan vertices: {report.orphan_vertices}")
    print(f"  Euler characteristic: {report.euler_characteristic}")
    print(f"  signed volume: {report.signed_volume:.9e} m^3")
    print("  status: healthy closed outward-oriented surface")
    return report

def generate_mesh(
    radius: float,
    target_edge_length: float,
    output_path: Path,
    gui: bool = False,
    run_health_check: bool = True,
    repo_root: Path | None = None
) -> Path:
    """Generate one OBJ mesh."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minimum_mesh_size = 0.8 * target_edge_length

    print("Generating sphere mesh")
    print(f"  radius: {radius:.9e} m")
    print(f"  target edge length: {target_edge_length:.9e} m")
    print(f"  output: {output_path}")

    SphereGenerator(
        SphereParams(radius=radius),
        CommonMeshOptions(
            output_obj=str(output_path),
            mesh_size_min=minimum_mesh_size,
            mesh_size_max=target_edge_length,
            headless=not gui,
            model_name="sphere_mesh",
            write_step=False,
        )
    ).generate()

    report = validate_sie_mesh(output_path)

    if run_health_check and repo_root is not None:
        health_tool = repo_root / "build" / "applications" / "mesh_apps" / "mesh_obj_loader_tool"
        if health_tool.is_file():
            subprocess.run([str(health_tool), str(output_path)], check=True)
        else:
            print("Mesh-health tool not found.")

    return output_path
