#!/usr/bin/env python3
"""Generate the Gmsh sphere used by the omega-biisotropic SIE workflow."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from sie_mom.mesh.obj_mesh import (
    MeshValidationReport,
    load_triangle_obj,
    validate_triangle_mesh,
)

try:
    from .workflow import (
        APPLICATION_DIR,
        DipoleMatchingCase,
        WorkflowConfiguration,
        load_workflow_configuration,
    )
except ImportError:  # Support direct execution from the repository root.
    from workflow import (  # type: ignore[no-redef]
        APPLICATION_DIR,
        DipoleMatchingCase,
        WorkflowConfiguration,
        load_workflow_configuration,
    )

REPOSITORY_ROOT = APPLICATION_DIR.parents[1]
DEFAULT_CONFIGURATION = APPLICATION_DIR / "configurations" / "budhu_10ghz_eps7.json"


def shortest_wavelength(case: DipoleMatchingCase) -> float:
    """Return the shortest wavelength among the interior and exterior media."""

    return 2.0 * np.pi / max(case.k_interior, case.k_exterior)


def target_edge_length(case: DipoleMatchingCase, points_per_wavelength: float) -> float:
    """Return the Gmsh target edge length based on the shortest wavelength."""

    if points_per_wavelength <= 0.0:
        raise ValueError("points_per_wavelength must be positive")
    return shortest_wavelength(case) / points_per_wavelength


def _application_path(path: Path) -> Path:
    return path if path.is_absolute() else APPLICATION_DIR / path


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


def _print_dense_system_estimate(report: MeshValidationReport) -> None:
    faces = report.face_count
    edges = report.edge_count
    unknowns = 4 * edges
    matrix_gib = unknowns**2 * np.dtype(np.complex128).itemsize / 1024**3
    print("Estimated dense GSTC system:")
    print(f"  faces: {faces}")
    print(f"  edges: {edges}")
    print(f"  unknowns: {unknowns}")
    print(f"  complex-double matrix: {matrix_gib:.3f} GiB")
    print("  (solver workspace and other GPU arrays require additional memory)")


def generate_mesh(
    configuration: WorkflowConfiguration,
    *,
    output: Path | None = None,
    points_per_wavelength: float | None = None,
    gui: bool = False,
    run_health_check: bool = True,
) -> Path:
    """Generate one OBJ mesh from the shared physical configuration."""

    try:
        from sie_mom.mesh.mesh_generators import (
            CommonMeshOptions,
            SphereGenerator,
            SphereParams,
            SphericalPatchRefinement,
        )
    except ModuleNotFoundError as error:
        if error.name == "gmsh":
            raise RuntimeError(
                "Gmsh is required; install the project with the 'mesh' extra"
            ) from error
        raise

    ppw = (
        configuration.sie.mesh.points_per_wavelength
        if points_per_wavelength is None
        else points_per_wavelength
    )
    mesh_size = target_edge_length(configuration.case, ppw)
    patch_refinement = None
    minimum_mesh_size = 0.8 * mesh_size
    cap_mesh = configuration.sie.mesh.cap_refinement
    if cap_mesh is not None:
        cap = configuration.cap_perturbation
        if cap is None:  # Guard for programmatic callers bypassing validation.
            raise ValueError("cap mesh refinement requires cap_perturbation")
        local_mesh_size = target_edge_length(
            configuration.case, cap_mesh.points_per_wavelength
        )
        theta = np.deg2rad(cap.theta_degrees)
        phi = np.deg2rad(cap.phi_degrees)
        center = configuration.case.radius * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        def chord_radius(angle_degrees: float) -> float:
            return 2.0 * configuration.case.radius * np.sin(
                0.5 * np.deg2rad(angle_degrees)
            )

        patch_refinement = SphericalPatchRefinement(
            center=tuple(float(value) for value in center),
            inner_chord_radius=chord_radius(
                cap_mesh.fully_refined_radius_degrees
            ),
            outer_chord_radius=chord_radius(cap_mesh.transition_radius_degrees),
            mesh_size=local_mesh_size,
        )
        minimum_mesh_size = 0.8 * local_mesh_size
    output_path = _application_path(Path(configuration.sie.mesh.path) if output is None else output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Omega-biisotropic sphere mesh")
    print(f"  case_id: {configuration.case_id}")
    print(f"  radius: {configuration.case.radius:.9e} m")
    print(f"  shortest wavelength: {shortest_wavelength(configuration.case):.9e} m")
    print(f"  points per wavelength: {ppw:g}")
    print(f"  target edge length: {mesh_size:.9e} m")
    if cap_mesh is not None and patch_refinement is not None:
        print("  cap-local refinement:")
        print(f"    local points per wavelength: {cap_mesh.points_per_wavelength:g}")
        print(f"    local target edge length: {patch_refinement.mesh_size:.9e} m")
        print(
            "    fully refined angular radius: "
            f"{cap_mesh.fully_refined_radius_degrees:g} deg"
        )
        print(
            "    transition angular radius: "
            f"{cap_mesh.transition_radius_degrees:g} deg"
        )
    print(f"  output: {output_path}")

    SphereGenerator(
        SphereParams(radius=configuration.case.radius),
        CommonMeshOptions(
            output_obj=str(output_path),
            mesh_size_min=minimum_mesh_size,
            mesh_size_max=mesh_size,
            headless=not gui,
            model_name=f"omega_{configuration.case_id}",
            write_step=False,
        ),
        patch_refinement=patch_refinement,
    ).generate()

    report = validate_sie_mesh(output_path)

    if run_health_check:
        health_tool = (
            REPOSITORY_ROOT / "build" / "applications" / "mesh_apps" / "mesh_obj_loader_tool"
        )
        if health_tool.is_file():
            subprocess.run([str(health_tool), str(output_path)], check=True)
        else:
            print(
                "Mesh-health tool not found; build target "
                "'mesh_obj_loader_tool' to run the C++ topology checks."
            )

    _print_dense_system_estimate(report)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIGURATION,
        help="Shared application configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Temporary mesh-path override; relative paths are application-relative",
    )
    parser.add_argument(
        "--ppw",
        type=float,
        help="Temporary points-per-shortest-wavelength override",
    )
    parser.add_argument("--gui", action="store_true", help="Open the Gmsh GUI")
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Do not run the C++ OBJ topology checker",
    )
    arguments = parser.parse_args()

    configuration = load_workflow_configuration(arguments.config)
    generate_mesh(
        configuration,
        output=arguments.output,
        points_per_wavelength=arguments.ppw,
        gui=arguments.gui,
        run_health_check=not arguments.no_health_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
