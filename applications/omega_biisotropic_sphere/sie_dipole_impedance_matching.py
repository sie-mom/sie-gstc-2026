#!/usr/bin/env python3
"""Run the complete SIE validation for the spherical dipole-matching case."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

try:
    from .dipole_impedance_matching import prescribed_total_fields_cartesian
    from .plot_dipole_impedance_matching import create_plots
    from .prepare_sie_inputs import prepare_sie_inputs
    from .workflow import (
        APPLICATION_DIR,
        WorkflowConfiguration,
        default_run_directory,
        load_field_csv,
        load_workflow_configuration,
        write_field_csv,
    )
except ImportError:  # Support direct execution from the repository root.
    from dipole_impedance_matching import (  # type: ignore[no-redef]
        prescribed_total_fields_cartesian,
    )
    from plot_dipole_impedance_matching import create_plots  # type: ignore[no-redef]
    from prepare_sie_inputs import prepare_sie_inputs  # type: ignore[no-redef]
    from workflow import (  # type: ignore[no-redef]
        APPLICATION_DIR,
        WorkflowConfiguration,
        default_run_directory,
        load_field_csv,
        load_workflow_configuration,
        write_field_csv,
    )

DEFAULT_CONFIGURATION = (
    APPLICATION_DIR / "configurations" / "budhu_10ghz_eps7.json"
)
DEFAULT_EXECUTABLE = (
    APPLICATION_DIR.parents[1]
    / "build"
    / "applications"
    / "omega_biisotropic_sphere"
    / "omega_sphere_sie"
)


def _application_path(path: Path) -> Path:
    return path if path.is_absolute() else APPLICATION_DIR / path


def _write_prescribed_reference(
    computed_path: Path,
    reference_path: Path,
    configuration: WorkflowConfiguration,
) -> None:
    computed = load_field_csv(computed_path)
    electric, magnetic = prescribed_total_fields_cartesian(
        computed["x"], computed["y"], computed["z"], configuration.case
    )
    expected_region = np.where(
        np.sqrt(computed["x"] ** 2 + computed["y"] ** 2 + computed["z"] ** 2)
        < configuration.case.radius,
        1,
        2,
    )
    np.testing.assert_array_equal(computed["region"], expected_region)
    write_field_csv(
        reference_path,
        computed["x"],
        computed["y"],
        computed["z"],
        computed["region"],
        electric,
        magnetic,
    )


def _write_run_manifest(
    run_dir: Path,
    configuration: WorkflowConfiguration,
) -> Path:
    case = configuration.case
    sampling = configuration.sampling
    configuration_path = run_dir / "configuration.json"
    with configuration_path.open("w", encoding="utf-8") as stream:
        json.dump(configuration.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")

    manifest = {
        "schema_version": 1,
        "case": "dipole_impedance_matching",
        "case_id": configuration.case_id,
        "backend": "sie",
        "phasor_convention": "exp(+j omega t)",
        "field_data": {
            "coordinate_system": "Cartesian",
            "coordinate_unit": "m",
            "electric_field_unit": "V/m",
            "magnetic_field_unit": "A/m",
            "regions": {"1": "interior", "2": "exterior"},
            "physical_construction": (
                "interior representation for r<a; exterior representation for r>a"
            ),
            "representation_sum": "diagnostic only; not a physical construction",
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
        "extraction_method": configuration.extraction_method,
        "cap_perturbation": configuration.to_dict().get("cap_perturbation"),
        "reference_label": configuration.reference_label,
        "solver": {
            "backend": "sie",
            "settings": {
                "method": configuration.sie.solver,
                "scaling": configuration.sie.scaling,
                "quadrature_rule": configuration.sie.quadrature_rule,
            },
        },
        "sampling": {
            "zx": {
                "shape": [sampling.zx_grid_size, sampling.zx_grid_size],
                "span_radii": sampling.zx_span_radii,
            },
            "radial": {
                "points": sampling.radial_points,
                "theta_degrees": sampling.radial_theta_degrees,
                "phi_degrees": sampling.radial_phi_degrees,
                "minimum_radius_over_a": sampling.radial_min_radii,
                "maximum_radius_over_a": sampling.radial_max_radii,
            },
        },
        "files": {
            "configuration": "configuration.json",
            "sie_inputs": "sie_inputs.json",
            "solver_metrics": "solver_metrics.json",
            "currents": "currents.bin",
            "reference_zx": "fields/prescribed_zx.csv",
            "computed_zx": "fields/sie_physical_zx.csv",
            "reference_radial": "fields/prescribed_radial.csv",
            "computed_radial": "fields/sie_physical_radial.csv",
            "interior_representation_zx": (
                "fields/sie_interior_representation_zx.csv"
            ),
            "exterior_representation_zx": (
                "fields/sie_exterior_representation_zx.csv"
            ),
            "interior_representation_radial": (
                "fields/sie_interior_representation_radial.csv"
            ),
            "exterior_representation_radial": (
                "fields/sie_exterior_representation_radial.csv"
            ),
        },
    }
    manifest_path = run_dir / "run.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest_path


def run_sie_validation(
    configuration: WorkflowConfiguration,
    *,
    run_dir: Path,
    executable: Path = DEFAULT_EXECUTABLE,
    use_existing_mesh: bool = False,
    reuse_inputs: bool = False,
    compute_residual: bool = True,
    create_figures: bool = True,
    svwf_run: Path | None = None,
) -> Path:
    """Prepare inputs, run the CUDA executable, and produce comparable outputs."""

    run_dir = run_dir.resolve()
    mesh_path = _application_path(Path(configuration.sie.mesh.path)).resolve()
    tensor_path = run_dir / "face_tensors.bin"
    input_manifest_path = run_dir / "sie_inputs.json"
    if reuse_inputs:
        for path in (mesh_path, tensor_path, input_manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"cannot reuse missing SIE input: {path}")
        with input_manifest_path.open(encoding="utf-8") as stream:
            saved_configuration = json.load(stream)["configuration"]
        requested_configuration = configuration.to_dict()
        for section in ("physical", "extraction_method", "cap_perturbation"):
            if saved_configuration.get(section) != requested_configuration.get(section):
                raise ValueError(
                    f"cannot reuse SIE tensors with different {section}; omit --reuse-inputs"
                )
    else:
        tensor_path, input_manifest_path = prepare_sie_inputs(
            configuration,
            run_dir=run_dir,
            use_existing_mesh=use_existing_mesh,
        )

    executable = executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"SIE executable does not exist: {executable}\n"
            "Build it with: cmake --build build --target omega_sphere_sie -j2"
        )

    case = configuration.case
    sampling = configuration.sampling
    command = [
        str(executable),
        "--mesh",
        str(mesh_path),
        "--tensors",
        str(tensor_path),
        "--output-dir",
        str(run_dir),
        "--frequency-hz",
        str(case.frequency),
        "--epsilon-r-interior",
        str(case.relative_permittivity_interior),
        "--mu-r-interior",
        str(case.relative_permeability_interior),
        "--epsilon-r-exterior",
        str(case.relative_permittivity_exterior),
        "--mu-r-exterior",
        str(case.relative_permeability_exterior),
        "--effective-moment-interior",
        str(case.effective_moment_interior),
        "--quadrature-rule",
        str(configuration.sie.quadrature_rule),
        "--zx-grid-size",
        str(sampling.zx_grid_size),
        "--zx-span-radii",
        str(sampling.zx_span_radii),
        "--radial-points",
        str(sampling.radial_points),
        "--radial-theta-degrees",
        str(sampling.radial_theta_degrees),
        "--radial-phi-degrees",
        str(sampling.radial_phi_degrees),
        "--radial-min-radii",
        str(sampling.radial_min_radii),
        "--radial-max-radii",
        str(sampling.radial_max_radii),
    ]
    if not compute_residual:
        command.append("--no-residual")
    print("Running omega sphere SIE")
    print(f"  command: {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)

    fields_dir = run_dir / "fields"
    _write_prescribed_reference(
        fields_dir / "sie_physical_zx.csv",
        fields_dir / "prescribed_zx.csv",
        configuration,
    )
    _write_prescribed_reference(
        fields_dir / "sie_physical_radial.csv",
        fields_dir / "prescribed_radial.csv",
        configuration,
    )
    manifest_path = _write_run_manifest(run_dir, configuration)
    if create_figures:
        create_plots(run_dir, comparison_run=svwf_run)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--use-existing-mesh",
        action="store_true",
        help="Validate and use the configured mesh instead of regenerating it",
    )
    parser.add_argument(
        "--reuse-inputs",
        action="store_true",
        help="Reuse existing mesh, face_tensors.bin, and sie_inputs.json",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Skip the costly post-solve residual reassembly",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--svwf-run",
        type=Path,
        help="Optional SVWF run to add to the analytical/SIE plots",
    )
    arguments = parser.parse_args()

    configuration = load_workflow_configuration(arguments.config)
    run_dir = arguments.run_dir or default_run_directory(
        configuration.case_id, "sie", configuration.extraction_method
    )
    manifest = run_sie_validation(
        configuration,
        run_dir=run_dir,
        executable=arguments.executable,
        use_existing_mesh=arguments.use_existing_mesh,
        reuse_inputs=arguments.reuse_inputs,
        compute_residual=not arguments.no_residual,
        create_figures=not arguments.no_plots,
        svwf_run=arguments.svwf_run,
    )
    print("Completed spherical omega SIE validation")
    print(f"  manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
