"""Run the configuration-driven spherical anisotropic GSTC SIE workflow."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from sie_mom.mesh.generate_sphere_mesh import generate_mesh

if __package__:
    from .prepare_sie_inputs import (
        ParameterEvaluator,
        export_sie_inputs,
        resolve_parameter_source,
    )
    from .workflow import (
        DEFAULT_CONFIGURATION,
        REPO_ROOT,
        DipoleFieldDefinition,
        PlaneWaveFieldDefinition,
        WorkflowConfiguration,
        create_run_directory,
        default_run_directory,
        load_workflow_configuration,
        write_json,
    )
else:
    from prepare_sie_inputs import ParameterEvaluator, export_sie_inputs, resolve_parameter_source
    from workflow import (
        DEFAULT_CONFIGURATION,
        REPO_ROOT,
        DipoleFieldDefinition,
        PlaneWaveFieldDefinition,
        WorkflowConfiguration,
        create_run_directory,
        default_run_directory,
        load_workflow_configuration,
        write_json,
    )

DEFAULT_EXECUTABLE = (
    REPO_ROOT / "build" / "applications" / "anisotropic_gstc_sphere" / "anisotropic_sphere_app"
)


def _solver_cases_arguments(configuration: WorkflowConfiguration) -> list[str]:
    """Resolve independent incident fields to the real-valued C++ source types."""

    if not configuration.simulation.cases:
        raise ValueError("C++ batch execution requires at least one simulation case")
    arguments = []
    for simulation_case in configuration.simulation.cases:
        arguments.extend(("--case", simulation_case.id))
        for excitation in simulation_case.excitations:
            field_id = excitation.field.field_id
            definition = configuration.fields[field_id]
            if isinstance(definition, PlaneWaveFieldDefinition):
                amplitude = definition.electric_amplitude
                prefix = ["--plane-wave", excitation.region]
                parameters = definition.propagation_direction
            elif isinstance(definition, DipoleFieldDefinition):
                amplitude = definition.effective_moment
                prefix = ["--dipole", excitation.region, definition.dipole_type]
                parameters = definition.source_position
            else:
                raise ValueError(f"unsupported C++ incident field type: {field_id}")
            if any(component.imag != 0 for component in amplitude):
                raise ValueError(
                    f"C++ incident field {field_id} requires real amplitude components; "
                    "use --prepare-only for complex incident fields"
                )
            if not any(component.real != 0 for component in amplitude):
                raise ValueError(f"C++ incident field {field_id} must have a nonzero amplitude")
            arguments.extend(prefix)
            arguments.extend(str(component.real) for component in amplitude)
            arguments.extend(str(value) for value in parameters)
    return arguments


def solver_command(
    configuration: WorkflowConfiguration,
    *,
    run_dir: Path,
    executable: Path,
) -> list[str]:
    """Translate shared settings and every simulation case to one C++ batch command."""

    case = configuration.case
    case_arguments = _solver_cases_arguments(configuration)
    command = [
        str(executable),
        "--mesh",
        str(configuration.mesh_path.resolve()),
        "--tensors",
        str(run_dir / "face_tensors.bin"),
        "--output-dir",
        str(run_dir),
        "--frequency-hz",
        str(case.frequency_hz),
        "--epsilon-r-interior",
        str(case.relative_permittivity_interior),
        "--mu-r-interior",
        str(case.relative_permeability_interior),
        "--epsilon-r-exterior",
        str(case.relative_permittivity_exterior),
        "--mu-r-exterior",
        str(case.relative_permeability_exterior),
        "--quadrature-rule",
        str(configuration.sie.quadrature_rule),
        "--singularity-threshold-lambda0",
        str(configuration.sie.singularity_threshold_lambda0),
        "--basis",
        configuration.sie.basis,
    ]
    command.extend(case_arguments)
    for name, value in asdict(configuration.sampling).items():
        command.extend(("--" + name.replace("_", "-"), str(value)))
    if not configuration.sie.compute_residual:
        command.append("--no-residual")
    if not configuration.sie.evaluate_fields:
        command.append("--no-fields")
    return command


def _run_manifest(
    configuration: WorkflowConfiguration, command: list[str] | None
) -> dict[str, Any]:
    """Record shared inputs and independent case outputs, relative to the run root."""

    resolved = configuration.to_dict()
    simulation = resolved["simulation"]
    for simulation_case in simulation["cases"]:
        output_dir = f"cases/{simulation_case['id']}"
        files = {
            "solver_metrics": f"{output_dir}/solver_metrics.json",
            "currents": f"{output_dir}/currents.bin",
            "plots": f"{output_dir}/plots",
        }
        if configuration.sie.evaluate_fields:
            for cut in ("zx", "radial"):
                files[f"computed_{cut}"] = f"{output_dir}/fields/sie_physical_{cut}.csv"
                for region in ("interior", "exterior"):
                    files[f"{region}_representation_{cut}"] = (
                        f"{output_dir}/fields/sie_{region}_representation_{cut}.csv"
                    )
        simulation_case.update(status="pending", command=command, files=files)
    return {
        "schema_version": 2,
        "application": "anisotropic_gstc_sphere",
        "case_id": configuration.case_id,
        "backend": "sie",
        "status": "preparing",
        "phasor_convention": "exp(+j omega t)",
        "tangent_basis": "theta, phi; outward radial normal",
        "frequency_hz": configuration.case.frequency_hz,
        "radius_m": configuration.case.radius_m,
        "fields": resolved["fields"],
        "transformations": resolved["transformations"],
        "simulation": simulation,
        "extraction_method": configuration.extraction_method,
        "parameter_source": {
            "method": "local_extraction", "extraction_method": configuration.extraction_method,
        },
        "tensor_structure": configuration.tensor_structure,
        "solver": {
            "method": "dense_lu_batch",
            "case_count": len(configuration.simulation.cases),
            "scaling": "eta0",
            "quadrature_rule": configuration.sie.quadrature_rule,
            "singularity_threshold_lambda0": configuration.sie.singularity_threshold_lambda0,
            "compute_residual": configuration.sie.compute_residual,
            "evaluate_fields": configuration.sie.evaluate_fields,
        },
        "sampling": asdict(configuration.sampling),
        "field_data": {
            "coordinate_system": "Cartesian",
            "coordinate_unit": "m",
            "electric_field_unit": "V/m",
            "magnetic_field_unit": "A/m",
            "regions": {"1": "interior", "2": "exterior"},
            "physical_construction": "interior for r<a; exterior for r>=a",
            "representation_sum": "diagnostic only; not the physical field",
        },
        "files": {
            "configuration": "configuration.json",
            "sie_inputs": "sie_inputs.json",
            "face_tensors": "face_tensors.bin",
            "batch_metrics": "batch_metrics.json",
        },
    }


def run_sie_workflow(
    configuration: WorkflowConfiguration,
    *,
    run_dir: Path,
    executable: Path = DEFAULT_EXECUTABLE,
    use_existing_mesh: bool = False,
    prepare_only: bool = False,
    parameter_evaluator: ParameterEvaluator | None = None,
    parameter_source: dict[str, Any] | None = None,
) -> Path:
    """Prepare one surface map and optionally solve all cases in one C++ batch.

    The default extracts local parameters from prescribed fields. An optional
    ``parameter_evaluator(theta, phi)`` supplies physical SurfaceParameters on
    mesh-face directions instead. Only preparation changes; run management,
    export, source arguments, and C++ execution are shared. ``parameter_source``
    records evaluator provenance, without changing the configuration schema.
    """

    source = resolve_parameter_source(configuration, parameter_evaluator, parameter_source)
    run_dir = run_dir.resolve()
    executable = executable.resolve()
    mesh_path = configuration.mesh_path.resolve()
    if use_existing_mesh:
        if not mesh_path.is_file():
            raise FileNotFoundError(f"configured mesh does not exist: {mesh_path}")
    elif mesh_path.exists():
        raise FileExistsError(f"mesh already exists: {mesh_path}; use --use-existing-mesh")
    if not prepare_only and not executable.is_file():
        raise FileNotFoundError(
            f"SIE executable does not exist: {executable}\n"
            "Build with: cmake --build build --target anisotropic_sphere_app -j2"
        )

    command = (
        None
        if prepare_only
        else solver_command(configuration, run_dir=run_dir, executable=executable)
    )
    create_run_directory(run_dir)
    write_json(run_dir / "configuration.json", configuration.to_dict())
    manifest = _run_manifest(configuration, command)
    manifest["parameter_source"] = source
    if parameter_evaluator is not None:
        manifest["extraction_method"] = None
    manifest_path = run_dir / "run.json"
    write_json(manifest_path, manifest)
    try:
        if not use_existing_mesh:
            print("=== 1. Meshing ===", flush=True)
            generate_mesh(
                configuration.case.radius_m,
                configuration.target_edge_length,
                mesh_path,
                repo_root=REPO_ROOT,
            )
        print("\n=== 2. Surface Parameter Preparation ===", flush=True)
        _, input_manifest = export_sie_inputs(
            configuration, mesh_path=mesh_path, run_dir=run_dir,
            parameter_evaluator=parameter_evaluator, parameter_source=parameter_source,
        )
        with input_manifest.open(encoding="utf-8") as stream:
            inputs = json.load(stream)
        manifest["mesh"] = inputs["mesh"]
        manifest["prescribed_state_ids"] = inputs["prescribed_state_ids"]
        manifest["parameter_source"] = inputs["parameter_source"]
        manifest["extraction_method"] = inputs["extraction_method"]
        manifest["status"] = "prepared"
        for simulation_case in manifest["simulation"]["cases"]:
            (run_dir / simulation_case["files"]["plots"]).mkdir(parents=True)
        write_json(manifest_path, manifest)
        if prepare_only:
            return manifest_path

        print("\n=== 3. C++ Solver ===", flush=True)
        print(f"Running: {shlex.join(command)}", flush=True)
        for simulation_case in manifest["simulation"]["cases"]:
            simulation_case["status"] = "running"
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        subprocess.run(command, check=True)
        batch_metrics = run_dir / manifest["files"]["batch_metrics"]
        if not batch_metrics.is_file():
            raise FileNotFoundError(f"solver did not produce required output: {batch_metrics}")
        for simulation_case in manifest["simulation"]["cases"]:
            for name, file_name in simulation_case["files"].items():
                path = run_dir / file_name
                if name != "plots" and not path.is_file():
                    raise FileNotFoundError(f"solver did not produce required output: {path}")
            simulation_case["status"] = "completed"
        manifest["status"] = "completed"
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        for simulation_case in manifest["simulation"]["cases"]:
            if simulation_case["status"] == "running":
                simulation_case["status"] = "failed"
                simulation_case["error"] = manifest["error"].copy()
        write_json(manifest_path, manifest)
        raise
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-dir", "--output-dir", dest="run_dir", type=Path)
    destination.add_argument("--run-name")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--singularity-threshold-lambda0",
        type=float,
        help="Override sie.singularity_threshold_lambda0 for this run",
    )
    parser.add_argument(
        "--basis",
        choices=["spherical", "face"],
        help="Override tangent basis for GSTC tensors ('spherical' or 'face')",
    )
    parser.add_argument("--use-existing-mesh", action="store_true")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Save inputs without running C++"
    )
    args = parser.parse_args()
    configuration = load_workflow_configuration(args.config)
    if args.singularity_threshold_lambda0 is not None:
        try:
            configuration = replace(
                configuration,
                sie=replace(
                    configuration.sie,
                    singularity_threshold_lambda0=args.singularity_threshold_lambda0,
                ),
            )
        except ValueError as error:
            parser.error(str(error))
    if args.basis is not None:
        configuration = replace(
            configuration,
            sie=replace(
                configuration.sie,
                basis=args.basis,
            ),
        )
    run_dir = args.run_dir or default_run_directory(
        configuration.case_id, args.run_name or configuration.run_name
    )
    manifest = run_sie_workflow(
        configuration,
        run_dir=run_dir,
        executable=args.executable,
        use_existing_mesh=args.use_existing_mesh,
        prepare_only=args.prepare_only,
    )
    print(f"{'Prepared' if args.prepare_only else 'Completed'} spherical GSTC SIE run")
    print(f"  manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
