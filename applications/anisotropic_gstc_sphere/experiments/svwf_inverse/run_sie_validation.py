"""Run the existing SIE workflow with a saved inverse experiment's surface.

Reconstruct physical parameters from saved angular coefficients at mesh-face
directions. Do not re-extract parameters, optimize, or solve another SVWF problem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from ...sie_gstc_sphere import DEFAULT_EXECUTABLE, run_sie_workflow
    from .inverse_synthesis import load_parameter_evaluator
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from applications.anisotropic_gstc_sphere.experiments.svwf_inverse.inverse_synthesis import (
        load_parameter_evaluator,
    )
    from applications.anisotropic_gstc_sphere.sie_gstc_sphere import (
        DEFAULT_EXECUTABLE,
        run_sie_workflow,
    )


def run_sie_validation(
    experiment_dir, *, run_dir=None, executable=DEFAULT_EXECUTABLE, prepare_only=False,
):
    """Supply saved parameters to the shared SIE workflow; preserve synthesis data."""
    experiment_dir = Path(experiment_dir).resolve()
    configuration, evaluator, source = load_parameter_evaluator(experiment_dir)
    return run_sie_workflow(
        configuration,
        run_dir=Path(run_dir) if run_dir is not None else experiment_dir / "sie_validation",
        executable=Path(executable),
        use_existing_mesh=configuration.mesh_path.is_file(),
        prepare_only=prepare_only,
        parameter_evaluator=evaluator,
        parameter_source=source,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="New or empty SIE output directory")
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Export tensors without running C++"
    )
    args = parser.parse_args()
    try:
        manifest = run_sie_validation(
            args.experiment_dir, run_dir=args.run_dir, executable=args.executable,
            prepare_only=args.prepare_only,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"{'Prepared' if args.prepare_only else 'Completed'} SIE validation: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
