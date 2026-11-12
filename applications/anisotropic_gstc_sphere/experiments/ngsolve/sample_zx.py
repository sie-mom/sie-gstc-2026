#!/usr/bin/env python3
# ruff: noqa: E402
"""Sample physical ZX fields from a solved or checkpointed NGSolve sphere."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[4]
if not __package__:
    sys.path.insert(0, str(REPOSITORY))
    sys.path.insert(0, str(REPOSITORY / "src"))
import ngsolve as ng
import numpy as np

from applications.anisotropic_gstc_sphere.validate_ngsolve import sample_zx
from applications.anisotropic_gstc_sphere.workflow import load_workflow_configuration
from sie_mom.validation.ngsolve import model as m


def resample(run_dir, output_dir, grid_size=None, span_radii=None):
    """Reload saved FE coefficients; no matrix assembly, factorization, or solve."""
    report = json.loads((run_dir / "metrics.json").read_text())
    if report["ngsolve"] != ng.__version__:
        raise ValueError("checkpoint loading requires the NGSolve version used for the solve")
    if not (run_dir / "mesh.vol.gz").is_file():
        raise ValueError(
            "run has no mesh checkpoint; repeat this older run with the updated runner"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    config = load_workflow_configuration(run_dir / "configuration.json")
    mesh = ng.Mesh(str(run_dir / "mesh.vol.gz"))
    space = m.make_space(mesh, report["settings"]["order"])
    if space.ndof != report["dofs"]:
        raise ValueError("checkpoint DOF count differs from the saved space")
    output_dir.mkdir(parents=True, exist_ok=True)
    system = SimpleNamespace(config=config, mesh=mesh)
    for case_id in report["cases"]:
        solution = ng.GridFunction(space)
        solution.Load(str(run_dir / case_id / "solution.sol"))
        fields = m.total_fields(system, solution, *m.incident_cf(config, case_id))
        data = sample_zx(
            system, fields, config, report["settings"]["air_lambda"], grid_size, span_radii
        )
        directory = output_dir / case_id
        directory.mkdir()
        np.savez(directory / "zx_fields.npz", **data)
        print(f"{case_id}: sampled {int(data['valid'].sum())} valid ZX points", flush=True)
    (output_dir / "sampling.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir.resolve()),
                "operation": "checkpoint field evaluation only",
                "grid_size": grid_size,
                "span_radii": span_radii,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--span-radii", type=float)
    args = parser.parse_args()
    resample(args.run_dir, args.output_dir, args.grid_size, args.span_radii)
