#!/usr/bin/env python3
# ruff: noqa: E402
"""Compatibility entry point for the original sphere experiment.

Prefer python -m applications.anisotropic_gstc_sphere.validate_ngsolve.
"""

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
for path in (REPOSITORY, REPOSITORY / "src"):
    sys.path.insert(0, str(path))
from applications.anisotropic_gstc_sphere.validate_ngsolve import (
    diagnostics,
    parser,
    relative_difference,
    run,
    sample_radial,
    sphere_quadrature,
)

if __name__ == "__main__":
    run(parser().parse_args())

__all__ = [
    "sphere_quadrature",
    "relative_difference",
    "diagnostics",
    "sample_radial",
    "parser",
    "run",
]
