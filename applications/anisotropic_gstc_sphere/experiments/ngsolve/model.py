# ruff: noqa: E402
"""Compatibility exports for the original spherical NGSolve experiment.

FEM implementation: sie_mom.validation.ngsolve. Sphere adapter/runner:
applications.anisotropic_gstc_sphere.validate_ngsolve. Old commands and tests
continue to use this module without duplicated numerical implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[4]
APPLICATION = Path(__file__).resolve().parents[2]
for path in (REPOSITORY, REPOSITORY / "src", APPLICATION):
    sys.path.insert(0, str(path))
from applications.anisotropic_gstc_sphere.validate_ngsolve import (
    DEFAULT_INPUT,
    load_benchmark,
    load_workflow_configuration,
    sha256,
)
from applications.anisotropic_gstc_sphere.validate_ngsolve import (
    assemble_sphere as assemble,
)
from applications.anisotropic_gstc_sphere.validate_ngsolve import (
    incident_cf_sphere as incident_cf,
)
from sie_mom.mesh.obj_mesh import load_triangle_obj
from sie_mom.validation.ngsolve.model import (
    ETA0,
    R,
    System,
    dipole_cf,
    direct_solver,
    evaluate,
    k0_of,
    local,
    make_space,
    medium,
    normalized_map,
    residual,
    rotate,
    sheet_admittance,
    split,
    total_fields,
)
from sie_mom.validation.ngsolve.tensor_transfer import spherical_tangent_frame as tangent_frame
from sie_mom.validation.ngsolve.volume_mesh import make_sphere_mesh as make_mesh

__all__ = [
    "ETA0",
    "R",
    "System",
    "normalized_map",
    "sheet_admittance",
    "k0_of",
    "direct_solver",
    "medium",
    "dipole_cf",
    "local",
    "rotate",
    "split",
    "make_space",
    "total_fields",
    "evaluate",
    "residual",
    "tangent_frame",
    "make_mesh",
    "load_triangle_obj",
    "DEFAULT_INPUT",
    "sha256",
    "load_benchmark",
    "load_workflow_configuration",
    "incident_cf",
    "assemble",
]
