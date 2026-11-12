"""Validate physical GSTC maps and pack maps/inverses in SIE face order.

The record layout is shared by spherical and non-spherical applications:
32 little-endian complex128 values per face, the row-major 4x4 map followed
by its full inverse. This module never reorders faces or changes a fitted map.
"""

import numpy as np
from scipy.linalg import solve

from .surface_parameters import SurfaceParameters

CONSTRAINT_TOLERANCE = 1e-10
INVERSE_TOLERANCE = 1e-8


def prepare_tensor_records(
    parameters, face_count, *, reference_impedance, extraction_method, tensor_structure
):
    """Validate and invert physical maps; return the existing binary record layout.

    Conditioning and constraint checks use [[eta*Y, chi], [gamma, Z/eta]].
    Lossless maps use a Hermitian solve of i*balanced in complex128.
    Numerical singularity uses s_min <= 4*eps*s_max. Inversion must satisfy
    both balanced identity products within INVERSE_TOLERANCE; no pseudoinverse
    or material projection is used. Export retains the physical, unscaled map.
    """
    if not np.isfinite(reference_impedance) or reference_impedance <= 0:
        raise ValueError("reference_impedance must be finite and positive")
    if not isinstance(parameters, SurfaceParameters):
        raise ValueError("parameter_evaluator must return SurfaceParameters")
    for name in ("Y_ee", "Y_em", "Y_me", "Y_mm"):
        block = np.asarray(getattr(parameters, name))
        if (
            block.shape != (2, 2, face_count)
            or not np.issubdtype(block.dtype, np.number)
            or not np.isfinite(block).all()
        ):
            raise ValueError(f"surface block {name} must be finite with shape (2, 2, {face_count})")
    matrix = np.moveaxis(parameters.as_matrix(), -1, 0).astype(complex)
    eta = reference_impedance
    balanced = matrix.copy()
    balanced[:, :2, :2] *= eta
    balanced[:, 2:, 2:] /= eta
    if not np.isfinite(balanced).all():
        raise ValueError("impedance-balanced surface maps must be finite")
    scale = np.maximum(np.max(np.abs(balanced), axis=(1, 2)), np.finfo(float).tiny)
    reciprocity = max(
        float(np.max(np.max(np.abs(defect), axis=(1, 2)) / scale))
        for defect in (
            balanced[:, :2, :2] - balanced[:, :2, :2].swapaxes(1, 2),
            balanced[:, 2:, 2:] - balanced[:, 2:, 2:].swapaxes(1, 2),
            balanced[:, 2:, :2] + balanced[:, :2, 2:].swapaxes(1, 2),
        )
    )
    losslessness = float(
        np.max(np.max(np.abs(balanced + balanced.conj().swapaxes(1, 2)), axis=(1, 2)) / scale)
    )
    if reciprocity > CONSTRAINT_TOLERANCE:
        raise ValueError("surface maps do not satisfy reciprocity")
    if extraction_method == "reciprocal_lossless" and losslessness > CONSTRAINT_TOLERANCE:
        raise ValueError("surface maps do not satisfy losslessness")
    if tensor_structure == "diagonal":
        off_diagonal = balanced.copy()
        off_diagonal[:, np.arange(4), np.arange(4)] = 0
        if np.max(np.max(np.abs(off_diagonal), axis=(1, 2)) / scale) > CONSTRAINT_TOLERANCE:
            raise ValueError("surface maps do not satisfy the configured diagonal structure")
    singular_values = np.linalg.svd(balanced, compute_uv=False)
    singular = singular_values[:, -1] <= 4 * np.finfo(float).eps * singular_values[:, 0]
    if np.any(singular):
        raise ValueError(
            f"surface map is numerically singular at {np.count_nonzero(singular)} faces "
            f"(first face index {np.flatnonzero(singular)[0]}); the current SIE interface "
            "requires its full inverse"
        )
    try:
        # With T = diag(sqrt(eta), sqrt(eta), 1/sqrt(eta), 1/sqrt(eta)),
        # balanced = T @ matrix @ T, hence matrix^-1 = T @ balanced^-1 @ T.
        # Invert before restoring physical units to avoid amplifying the
        # electric/magnetic unit imbalance in the numerical factorization.
        if extraction_method == "reciprocal_lossless":
            # B^H = -B, so i*B is Hermitian and (i*B) X = i*I gives B^-1.
            hermitian_rhs = 1j * np.eye(4, dtype=complex)
            inverse = np.stack(
                [solve(1j * face_map, hermitian_rhs, assume_a="her") for face_map in balanced]
            )
        else:
            inverse = np.linalg.inv(balanced)
        inverse[:, :2, :2] *= eta
        inverse[:, 2:, 2:] /= eta
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "surface map is singular; the current SIE interface requires its full inverse"
        ) from error
    if not np.isfinite(inverse).all():
        raise ValueError("surface map inverse must be finite")
    balanced_inverse = inverse.copy()
    balanced_inverse[:, :2, :2] /= eta
    balanced_inverse[:, 2:, 2:] *= eta
    if not np.isfinite(balanced_inverse).all():
        raise ValueError("impedance-balanced inverse must be finite")
    identity = np.eye(4, dtype=complex)
    residual = float(
        np.max(
            [
                np.linalg.norm(product - identity, axis=(1, 2))
                for product in (balanced @ balanced_inverse, balanced_inverse @ balanced)
            ]
        )
    )
    if not np.isfinite(residual) or residual > INVERSE_TOLERANCE:
        raise ValueError(
            f"surface map inversion residual {residual:.3g} exceeds {INVERSE_TOLERANCE}"
        )
    physical_residual = float(np.max(np.linalg.norm(matrix @ inverse - identity, axis=(1, 2))))
    if not np.isfinite(physical_residual):
        raise ValueError("physical surface map inversion residual must be finite")
    records = np.concatenate((matrix.reshape((-1, 16)), inverse.reshape((-1, 16))), axis=1).astype(
        np.dtype("<c16")
    )
    return records, {
        "all_tensors_finite": True,
        "relative_constraint_tolerance": CONSTRAINT_TOLERANCE,
        "balanced_inverse_tolerance": INVERSE_TOLERANCE,
        "maximum_relative_reciprocity_residual": reciprocity,
        "maximum_relative_skew_hermitian_residual": losslessness,
        "maximum_map_condition_number": float(
            np.max(singular_values[:, 0] / singular_values[:, -1])
        ),
        "maximum_balanced_inverse_residual": residual,
        "maximum_inverse_residual": physical_residual,
    }
