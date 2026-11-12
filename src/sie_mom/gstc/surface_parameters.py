"""Generic extraction of reciprocal tangential surface parameter tensors.

The physical fields-to-currents map is ``[J_s; M_s] = Lambda [E_avg; H_avg]``.
Its blocks are electric admittance ``Y_ee``, magnetic impedance ``Y_mm``, and
dimensionless coupling tensors ``Y_em`` and ``Y_me``. These are surface
parameters, not polarization susceptibilities.

Each prescribed state supplies component-first ``(2, ...)`` arrays in the
same oriented orthonormal tangent basis, with ``n x (u, v) = (-v, u)`` and
the ``exp(+j omega t)`` convention. The extraction has no dependence on the
surface geometry, source type, or prescribed field transformation.

The reciprocal-lossless extractor enforces a skew-Hermitian map at each sample:
``Lambda.conj().T = -Lambda``. This gives zero time-average absorbed power for
every local field state. The reciprocal-complex extractor does not impose
this condition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .omega_biisotropic import BoundaryQuantities

ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
TensorStructure = Literal["general", "diagonal"]


@dataclass(frozen=True)
class SurfaceParameters:
    """Physical surface parameter blocks with shape ``(2, 2, ...)``."""

    Y_ee: ComplexArray
    Y_em: ComplexArray
    Y_me: ComplexArray
    Y_mm: ComplexArray

    def as_matrix(self) -> ComplexArray:
        """Return the complete fields-to-currents map, shape ``(4, 4, ...)``."""

        return np.concatenate(
            (
                np.concatenate((self.Y_ee, self.Y_em), axis=1),
                np.concatenate((self.Y_me, self.Y_mm), axis=1),
            ),
            axis=0,
        )


@dataclass(frozen=True)
class SurfaceParameterExtraction:
    """Surface parameters and diagnostics at each surface sample.

    ``unique`` means the weighted coefficient matrix has full column rank;
    it does not mean the prescribed fields were fitted exactly. Rank-deficient
    samples retain their minimum-norm fit and have infinite condition number.
    Residuals weight magnetic currents by ``1 / reference_impedance``. The
    per-state relative residual has shape ``(number_of_states, ...)``.
    """

    parameters: SurfaceParameters
    rank: NDArray[np.int64]
    singular_values: RealArray
    condition_number: RealArray
    unique: NDArray[np.bool_]
    residual_norm: RealArray
    relative_residual: RealArray
    state_relative_residual: RealArray


def _stack_states(
    states: Sequence[BoundaryQuantities],
) -> tuple[tuple[int, ...], ComplexArray, ComplexArray]:
    if not states:
        raise ValueError("at least one prescribed field state is required")

    expected_shape = None
    fields = []
    currents = []
    for state_index, state in enumerate(states):
        arrays = []
        for name in (
            "electric_average",
            "magnetic_average",
            "electric_current",
            "magnetic_current",
        ):
            array = np.asarray(getattr(state, name), dtype=complex)
            if array.ndim == 0 or array.shape[0] != 2 or array.size == 0:
                raise ValueError(f"state {state_index} {name} must have nonempty shape (2, ...)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"state {state_index} {name} must be finite")
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError("all fields in all states must have the same shape")
            arrays.append(array)
        fields.append(np.concatenate(arrays[:2], axis=0).reshape(4, -1))
        currents.append(np.concatenate(arrays[2:], axis=0).reshape(4, -1))

    return expected_shape[1:], np.stack(fields, axis=1), np.stack(currents, axis=1)


def _reciprocal_basis(structure: TensorStructure, *, lossless: bool) -> ComplexArray:
    if structure == "general":
        self_entries = ((0, 0), (1, 1), (0, 1), (2, 2), (3, 3), (2, 3))
        coupling_entries = ((0, 2), (0, 3), (1, 2), (1, 3))
    elif structure == "diagonal":
        self_entries = ((0, 0), (1, 1), (2, 2), (3, 3))
        coupling_entries = ()
    else:
        raise ValueError("structure must be 'general' or 'diagonal'")

    basis = np.zeros((len(self_entries) + len(coupling_entries), 4, 4), complex)
    for parameter, (row, column) in enumerate(self_entries):
        basis[parameter, row, column] = 1j if lossless else 1.0
        basis[parameter, column, row] = basis[parameter, row, column]
    for parameter, (row, column) in enumerate(coupling_entries, len(self_entries)):
        basis[parameter, row, column] = 1.0
        basis[parameter, column, row] = -1.0
    return basis


def _extract_reciprocal(
    states: Sequence[BoundaryQuantities],
    *,
    reference_impedance: float,
    structure: TensorStructure,
    lossless: bool,
    rcond: float | None,
) -> SurfaceParameterExtraction:
    """Fit one reciprocal surface map per sample across all prescribed states.

    ``states`` accepts any nonempty collection of transformations. Each state
    supplies tangential averages and currents with the same shape ``(2, ...)``
    and the same oriented orthonormal tangent basis. Define the state vectors
    ``f_s = [E_avg; H_avg]`` and ``i_s = [J_s; M_s]``.

    Construct ``Lambda(q) = sum_k q_k B_k`` from reciprocal basis matrices:
    symmetric self-blocks and ``Y_me = -Y_em.T``. General and diagonal structures
    have ten and four independent coefficients, respectively; the diagonal
    structure has diagonal self-blocks and zero coupling. For N states and P
    coefficients, each sample's matrix has shape ``(4*N, P)``; column k stacks
    ``B_k @ f_s`` in state order, matching the stacked current vectors.

    Minimize ``sum_s ||D @ (Lambda(q) @ f_s - i_s)||_2**2``, where
    ``D = diag(1, 1, 1/eta_ref, 1/eta_ref)``. The finite positive
    ``reference_impedance`` is eta_ref in ohms: it divides both sides of the
    magnetic-current equations, converting their residuals from V/m to A/m.
    Fields and independent coefficients are not normalized.

    With ``lossless=False``, solve directly for complex coefficients. With
    ``lossless=True``, the basis uses imaginary self-blocks and real reciprocal
    coupling; coefficients are real. Stack real and imaginary equation parts
    into a real ``(8*N, P)`` system acting on the same coefficients. Prescribed
    fields and the resulting surface map remain complex-valued.

    Solve each sample independently with ``numpy.linalg.lstsq``. ``rcond``
    controls its relative singular-value cutoff; None uses NumPy's default.
    Rank-deficient fits select the minimum Euclidean norm of the independent
    coefficients, without a regularization term.

    Return physical tensor blocks in the original sample shape, solver rank
    and singular values, and weighted aggregate/per-state residual diagnostics.
    ``unique`` indicates full column rank, not an exact fit; rank-deficient
    samples report infinite condition number. Relative residuals divide by the
    corresponding weighted current norms, floored at the smallest positive
    normal float to handle zero currents.
    """
    if not np.isfinite(reference_impedance) or reference_impedance <= 0.0:
        raise ValueError("reference_impedance must be finite and positive")
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0.0):
        raise ValueError("rcond must be finite and nonnegative, or None")

    basis = _reciprocal_basis(structure, lossless=lossless)
    sample_shape, fields, currents = _stack_states(states)
    number_of_states = fields.shape[1]
    number_of_points = fields.shape[2]
    number_of_parameters = basis.shape[0]

    # Column k is B_k @ f_s; rows group each state's [J_s; M_s] equations.
    # coefficient are obtained from the basis and the stacked fields
    coefficients = np.einsum("kij,jsn->nsik", basis, fields).reshape(
        number_of_points, 4 * number_of_states, number_of_parameters
    )
    sources = currents.transpose(2, 1, 0).reshape(number_of_points, -1)
    # Scale both sides of the magnetic-current equations from V/m to A/m.
    current_weights = np.array((1.0, 1.0, 1.0 / reference_impedance, 1.0 / reference_impedance))
    row_weights = np.tile(current_weights, number_of_states)
    coefficients *= row_weights[None, :, None]
    sources = sources * row_weights

    parameter_values = np.empty((number_of_parameters, number_of_points), complex)
    ranks = np.empty(number_of_points, dtype=np.int64)
    singular_values = np.zeros((number_of_parameters, number_of_points))
    condition_numbers = np.full(number_of_points, np.inf)
    for point_index in range(number_of_points):
        coefficient = coefficients[point_index]
        source = sources[point_index]
        if lossless:
            # Split complex equations to solve for real lossless coefficients.
            # stack coefficients (fields) to separate real and imaginary parts
            coefficient = np.vstack((coefficient.real, coefficient.imag))
            # stack source (currents) to separate real and imaginary parts
            source = np.concatenate((source.real, source.imag))
        values, _, rank, singular = np.linalg.lstsq(coefficient, source, rcond=rcond)
        parameter_values[:, point_index] = values
        ranks[point_index] = rank
        singular_values[: singular.size, point_index] = singular
        if rank == number_of_parameters:
            condition_numbers[point_index] = singular[0] / singular[-1]

    matrix = np.einsum("kn,kij->ijn", parameter_values, basis)
    reconstructed = np.einsum("ijn,jsn->isn", matrix, fields)
    weighted_residual = (reconstructed - currents) * current_weights[:, None, None]
    weighted_current = currents * current_weights[:, None, None]
    state_norms = np.linalg.norm(weighted_residual, axis=0)
    state_source_norms = np.linalg.norm(weighted_current, axis=0)
    residual_norms = np.linalg.norm(state_norms, axis=0)
    source_norms = np.linalg.norm(state_source_norms, axis=0)
    floor = np.finfo(float).tiny
    matrix = matrix.reshape((4, 4) + sample_shape)

    return SurfaceParameterExtraction(
        parameters=SurfaceParameters(
            Y_ee=matrix[:2, :2],
            Y_em=matrix[:2, 2:],
            Y_me=matrix[2:, :2],
            Y_mm=matrix[2:, 2:],
        ),
        rank=ranks.reshape(sample_shape),
        singular_values=singular_values.reshape((number_of_parameters,) + sample_shape),
        condition_number=condition_numbers.reshape(sample_shape),
        unique=(ranks == number_of_parameters).reshape(sample_shape),
        residual_norm=residual_norms.reshape(sample_shape),
        relative_residual=(residual_norms / np.maximum(source_norms, floor)).reshape(sample_shape),
        state_relative_residual=(state_norms / np.maximum(state_source_norms, floor)).reshape(
            (number_of_states,) + sample_shape
        ),
    )


def extract_reciprocal_complex_least_squares(
    states: Sequence[BoundaryQuantities],
    *,
    reference_impedance: float,
    structure: TensorStructure = "general",
    rcond: float | None = None,
) -> SurfaceParameterExtraction:
    """Fit reciprocal complex surface parameters using an SVD solve.

    Self-blocks are symmetric and ``Y_me = -Y_em.T``. General and diagonal
    structures have ten and four complex unknowns, respectively. Passivity
    and losslessness are not imposed. The required reference impedance, in
    ohms, divides the magnetic-current equations to give both residual blocks
    units of A/m.

    Rank-deficient inputs return the minimum Euclidean norm of the independent
    physical coefficients, matching the legacy complex extraction convention.
    Columns are not normalized: changing that convention would select a
    different sheet for underdetermined data. ``unique`` and residuals must
    therefore be assessed separately.
    """

    return _extract_reciprocal(
        states,
        reference_impedance=reference_impedance,
        structure=structure,
        lossless=False,
        rcond=rcond,
    )


def extract_reciprocal_lossless_least_squares(
    states: Sequence[BoundaryQuantities],
    *,
    reference_impedance: float,
    structure: TensorStructure = "general",
    rcond: float | None = None,
) -> SurfaceParameterExtraction:
    """Fit reciprocal passive-lossless surface parameters using a real SVD solve.

    The map is ``[[j B, K], [-K.T, j X]]`` with real symmetric ``B`` and ``X``
    and real ``K``. Thus it absorbs or generates no time-average real power
    for any local field state. General and diagonal structures have ten and
    four real unknowns, respectively. The complex equations are split into
    real and imaginary parts before solving for these real coefficients.

    Reference-impedance weighting and minimum-norm handling follow the complex
    extractor. Incompatible targets retain a nonzero fitting residual even
    though the fitted surface parameters satisfy the material constraints.
    """

    return _extract_reciprocal(
        states,
        reference_impedance=reference_impedance,
        structure=structure,
        lossless=True,
        rcond=rcond,
    )
