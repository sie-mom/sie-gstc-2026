"""Extraction utilities for reciprocal lossless omega-biisotropic sheets.

All vector arrays use component-first shape ``(2, ...)`` in an oriented local
tangential basis ``(u, v)`` for which

``n_hat x (a_u, a_v) = (-a_v, a_u)``.

The module uses the outgoing-wave ``exp(+j omega t)`` convention.  A
reciprocal lossless omega sheet is consequently parameterized by
``Y = j B``, ``Z = j X``, and real ``chi``.

The extraction routines are independent of how the interior and exterior
fields were generated; they require only prescribed tangential fields that
follow the conventions above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]


@dataclass(frozen=True)
class BoundaryQuantities:
    """Average tangential fields and equivalent sheet currents."""

    electric_average: ComplexArray
    magnetic_average: ComplexArray
    electric_current: ComplexArray
    magnetic_current: ComplexArray


@dataclass(frozen=True)
class OmegaParameters:
    """Real lossless omega parameters at one or more surface points."""

    B: RealArray
    X: RealArray
    chi: RealArray

    @property
    def Y(self) -> ComplexArray:
        """Electric sheet admittance ``Y = j B``."""

        return 1j * self.B

    @property
    def Z(self) -> ComplexArray:
        """Magnetic sheet impedance ``Z = j X``."""

        return 1j * self.X


@dataclass(frozen=True)
class ProjectedExtraction:
    """Result and diagnostics from the projected analytical formulas."""

    parameters: OmegaParameters
    chi_electric: RealArray
    chi_magnetic: RealArray
    valid: NDArray[np.bool_]
    residual_norm: RealArray
    relative_residual: RealArray


@dataclass(frozen=True)
class LeastSquaresExtraction:
    """Result and diagnostics from the constrained real least-squares solve."""

    parameters: OmegaParameters
    rank: IntegerArray
    singular_values: RealArray
    condition_number: RealArray
    valid: NDArray[np.bool_]
    residual_norm: RealArray
    relative_residual: RealArray


def _as_tangential_field(name: str, field: ArrayLike) -> ComplexArray:
    array = np.asarray(field, dtype=complex)
    if array.ndim == 0 or array.shape[0] != 2:
        raise ValueError(f"{name} must have component-first shape (2, ...)")
    return array


def _validate_same_shape(**fields: ComplexArray) -> tuple[int, ...]:
    shapes = {field.shape for field in fields.values()}
    if len(shapes) != 1:
        description = ", ".join(
            f"{name}={field.shape}" for name, field in fields.items()
        )
        raise ValueError(
            f"all tangential fields must have the same shape; {description}"
        )
    return next(iter(shapes))[1:]


def _normal_cross(field: ComplexArray) -> ComplexArray:
    """Return ``n_hat x field`` in the oriented local tangential basis."""

    return np.stack((-field[1], field[0]))


def _dot(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    return np.sum(left * right, axis=0)


def form_boundary_quantities(
    electric_interior: ArrayLike,
    magnetic_interior: ArrayLike,
    electric_exterior: ArrayLike,
    magnetic_exterior: ArrayLike,
) -> BoundaryQuantities:
    """Form average fields and sheet currents from prescribed boundary fields.

    The unit normal points from the interior region to the exterior region.
    With ``Delta F = F_exterior - F_interior``, the definitions are

    ``J_s = n_hat x Delta H`` and ``M_s = -n_hat x Delta E``.
    """

    electric_interior_array = _as_tangential_field(
        "electric_interior", electric_interior
    )
    magnetic_interior_array = _as_tangential_field(
        "magnetic_interior", magnetic_interior
    )
    electric_exterior_array = _as_tangential_field(
        "electric_exterior", electric_exterior
    )
    magnetic_exterior_array = _as_tangential_field(
        "magnetic_exterior", magnetic_exterior
    )
    _validate_same_shape(
        electric_interior=electric_interior_array,
        magnetic_interior=magnetic_interior_array,
        electric_exterior=electric_exterior_array,
        magnetic_exterior=magnetic_exterior_array,
    )

    electric_average = 0.5 * (
        electric_exterior_array + electric_interior_array
    )
    magnetic_average = 0.5 * (
        magnetic_exterior_array + magnetic_interior_array
    )
    electric_current = _normal_cross(
        magnetic_exterior_array - magnetic_interior_array
    )
    magnetic_current = -_normal_cross(
        electric_exterior_array - electric_interior_array
    )

    return BoundaryQuantities(
        electric_average=electric_average,
        magnetic_average=magnetic_average,
        electric_current=electric_current,
        magnetic_current=magnetic_current,
    )


def normal_power_density(electric: ArrayLike, magnetic: ArrayLike) -> RealArray:
    """Return outward time-average real power density ``n_hat . S``."""

    electric_array = _as_tangential_field("electric", electric)
    magnetic_array = _as_tangential_field("magnetic", magnetic)
    _validate_same_shape(electric=electric_array, magnetic=magnetic_array)
    return 0.5 * np.real(
        electric_array[0] * np.conjugate(magnetic_array[1])
        - electric_array[1] * np.conjugate(magnetic_array[0])
    )


def absorbed_power_density(quantities: BoundaryQuantities) -> RealArray:
    """Return time-average real power density absorbed by the sheet."""

    return 0.5 * np.real(
        _dot(
            quantities.electric_current,
            np.conjugate(quantities.electric_average),
        )
        + _dot(
            quantities.magnetic_current,
            np.conjugate(quantities.magnetic_average),
        )
    )


def reconstruct_omega_currents(
    quantities: BoundaryQuantities,
    parameters: OmegaParameters,
) -> tuple[ComplexArray, ComplexArray]:
    """Reconstruct ``(J_s, M_s)`` from lossless omega parameters."""

    electric_average = quantities.electric_average
    magnetic_average = quantities.magnetic_average
    sample_shape = electric_average.shape[1:]

    B, X, chi = np.broadcast_arrays(parameters.B, parameters.X, parameters.chi)
    if B.shape != sample_shape:
        raise ValueError(
            "parameter arrays must match the boundary-field sample shape "
            f"{sample_shape}; received {B.shape}"
        )

    electric_current = (
        1j * B * electric_average - chi * _normal_cross(magnetic_average)
    )
    magnetic_current = (
        -chi * _normal_cross(electric_average) + 1j * X * magnetic_average
    )
    return electric_current, magnetic_current


def _current_residuals(
    quantities: BoundaryQuantities,
    parameters: OmegaParameters,
) -> tuple[RealArray, RealArray]:
    electric_reconstructed, magnetic_reconstructed = reconstruct_omega_currents(
        quantities, parameters
    )
    residual_squared = np.sum(
        np.abs(electric_reconstructed - quantities.electric_current) ** 2,
        axis=0,
    ) + np.sum(
        np.abs(magnetic_reconstructed - quantities.magnetic_current) ** 2,
        axis=0,
    )
    source_squared = np.sum(
        np.abs(quantities.electric_current) ** 2, axis=0
    ) + np.sum(np.abs(quantities.magnetic_current) ** 2, axis=0)
    residual = np.sqrt(residual_squared)
    source_norm = np.sqrt(source_squared)
    relative = np.full(residual.shape, np.nan, dtype=float)
    np.divide(residual, source_norm, out=relative, where=source_norm > 0.0)
    return residual, relative


def extract_lossless_omega_projected(
    quantities: BoundaryQuantities,
    *,
    relative_tolerance: float = 1e-12,
) -> ProjectedExtraction:
    """Extract ``B``, ``X``, and ``chi`` using the projected GSTC formulas.

    The reported coupling is the symmetric value
    ``chi = (chi_electric + chi_magnetic) / 2``.  A point is invalid when a
    projected denominator vanishes or when ``Re(q)`` is too small relative to
    ``|E_avg| |H_avg|``.
    """

    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be nonnegative")

    electric_average = quantities.electric_average
    magnetic_average = quantities.magnetic_average
    electric_norm_squared = np.real(
        _dot(electric_average, np.conjugate(electric_average))
    )
    magnetic_norm_squared = np.real(
        _dot(magnetic_average, np.conjugate(magnetic_average))
    )
    q = _dot(_normal_cross(electric_average), np.conjugate(magnetic_average))
    q_scale = np.sqrt(electric_norm_squared * magnetic_norm_squared)

    electric_projection = _dot(
        quantities.electric_current, np.conjugate(electric_average)
    )
    magnetic_projection = _dot(
        quantities.magnetic_current, np.conjugate(magnetic_average)
    )

    nonzero_fields = (electric_norm_squared > 0.0) & (magnetic_norm_squared > 0.0)
    coupling_is_resolved = np.abs(np.real(q)) > relative_tolerance * q_scale
    valid = nonzero_fields & coupling_is_resolved

    result_shape = electric_norm_squared.shape
    chi_electric = np.full(result_shape, np.nan, dtype=float)
    chi_magnetic = np.full(result_shape, np.nan, dtype=float)
    np.divide(
        np.real(electric_projection),
        np.real(q),
        out=chi_electric,
        where=valid,
    )
    np.divide(
        -np.real(magnetic_projection),
        np.real(q),
        out=chi_magnetic,
        where=valid,
    )
    chi = 0.5 * (chi_electric + chi_magnetic)

    B = np.full(result_shape, np.nan, dtype=float)
    X = np.full(result_shape, np.nan, dtype=float)
    np.divide(
        np.imag(electric_projection) + chi * np.imag(q),
        electric_norm_squared,
        out=B,
        where=valid,
    )
    np.divide(
        np.imag(magnetic_projection) + chi * np.imag(q),
        magnetic_norm_squared,
        out=X,
        where=valid,
    )

    parameters = OmegaParameters(B=B, X=X, chi=chi)
    residual_norm, relative_residual = _current_residuals(quantities, parameters)
    return ProjectedExtraction(
        parameters=parameters,
        chi_electric=chi_electric,
        chi_magnetic=chi_magnetic,
        valid=valid,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
    )


def extract_lossless_omega_least_squares(
    quantities: BoundaryQuantities,
    *,
    rcond: float | None = None,
) -> LeastSquaresExtraction:
    """Extract all lossless omega parameters through a real SVD solve.

    At each surface point the complex system ``C p = s`` is converted to

    ``[Re(C); Im(C)] p = [Re(s); Im(s)]``,

    where ``p = (B, X, chi)`` is explicitly real.  Columns are normalized
    before calling :func:`numpy.linalg.lstsq` and the solution is unscaled
    afterward.  Rank-deficient points are marked invalid rather than assigned
    an arbitrary minimum-norm parameter vector.
    """

    sample_shape = quantities.electric_average.shape[1:]
    electric_average = quantities.electric_average.reshape(2, -1)
    magnetic_average = quantities.magnetic_average.reshape(2, -1)
    electric_current = quantities.electric_current.reshape(2, -1)
    magnetic_current = quantities.magnetic_current.reshape(2, -1)
    number_of_points = electric_average.shape[1]

    parameter_values = np.full((3, number_of_points), np.nan, dtype=float)
    ranks = np.zeros(number_of_points, dtype=np.int64)
    singular_values = np.full((3, number_of_points), np.nan, dtype=float)
    condition_numbers = np.full(number_of_points, np.inf, dtype=float)
    valid = np.zeros(number_of_points, dtype=bool)

    for point_index in range(number_of_points):
        electric = electric_average[:, point_index]
        magnetic = magnetic_average[:, point_index]
        source = np.concatenate(
            (
                electric_current[:, point_index],
                magnetic_current[:, point_index],
            )
        )

        coefficient = np.zeros((4, 3), dtype=complex)
        coefficient[:2, 0] = 1j * electric
        coefficient[2:, 1] = 1j * magnetic
        coefficient[:2, 2] = -_normal_cross(magnetic)
        coefficient[2:, 2] = -_normal_cross(electric)

        real_coefficient = np.vstack((coefficient.real, coefficient.imag))
        real_source = np.concatenate((source.real, source.imag))
        column_norms = np.linalg.norm(real_coefficient, axis=0)
        if np.any(column_norms == 0.0):
            continue

        scaled_coefficient = real_coefficient / column_norms
        scaled_parameters, _, rank, values = np.linalg.lstsq(
            scaled_coefficient, real_source, rcond=rcond
        )
        ranks[point_index] = rank
        singular_values[: values.size, point_index] = values
        if rank < 3:
            continue

        parameter_values[:, point_index] = scaled_parameters / column_norms
        condition_numbers[point_index] = values[0] / values[-1]
        valid[point_index] = True

    parameters = OmegaParameters(
        B=parameter_values[0].reshape(sample_shape),
        X=parameter_values[1].reshape(sample_shape),
        chi=parameter_values[2].reshape(sample_shape),
    )
    residual_norm, relative_residual = _current_residuals(quantities, parameters)
    return LeastSquaresExtraction(
        parameters=parameters,
        rank=ranks.reshape(sample_shape),
        singular_values=singular_values.reshape((3,) + sample_shape),
        condition_number=condition_numbers.reshape(sample_shape),
        valid=valid.reshape(sample_shape),
        residual_norm=residual_norm,
        relative_residual=relative_residual,
    )


def omega_tensor_blocks(parameters: OmegaParameters) -> dict[str, ComplexArray]:
    """Return the ``Y``, ``Chi``, ``Gamma``, and ``Z`` blocks used by SVWF.

    For the local rotation matrix ``N = [[0, -1], [1, 0]]``, both reciprocal
    omega cross-coupling blocks are ``-chi N``.
    """

    B, X, chi = np.broadcast_arrays(parameters.B, parameters.X, parameters.chi)
    tensor_shape = (2, 2) + B.shape
    electric = np.zeros(tensor_shape, dtype=complex)
    magnetic = np.zeros(tensor_shape, dtype=complex)
    coupling = np.zeros(tensor_shape, dtype=complex)

    electric[0, 0] = 1j * B
    electric[1, 1] = 1j * B
    magnetic[0, 0] = 1j * X
    magnetic[1, 1] = 1j * X
    coupling[0, 1] = chi
    coupling[1, 0] = -chi

    return {
        "Y": electric,
        "Chi": coupling,
        "Gamma": coupling.copy(),
        "Z": magnetic,
    }
