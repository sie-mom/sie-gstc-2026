"""SIE tensor records retain physical units and reject unsafe inverses."""

import numpy as np
import pytest

from sie_mom.gstc import SurfaceParameters
from sie_mom.gstc.tensor_io import INVERSE_TOLERANCE, prepare_tensor_records


def parameters_from_matrix(matrix):
    return SurfaceParameters(
        matrix[:2, :2, None],
        matrix[:2, 2:, None],
        matrix[2:, :2, None],
        matrix[2:, 2:, None],
    )


def test_balanced_inversion_preserves_units_for_an_ill_scaled_lossless_map():
    eta = 376.730313668
    rng = np.random.default_rng(321)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    symmetric = (rotation * np.array([1, 0.5, 0.2, 1e-6])) @ rotation.T
    phase = np.diag([1, 1, 1j, 1j])
    balanced = 1j * phase @ symmetric @ phase.conj().T
    matrix = balanced.copy()
    matrix[:2, :2] /= eta
    matrix[2:, 2:] *= eta
    assert np.linalg.cond(matrix) > 1e10
    records, diagnostics = prepare_tensor_records(
        parameters_from_matrix(matrix),
        1,
        reference_impedance=eta,
        extraction_method="reciprocal_lossless",
        tensor_structure="general",
    )
    assert records.dtype == np.dtype("<c16")
    np.testing.assert_array_equal(records[0, :16].reshape(4, 4), matrix)
    inverse = records[0, 16:].reshape(4, 4)
    balanced_inverse = inverse.copy()
    balanced_inverse[:2, :2] /= eta
    balanced_inverse[2:, 2:] *= eta
    for product in (balanced @ balanced_inverse, balanced_inverse @ balanced):
        assert np.linalg.norm(product - np.eye(4)) < INVERSE_TOLERANCE
    assert diagnostics["maximum_balanced_inverse_residual"] < INVERSE_TOLERANCE


def test_singular_map_is_rejected_without_a_pseudoinverse():
    with pytest.raises(ValueError, match="numerically singular"):
        prepare_tensor_records(
            parameters_from_matrix(1j * np.diag([1, 2, 3, 0])),
            1,
            reference_impedance=376.73,
            extraction_method="reciprocal_lossless",
            tensor_structure="general",
        )


def test_windows_cone_face_442_exports_with_the_existing_two_sided_tolerance():
    # Fitted physical map from the 3558-face Windows cone-sphere mesh. Its
    # balanced condition number is about 4.06e5; the previous general inverse
    # failed the right identity check despite a small left identity residual.
    eta = 376.7303134118051
    matrix = np.array(
        [
            [-1.6650762592296j, 2.410905448629344j, -54.19039409513286, 83.03058392494242],
            [2.410905448629344j, -3.490763303997856j, 79.10976424178277, -121.159795505059],
            [54.19039409513286, -79.10976424178277, 240794.9202729676j, -348739.7436481681j],
            [-83.03058392494242, 121.159795505059, -348739.7436481681j, 505068.5555485471j],
        ],
        dtype=complex,
    )
    maps = np.stack((matrix, 2 * matrix), axis=-1)
    parameters = SurfaceParameters(maps[:2, :2], maps[:2, 2:], maps[2:, :2], maps[2:, 2:])
    records, diagnostics = prepare_tensor_records(
        parameters,
        2,
        reference_impedance=eta,
        extraction_method="reciprocal_lossless",
        tensor_structure="general",
    )
    assert records.dtype == np.dtype("<c16")
    np.testing.assert_array_equal(records[:, :16].reshape(2, 4, 4), np.moveaxis(maps, -1, 0))
    balanced = np.moveaxis(maps, -1, 0).copy()
    balanced[:, :2, :2] *= eta
    balanced[:, 2:, 2:] /= eta
    inverse = records[:, 16:].reshape(2, 4, 4).copy()
    inverse[:, :2, :2] /= eta
    inverse[:, 2:, 2:] *= eta
    for product in (balanced @ inverse, inverse @ balanced):
        assert np.max(np.linalg.norm(product - np.eye(4), axis=(1, 2))) < 1e-8
    assert diagnostics["balanced_inverse_tolerance"] == 1e-8


def test_general_reciprocal_complex_map_keeps_its_full_complex_inverse():
    matrix = np.diag([1 + 2j, 2 - 1j, 3 + 0.5j, 4 - 2j])
    records, _ = prepare_tensor_records(
        parameters_from_matrix(matrix),
        1,
        reference_impedance=376.7303134118051,
        extraction_method="reciprocal_complex",
        tensor_structure="general",
    )
    np.testing.assert_array_equal(records[0, :16].reshape(4, 4), matrix)
    np.testing.assert_allclose(
        records[0, 16:].reshape(4, 4), np.linalg.inv(matrix), rtol=1e-14, atol=0
    )


def test_lossless_inversion_rejects_a_map_with_a_real_diagonal_component():
    matrix = np.diag([0.1 + 1j, 2j, 3j, 4j])
    with pytest.raises(ValueError, match="losslessness"):
        prepare_tensor_records(
            parameters_from_matrix(matrix),
            1,
            reference_impedance=1,
            extraction_method="reciprocal_lossless",
            tensor_structure="general",
        )
