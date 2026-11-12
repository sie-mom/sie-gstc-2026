import json

import numpy as np
import pytest

from applications.anisotropic_gstc_sphere.prepare_sie_inputs import export_sie_inputs
from applications.anisotropic_gstc_sphere.workflow import (
    DEFAULT_CONFIGURATION,
    load_workflow_configuration,
)
from sie_mom.gstc import (
    BoundaryQuantities,
    OmegaParameters,
    absorbed_power_density,
    extract_reciprocal_complex_least_squares,
    extract_reciprocal_lossless_least_squares,
    form_boundary_quantities,
    omega_tensor_blocks,
)

EXTRACTORS = (
    extract_reciprocal_complex_least_squares,
    extract_reciprocal_lossless_least_squares,
)


def known_matrix(*, lossless, sample_shape=()):
    electric = np.array([[0.2, -0.07], [-0.07, 0.5]])
    magnetic = np.array([[0.9, 0.13], [0.13, -0.3]])
    coupling = np.array([[0.12, -0.4], [0.25, 0.08]])
    if lossless:
        electric = 1j * electric
        magnetic = 1j * magnetic
    else:
        electric = electric + 1j * np.array([[0.03, 0.06], [0.06, -0.02]])
        magnetic = magnetic + 1j * np.array([[0.11, -0.05], [-0.05, 0.07]])
        coupling = coupling + 1j * np.array([[0.04, -0.03], [0.1, -0.02]])
    matrix = np.block([[electric, coupling], [-coupling.T, magnetic]])
    if sample_shape:
        scales = np.linspace(0.8, 1.6, np.prod(sample_shape)).reshape(sample_shape)
        matrix = matrix.reshape((4, 4) + (1,) * len(sample_shape)) * scales
    return matrix


def field_states(number_of_states, sample_shape=()):
    rng = np.random.default_rng(391)
    shape = (4, number_of_states) + sample_shape
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def boundary_states(matrix, fields):
    currents = np.einsum("ij...,js...->is...", matrix, fields)
    states = []
    for state in range(fields.shape[1]):
        electric, magnetic = fields[:2, state], fields[2:, state]
        electric_current, magnetic_current = currents[:2, state], currents[2:, state]
        electric_jump = np.stack((-magnetic_current[1], magnetic_current[0]))
        magnetic_jump = np.stack((electric_current[1], -electric_current[0]))
        states.append(
            form_boundary_quantities(
                electric - 0.5 * electric_jump,
                magnetic - 0.5 * magnetic_jump,
                electric + 0.5 * electric_jump,
                magnetic + 0.5 * magnetic_jump,
            )
        )
    return states


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_recovers_general_surface_parameters_at_multiple_samples(extractor):
    lossless = extractor is extract_reciprocal_lossless_least_squares
    expected = known_matrix(lossless=lossless, sample_shape=(2, 3))
    states = boundary_states(expected, field_states(5, (2, 3)))
    result = extractor(states, reference_impedance=376.7303)

    np.testing.assert_allclose(result.parameters.as_matrix(), expected, atol=2e-12)
    np.testing.assert_allclose(result.relative_residual, 0.0, atol=2e-12)
    np.testing.assert_allclose(result.state_relative_residual, 0.0, atol=2e-12)
    assert result.rank.shape == (2, 3)
    assert result.singular_values.shape == (10, 2, 3)
    assert result.state_relative_residual.shape == (5, 2, 3)
    assert np.all(result.rank == 10)
    assert np.all(result.unique)
    assert np.all(np.isfinite(result.condition_number))
    np.testing.assert_allclose(result.parameters.Y_ee, np.swapaxes(result.parameters.Y_ee, 0, 1))
    np.testing.assert_allclose(result.parameters.Y_mm, np.swapaxes(result.parameters.Y_mm, 0, 1))
    np.testing.assert_allclose(result.parameters.Y_me, -np.swapaxes(result.parameters.Y_em, 0, 1))


def test_lossless_recovery_from_two_states_has_real_coupling_and_zero_real_power():
    expected = known_matrix(lossless=True)
    states = boundary_states(expected, field_states(2))
    result = extract_reciprocal_lossless_least_squares(states, reference_impedance=1.0)
    matrix = result.parameters.as_matrix()

    assert result.unique
    np.testing.assert_allclose(matrix, expected, atol=2e-14)
    np.testing.assert_allclose(result.parameters.Y_ee.real, 0.0)
    np.testing.assert_allclose(result.parameters.Y_mm.real, 0.0)
    np.testing.assert_allclose(result.parameters.Y_em.imag, 0.0)
    np.testing.assert_allclose(result.parameters.Y_me.imag, 0.0)
    np.testing.assert_allclose(matrix + matrix.conj().T, 0.0, atol=2e-14)
    probes = boundary_states(matrix, field_states(12))
    for state in probes:
        np.testing.assert_allclose(absorbed_power_density(state), 0.0, atol=2e-14)


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_diagonal_structure_can_be_identified_with_one_state(extractor):
    diagonal = np.array([0.2, 0.5, -0.7, 0.9])
    if extractor is extract_reciprocal_lossless_least_squares:
        diagonal = 1j * diagonal
    else:
        diagonal = diagonal + 1j * np.array([0.1, -0.2, 0.3, -0.4])
    expected = np.diag(diagonal)
    result = extractor(
        boundary_states(expected, field_states(1)), reference_impedance=50.0, structure="diagonal"
    )

    assert result.unique
    assert result.rank == 4
    np.testing.assert_allclose(result.parameters.as_matrix(), expected, atol=2e-14)


def test_complex_two_state_fit_reports_nonuniqueness_even_when_residual_is_zero():
    states = boundary_states(known_matrix(lossless=False), field_states(2))
    result = extract_reciprocal_complex_least_squares(states, reference_impedance=1.0)

    assert not result.unique
    assert result.rank < 10
    assert np.isinf(result.condition_number)
    np.testing.assert_allclose(result.relative_residual, 0.0, atol=2e-14)
    np.testing.assert_allclose(result.state_relative_residual, 0.0, atol=2e-14)


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_zero_states_report_degeneracy_without_nan_parameters(extractor):
    zero = np.zeros((2, 3), complex)
    states = [BoundaryQuantities(zero, zero, zero, zero)]
    result = extractor(states, reference_impedance=50.0)

    assert np.all(result.rank == 0)
    assert not np.any(result.unique)
    assert np.all(np.isinf(result.condition_number))
    np.testing.assert_array_equal(result.parameters.as_matrix(), np.zeros((4, 4, 3)))
    np.testing.assert_array_equal(result.relative_residual, 0.0)


def test_imaginary_reciprocal_coupling_is_not_silently_accepted_as_lossless():
    target = known_matrix(lossless=True)
    target[:2, 2:] *= 1j
    target[2:, :2] *= 1j
    states = boundary_states(target, np.eye(4, dtype=complex))
    complex_result = extract_reciprocal_complex_least_squares(states, reference_impedance=1.0)
    lossless_result = extract_reciprocal_lossless_least_squares(states, reference_impedance=1.0)

    np.testing.assert_allclose(complex_result.relative_residual, 0.0, atol=2e-14)
    assert lossless_result.relative_residual > 0.1
    assert lossless_result.unique
    matrix = lossless_result.parameters.as_matrix()
    np.testing.assert_allclose(matrix + matrix.conj().T, 0.0, atol=2e-14)


def test_reports_residuals_for_every_prescribed_state():
    states = boundary_states(known_matrix(lossless=False), field_states(5))
    last = states[-1]
    states[-1] = BoundaryQuantities(
        last.electric_average,
        last.magnetic_average,
        last.electric_current + np.array([0.7, -0.4j]),
        last.magnetic_current,
    )
    result = extract_reciprocal_complex_least_squares(states, reference_impedance=1.0)

    assert result.unique
    assert result.relative_residual > 0.01
    assert result.state_relative_residual.shape == (5,)
    assert result.state_relative_residual[-1] > 0.01


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_extraction_is_covariant_under_oriented_tangent_basis_rotation(extractor):
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    zero = np.zeros((2, 2))
    transform = np.block([[rotation, zero], [zero, rotation]])
    matrix = known_matrix(lossless=extractor is EXTRACTORS[1])
    rotated_matrix = transform @ matrix @ transform.T
    rotated_fields = transform @ field_states(4)
    result = extractor(boundary_states(rotated_matrix, rotated_fields), reference_impedance=50.0)

    assert result.unique
    np.testing.assert_allclose(result.parameters.as_matrix(), rotated_matrix, atol=2e-13)


def test_general_lossless_extraction_contains_the_existing_omega_specialization():
    blocks = omega_tensor_blocks(
        OmegaParameters(B=np.array(0.3), X=np.array(1.7), chi=np.array(-0.2))
    )
    matrix = np.block([[blocks["Y"], blocks["Chi"]], [blocks["Gamma"], blocks["Z"]]])
    result = extract_reciprocal_lossless_least_squares(
        boundary_states(matrix, field_states(4)), reference_impedance=50.0
    )

    assert result.unique
    np.testing.assert_allclose(result.parameters.as_matrix(), matrix, atol=2e-13)


@pytest.mark.parametrize("extractor", EXTRACTORS)
@pytest.mark.parametrize("impedance", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_reference_impedance(extractor, impedance):
    with pytest.raises(ValueError, match="reference_impedance"):
        extractor([], reference_impedance=impedance)


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_rejects_missing_states_and_unknown_structure(extractor):
    with pytest.raises(ValueError, match="at least one"):
        extractor([], reference_impedance=50.0)
    with pytest.raises(ValueError, match="structure"):
        extractor([], reference_impedance=50.0, structure="unknown")


@pytest.mark.parametrize("bad", [np.ones((3,)), np.ones((2, 0)), np.full((2, 3), np.nan)])
def test_rejects_invalid_field_arrays(bad):
    good = np.ones((2, 3), complex)
    with pytest.raises(ValueError):
        extract_reciprocal_complex_least_squares(
            [BoundaryQuantities(good, good, good, bad)], reference_impedance=50.0
        )


def test_rejects_mismatched_shapes_between_states():
    first = np.ones((2, 3), complex)
    second = np.ones((2, 4), complex)
    states = [
        BoundaryQuantities(first, first, first, first),
        BoundaryQuantities(second, second, second, second),
    ]
    with pytest.raises(ValueError, match="same shape"):
        extract_reciprocal_complex_least_squares(states, reference_impedance=50.0)


def test_application_exports_complex_reference_and_all_state_diagnostics(tmp_path):
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        "v 1 0 0\nv -1 0 0\nv 0 1 0\nv 0 -1 0\nv 0 0 1\nv 0 0 -1\n"
        "f 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\n"
        "f 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n",
        encoding="ascii",
    )
    output = tmp_path / "inputs"
    configuration = load_workflow_configuration(DEFAULT_CONFIGURATION)
    export_sie_inputs(configuration, mesh_path=mesh_path, run_dir=output)
    records = np.fromfile(output / "face_tensors.bin", dtype="<c16").reshape(8, 32)
    matrix = records[:, :16].reshape(8, 4, 4)
    inverse = records[:, 16:].reshape(8, 4, 4)
    np.testing.assert_allclose(matrix @ inverse, np.broadcast_to(np.eye(4), (8, 4, 4)), atol=2e-12)
    manifest = json.loads((output / "sie_inputs.json").read_text())
    assert manifest["extraction_method"] == "reciprocal_complex"
    assert manifest["diagnostics"]["non_unique_face_count"] == 8
    assert len(manifest["diagnostics"]["maximum_state_relative_residual"]) == 2
    assert manifest["diagnostics"]["maximum_relative_residual"] < 1e-10
