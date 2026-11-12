import numpy as np
import pytest

from spherical_expansion.svwf_sphere import compute_fields, compute_fields_chunked


def test_chunked_process_fields_match_serial_order_and_values():
    point_count = 23
    radius = np.linspace(0.4, 1.7, point_count)
    theta = np.linspace(0.1, np.pi - 0.1, point_count)
    phi = np.linspace(-np.pi, np.pi, point_count, endpoint=False)
    modes = [(m, n) for n in range(1, 4) for m in range(-n, n + 1)]
    a_coefficients = {
        mode: complex(index / 13, -index / 17)
        for index, mode in enumerate(modes, start=1)
    }
    b_coefficients = {
        mode: complex(-index / 19, index / 23)
        for index, mode in enumerate(modes, start=1)
    }

    serial = compute_fields(
        3.2,
        radius,
        theta,
        phi,
        a_coefficients,
        b_coefficients,
        kind=3,
        eta=2.5,
        grid=False,
    )
    parallel = compute_fields_chunked(
        3.2,
        radius,
        theta,
        phi,
        a_coefficients,
        b_coefficients,
        kind=3,
        eta=2.5,
        workers=2,
        chunk_size=5,
    )

    for serial_field, parallel_field in zip(serial, parallel):
        for serial_component, parallel_component in zip(
            serial_field, parallel_field
        ):
            np.testing.assert_allclose(
                parallel_component,
                serial_component,
                rtol=2e-15,
                atol=2e-15,
            )


@pytest.mark.parametrize(
    ("workers", "chunk_size", "message"),
    ((0, None, "workers"), (1, 0, "chunk_size")),
)
def test_chunked_process_fields_validate_controls(workers, chunk_size, message):
    coordinates = np.ones(2)
    with pytest.raises(ValueError, match=message):
        compute_fields_chunked(
            1.0,
            coordinates,
            coordinates,
            coordinates,
            {(0, 1): 1.0},
            {(0, 1): 0.0},
            workers=workers,
            chunk_size=chunk_size,
        )
