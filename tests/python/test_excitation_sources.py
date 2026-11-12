import numpy as np

from sie_mom.excitation import (
    evaluate_dipole_fields,
    evaluate_plane_wave_fields,
)
from spherical_expansion.solver import evaluate_incident_plane_wave


def test_package_exports_existing_field_implementations():
    from sie_mom.excitation import fields

    assert evaluate_dipole_fields is fields.evaluate_dipole_fields
    assert evaluate_plane_wave_fields is fields.evaluate_plane_wave_fields


def test_plane_wave_matches_legacy_reference_values():
    theta = np.array([0.4, 1.1])
    phi = np.array([-0.2, 0.8])

    electric, magnetic = evaluate_plane_wave_fields(
        2.1,
        1.3,
        theta,
        phi,
        250.0,
    )

    expected_electric = np.array(
        [
            [
                -0.30904010105437385 + 0.22395411035363358j,
                0.20265711768861103 + 0.5869067247137751j,
            ],
            [
                -0.7309485757049575 + 0.5297012828683273j,
                0.10314600920318241 + 0.2987168036296352j,
            ],
            [
                -0.16086948931820835 + 0.1165783444944056j,
                -0.2341358381718291 - 0.6780709184401138j,
            ],
        ]
    )
    expected_magnetic = np.array(
        [
            [
                0.00025058211943334 - 0.00018159098264839j,
                0.00083465432893155 + 0.00241720717262979j,
            ],
            [
                0.00059268244694466 - 0.00042950306343691j,
                0.00042481243232587 + 0.00123028135462366j,
            ],
            [
                -0.00317437642225317 + 0.0023003961141217j,
                0.00090958457827207 + 0.00263420950506206j,
            ],
        ]
    )

    np.testing.assert_allclose(electric, expected_electric, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(magnetic, expected_magnetic, rtol=1e-13, atol=1e-14)


def test_plane_wave_grid_has_component_first_shape():
    electric, magnetic = evaluate_plane_wave_fields(
        1.4,
        2.0,
        np.array([0.3, 1.0]),
        np.array([-0.4, 0.2, 1.2]),
        377.0,
        grid=True,
    )

    assert electric.shape == (3, 2, 3)
    assert magnetic.shape == (3, 2, 3)


def test_plane_wave_accepts_arbitrary_transverse_cartesian_vectors():
    electric, magnetic = evaluate_plane_wave_fields(
        1.4,
        2.0,
        np.array([0.3, 1.0]),
        np.array([-0.4, 0.2]),
        377.0,
        electric_field=(0.0, 0.0, 2.0),
        propagation_direction=(3.0, 0.0, 0.0),
    )

    assert electric.shape == (3, 2)
    assert magnetic.shape == (3, 2)
    assert np.all(np.isfinite(electric))
    assert np.all(np.isfinite(magnetic))


def test_spherical_expansion_plane_wave_wrapper_preserves_grid_api():
    theta = np.array([0.4, 1.1])
    phi = np.array([-0.2, 0.8, 1.4])
    electric, magnetic = evaluate_incident_plane_wave(
        2.1,
        1.3,
        theta,
        phi,
        E0=1.7,
        eta=250.0,
        grid=True,
    )

    expected_electric, expected_magnetic = evaluate_plane_wave_fields(
        2.1,
        1.3,
        theta,
        phi,
        250.0,
        electric_field=(1.7, 0.0, 0.0),
        grid=True,
    )

    assert isinstance(electric, tuple)
    assert isinstance(magnetic, tuple)
    np.testing.assert_allclose(np.asarray(electric), expected_electric)
    np.testing.assert_allclose(np.asarray(magnetic), expected_magnetic)


def test_electric_dipole_matches_legacy_reference_values():
    theta = np.array([0.4, 1.1, 2.2])
    phi = np.array([0.2, -0.7, 1.4])

    electric, magnetic = evaluate_dipole_fields(
        2.3,
        1.7,
        theta,
        phi,
        376.730313668,
        effective_moment=(0.3, -0.2, 0.7),
        source_position=(0.1, -0.05, 0.2),
    )

    expected_electric = np.array(
        [
            [
                -0.8766734018716473 - 1.3279877234972548j,
                -0.8420366206748314 - 0.9038366460445865j,
                0.7511603635686179 + 0.3139138131114426j,
            ],
            [
                0.10718785495576433 - 0.06605161598780035j,
                1.3010096499799588 - 1.1260971925637315j,
                0.3106282677581854 - 1.1697442312939414j,
            ],
            [
                0.7904038876480776 - 0.606142556721735j,
                -0.12206195660425856 + 0.12005888205059356j,
                0.21551816503149507 - 0.7947412065810058j,
            ],
        ]
    )
    expected_magnetic = np.array(
        [
            [
                -1.9627120808407366e-5 + 1.2160576651332037e-5j,
                3.5265411385621242e-5 - 2.5659567314989209e-5j,
                -4.0024636701953037e-6 + 1.7788381403479247e-5j,
            ],
            [
                -2.4312120094536153e-3 + 1.5063309736156264e-3j,
                3.9776406562908034e-4 - 2.8941825478475834e-4j,
                -5.0843710879607137e-4 + 2.2596765283082335e-3j,
            ],
            [
                3.0793095690888345e-4 - 1.9078794293682084e-4j,
                3.996332137705714e-3 - 2.9077827103506718e-3j,
                7.467650367729249e-4 - 3.3188911599169495e-3j,
            ],
        ]
    )

    np.testing.assert_allclose(electric, expected_electric, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(magnetic, expected_magnetic, rtol=1e-13, atol=1e-14)


def test_magnetic_dipole_uses_electromagnetic_duality():
    eta = 220.0
    arguments = dict(
        k=1.8,
        radius=np.array([1.2, 1.6]),
        theta=np.array([0.6, 1.4]),
        phi=np.array([-0.3, 0.9]),
        eta=eta,
        effective_moment=(0.2, 0.4, -0.1),
        source_position=(-0.1, 0.2, 0.05),
    )

    electric_e, magnetic_e = evaluate_dipole_fields(**arguments)
    electric_m, magnetic_m = evaluate_dipole_fields(
        **arguments, dipole_type="magnetic"
    )

    np.testing.assert_allclose(electric_m, -eta * magnetic_e)
    np.testing.assert_allclose(magnetic_m, electric_e / eta)


def test_dipole_field_output_preserves_broadcast_shape():
    radius = np.array([[1.0], [1.5]])
    theta = np.array([[0.4], [1.2]])
    phi = np.array([[0.1, 0.7, 1.4]])

    electric, magnetic = evaluate_dipole_fields(2.0, radius, theta, phi, 377.0)

    assert electric.shape == (3, 2, 3)
    assert magnetic.shape == (3, 2, 3)
