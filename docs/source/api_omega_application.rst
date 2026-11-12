.. _api_omega_application:

Omega-Biisotropic Sphere Application API (Python)
==================================================

These application-level routines construct the prescribed dipole
transformation, extract its omega-sheet parameters, and run the SVWF
validation.  The source links generated beside each signature show the
explicit phasor-convention conversions used by the workflow.

.. autofunction:: applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching.prescribed_dipole_fields

.. autofunction:: applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching.extract_case_parameters

.. autofunction:: applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching.run_svwf_validation

.. autofunction:: applications.omega_biisotropic_sphere.svwf_dipole_impedance_matching._evaluate_svwf_total_fields

