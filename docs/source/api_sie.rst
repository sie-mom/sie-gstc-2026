.. _api_sie:

SIE Solver API (C++/CUDA)
=========================

This section documents the C++ and CUDA implementation of the Surface Integral Equation (SIE) solver.

CUDA Kernels
------------

Dielectric Assembly
^^^^^^^^^^^^^^^^^^^

The launch functions dispatch the corresponding CUDA assembly kernels in
``src/sie_mom/solvers/dielectric/dielectric_kernels.cu``.

.. doxygenfunction:: sie_mom::solvers::dielectric::t_pmchwt_kernel
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::dielectric::launch_t_pmchwt_kernel
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::dielectric::n_pmchwt_kernel
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::dielectric::launch_n_pmchwt_kernel
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::dielectric::n_muller_kernel
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::dielectric::launch_n_muller_kernel
   :project: sie_mom

GSTC Assembly
^^^^^^^^^^^^^

The matrix kernels are defined in ``include/sie_mom/cuda/sie_kernels_assembly.cuh``;
the excitation kernel is defined in ``include/sie_mom/cuda/sie_kernels_utils.cuh``.

.. doxygenfunction:: assemble_Z_calderon_kernel
   :project: sie_mom

.. doxygenfunction:: assemble_Z_calderon_eta0_scaled_kernel
   :project: sie_mom

.. doxygenfunction:: assemble_Z_mass_kernel
   :project: sie_mom

.. doxygenfunction:: assemble_Z_mass_eta0_scaled_kernel
   :project: sie_mom

.. doxygenfunction:: assemble_V_vector_eta0_scaled_kernel
   :project: sie_mom

CUDA Helper Functions
---------------------

This section documents the helper functions used by the main CUDA kernels. They are organized by functionality.

Mesh Geometry
^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/mesh_geometry.cuh
   :project: sie_mom

Semi-Analytic Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/semi_analytic_integrals.cuh
   :project: sie_mom

Standard Quadrature
^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/dunavant_quadrature.cuh
   :project: sie_mom

Math Utilities
^^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/math_utils.cuh
   :project: sie_mom

RWG Basis Functions Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/rwg_basis_utils.cuh
   :project: sie_mom

Green's Function Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfile:: include/sie_mom/cuda/green_functions_utils.cuh
   :project: sie_mom

GSTC Tensor & Basis Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: sie_mom::GstcBasisType
   :project: sie_mom

.. doxygenfunction:: sie_mom::get_face_basis_vectors
   :project: sie_mom

.. doxygenfunction:: sie_mom::get_spherical_basis_vectors
   :project: sie_mom

.. doxygenfunction:: sie_mom::get_local_basis_vectors
   :project: sie_mom

.. doxygenfunction:: sie_mom::rotate_tensor_2x2_to_3x3
   :project: sie_mom

.. doxygenfunction:: sie_mom::apply_perp_transform
   :project: sie_mom

.. doxygenfunction:: sie_mom::copy_tensor_3x3
   :project: sie_mom

Dielectric Solver
-----------------

.. doxygenfunction:: sie_mom::solvers::solve_dielectric_dense
   :project: sie_mom

.. doxygenenum:: sie_mom::solvers::DielectricFormulation
   :project: sie_mom

.. doxygenenum:: sie_mom::solvers::NtlIntegralForm
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::DielectricOptions
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::DielectricResult
   :project: sie_mom

GSTC Solvers
------------

.. doxygenfunction:: sie_mom::solvers::solve_gstc_dense
   :project: sie_mom

.. doxygenfunction:: sie_mom::solvers::solve_gstc_dense_batch
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieOptions
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieResult
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcExcitationCase
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieBatchResult
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieCaseResult
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieCaseMetrics
   :project: sie_mom

.. doxygenstruct:: sie_mom::solvers::GstcSieBatchMetrics
   :project: sie_mom


References
----------

David Dunavant, "High Degree Efficient Symmetrical Gaussian Quadrature Rules for the Triangle," International Journal for Numerical Methods in Engineering, Volume 21, 1985, pages 1129-1148.

James Lyness, Dennis Jespersen, "Moderate Degree Symmetric Quadrature Rules for the Triangle," Journal of the Institute of Mathematics and its Applications, Volume 15, Number 1, February 1975, pages 19-32.
