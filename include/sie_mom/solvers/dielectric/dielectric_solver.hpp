#pragma once

#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/excitation/excitation.hpp"

#include <complex>
#include <vector>

namespace sie_mom {
namespace solvers {

/**
 * @brief Specifies the testing formulation for the dielectric solver.
 */
enum class DielectricFormulation {
    T_PMCHWT, ///< Tested with standard RWG functions (tangential continuity)
    N_PMCHWT, ///< Tested with RWG x n functions (rotated)
    N_Muller  ///< Tested with RWG x n functions using the Muller formulation.
};

/** Selects the nTL integral for N-PMCHWT; nTK is the same in both forms. */
enum class NtlIntegralForm {
    Line,    ///< Form A: testing-edge line integral
    Surface  ///< Form B: double-area integral
};

/**
 * @brief Options for the dense dielectric solver.
 */
struct DielectricOptions {
    real_t frequency_hz = 0;
    IsotropicMedium interior;
    IsotropicMedium exterior;

    int quadrature_rule = 17;
    real_t singularity_threshold_lambda0 = 0.1;
    bool compute_residual = true;

    real_t vacuum_permittivity = 8.8541878128e-12;
    real_t vacuum_permeability = 1.25663706212e-6;

    DielectricFormulation formulation = DielectricFormulation::T_PMCHWT;
    NtlIntegralForm ntl_form = NtlIntegralForm::Surface;
    int line_quadrature_points = 20; ///< Used only for N-PMCHWT Form A.
};

/**
 * @brief Metrics and performance data collected during the solver run.
 */
struct DielectricMetrics {
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::size_t edge_count = 0;
    std::size_t matrix_dimension = 0;
    std::size_t dense_matrix_bytes = 0;

    double device_preparation_seconds = 0.0;
    double assembly_seconds = 0.0;
    double solve_seconds = 0.0;
    double residual_seconds = 0.0;

    bool residual_computed = false;
    double absolute_residual = 0.0;
    double relative_residual = 0.0;

    double maximum_j_coefficient = 0.0;
    double maximum_m_coefficient = 0.0;
};

/**
 * @brief Results from the dense dielectric solver.
 */
struct DielectricResult {
    /// Physical interior-oriented currents: [J1, M1]. The exterior pair is [-J1, -M1].
    /// Each block has one coefficient per mesh edge.
    std::vector<std::complex<real_t>> physical_currents;
    DielectricMetrics metrics;
};

/**
 * @brief Solves the linear system for a dielectric body using a dense matrix formulation.
 *
 * This function handles T-PMCHWT, N-PMCHWT, and N-Müller formulations based on the provided
 * options. It delegates the massively parallel matrix assembly to the GPU and utilizes cuSolverDn
 * to factorize and solve the resulting dense linear system.
 *
 * @param mesh The triangulated surface mesh of the dielectric body.
 * @param options Configuration options, materials, and formulation settings.
 * @param source Prepared excitation reused for incident-field evaluation.
 * @return DielectricResult containing the raw physical J and M current coefficients and solver metrics.
 */
DielectricResult solve_dielectric_dense(
    const mesh::MeshData& mesh,
    const DielectricOptions& options,
    const excitation::PreparedExcitation& source
);

} // namespace solvers
} // namespace sie_mom
