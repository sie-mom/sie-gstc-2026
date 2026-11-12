#pragma once

#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/spherical_surface.hpp"

#include <complex>
#include <cstddef>
#include <vector>

namespace sie_mom::solvers {

enum class PecFormulation { TEfie, NEfie, NMfie };
enum class PecNtlForm { Line, Surface };

/** Exterior-region PEC EFIE or MFIE; the unknown is one electric-current coefficient per edge. */
struct PecOptions {
    real_t frequency_hz = real_t(0);
    IsotropicMedium exterior;
    int quadrature_rule = 7;
    real_t singularity_threshold_lambda0 = real_t(0.1);
    bool compute_residual = true;
    bool capture_assembled_matrix = false;
    PecFormulation formulation = PecFormulation::TEfie;
    PecNtlForm ntl_form = PecNtlForm::Surface;
    int line_quadrature_points = 20;
    real_t vacuum_permittivity = real_t(8.8541878128e-12);
    real_t vacuum_permeability = real_t(1.25663706212e-6);
};

struct PecMetrics {
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::size_t edge_count = 0;
    std::size_t matrix_dimension = 0;
    std::size_t dense_matrix_bytes = 0;
    double assembly_seconds = 0.0;
    double solve_seconds = 0.0;
    double residual_seconds = 0.0;
    bool residual_computed = false;
    double absolute_residual = 0.0;
    double relative_residual = 0.0;
    double maximum_j_coefficient = 0.0;
};

struct PecResult {
    /** Exterior-oriented physical current J2, in mesh edge order. */
    std::vector<std::complex<real_t>> exterior_current;
    /** Optional final LHS matrix before LU, column-major in mesh edge order. */
    std::vector<std::complex<real_t>> assembled_matrix;
    PecMetrics metrics;
};

/** Solve the selected exterior PEC equation for the RWG coefficients of J2.
 * T-EFIE uses TL; N-EFIE uses nTL Form A or B.
 * N-MFIE uses 1/2 M - nTK from the Form A routine with zero 1D points.
 */
PecResult solve_pec_dense(
    const mesh::MeshData& mesh,
    const PecOptions& options,
    const excitation::PreparedExcitation& source
);

} // namespace sie_mom::solvers
