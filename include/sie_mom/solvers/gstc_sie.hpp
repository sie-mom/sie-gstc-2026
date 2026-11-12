#pragma once

/**
 * @file gstc_sie.hpp
 * @brief Dense GPU-accelerated multitrace SIE-GSTC solver for closed triangular metasurfaces.
 *
 * Implements single-excitation and multi-excitation batch dense solves for closed,
 * consistently outward-oriented triangular surface meshes governed by Generalized Sheet
 * Transition Conditions (GSTCs). Supports both the geometry-agnostic intrinsic face-edge
 * tangent frame (`GstcBasisType::face`, derived from ordered vertices `(v0, v1, v2)`)
 * and the origin-centered spherical tangent frame (`GstcBasisType::spherical`).
 */

#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/spherical_surface.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <vector>

namespace sie_mom {
namespace solvers {

using sie_mom::GstcBasisType;

/**
 * @brief Numerical, physical, and coordinate-frame inputs for the eta0-scaled dense GSTC SIE.
 */
struct GstcSieOptions {
    real_t frequency_hz = real_t(0);
    IsotropicMedium interior;
    IsotropicMedium exterior;

    int quadrature_rule = 17;
    real_t singularity_threshold_lambda0 = real_t(0.1);
    bool tensors_are_uniform = false;
    bool compute_residual = true;

    /**
     * Local tangential coordinate frame in which the input 2x2 GSTC tensor blocks
     * are expressed on each face:
     * - `GstcBasisType::face` (default): Intrinsic triangle-edge frame `(t1, t2)`
     *   computed on the fly from ordered face vertices `(v0, v1, v2)` via
     *   `get_face_basis_vectors`. Valid for arbitrary closed triangular meshes;
     *   requires parameter synthesis to preserve the exact `(i0, i1, i2)` face order.
     * - `GstcBasisType::spherical`: Origin-centered `(theta_hat, phi_hat)` frame
     *   at the face centroid via `get_spherical_basis_vectors`.
     */
    GstcBasisType basis = GstcBasisType::face;

    // Explicit constants keep the C++ solve consistent with the Python case.
    real_t vacuum_permittivity = real_t(8.8541878188e-12);
    real_t vacuum_permeability = real_t(1.25663706127e-6);
};

/**
 * @brief Physical parameters derived from GstcSieOptions for SIE matrix assembly.
 */
struct PhysicalParameters {
    real_t omega;                  ///< Angular frequency [rad/s]
    real_t k_interior;              ///< Interior wavenumber [rad/m]
    real_t k_exterior;              ///< Exterior wavenumber [rad/m]
    real_t eta0;                    ///< Characteristic impedance of vacuum [Ohm]
    real_t singularity_threshold;   ///< Quadrature singularity distance threshold
};

/**
 * @brief Computes physical wave parameters from user options.
 * @param options Solver configuration options.
 * @return Computed PhysicalParameters struct.
 */
inline PhysicalParameters make_physical_parameters(const GstcSieOptions& options) {
    const real_t pi = std::acos(real_t(-1));
    const real_t omega = real_t(2) * pi * options.frequency_hz;
    const real_t epsilon_interior =
        options.interior.relative_permittivity * options.vacuum_permittivity;
    const real_t mu_interior =
        options.interior.relative_permeability * options.vacuum_permeability;
    const real_t epsilon_exterior =
        options.exterior.relative_permittivity * options.vacuum_permittivity;
    const real_t mu_exterior =
        options.exterior.relative_permeability * options.vacuum_permeability;
    const real_t eta0 = std::sqrt(
        options.vacuum_permeability / options.vacuum_permittivity
    );
    const real_t c0 = real_t(1) / std::sqrt(
        options.vacuum_permeability * options.vacuum_permittivity
    );

    return {
        omega,
        omega * std::sqrt(mu_interior * epsilon_interior),
        omega * std::sqrt(mu_exterior * epsilon_exterior),
        eta0,
        options.singularity_threshold_lambda0 * c0 / options.frequency_hz,
    };
}

/**
 * @brief Timings and algebraic diagnostics returned by the dense solve.
 *
 * The RHS and residual norms refer to the internally scaled system
 * `D A D x_tilde = D b`, not to a mixture of differently scaled blocks.
 */
struct GstcSieMetrics {
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::size_t edge_count = 0;
    std::size_t matrix_dimension = 0;
    std::size_t dense_matrix_bytes = 0;

    double device_preparation_seconds = 0.0;
    double assembly_seconds = 0.0;
    double solve_seconds = 0.0;
    double residual_seconds = 0.0;

    double rhs_norm = 0.0;
    double absolute_residual = 0.0;
    double relative_residual = 0.0;
    bool residual_computed = false;

    double maximum_j1_coefficient = 0.0;
    double maximum_m1_coefficient = 0.0;
    double maximum_j2_coefficient = 0.0;
    double maximum_m2_coefficient = 0.0;
};

/**
 * @brief Result of the eta0-scaled solve.
 *
 * `physical_currents` is ordered as `[J1, M1, J2, M2]`, with one coefficient per
 * RWG edge in each block. Magnetic-current coefficients have already been
 * multiplied by `eta0`; no numerical scaling remains in the returned vector.
 */
struct GstcSieResult {
    std::vector<std::complex<real_t>> physical_currents;
    GstcSieMetrics metrics;
};

/**
 * @brief Incident excitations forming one RHS, not prescribed GSTC total fields.
 *
 * An absent region contributes zero. A solve requires at least one source,
 * each matching its slot's region and the solver's frequency/medium settings.
 * Case IDs, field references, and output paths belong to the application.
 */
struct GstcExcitationCase {
    std::optional<excitation::PreparedExcitation> interior;
    std::optional<excitation::PreparedExcitation> exterior;
};

/** Shared problem dimensions and timings, recorded once for the whole batch. */
struct GstcSieBatchMetrics {
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::size_t edge_count = 0;
    std::size_t matrix_dimension = 0;
    std::size_t dense_matrix_bytes = 0;
    std::size_t case_count = 0;

    double device_preparation_seconds = 0.0;
    double matrix_assembly_seconds = 0.0;
    double factorization_seconds = 0.0;
    double solve_seconds = 0.0; // All RHS solves; excludes LU factorization.
    double residual_matrix_assembly_seconds = 0.0;
};

/**
 * @brief Per-case RHS assembly, residual evaluation, and physical-current diagnostics.
 *
 * Norms use the eta0-scaled system, as in `GstcSieMetrics`. Shared timings are
 * not copied here or divided into artificial per-case solve times.
 */
struct GstcSieCaseMetrics {
    double rhs_assembly_seconds = 0.0;
    double residual_seconds = 0.0;

    double rhs_norm = 0.0;
    double absolute_residual = 0.0;
    double relative_residual = 0.0;
    bool residual_computed = false;

    double maximum_j1_coefficient = 0.0;
    double maximum_m1_coefficient = 0.0;
    double maximum_j2_coefficient = 0.0;
    double maximum_m2_coefficient = 0.0;
};

/** Physical currents use the same `[J1, M1, J2, M2]` ordering as the single solve. */
struct GstcSieCaseResult {
    std::vector<std::complex<real_t>> physical_currents;
    GstcSieCaseMetrics metrics;
};

/** One result per input case, in input order, with shared timings kept separate. */
struct GstcSieBatchResult {
    std::vector<GstcSieCaseResult> cases;
    GstcSieBatchMetrics metrics;
};

/**
 * @brief Solve the dense reciprocal/general GSTC SIE on a closed triangular surface mesh.
 *
 * `gstc_tensors` contains physical, unscaled fields-to-currents maps followed by
 * their inverses, expressed in the local tangential frame selected by `options.basis`
 * (`GstcBasisType::face` or `GstcBasisType::spherical`). It must contain 32 values for
 * a uniform sheet or `32 * mesh.faces.size()` values for per-face data, according to
 * `options.tensors_are_uniform`. Phasors follow `exp(+j omega t)`.
 * `source` is prepared once and reused for incident-field evaluation.
 */
GstcSieResult solve_gstc_dense(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& gstc_tensors,
    const GstcSieOptions& options,
    const excitation::PreparedExcitation& source
);

/**
 * @brief Solve multiple excitation cases with one shared matrix and LU factorization.
 *
 * All cases share `mesh`, `gstc_tensors`, and `options` (including `options.basis`).
 * Each case forms one RHS from its interior/exterior incident contributions. Tensor
 * layout, phasor convention, and physical-current scaling match `solve_gstc_dense`.
 * The batch must be nonempty, and sources must satisfy the case contract.
 *
 * Assembles and factorizes once, then solves all RHS together. Each assembled
 * RHS must have a finite, nonzero norm. If residuals are requested, reassembles
 * the original matrix once in the same storage after all solves; no second
 * dense matrix is required.
 */
GstcSieBatchResult solve_gstc_dense_batch(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& gstc_tensors,
    const GstcSieOptions& options,
    const std::vector<GstcExcitationCase>& cases
);

} // namespace solvers
} // namespace sie_mom

