#ifndef DUNAVANT_QUADRATURE_CUH
#define DUNAVANT_QUADRATURE_CUH

#include "platform.cuh"
#include "math_utils.cuh"
#include "mesh_geometry.cuh"
#include "rwg_basis_utils.cuh"
#include "semi_analytic_integrals.cuh"
#include "green_functions_utils.cuh"
#include "sie_mom/excitation/source_functors.cuh"
#include <type_traits>

//// Begin Generic modifier/operator quadrature helpers
// These helpers and the generic calculateDoubleQuadrature overload are used by tests.
// Review those uses before considering removal.

namespace detail
{
    template <typename T>
    struct always_false : std::false_type {};
}


template <typename ModifierValue, typename KernelValue>
__host__ __device__ inline complex__float evaluateModifierOperator(
    const ModifierValue& modifier_value,
    const KernelValue& kernel_value)
{
    if constexpr (std::is_same<ModifierValue, real_t>::value &&
                  std::is_same<KernelValue, complex__float>::value)
    {
        return complexScale(kernel_value, modifier_value);
    }
    else if constexpr (std::is_same<ModifierValue, vec3_t>::value &&
                       std::is_same<KernelValue, cvec3_t>::value)
    {
        return dot(modifier_value, kernel_value);
    }
    else
    {
        static_assert(detail::always_false<ModifierValue>::value,
                      "Unsupported modifier/operator combination");
        return make_complex(0.f, 0.f);
    }
}

namespace detail
{
    template <typename ModifierKernel>
    __host__ __device__ inline auto invokeModifier(const ModifierKernel& modifier,
                                                   int edge_j,
                                                   int edge_i,
                                                   const vec3_t& rm,
                                                   const vec3_t& rm_free,
                                                   real_t emj,
                                                   const vec3_t& rn,
                                                   const vec3_t& rn_free,
                                                   real_t eni,
                                                   real_t k)
    {
        if constexpr (std::is_invocable_v<ModifierKernel, int, int, const vec3_t&, const vec3_t&, real_t,
                                           const vec3_t&, const vec3_t&, real_t, real_t>)
        {
            return modifier(edge_j, edge_i, rm, rm_free, emj, rn, rn_free, eni, k);
        }
        else if constexpr (std::is_invocable_v<ModifierKernel, const vec3_t&, const vec3_t&,
                                                const vec3_t&, const vec3_t&, real_t>)
        {
            return modifier(rm, rm_free, rn, rn_free, k);
        }
        else if constexpr (std::is_invocable_v<ModifierKernel, const vec3_t&, const vec3_t&,
                                                const vec3_t&, const vec3_t&>)
        {
            return modifier(rm, rm_free, rn, rn_free);
        }
        else
        {
            static_assert(detail::always_false<ModifierKernel>::value,
                          "Unsupported modifier functor signature");
        }
    }

    template <typename OperatorKernel>
    __host__ __device__ inline auto invokeOperator(const OperatorKernel& op,
                                                   const vec3_t& rm,
                                                   const vec3_t& rn,
                                                   real_t k)
    {
        if constexpr (std::is_invocable_v<OperatorKernel, const vec3_t&, const vec3_t&, real_t>)
        {
            return op(rm, rn, k);
        }
        else if constexpr (std::is_invocable_v<OperatorKernel, const vec3_t&, const vec3_t&>)
        {
            return op(rm, rn);
        }
        else
        {
            static_assert(detail::always_false<OperatorKernel>::value,
                          "Unsupported operator functor signature");
        }
    }
}

// Create double quadrature of the type M*D

template <typename ModifierKernel, typename OperatorKernel>
__host__ __device__ inline void calculateDoubleQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    real_t k,
    const ModifierKernel& modifier_functor,
    const OperatorKernel& operator_functor,
    complex__float result_matrix[3][3])
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix[row][col] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop.
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi_p = static_cast<real_t>(ref_points[2 * p + 0]);
        const real_t eta_p = static_cast<real_t>(ref_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi_p, eta_p, rm_p);

        for (int q = 0; q < num_quad_points; ++q)
        {
            const real_t xi_q = static_cast<real_t>(ref_points[2 * q + 0]);
            const real_t eta_q = static_cast<real_t>(ref_points[2 * q + 1]);
            const real_t w_q = static_cast<real_t>(quad_weights[q]);

            vec3_t rn_q;
            mapTriQuadPoint(source_tri, xi_q, eta_q, rn_q);

            const auto kernel_value = detail::invokeOperator(operator_functor, rm_p, rn_q, k);
            const real_t combined_weight = w_p * w_q;

            for (int j = 0; j < 3; ++j)
            {
                const vec3_t& rm_free = test_free_vertices[j];
                const real_t emj = test_edge_lengths[j];

                for (int i = 0; i < 3; ++i)
                {
                    const vec3_t& rn_free = source_free_vertices[i];
                    const real_t eni = source_edge_lengths[i];

                    const auto modifier_value = detail::invokeModifier(
                        modifier_functor, j, i, rm_p, rm_free, emj, rn_q, rn_free, eni, k);

                    const complex__float integrand = evaluateModifierOperator(modifier_value, kernel_value);
                    const complex__float weighted_integrand = complexScale(integrand, combined_weight);
                    complexAddInPlace(result_matrix[j][i], weighted_integrand);
                }
            }
            
        }
    }

    // Apply the final scaling factor (edge lengths) to all elements
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            result_matrix[j][i] = complexScale(result_matrix[j][i], emj * eni);
        }
    }
}

//// End Generic modifier/operator quadrature helpers

//// Begin Standard quadrature for TL and TK operators

/**
 * @brief Computes the double surface numerical quadrature for the standard TL and TK operators.
 *
 * This routine evaluates the interactions for the standard TL and TK operators using 
 * Dunavant quadrature. It is designed to evaluate either the standard kernels (for far-field 
 * interactions) or the analytically extracted regular parts of the kernels (for near-field 
 * interactions) by toggling the `is_singular` flag.
 */
__host__ __device__ inline void calculateDoubleQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    real_t k1,
    real_t k2,
    bool is_singular,
    complex__float result_matrix_tl_k1[3][3],
    complex__float result_matrix_tl_k2[3][3],
    complex__float result_matrix_tk_k1[3][3],
    complex__float result_matrix_tk_k2[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix_tl_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_tl_k2[row][col] = make_complex(0.f, 0.f);
            result_matrix_tk_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_tk_k2[row][col] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop.
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi_p = static_cast<real_t>(ref_points[2 * p + 0]);
        const real_t eta_p = static_cast<real_t>(ref_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi_p, eta_p, rm_p);

        for (int q = 0; q < num_quad_points; ++q)
        {
            const real_t xi_q = static_cast<real_t>(ref_points[2 * q + 0]);
            const real_t eta_q = static_cast<real_t>(ref_points[2 * q + 1]);
            const real_t w_q = static_cast<real_t>(quad_weights[q]);

            vec3_t rn_q;
            mapTriQuadPoint(source_tri, xi_q, eta_q, rn_q);

            const IntegrandKernels kernels = is_singular ? 
                                             integrandKernelsRegular(rm_p, rn_q, k1, k2) :
                                             integrandKernels(rm_p, rn_q, k1, k2);

            const real_t combined_weight = w_p * w_q;

            for (int j = 0; j < 3; ++j)
            {
                const vec3_t& rm_free = test_free_vertices[j];
                const vec3_t rho_mj_p = rhoRwg(rm_p, rm_free);

                for (int i = 0; i < 3; ++i)
                {
                    const vec3_t& rn_free = source_free_vertices[i];
                    const vec3_t rho_ni_q = rhoRwg(rn_q, rn_free);

                    // --- Calculate Modifiers ---
                    const real_t dot_basis_rwg = real_t(0.25) * dot(rho_mj_p, rho_ni_q);
                    const real_t tl_modifier_value1 = dot_basis_rwg - real_t(1) / (k1 * k1);
                    const real_t tl_modifier_value2 = dot_basis_rwg - real_t(1) / (k2 * k2);
                    const vec3_t tk_modifier_value = 0.25f * cross(rho_mj_p, rho_ni_q);

                    // --- Calculate and weight each integrand ---
                    const complex__float integrand_tl1 = complexScale(kernels.tl_k1, tl_modifier_value1);
                    const complex__float integrand_tl2 = complexScale(kernels.tl_k2, tl_modifier_value2);
                    const complex__float integrand_tk1 = dot(tk_modifier_value, kernels.tk_k1);
                    const complex__float integrand_tk2 = dot(tk_modifier_value, kernels.tk_k2);

                    complexAddInPlace(result_matrix_tl_k1[j][i], complexScale(integrand_tl1, combined_weight));
                    complexAddInPlace(result_matrix_tl_k2[j][i], complexScale(integrand_tl2, combined_weight));

                    // For coincident triangles (self-term), the surface integral part of the TK operator is zero.
                    // We skip the accumulation to avoid adding numerical noise.
                    if (!(test_tri_indices[0] == source_tri_indices[0] &&
                          test_tri_indices[1] == source_tri_indices[1] &&
                          test_tri_indices[2] == source_tri_indices[2])) {
                        // --- Accumulate results into their respective matrices (with quad weights only) ---
                        // Operators are orincipla value, so the self test is extrated ouside as +/- 1/2
                        complexAddInPlace(result_matrix_tk_k1[j][i], complexScale(integrand_tk1, combined_weight));
                        complexAddInPlace(result_matrix_tk_k2[j][i], complexScale(integrand_tk2, combined_weight));
                    }
                }
            }
        }
    }

    // Apply the final scaling factor (edge lengths) to all elements
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            const real_t final_scaling = emj * eni;
            result_matrix_tl_k1[j][i] = complexScale(result_matrix_tl_k1[j][i], final_scaling);
            result_matrix_tl_k2[j][i] = complexScale(result_matrix_tl_k2[j][i], final_scaling);
            result_matrix_tk_k1[j][i] = complexScale(result_matrix_tk_k1[j][i], final_scaling);
            result_matrix_tk_k2[j][i] = complexScale(result_matrix_tk_k2[j][i], final_scaling);
        }
    }
}

//// End Standard quadrature for TL and TK operators


//// Begin Standard quadrature for nTL and nTK operators
/**
 * @brief Computes the double surface numerical quadrature for the nTL and nTK operators.
 * Form B of the nTL and nTK operators.
 * Evaluates the interactions for the nTL and nTK operators, which are used in 
 * formulations like N-PMCHWT. Unlike the standard TL operator, nTL avoids surface 
 * integration by parts on the testing triangle, meaning the gradient of the scalar 
 * potential remains on the Green's function.
 *
 * If `is_singular` is true, the routine evaluates the analytically extracted regular 
 * parts of the integrands (using `integrandKernelsRegular`). Otherwise, it evaluates 
 * the standard integrands.
 */
__host__ __device__ inline void calculateDoubleQuadrature_nTL_nTK(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    real_t k1,
    real_t k2,
    bool is_singular,
    complex__float result_matrix_ntl_k1[3][3],
    complex__float result_matrix_ntl_k2[3][3],
    complex__float result_matrix_ntk_k1[3][3],
    complex__float result_matrix_ntk_k2[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix_ntl_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntl_k2[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntk_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntk_k2[row][col] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate testing face normal
    const vec3_t e1 = test_tri[1] - test_tri[0];
    const vec3_t e2 = test_tri[2] - test_tri[0];
    const vec3_t n_m = normalize(cross(e1, e2));

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop.
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    const bool is_self =
        test_tri_indices[0] == source_tri_indices[0] &&
        test_tri_indices[1] == source_tri_indices[1] &&
        test_tri_indices[2] == source_tri_indices[2];

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi_p = static_cast<real_t>(ref_points[2 * p + 0]);
        const real_t eta_p = static_cast<real_t>(ref_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi_p, eta_p, rm_p);

        for (int q = 0; q < num_quad_points; ++q)
        {
            const real_t xi_q = static_cast<real_t>(ref_points[2 * q + 0]);
            const real_t eta_q = static_cast<real_t>(ref_points[2 * q + 1]);
            const real_t w_q = static_cast<real_t>(quad_weights[q]);

            vec3_t rn_q;
            mapTriQuadPoint(source_tri, xi_q, eta_q, rn_q);

            const IntegrandKernels kernels = is_singular ? 
                                             integrandKernelsRegular(rm_p, rn_q, k1, k2) :
                                             integrandKernels(rm_p, rn_q, k1, k2);

            const real_t combined_weight = w_p * w_q;

            for (int j = 0; j < 3; ++j)
            {
                const vec3_t& rm_free = test_free_vertices[j];
                const vec3_t rho_mj_p = rhoRwg(rm_p, rm_free);
                const vec3_t tau_mj_p = cross(rho_mj_p, n_m);

                for (int i = 0; i < 3; ++i)
                {
                    const vec3_t& rn_free = source_free_vertices[i];
                    const vec3_t rho_ni_q = rhoRwg(rn_q, rn_free);

                    // --- Calculate Modifiers for nTL ---
                    // nTL = 1/4 * (tau . rho) * G - 1/(2*k^2) * tau . nabla G
                    const real_t dot_tau_rho = dot(tau_mj_p, rho_ni_q);
                    
                    complex__float ntl1 = complexScale(kernels.tl_k1, real_t(0.25) * dot_tau_rho);
                    complex__float ntl2 = complexScale(kernels.tl_k2, real_t(0.25) * dot_tau_rho);
                    
                    // Note: kernels.tk_k1 contains \nabla' G. This is why it is used in the nTL 
                    // scalar potential term, which requires the evaluation of \tau \cdot \nabla' G.
                    // Diagnostic: omit this quadrature routine's scalar self contribution.
                    // The separately extracted singular self term is skipped in its own routine.
                    // if (!is_self) {
                        const complex__float grad1 = dot(tau_mj_p, kernels.tk_k1);
                        const complex__float grad2 = dot(tau_mj_p, kernels.tk_k2);

                        ntl1 = complexAdd(ntl1, complexScale(grad1, real_t(-0.5) / (k1 * k1)));
                        ntl2 = complexAdd(ntl2, complexScale(grad2, real_t(-0.5) / (k2 * k2)));
                    //}

                    // --- Calculate Modifiers for nTK ---
                    // nTK = 1/4 * (tau x rho) . nabla G
                    // We apply the vector identity (tau x rho) . nabla G = (tau x rho) . kernels.tk
                    const vec3_t ntk_modifier = real_t(0.25) * cross(tau_mj_p, rho_ni_q);
                    const complex__float ntk1 = dot(ntk_modifier, kernels.tk_k1);
                    const complex__float ntk2 = dot(ntk_modifier, kernels.tk_k2);

                    complexAddInPlace(result_matrix_ntl_k1[j][i], complexScale(ntl1, combined_weight));
                    complexAddInPlace(result_matrix_ntl_k2[j][i], complexScale(ntl2, combined_weight));

                    // For coincident triangles (self-term), the surface integral part of the nTK operator is zero.
                    // We skip the accumulation to avoid adding numerical noise.
                    if (!is_self) {
                        complexAddInPlace(result_matrix_ntk_k1[j][i], complexScale(ntk1, combined_weight));
                        complexAddInPlace(result_matrix_ntk_k2[j][i], complexScale(ntk2, combined_weight));
                    }
                }
            }
        }
    }

    // Apply the final scaling factor (edge lengths) to all elements
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            const real_t final_scaling = emj * eni;
            result_matrix_ntl_k1[j][i] = complexScale(result_matrix_ntl_k1[j][i], final_scaling);
            result_matrix_ntl_k2[j][i] = complexScale(result_matrix_ntl_k2[j][i], final_scaling);
            result_matrix_ntk_k1[j][i] = complexScale(result_matrix_ntk_k1[j][i], final_scaling);
            result_matrix_ntk_k2[j][i] = complexScale(result_matrix_ntk_k2[j][i], final_scaling);
        }
    }
}

/**
 * @brief Computes complete Form A nTL and unchanged double-area nTK quadrature.
 *
 * The nTL vector term and the full nTK operator use double-area Dunavant
 * quadrature. The nTL scalar term uses Gauss-Legendre quadrature on each
 * testing edge coupled with Dunavant quadrature on the source triangle.
 *
 * @param dunavant_nodes Reference points (xi, eta) for both 2D triangle integrals.
 * @param gauss_nodes Reference points t in [0,1] for the 1D testing edge quadrature.
 */
__host__ __device__ inline void calculateFormAQuadrature_nTL_nTK(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* dunavant_nodes,
    const float* dunavant_weights,
    int num_2d_points,
    const float* gauss_nodes,
    const float* gauss_weights,
    int num_1d_points,
    real_t k1,
    real_t k2,
    bool is_singular,
    complex__float result_matrix_ntl_k1[3][3],
    complex__float result_matrix_ntl_k2[3][3],
    complex__float result_matrix_ntk_k1[3][3],
    complex__float result_matrix_ntk_k2[3][3])
{
    // Initialize results to zero
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            result_matrix_ntl_k1[j][i] = make_complex(0.f, 0.f);
            result_matrix_ntl_k2[j][i] = make_complex(0.f, 0.f);
            result_matrix_ntk_k1[j][i] = make_complex(0.f, 0.f);
            result_matrix_ntk_k2[j][i] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate test triangle normal and area
    const vec3_t e1 = test_tri[1] - test_tri[0];
    const vec3_t e2 = test_tri[2] - test_tri[0];
    const vec3_t cross_e1_e2 = cross(e1, e2);
    const real_t twice_A_m = length(cross_e1_e2);
    const vec3_t n_m = cross_e1_e2 / twice_A_m;

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        // Source free vertices enter nTL and nTK; edge lengths scale both.
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    const bool is_self =
        test_tri_indices[0] == source_tri_indices[0] &&
        test_tri_indices[1] == source_tri_indices[1] &&
        test_tri_indices[2] == source_tri_indices[2];

    // Form A retains the double-area nTL vector term and all of nTK from Form B.
    // Only the nTL scalar term is converted into a boundary integral.
    for (int p = 0; p < num_2d_points; ++p) {
        const real_t xi_p = static_cast<real_t>(dunavant_nodes[2 * p]);
        const real_t eta_p = static_cast<real_t>(dunavant_nodes[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(dunavant_weights[p]);
        vec3_t r_mp;
        mapTriQuadPoint(test_tri, xi_p, eta_p, r_mp);

        for (int q = 0; q < num_2d_points; ++q) {
            const real_t xi_q = static_cast<real_t>(dunavant_nodes[2 * q]);
            const real_t eta_q = static_cast<real_t>(dunavant_nodes[2 * q + 1]);
            const real_t w_q = static_cast<real_t>(dunavant_weights[q]);
            vec3_t r_nq;
            mapTriQuadPoint(source_tri, xi_q, eta_q, r_nq);

            const IntegrandKernels kernels = is_singular ?
                integrandKernelsRegular(r_mp, r_nq, k1, k2) :
                integrandKernels(r_mp, r_nq, k1, k2);
            const real_t area_weight = w_p * w_q;
            const real_t vector_weight = real_t(0.25) * area_weight;

            for (int j = 0; j < 3; ++j) {
                const vec3_t tau_mj = cross(rhoRwg(r_mp, test_free_vertices[j]), n_m);
                for (int i = 0; i < 3; ++i) {
                    const vec3_t rho_ni = rhoRwg(r_nq, source_free_vertices[i]);
                    const real_t factor = vector_weight * dot(tau_mj, rho_ni);
                    complexAddInPlace(result_matrix_ntl_k1[j][i], complexScale(kernels.tl_k1, factor));
                    complexAddInPlace(result_matrix_ntl_k2[j][i], complexScale(kernels.tl_k2, factor));

                    // Same nTK expression and coincident-triangle skip as Form B.
                    if (!is_self) {
                        const vec3_t ntk_modifier = real_t(0.25) * cross(tau_mj, rho_ni);
                        const complex__float ntk1 = dot(ntk_modifier, kernels.tk_k1);
                        const complex__float ntk2 = dot(ntk_modifier, kernels.tk_k2);
                        complexAddInPlace(result_matrix_ntk_k1[j][i],
                                          complexScale(ntk1, area_weight));
                        complexAddInPlace(result_matrix_ntk_k2[j][i],
                                          complexScale(ntk2, area_weight));
                    }
                }
            }
        }
    }

    // Temporary diagnostic: area-area contribution for the four face pairs
    // supporting edge 212 of the level-3 sphere. The line loop has not run.
    const int debug_test_face =
        test_tri_indices[0] == 219 && test_tri_indices[1] == 218 &&
        test_tri_indices[2] == 59 ? 130 :
        test_tri_indices[0] == 193 && test_tri_indices[1] == 218 &&
        test_tri_indices[2] == 219 ? 131 : -1;
    const int debug_source_face =
        source_tri_indices[0] == 219 && source_tri_indices[1] == 218 &&
        source_tri_indices[2] == 59 ? 130 :
        source_tri_indices[0] == 193 && source_tri_indices[1] == 218 &&
        source_tri_indices[2] == 219 ? 131 : -1;
    if (is_singular && debug_test_face >= 0 && debug_source_face >= 0) {
        const int j = debug_test_face == 130 ? 0 : 1;
        const int i = debug_source_face == 130 ? 0 : 1;
        const complex__float area_unscaled = result_matrix_ntl_k2[j][i];
        const complex__float area_scaled = complexScale(
            area_unscaled, test_edge_lengths[j] * source_edge_lengths[i]);
        printf("Form A area-only edge 212: m=%d n=%d j=%d i=%d "
               "unscaled=(%.17g, %.17g) scaled=(%.17g, %.17g)\n",
               debug_test_face, debug_source_face, j, i,
               static_cast<double>(area_unscaled.x),
               static_cast<double>(area_unscaled.y),
               static_cast<double>(area_scaled.x),
               static_cast<double>(area_scaled.y));
    }

    // Diagnostic: omit this quadrature routine's scalar self line integral.
    // The separately extracted singular self line term is skipped in its own routine.
    // const int line_edge_count = is_self ? 0 : 3;
    const int line_edge_count = 3;
    for (int edge_idx = 0; edge_idx < line_edge_count; ++edge_idx) {
        const vec3_t vA = test_tri[edge_idx];
        const vec3_t vB = test_tri[(edge_idx + 1) % 3];
        const vec3_t edge_vec = vB - vA;
        const real_t L_e = length(edge_vec);
        const vec3_t nu_e = cross(edge_vec, n_m) / L_e;

        // Common formula prefactor without e_mj, e_ni (applied at the end)
        const real_t geom_factor = L_e / twice_A_m;

        for (int q = 0; q < num_1d_points; ++q) {
            const real_t t_q = static_cast<real_t>(gauss_nodes[q]);
            const real_t w_q = static_cast<real_t>(gauss_weights[q]);

            vec3_t r_eq;
            mapEdgeQuadPoint(vA, vB, t_q, r_eq);

            for (int p = 0; p < num_2d_points; ++p) {
                const real_t xi_p = static_cast<real_t>(dunavant_nodes[2 * p + 0]);
                const real_t eta_p = static_cast<real_t>(dunavant_nodes[2 * p + 1]);
                const real_t w_p = static_cast<real_t>(dunavant_weights[p]);

                vec3_t r_np;
                mapTriQuadPoint(source_tri, xi_p, eta_p, r_np);

                const IntegrandKernels kernels = is_singular ?
                                                 integrandKernelsRegular(r_eq, r_np, k1, k2) :
                                                 integrandKernels(r_eq, r_np, k1, k2);

                const real_t combined_weight = w_q * w_p;

                for (int j = 0; j < 3; ++j) {
                    const vec3_t rho_mj = rhoRwg(r_eq, test_free_vertices[j]);
                    const vec3_t tau_mj = cross(rho_mj, n_m);
                    const real_t test_dot = dot(tau_mj, nu_e);

                    const real_t final_factor = geom_factor * test_dot;

                    const complex__float ntl1 = complexScale(kernels.tl_k1, final_factor / (k1 * k1));
                    const complex__float ntl2 = complexScale(kernels.tl_k2, final_factor / (k2 * k2));

                    const complex__float ntl1_w = complexScale(ntl1, combined_weight);
                    const complex__float ntl2_w = complexScale(ntl2, combined_weight);

                    for (int i = 0; i < 3; ++i) {
                        // The source function divergence \nabla_S' \cdot f_{ni} = e_{ni} / A_n
                        // has no spatial dependence. The e_{ni} is applied at the end, and A_n
                        // implicitly cancels with the 2D Dunavant integral's area Jacobian.
                        complexAddInPlace(result_matrix_ntl_k1[j][i], ntl1_w);
                        complexAddInPlace(result_matrix_ntl_k2[j][i], ntl2_w);
                    }
                }
            }
        }
    }

    // Apply the final scaling factor (edge lengths e_mj * e_ni) to all elements
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            const real_t final_scaling = emj * eni;
            result_matrix_ntl_k1[j][i] = complexScale(result_matrix_ntl_k1[j][i], final_scaling);
            result_matrix_ntl_k2[j][i] = complexScale(result_matrix_ntl_k2[j][i], final_scaling);
            result_matrix_ntk_k1[j][i] = complexScale(result_matrix_ntk_k1[j][i], final_scaling);
            result_matrix_ntk_k2[j][i] = complexScale(result_matrix_ntk_k2[j][i], final_scaling);
        }
    }
}
//// End Standard quadrature for nTL and nTK operators

//// Begin Singular-part quadrature for TL and TK operators
        
/**
 * @brief This may be removed in the future if not needed.
 * @brief Computes the outer quadrature loop for the singular (Coulomb) part of an operator.
 *
 * This function provides a generic framework for the one-loop quadrature over the
 * testing triangle. It is templated on an `OperatorKernel` functor (typically a C++ lambda)
 * that encapsulates the specific calculations required for a given operator (e.g., TL or TK)
 * at each observation point.
 *
 * @tparam OperatorKernel A functor or lambda with the signature:
 *   `__device__ IntegrandComponents operator()(const vec3_t& r_obs) const;`
 *   It computes the necessary inner integral potentials at a given observation point `r_obs`.
 *   Any additional arguments needed by the kernel (like `k`, source triangle, etc.)
 *   should be captured by the lambda from its enclosing scope.
 *
 * @param test_tri The three vertices of the testing triangle.
 * @param test_free_vertex_idx The local index (0, 1, or 2) of the free vertex for the
 *                             testing basis function within the `test_tri` array.
 * @param quad_xi_eta Device pointer to the quadrature points (xi,eta) in reference coordinates.
 * @param quad_w Device pointer to the quadrature weights.
 * @param num_quad_points The number of points in the quadrature rule.
 * @param kernel_functor An instance of the operator-specific kernel logic (the lambda).
 * @return The raw, real-valued result of the quadrature summation.
 */

template <typename OperatorKernel>
__device__ inline real_t calculateSingularPartQuadrature(
    const vec3_t test_tri[3],
    int test_free_vertex_idx,
    const float* quad_xi_eta,
    const float* quad_w,
    int num_quad_points,
    const OperatorKernel& kernel_functor)
{
    real_t total_integral = real_t(0);

    const vec3_t& free_vertex = test_tri[test_free_vertex_idx];

    for (int p = 0; p < num_quad_points; ++p)
    {
        // 1. Get current quadrature point in reference coordinates
        const real_t xi = static_cast<real_t>(quad_xi_eta[2 * p + 0]);
        const real_t eta = static_cast<real_t>(quad_xi_eta[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_w[p]);

        // 2. Map to physical observation point on the testing triangle
        vec3_t r_mp;
        mapTriQuadPoint(test_tri, xi, eta, r_mp);

        // 3. Evaluate the testing basis function vector at this point
        const vec3_t rho_mj_p = rhoRwg(r_mp, free_vertex);

        // 4. Call the operator-specific lambda to get the inner integral components
        const IntegrandComponents components = kernel_functor(r_mp);

        // 5. Compute the contribution for this point
        const real_t contribution = dot(rho_mj_p, components.vector_part) + components.scalar_part;

        // 6. Accumulate the weighted result
        total_integral += w_p * contribution;
    }

    return total_integral;
}

/**
 * @brief Computes the double surface numerical quadrature for the singular (extracted) 
 * part of the standard TL and TK operators.
 *
 * This routine evaluates the inner integrals analytically using Wilton's potentials 
 * for strongly singular near-field/self-interactions, and evaluates the outer integral 
 * numerically via Dunavant quadrature.
 */
__host__ __device__ inline void calculateSingularPartQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    real_t k1,
    real_t k2,
    complex__float result_matrix_tl_k1[3][3],
    complex__float result_matrix_tl_k2[3][3],
    complex__float result_matrix_tk_k1[3][3],
    complex__float result_matrix_tk_k2[3][3])
{
    // Initialize all result matrices to zero to ensure a clean calculation
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix_tl_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_tl_k2[row][col] = make_complex(0.f, 0.f);
            result_matrix_tk_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_tk_k2[row][col] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop.
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi = static_cast<real_t>(ref_points[2 * p + 0]);
        const real_t eta = static_cast<real_t>(ref_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi, eta, rm_p);

        // Calculate all Wilton potentials once for the current observation point
        const WiltonPotentials pots = calculateAllWiltonPotentials(rm_p, source_tri);

        for (int j = 0; j < 3; ++j)
        {
            const vec3_t& rm_free = test_free_vertices[j];
            const vec3_t rho_mj_p = rhoRwg(rm_p, rm_free);

            for (int i = 0; i < 3; ++i)
            {
                const vec3_t& rn_free = source_free_vertices[i];

                // --- Efficiently compute potentials using the pre-calculated struct ---
                const vec3_t psi_1 = calculate_Psi_1(pots, rn_free);
                const real_t dot_basis_rwg = real_t(0.25) * dot(rho_mj_p, psi_1);
                const vec3_t upsilon_coul_k1 = calculate_Upsilon_coul(pots, rn_free, rm_p, k1);
                const vec3_t upsilon_coul_k2 = calculate_Upsilon_coul(pots, rn_free, rm_p, k2);

                // --- Calculate contributions for TL and TK ---
                
                const real_t tl_singular_contrib_1 = dot_basis_rwg - real_t(1)/(k1*k1)*pots.phi_0;
                const real_t tl_singular_contrib_2 = dot_basis_rwg - real_t(1)/(k2*k2)*pots.phi_0;
                const real_t tk_singular_contrib_1 = real_t(0.25) * dot(rho_mj_p, upsilon_coul_k1);
                const real_t tk_singular_contrib_2 = real_t(0.25) * dot(rho_mj_p, upsilon_coul_k2);

                // --- Accumulate results into their respective matrices (with quad weights only) ---
                // Note: The singular part is purely real.
                complexAddInPlace(result_matrix_tl_k1[j][i], make_complex(tl_singular_contrib_1 * w_p, 0.f));
                complexAddInPlace(result_matrix_tl_k2[j][i], make_complex(tl_singular_contrib_2 * w_p, 0.f));

                // For coincident triangles (self-term), the surface integral part of the TK operator is zero.
                // We skip the accumulation to avoid adding numerical noise from the semi-analytic formula.
                if (!(test_tri_indices[0] == source_tri_indices[0] &&
                      test_tri_indices[1] == source_tri_indices[1] &&
                      test_tri_indices[2] == source_tri_indices[2])) {
                    complexAddInPlace(result_matrix_tk_k1[j][i], make_complex(tk_singular_contrib_1 * w_p, 0.f));
                    complexAddInPlace(result_matrix_tk_k2[j][i], make_complex(tk_singular_contrib_2 * w_p, 0.f));
                }
            }
        }
    }

    const real_t source_area = getTriangleArea(source_tri);
    const real_t area_scaling = real_t(1) / (real_t(4) * real_t(M_PI) * source_area);

    // After the quadrature loop, apply all scaling factors
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            const real_t final_scaling = area_scaling * emj * eni;
            result_matrix_tl_k1[j][i] = complexScale(result_matrix_tl_k1[j][i], final_scaling);
            result_matrix_tl_k2[j][i] = complexScale(result_matrix_tl_k2[j][i], final_scaling);
            result_matrix_tk_k1[j][i] = complexScale(result_matrix_tk_k1[j][i], final_scaling);
            result_matrix_tk_k2[j][i] = complexScale(result_matrix_tk_k2[j][i], final_scaling);
        }
    }
}

//// End Singular-part quadrature for TL and TK operators


//// Begin Singular-part quadrature for nTL and nTK operators
/**
 * @brief Computes the double surface numerical quadrature for the singular (extracted) 
 * part of the nTL and nTK operators (Form B).
 *
 * Implements the theoretical principal-value singular extractions for N-PMCHWT operators.
 * Specifically for nTL, the scalar potential gradient is bounded analytically. The jump 
 * limits are structurally separated from this PV evaluation.
 */
__host__ __device__ inline void calculateSingularPartQuadrature_nTL_nTK(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const vec3_t source_tri[3],
    const int   source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    real_t k1,
    real_t k2,
    complex__float result_matrix_ntl_k1[3][3],
    complex__float result_matrix_ntl_k2[3][3],
    complex__float result_matrix_ntk_k1[3][3],
    complex__float result_matrix_ntk_k2[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix_ntl_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntl_k2[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntk_k1[row][col] = make_complex(0.f, 0.f);
            result_matrix_ntk_k2[row][col] = make_complex(0.f, 0.f);
        }
    }

    // Pre-calculate test face normal
    const vec3_t test_e1 = test_tri[1] - test_tri[0];
    const vec3_t test_e2 = test_tri[2] - test_tri[0];
    const vec3_t n_m = normalize(cross(test_e1, test_e2));

    // Pre-calculate source face normal
    const vec3_t source_e1 = source_tri[1] - source_tri[0];
    const vec3_t source_e2 = source_tri[2] - source_tri[0];
    const vec3_t n_n = normalize(cross(source_e1, source_e2));

    // Pre-calculate free vertices and edge lengths to avoid redundant calls in the loop.
    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
    }

    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri, i, source_free_vertices[i]);
    }

    const bool is_self =
        test_tri_indices[0] == source_tri_indices[0] &&
        test_tri_indices[1] == source_tri_indices[1] &&
        test_tri_indices[2] == source_tri_indices[2];

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi = static_cast<real_t>(ref_points[2 * p + 0]);
        const real_t eta = static_cast<real_t>(ref_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi, eta, rm_p);

        // Calculate all Wilton potentials once for the current observation point
        const WiltonPotentials pots = calculateAllWiltonPotentials(rm_p, source_tri);

        // Compute the true phi_1 vector potential from its base
        // phi_1(r) = phi_1_base - d * n * phi_0
        const vec3_t phi_1 = calculate_phi_1(pots, n_n);

        // Compute the extracted gradient of the scalar potential: phi_3 - R_i * phi_2
        // As proven in the theory, this simplifies to phi_3_base - d * phi_2 * n_n
        const vec3_t nabla_G_extracted = pots.phi_3_base - (pots.d * pots.phi_2) * n_n;

        for (int j = 0; j < 3; ++j)
        {
            const vec3_t& rm_free = test_free_vertices[j];
            const vec3_t rho_mj_p = rhoRwg(rm_p, rm_free);
            const vec3_t tau_mj_p = cross(rho_mj_p, n_m);

            for (int i = 0; i < 3; ++i)
            {
                const vec3_t& rn_free = source_free_vertices[i];
                const vec3_t R_i = rm_p - rn_free;

                // --- Calculate contributions for nTL ---
                // V_sing bracket = phi_1 + 0.5 * R_i * phi_0 + 1/k^2 * (phi_3 - R_i * phi_2)
                const vec3_t vec_pot_bracket = phi_1 + 0.5f * R_i * pots.phi_0;
                
                // Diagnostic: keep the vector potential but omit the extracted
                // scalar gradient for an identical test/source triangle.
                const vec3_t V_bracket_k1 = is_self ? vec_pot_bracket
                    : vec_pot_bracket + (1.0f / (k1 * k1)) * nabla_G_extracted;
                const vec3_t V_bracket_k2 = is_self ? vec_pot_bracket
                    : vec_pot_bracket + (1.0f / (k2 * k2)) * nabla_G_extracted;

                // Multiply by 0.5 to yield the final e_mj * e_ni / (8 * pi * A_n) after area scaling
                const real_t ntl_singular_contrib_1 = 0.5f * dot(tau_mj_p, V_bracket_k1);
                const real_t ntl_singular_contrib_2 = 0.5f * dot(tau_mj_p, V_bracket_k2);

                // --- Calculate contributions for nTK ---
                // nTK evaluates directly against Upsilon_coul exactly like standard TK
                const vec3_t upsilon_coul_k1 = calculate_Upsilon_coul(pots, rn_free, rm_p, k1);
                const vec3_t upsilon_coul_k2 = calculate_Upsilon_coul(pots, rn_free, rm_p, k2);

                // Multiply by 0.25 to yield the final e_mj * e_ni / (16 * pi * A_n) after area scaling
                const real_t ntk_singular_contrib_1 = 0.25f * dot(tau_mj_p, upsilon_coul_k1);
                const real_t ntk_singular_contrib_2 = 0.25f * dot(tau_mj_p, upsilon_coul_k2);

                // --- Accumulate results into their respective matrices (with quad weights only) ---
                complexAddInPlace(result_matrix_ntl_k1[j][i], make_complex(ntl_singular_contrib_1 * w_p, 0.f));
                complexAddInPlace(result_matrix_ntl_k2[j][i], make_complex(ntl_singular_contrib_2 * w_p, 0.f));

                // For coincident triangles (self-term), the surface integral PV of the nTK operator is zero.
                // We skip the accumulation to avoid adding numerical noise. The jump terms 
                // are assembled explicitly elsewhere.
               if (!is_self) {
                    complexAddInPlace(result_matrix_ntk_k1[j][i], make_complex(ntk_singular_contrib_1 * w_p, 0.f));
                    complexAddInPlace(result_matrix_ntk_k2[j][i], make_complex(ntk_singular_contrib_2 * w_p, 0.f));
                }
            }
        }
    }

    const real_t source_area = getTriangleArea(source_tri);
    // Base area scaling: e_mj * e_ni / (4 * pi * A_n)
    const real_t area_scaling = real_t(1) / (real_t(4) * real_t(M_PI) * source_area);

    // After the quadrature loop, apply all scaling factors
    for (int j = 0; j < 3; ++j) {
        const real_t emj = test_edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const real_t eni = source_edge_lengths[i];
            const real_t final_scaling = area_scaling * emj * eni;
            result_matrix_ntl_k1[j][i] = complexScale(result_matrix_ntl_k1[j][i], final_scaling);
            result_matrix_ntl_k2[j][i] = complexScale(result_matrix_ntl_k2[j][i], final_scaling);
            result_matrix_ntk_k1[j][i] = complexScale(result_matrix_ntk_k1[j][i], final_scaling);
            result_matrix_ntk_k2[j][i] = complexScale(result_matrix_ntk_k2[j][i], final_scaling);
        }
    }
}

/**
 * @brief Computes the extracted singular Form A nTL and nTK contributions.
 *
 * The nTL vector term and nTK use one outer Dunavant loop with analytical
 * source potentials. The nTL scalar term uses one outer Gauss-Legendre loop
 * along each testing edge with the analytical source potential phi_0.
 * No source quadrature loop is needed in either contribution.
 */
__host__ __device__ inline void calculateFormASingularPartQuadrature_nTL_nTK(
    const vec3_t test_tri[3],
    const int test_tri_indices[3],
    const vec3_t source_tri[3],
    const int source_tri_indices[3],
    const float* dunavant_nodes,
    const float* dunavant_weights,
    int num_2d_points,
    const float* gauss_nodes,
    const float* gauss_weights,
    int num_1d_points,
    real_t k1,
    real_t k2,
    complex__float result_matrix_ntl_k1[3][3],
    complex__float result_matrix_ntl_k2[3][3],
    complex__float result_matrix_ntk_k1[3][3],
    complex__float result_matrix_ntk_k2[3][3])
{
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            result_matrix_ntl_k1[j][i] = make_complex(real_t(0), real_t(0));
            result_matrix_ntl_k2[j][i] = make_complex(real_t(0), real_t(0));
            result_matrix_ntk_k1[j][i] = make_complex(real_t(0), real_t(0));
            result_matrix_ntk_k2[j][i] = make_complex(real_t(0), real_t(0));
        }
    }

    const vec3_t test_cross = cross(test_tri[1] - test_tri[0],
                                    test_tri[2] - test_tri[0]);
    const real_t twice_A_m = length(test_cross);
    const real_t A_m = real_t(0.5) * twice_A_m;
    const vec3_t n_m = test_cross / twice_A_m;
    const real_t A_n = getTriangleArea(source_tri);
    const real_t pi = real_t(M_PI);
    const bool is_self =
        test_tri_indices[0] == source_tri_indices[0] &&
        test_tri_indices[1] == source_tri_indices[1] &&
        test_tri_indices[2] == source_tri_indices[2];

    vec3_t test_free_vertices[3];
    real_t test_edge_lengths[3];
    vec3_t source_free_vertices[3];
    real_t source_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] =
            getEdgeInfo(test_tri_indices, test_tri, j, test_free_vertices[j]);
        source_edge_lengths[j] =
            getEdgeInfo(source_tri_indices, source_tri, j, source_free_vertices[j]);
    }

    // One outer surface loop: Psi_1 supplies the nTL vector term, and
    // Upsilon_coul supplies nTK. Both inner source integrals are analytic.
    for (int p = 0; p < num_2d_points; ++p) {
        const real_t xi = static_cast<real_t>(dunavant_nodes[2 * p]);
        const real_t eta = static_cast<real_t>(dunavant_nodes[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(dunavant_weights[p]);
        vec3_t r_mp;
        mapTriQuadPoint(test_tri, xi, eta, r_mp);
        const WiltonPotentials pots = calculateAllWiltonPotentials(r_mp, source_tri);

        for (int j = 0; j < 3; ++j) {
            const vec3_t tau_mj =
                cross(rhoRwg(r_mp, test_free_vertices[j]), n_m);
            for (int i = 0; i < 3; ++i) {
                const vec3_t psi_1 = calculate_Psi_1(pots, source_free_vertices[i]);
                const real_t area_ntl = real_t(0.25) * w_p * dot(tau_mj, psi_1);
                complexAddInPlace(result_matrix_ntl_k1[j][i],
                                  make_complex(area_ntl, real_t(0)));
                complexAddInPlace(result_matrix_ntl_k2[j][i],
                                  make_complex(area_ntl, real_t(0)));

                // Keep the Form B nTK expression and self-term PV skip.
                if (!is_self) {
                    const vec3_t upsilon_k1 =
                        calculate_Upsilon_coul(pots, source_free_vertices[i], r_mp, k1);
                    const vec3_t upsilon_k2 =
                        calculate_Upsilon_coul(pots, source_free_vertices[i], r_mp, k2);
                    const real_t area_ntk_k1 =
                        real_t(0.25) * w_p * dot(tau_mj, upsilon_k1);
                    const real_t area_ntk_k2 =
                        real_t(0.25) * w_p * dot(tau_mj, upsilon_k2);
                    complexAddInPlace(result_matrix_ntk_k1[j][i],
                                      make_complex(area_ntk_k1, real_t(0)));
                    complexAddInPlace(result_matrix_ntk_k2[j][i],
                                      make_complex(area_ntk_k2, real_t(0)));
                }
            }
        }
    }

    // One outer line loop: phi_0 replaces the source surface integral.
    // It is independent of the source basis index i.
    real_t line_sum[3] = {real_t(0), real_t(0), real_t(0)};
    // Diagnostic: omit the extracted scalar line term for an identical face.
    // const int line_edge_count = is_self ? 0 : 3;
    const int line_edge_count = 3;
    for (int edge = 0; edge < line_edge_count; ++edge) {
        const vec3_t vA = test_tri[edge];
        const vec3_t vB = test_tri[(edge + 1) % 3];
        const vec3_t edge_vec = vB - vA;
        const real_t L_e = length(edge_vec);
        const vec3_t nu_e = cross(edge_vec, n_m) / L_e;
        for (int q = 0; q < num_1d_points; ++q) {
            const real_t t_q = static_cast<real_t>(gauss_nodes[q]);
            const real_t w_q = static_cast<real_t>(gauss_weights[q]);
            vec3_t r_eq;
            mapEdgeQuadPoint(vA, vB, t_q, r_eq);
            const real_t phi_0 = static_cast<real_t>(calculate_phi_0(r_eq, source_tri));
            for (int j = 0; j < 3; ++j) {
                const vec3_t tau_mj =
                    cross(rhoRwg(r_eq, test_free_vertices[j]), n_m);
                line_sum[j] += L_e * w_q * dot(tau_mj, nu_e) * phi_0;
            }
        }
    }

    // The area accumulator contains its 1/4 factor; the line accumulator
    // receives its 1/(2 A_m k^2) factor here.
    const real_t area_scale = real_t(1) / (real_t(4) * pi * A_n);
    const real_t line_scale = real_t(1) / (real_t(8) * pi * A_m * A_n);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const real_t edges = test_edge_lengths[j] * source_edge_lengths[i];
            const real_t line_k1 = edges * line_scale * line_sum[j] / (k1 * k1);
            const real_t line_k2 = edges * line_scale * line_sum[j] / (k2 * k2);
            result_matrix_ntl_k1[j][i] = complexAdd(
                complexScale(result_matrix_ntl_k1[j][i], edges * area_scale),
                make_complex(line_k1, real_t(0)));
            result_matrix_ntl_k2[j][i] = complexAdd(
                complexScale(result_matrix_ntl_k2[j][i], edges * area_scale),
                make_complex(line_k2, real_t(0)));
            result_matrix_ntk_k1[j][i] =
                complexScale(result_matrix_ntk_k1[j][i], edges * area_scale);
            result_matrix_ntk_k2[j][i] =
                complexScale(result_matrix_ntk_k2[j][i], edges * area_scale);
        }
    }
}

//// End Singular-part quadrature for nTL and nTK operators

//// Begin Local basis-pairing matrices
// Cross-identity matrix SIE: <f_m, n x f_n>
// WARNING: This is NOT the standard identity/Gram matrix <f_m, f_n>.
// Keep this only for formulations that explicitly require n x in the tested term.
// If not used, prefer removing to avoid confusion.
__device__ inline void calculateCrossIdentityQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    complex__float result_matrix[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix[row][col] = make_complex(0.f, 0.f);
        }
    }

    // removed reinterpret_cast
    const float tri_area = getTriangleArea(test_tri);
    const vec3_t e1 = test_tri[1] - test_tri[0];
    const vec3_t e2 = test_tri[2] - test_tri[0];
    const vec3_t n_hat = normalize(cross(e1, e2));


    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t free_vertices[3];
    float edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        edge_lengths[i] = getEdgeInfo(test_tri_indices, test_tri, i, free_vertices[i]);
    }

    // Loop over quadrature points (most expensive loop)
    for (int p = 0; p < num_quad_points; ++p)
    {
        const float2 ref_p = {ref_points[2 * p], ref_points[2 * p + 1]};
        const float w_p = quad_weights[p];

        vec3_t r_p;
        mapTriQuadPoint(test_tri, ref_p.x, ref_p.y, r_p);

        // Pre-calculate all rho vectors at the current quadrature point
        vec3_t rho_vectors[3];
        for (int i = 0; i < 3; ++i) {
            rho_vectors[i] = rhoRwg(r_p, free_vertices[i]);
        }

        // Now, compute the 3x3 matrix contributions at this point
        for (int j = 0; j < 3; ++j) // test edge
        {
            for (int i = 0; i < 3; ++i) // source edge
            {
                // Integrand: rho_mj . (n_hat x rho_ni)
                const vec3_t cross_prod = cross(n_hat, rho_vectors[i]);
                const float integrand = dot(rho_vectors[j], cross_prod);

                // Accumulate the weighted integrand (without final scaling yet)
                complexAddInPlace(result_matrix[j][i], make_complex(w_p * integrand, 0.f));
            }
        }
    }

    // Apply the final scaling factor to all elements
    const float scaling = 1.f / (4.f * tri_area);
    for (int j = 0; j < 3; ++j) {
        const float emj = edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const float eni = edge_lengths[i];
            const float final_scaling = scaling * emj * eni;
            result_matrix[j][i] = complexScale(result_matrix[j][i], final_scaling);
        }
    }
}

// Standard Mass Matrix (Gram matrix)
// Computes integral of f_m . f_n
__host__ __device__ inline void calculateMassMatrixQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    complex__float result_matrix[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix[row][col] = make_complex(0.f, 0.f);
        }
    }

    // removed reinterpret_cast
    const float tri_area = getTriangleArea(test_tri);
    
    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t free_vertices[3];
    float edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        edge_lengths[i] = getEdgeInfo(test_tri_indices, test_tri, i, free_vertices[i]);
    }

    // Loop over quadrature points (most expensive loop)
    for (int p = 0; p < num_quad_points; ++p)
    {
        const float2 ref_p = {ref_points[2 * p], ref_points[2 * p + 1]};
        const float w_p = quad_weights[p];

        vec3_t r_p;
        mapTriQuadPoint(test_tri, ref_p.x, ref_p.y, r_p);

        // Pre-calculate all rho vectors at the current quadrature point
        vec3_t rho_vectors[3];
        for (int i = 0; i < 3; ++i) {
            rho_vectors[i] = rhoRwg(r_p, free_vertices[i]);
        }

        // Now, compute the 3x3 matrix contributions at this point
        for (int j = 0; j < 3; ++j) // test edge
        {
            for (int i = 0; i < 3; ++i) // source edge
            {
                // Integrand: rho_mj . rho_ni
                const float integrand = dot(rho_vectors[j], rho_vectors[i]);

                // Accumulate the weighted integrand (without final scaling yet)
                complexAddInPlace(result_matrix[j][i], make_complex(w_p * integrand, 0.f));
            }
        }
    }

    // Apply the final scaling factor to all elements
    const float scaling = 1.f / (4.f * tri_area);
    for (int j = 0; j < 3; ++j) {
        const float emj = edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const float eni = edge_lengths[i];
            const float final_scaling = scaling * emj * eni;
            result_matrix[j][i] = complexScale(result_matrix[j][i], final_scaling);
        }
    }
}

//// End Local basis-pairing matrices

//// Begin GSTC tensor mass matrices
// Mass Matrix with general tensor T (e.g. for GSTC)
// Computes integral of f_m . (T . f_n)
__host__ __device__ inline void calculateMassMatrixQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    const complex__float tensor_T[3][3],
    complex__float result_matrix[3][3])
{
    // Initialize all result matrices to zero
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            result_matrix[row][col] = make_complex(0.f, 0.f);
        }
    }

    // removed reinterpret_cast
    const float tri_area = getTriangleArea(test_tri);
    
    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t free_vertices[3];
    float edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        edge_lengths[i] = getEdgeInfo(test_tri_indices, test_tri, i, free_vertices[i]);
    }

    // Loop over quadrature points
    for (int p = 0; p < num_quad_points; ++p)
    {
        const float2 ref_p = {ref_points[2 * p], ref_points[2 * p + 1]};
        const float w_p = quad_weights[p];

        vec3_t r_p;
        mapTriQuadPoint(test_tri, ref_p.x, ref_p.y, r_p);

        // Pre-calculate all rho vectors at the current quadrature point
        vec3_t rho_vectors[3];
        for (int i = 0; i < 3; ++i) {
            rho_vectors[i] = rhoRwg(r_p, free_vertices[i]);
        }

        // Now, compute the 3x3 matrix contributions at this point
        for (int j = 0; j < 3; ++j) // test edge (m)
        {
            const vec3_t& rho_m = rho_vectors[j];
            
            for (int i = 0; i < 3; ++i) // source edge (n)
            {
                const vec3_t& rho_n = rho_vectors[i];

                // Compute T . rho_n
                // T is 3x3 complex, rho_n is real vec3_t
                complex__float T_rho_n_x =  complexAdd(complexScale(tensor_T[0][0], rho_n.x), 
                                            complexAdd(complexScale(tensor_T[0][1], rho_n.y), complexScale(tensor_T[0][2], rho_n.z)));
                
                complex__float T_rho_n_y =  complexAdd(complexScale(tensor_T[1][0], rho_n.x), 
                                            complexAdd(complexScale(tensor_T[1][1], rho_n.y), complexScale(tensor_T[1][2], rho_n.z)));

                complex__float T_rho_n_z =  complexAdd(complexScale(tensor_T[2][0], rho_n.x), 
                                            complexAdd(complexScale(tensor_T[2][1], rho_n.y), complexScale(tensor_T[2][2], rho_n.z)));

                // Compute rho_m . (T . rho_n)
                complex__float integrand = complexAdd(complexScale(T_rho_n_x, rho_m.x),
                                           complexAdd(complexScale(T_rho_n_y, rho_m.y),
                                                      complexScale(T_rho_n_z, rho_m.z)));

                // Accumulate weighted
                complexAddInPlace(result_matrix[j][i], complexScale(integrand, w_p));
            }
        }
    }

    // Apply the final scaling factor to all elements
    const float scaling = 1.f / (4.f * tri_area);
    for (int j = 0; j < 3; ++j) {
        const float emj = edge_lengths[j];
        for (int i = 0; i < 3; ++i) {
            const float eni = edge_lengths[i];
            const float final_scaling = scaling * emj * eni;
            result_matrix[j][i] = complexScale(result_matrix[j][i], final_scaling);
        }
    }
}

// Batch Mass Matrix for 8 tensors (GSTC optimized)
// Computes integral of f_m . (T_k . f_n) for k=0..7 in a single pass
__host__ __device__ inline void calculateMassMatrixQuadrature(
    const vec3_t test_tri[3],
    const int   test_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    const complex__float tensors_T[8][3][3],
    complex__float results_matrices[8][3][3])
{
    // Initialize all result matrices to zero
    for(int k=0; k<8; ++k) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                results_matrices[k][row][col] = make_complex(0.f, 0.f);
            }
        }
    }

    // removed reinterpret_cast
    const float tri_area = getTriangleArea(test_tri);
    
    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t free_vertices[3];
    float edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        edge_lengths[i] = getEdgeInfo(test_tri_indices, test_tri, i, free_vertices[i]);
    }

    // Loop over quadrature points
    for (int p = 0; p < num_quad_points; ++p)
    {
        const float2 ref_p = {ref_points[2 * p], ref_points[2 * p + 1]};
        const float w_p = quad_weights[p];

        vec3_t r_p;
        mapTriQuadPoint(test_tri, ref_p.x, ref_p.y, r_p);

        // Pre-calculate all rho vectors at the current quadrature point
        vec3_t rho_vectors[3];
        for (int i = 0; i < 3; ++i) {
            rho_vectors[i] = rhoRwg(r_p, free_vertices[i]);
        }

        // Now, compute contributions for all 8 tensors
        for (int j = 0; j < 3; ++j) // test edge (m)
        {
            const vec3_t& rho_m = rho_vectors[j];
            
            for (int i = 0; i < 3; ++i) // source edge (n)
            {
                const vec3_t& rho_n = rho_vectors[i];

                for(int k=0; k<8; ++k) {
                    const complex__float* T_ptr = &tensors_T[k][0][0];
                    
                    complex__float T_rho_n_x =  complexAdd(complexScale(T_ptr[0], rho_n.x), 
                                                complexAdd(complexScale(T_ptr[1], rho_n.y), complexScale(T_ptr[2], rho_n.z)));
                    
                    complex__float T_rho_n_y =  complexAdd(complexScale(T_ptr[3], rho_n.x), 
                                                complexAdd(complexScale(T_ptr[4], rho_n.y), complexScale(T_ptr[5], rho_n.z)));

                    complex__float T_rho_n_z =  complexAdd(complexScale(T_ptr[6], rho_n.x), 
                                                complexAdd(complexScale(T_ptr[7], rho_n.y), complexScale(T_ptr[8], rho_n.z)));

                    complex__float integrand = complexAdd(complexScale(T_rho_n_x, rho_m.x),
                                               complexAdd(complexScale(T_rho_n_y, rho_m.y),
                                                          complexScale(T_rho_n_z, rho_m.z)));

                    complexAddInPlace(results_matrices[k][j][i], complexScale(integrand, w_p));
                }
            }
        }
    }

    // Apply the final scaling factor to all elements
    const float scaling = 1.f / (4.f * tri_area);
    for(int k=0; k<8; ++k) {
        for (int j = 0; j < 3; ++j) {
            const float emj = edge_lengths[j];
            for (int i = 0; i < 3; ++i) {
                const float eni = edge_lengths[i];
                const float final_scaling = scaling * emj * eni;
                results_matrices[k][j][i] = complexScale(results_matrices[k][j][i], final_scaling);
            }
        }
    }
}

//// End GSTC tensor mass matrices

//// Begin Incident-field excitation quadrature
enum class TestingFunction {
    RWG,
    NxRWG
};

/**
 * @brief Computes the local excitation vector contributions for a single triangle.
 *
 * This function calculates the integrals of the incident electric and magnetic
 * fields tested against the three half-RWG basis functions defined on a given
 * triangle. It uses a Dunavant quadrature rule for numerical integration.
 *
 * The incident field is provided by a generic `ExcitationFunctor`, which allows
 * this function to be used with any source type (e.g., PlaneWave, Dipole)
 * that can evaluate its E and H fields at a given point.
 *
 * @tparam TestFunc The testing function to use (RWG or NxRWG). Defaults to RWG.
 * @tparam ExcitationFunctor A functor (like a struct with an `operator()`) that
 *   evaluates the incident fields. It must have the signature:
 *   `__device__ void operator()(const Vector& r, Vector& E_inc, Vector& H_inc) const;`
 *
 * @param tri_vertices The three vertices of the triangle.
 * @param tri_indices The global vertex indices of the triangle.
 * @param ref_points Device pointer to the quadrature points in reference coordinates.
 * @param quad_weights Device pointer to the quadrature weights.
 * @param num_quad_points The number of points in the quadrature rule.
 * @param excitation_functor An instance of the functor that provides the incident fields.
 * @param V_E_local (Output) A 3-element array to store the local electric excitation vector contributions.
 * @param V_H_local (Output) A 3-element array to store the local magnetic excitation vector contributions.
 */
template <TestingFunction TestFunc = TestingFunction::RWG, typename ExcitationFunctor>
__host__ __device__ inline void calculateExcitationQuadrature(
    const vec3_t tri_vertices[3],
    const int tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    const ExcitationFunctor& excitation_functor,
    complex__float V_E_local[3],
    complex__float V_H_local[3])
{
    // Initialize local result vectors
    for (int i = 0; i < 3; ++i) {
        V_E_local[i] = make_complex(0.f, 0.f);
        V_H_local[i] = make_complex(0.f, 0.f);
    }

    // removed reinterpret_cast

    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t free_vertices[3];
    float edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        edge_lengths[i] = getEdgeInfo(tri_indices, tri_vertices, i, free_vertices[i]);
    }

    // Pre-calculate normal for NxRWG testing
    vec3_t n_m;
    if constexpr (TestFunc == TestingFunction::NxRWG) {
        n_m = normalize(cross(tri_vertices[1] - tri_vertices[0], tri_vertices[2] - tri_vertices[0]));
    }

    // Loop over quadrature points
    for (int p = 0; p < num_quad_points; ++p) {
        const float2 ref_p = {ref_points[2 * p], ref_points[2 * p + 1]};
        const float w_p = quad_weights[p];

        vec3_t r_p;
        mapTriQuadPoint(tri_vertices, ref_p.x, ref_p.y, r_p);

        // Evaluate incident fields at the quadrature point
        cvec3_t E_inc, H_inc;
        excitation_functor(r_p, E_inc, H_inc);

        // Test against the 3 half-RWG basis functions on this triangle
        for (int i = 0; i < 3; ++i) {
            vec3_t test_vec = rhoRwg(r_p, free_vertices[i]); // only rho part, no 2l/A factor

            if constexpr (TestFunc == TestingFunction::NxRWG) {
                test_vec = cross(test_vec, n_m);
            }

            // Integrand for V_E: (test_vec . E_inc)
            const complex__float dot_E = dot(test_vec, E_inc);
            complexAddInPlace(V_E_local[i], complexScale(dot_E, w_p));

            // Integrand for V_H: (test_vec . H_inc)
            const complex__float dot_H = dot(test_vec, H_inc);
            complexAddInPlace(V_H_local[i], complexScale(dot_H, w_p));
        }
    }

    // Apply the final scaling factor: 0.5 * edge_length
    for (int i = 0; i < 3; ++i) {
        const float scaling = 0.5f * edge_lengths[i];
        V_E_local[i] = complexScale(V_E_local[i], scaling);
        V_H_local[i] = complexScale(V_H_local[i], scaling);
    }
}


//// End Incident-field excitation quadrature

//// Begin Scattered-field quadrature
/**
 * @brief Computes the local L and K operator contributions for the scattered field from a single source triangle.
 *
 * This function calculates the integrals for the L and K operators, which are needed to find the
 * scattered E and H fields at a specific observation point. The calculation is done for a single
 * source triangle and its three associated half-RWG basis functions.
 *
 * @param obs_point The 3D coordinate of the point where the scattered field is being evaluated.
 * @param source_tri_v The three vertices of the source triangle.
 * @param source_tri_indices The global vertex indices of the source triangle.
 * @param ref_points Device pointer to the quadrature points in reference coordinates.
 * @param quad_weights Device pointer to the quadrature weights.
 * @param num_quad_points The number of points in the quadrature rule.
 * @param k The wavenumber of the medium in which the scattered field is being computed (k2).
 * @param L_results (Output) A 3-element array to store the vector results for the L operator for each of the 3 half-basis functions.
 * @param K_results (Output) A 3-element array to store the vector results for the K operator for each of the 3 half-basis functions.
 */
__host__ __device__ inline void calculateScatterFieldQuadrature(
    const vec3_t& obs_point,
    const vec3_t source_tri_v[3],
    const int source_tri_indices[3],
    const float* ref_points,
    const float* quad_weights,
    int num_quad_points,
    float k,
    bool is_regular,
    cvec3_t L_results[3],
    cvec3_t K_results[3])
{
    // Initialize local result vectors
    const complex__float zero = make_complex(0.f, 0.f);
    for (int i = 0; i < 3; ++i) {
        L_results[i] = make_cvec3(zero, zero, zero);
        K_results[i] = make_cvec3(zero, zero, zero);
    }

    // removed reinterpret_cast
    const float k_sq_inv = 1.f / (k * k);

    // Pre-calculate free vertices and edge lengths for all 3 edges
    vec3_t source_free_vertices[3];
    float source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri_v, i, source_free_vertices[i]);
    }

    // Loop over quadrature points
    for (int q = 0; q < num_quad_points; ++q) {
        const float2 ref_q = {ref_points[2 * q], ref_points[2 * q + 1]};
        const float w_q = quad_weights[q];

        vec3_t r_q;
        mapTriQuadPoint(source_tri_v, ref_q.x, ref_q.y, r_q);

        // Get Green's function and its gradient at the quadrature point (computed together for efficiency)
        greenAndGradGreen green_and_grad;
        if (is_regular) {
            green_and_grad = greenAndGradSourceGreenFunctionRegular(obs_point, r_q, k);
        } else {
            green_and_grad = greenAndGradSourceGreenFunction(obs_point, r_q, k);
        }
        const complex__float G = green_and_grad.green;
        const cvec3_t grad_G = green_and_grad.grad_green;

        // Pre-calculate all rho vectors at the current quadrature point
        vec3_t rho_vectors[3];
        for (int i = 0; i < 3; ++i) {
            rho_vectors[i] = r_q - source_free_vertices[i];
        }

        // Now, compute the 3x3 matrix contributions at this point
        for (int i = 0; i < 3; ++i) {
            // --- L operator part ---
            // Term 1: 0.5 * rho_i * G
            cvec3_t L_term1 = 0.5f * (rho_vectors[i] * G);
            // Term 2: -1/k^2 * grad_G
            cvec3_t L_term2 = -k_sq_inv * grad_G;
            // Accumulate weighted sum for L
            L_results[i] += w_q * (L_term1 + L_term2);

            // --- K operator part ---
            // Term: 0.5 * rho_i x grad_G
            cvec3_t K_term = 0.5f * cross(rho_vectors[i], grad_G);
            // Accumulate weighted sum for K
            K_results[i] += w_q * K_term;
        }
    }

    // Apply the final scaling factor (edge lengths) to all elements
    for (int i = 0; i < 3; ++i) {
        const float eni = source_edge_lengths[i];
        L_results[i] = eni * L_results[i];
        K_results[i] = eni * K_results[i];
    }
}




/**
 * @brief Computes the analytical singular parts of the scattered field (L and K operators).
 */
__host__ __device__ inline void calculateScatterFieldSingular(
    const vec3_t& obs_point,
    const vec3_t source_tri_v[3],
    const int source_tri_indices[3],
    float k,
    cvec3_t L_results[3],
    cvec3_t K_results[3])
{
    const complex__float zero = make_complex(0.f, 0.f);
    for (int i = 0; i < 3; ++i) {
        L_results[i] = make_cvec3(zero, zero, zero);
        K_results[i] = make_cvec3(zero, zero, zero);
    }

    vec3_t source_free_vertices[3];
    float source_edge_lengths[3];
    for (int i = 0; i < 3; ++i) {
        source_edge_lengths[i] = getEdgeInfo(source_tri_indices, source_tri_v, i, source_free_vertices[i]);
    }

    const float A = 0.5f * length(cross(source_tri_v[1] - source_tri_v[0], source_tri_v[2] - source_tri_v[0]));
    const float base_scale = 1.0f / (8.0f * static_cast<float>(M_PI) * A);

    WiltonPotentials pots = calculateAllWiltonPotentials(obs_point, source_tri_v);

    const float k_sq_inv = 1.f / (k * k);
    
    for (int i = 0; i < 3; ++i) {
        vec3_t R_free = obs_point - source_free_vertices[i];
        
        // --- L operator singular part ---
        vec3_t psi_1 = calculate_Psi_1(pots, source_free_vertices[i]);
        vec3_t phi_3 = pots.phi_3_base + (pots.r_pi - source_free_vertices[i]) * pots.phi_2;
        vec3_t L_vec_inner = psi_1 - 0.5f * R_free * pots.phi_0 + k_sq_inv * (phi_3 - R_free * pots.phi_2);
        
        // --- K operator singular part ---
        vec3_t K_vec = calculate_Upsilon_coul(pots, source_free_vertices[i], obs_point, k);
        
        float l_scale = source_edge_lengths[i] * base_scale;
        
        // l_scale applies e_ni / (8 * pi * A_n). 
        // L operator needs e_ni / (4 * pi * A_n), so we multiply by 2.0f.
        vec3_t L_vec = 2.0f * L_vec_inner;
        
        L_results[i] = make_cvec3(make_complex(L_vec.x * l_scale, 0.f),
                                  make_complex(L_vec.y * l_scale, 0.f),
                                  make_complex(L_vec.z * l_scale, 0.f));
                                  
        K_results[i] = make_cvec3(make_complex(K_vec.x * l_scale, 0.f),
                                  make_complex(K_vec.y * l_scale, 0.f),
                                  make_complex(K_vec.z * l_scale, 0.f));
    }
}

//// End Scattered-field quadrature

#endif // DUNAVANT_QUADRATURE_CUH
