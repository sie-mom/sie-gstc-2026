#include "catch.hpp"
#include <vector>
#include <array>
#include <iostream>

#include "sie_mom/cuda/platform.cuh"
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/quadrature/gauss_legendre_rule.hpp"
#include "sie_mom/cuda/rwg_basis_utils.cuh"
#include "sie_mom/cuda/green_functions_utils.cuh"




TEST_CASE("Test calculateSingularPartQuadrature on CPU", "[CPU][Quadrature]")
{
    // 1. Define a simple test triangle (e.g., with area 0.5)
    const vec3_t test_tri[3] = {
        {0.f, 0.f, 0.f}, // v0
        {1.f, 0.f, 0.f}, // v1
        {0.f, 1.f, 0.f}  // v2
    };
    const float triangle_area = 0.5f;

    // 2. Define a simple quadrature rule (e.g., 1-point rule at the centroid)
    const float quad_xi_eta[] = {1.f/3.f, 1.f/3.f}; // (xi, eta)
    const float quad_w[] = {1.f};
    const int num_quad_points = 1;

    // 3. Define the testing basis function details
    const int test_free_vertex_idx = 2; // Free vertex is v2

    // 4. Define the lambda function for the operator kernel
    auto simple_kernel = [&](const vec3_t& r_obs) -> IntegrandComponents {
        IntegrandComponents components;
        components.vector_part = {1.f, 0.f, 0.f};
        components.scalar_part = 2.f;
        return components;
    };

    // 5. Call the function under test
    float integral_value = calculateSingularPartQuadrature(
        test_tri,
        test_free_vertex_idx,
        quad_xi_eta,
        quad_w,
        num_quad_points,
        simple_kernel
    );

    float h_result = integral_value * triangle_area;

    // --- Calculate Expected Result ---
    // (Same logic as the CUDA test)
    // Quadrature point: r_mp = (1/3, 1/3, 0)
    // Free vertex: r_free = (0, 1, 0)
    // Testing basis function: rho_mj = (1/3, -2/3, 0)
    // Integrand: dot(<1/3, -2/3, 0>, <1, 0, 0>) + 2.0 = 1/3 + 2.0 = 7/3
    // Quadrature sum: 1.0 * (7/3) = 7/3
    // Final result (scaled by area): (7/3) * 0.5 = 7/6
    const float expected_result = 7.f / 6.f;

    REQUIRE(h_result == Approx(expected_result));
}


TEST_CASE("TL Operator: Standard Double Quadrature vs. Singularity Subtraction (Coplanar)", "[Quadrature][TL][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar/non-coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close but not touching.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.0f}, {1.f, 0.f, 0.0f}, {0.f, 1.f, 0.0f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};

    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // Choose basis functions (e.g., edge 0 on both triangles)
    const int test_edge_idx = 0;
    const int source_edge_idx = 0;

    // Get free vertex indices corresponding to the chosen edges
    vec3_t test_free_v, source_free_v;
    const float emj = getEdgeInfo(test_tri_indices, test_tri_v, test_edge_idx, test_free_v);
    const float eni = getEdgeInfo(source_tri_indices, source_tri_v, source_edge_idx, source_free_v);
    // For local edge 0, the free vertex is index 2
    const int test_free_vertex_idx = 2;
    const int source_free_vertex_idx = 2;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpq{}, ref_matrix);
    const complex__float ref_val = ref_matrix[test_edge_idx][source_edge_idx];

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpqRegular{}, reg_matrix);
    const complex__float reg_val = reg_matrix[test_edge_idx][source_edge_idx];

    // C. Singular (Coulomb) part using semi-analytic method
    TLSingularKernel tl_singular_kernel;
    tl_singular_kernel.source_tri = source_tri_v;
    tl_singular_kernel.source_free_vertex_idx = source_free_vertex_idx;
    tl_singular_kernel.k_inv_sq = 1.f / (k * k);

    float singular_integral = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx,
        quad_points.data(), quad_weights.data(), order_num,
        tl_singular_kernel);
    
    // The formula for the Coulomb part has a 1/(4*pi) scaling factor
    const float coul_scaling = 1.f / (4.f * M_PI)/source_area*emj*eni;
    const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

    // D. Combine regular and singular parts
    const complex__float combined_val = complexAdd(reg_val, singular_val);

    // 3. --- VALIDATION ---
    INFO("Reference: (" << ref_val.x << ", " << ref_val.y << ")");
    INFO("Combined:  (" << combined_val.x << ", " << combined_val.y << ")");
    REQUIRE(combined_val.x == Approx(ref_val.x).epsilon(1e-4));
    REQUIRE(combined_val.y == Approx(ref_val.y).epsilon(1e-4));
}


TEST_CASE("TL Operator: Standard Double Quadrature vs. Singularity Subtraction (Non-Coplanar)", "[Quadrature][TL][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar/non-coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close but not touching.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};

    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // Choose basis functions (e.g., edge 0 on both triangles)
    const int test_edge_idx = 0;
    const int source_edge_idx = 0;

    // Get free vertex indices corresponding to the chosen edges
    vec3_t test_free_v, source_free_v;
    const float emj = getEdgeInfo(test_tri_indices, test_tri_v, test_edge_idx, test_free_v);
    const float eni = getEdgeInfo(source_tri_indices, source_tri_v, source_edge_idx, source_free_v);
    // For local edge 0, the free vertex is index 2
    const int test_free_vertex_idx = 2;
    const int source_free_vertex_idx = 2;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpq{}, ref_matrix);
    const complex__float ref_val = ref_matrix[test_edge_idx][source_edge_idx];

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpqRegular{}, reg_matrix);
    const complex__float reg_val = reg_matrix[test_edge_idx][source_edge_idx];

    // C. Singular (Coulomb) part using semi-analytic method
    TLSingularKernel tl_singular_kernel;
    tl_singular_kernel.source_tri = source_tri_v;
    tl_singular_kernel.source_free_vertex_idx = source_free_vertex_idx;
    tl_singular_kernel.k_inv_sq = 1.f / (k * k);

    float singular_integral = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx,
        quad_points.data(), quad_weights.data(), order_num,
        tl_singular_kernel);
    
    // The formula for the Coulomb part has a 1/(4*pi) scaling factor
    const float coul_scaling = 1.f / (4.f * M_PI)/source_area*emj*eni;
    const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

    // D. Combine regular and singular parts
    const complex__float combined_val = complexAdd(reg_val, singular_val);

    // 3. --- VALIDATION ---
    INFO("Reference: (" << ref_val.x << ", " << ref_val.y << ")");
    INFO("Combined:  (" << combined_val.x << ", " << combined_val.y << ")");
    REQUIRE(combined_val.x == Approx(ref_val.x).epsilon(1e-4));
    REQUIRE(combined_val.y == Approx(ref_val.y).epsilon(1e-4));
}

TEST_CASE("TL Operator Full Matrix: Standard Double Quadrature vs. Singularity Subtraction (Non-Coplanar)", "[Quadrature][TL][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar/non-coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close and non-coplanar.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};
    
    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpq{}, ref_matrix);

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TLModifier{}, TLDpqRegular{}, reg_matrix);

    // C. Assemble the full 3x3 singular (Coulomb) matrix and add it to the regular matrix
    complex__float combined_matrix[3][3];
    TLSingularKernel tl_singular_kernel;
    tl_singular_kernel.source_tri = source_tri_v;
    tl_singular_kernel.k_inv_sq = 1.f / (k * k);

    for (int j = 0; j < 3; ++j)
    {
        int test_free_v_idx;
        const float emj = getEdgeInfo(test_tri_indices, test_tri_v, j, test_free_v_idx);
        for (int i = 0; i < 3; ++i)
        {
            int source_free_v_idx;
            const float eni = getEdgeInfo(source_tri_indices, source_tri_v, i, source_free_v_idx);
            tl_singular_kernel.source_free_vertex_idx = source_free_v_idx;

            // Calculate the raw singular integral for the (j,i) interaction
            float singular_integral = calculateSingularPartQuadrature(
                test_tri_v, test_free_v_idx,
                quad_points.data(), quad_weights.data(), order_num,
                tl_singular_kernel);

            // The formula for the Coulomb part has a 1/(4*pi) scaling factor
            const float coul_scaling = 1.f / (4.f * M_PI)/source_area*emj*eni;
            const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

            // Combine with the regular part for this matrix element
            combined_matrix[j][i] = complexAdd(reg_matrix[j][i], singular_val);
        }
    }

    // 3. --- VALIDATION ---
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("Matrix element (" << j << ", " << i << ")");
            REQUIRE(combined_matrix[j][i].x == Approx(ref_matrix[j][i].x).epsilon(1e-4));
            REQUIRE(combined_matrix[j][i].y == Approx(ref_matrix[j][i].y).epsilon(1e-4));
        }
    }
}

TEST_CASE("TK Operator: Standard Double Quadrature vs. Singularity Subtraction (Coplanar)", "[Quadrature][TK][SingularitySubtraction][Coplanar]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close but not touching.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.0f}, {1.f, 0.f, 0.0f}, {0.f, 1.f, 0.0f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};

    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // Choose basis functions (e.g., edge 0 on both triangles)
    const int test_edge_idx = 0;
    const int source_edge_idx = 0;

    // Get free vertex indices corresponding to the chosen edges
    vec3_t test_free_v, source_free_v;
    const float emj = getEdgeInfo(test_tri_indices, test_tri_v, test_edge_idx, test_free_v);
    const float eni = getEdgeInfo(source_tri_indices, source_tri_v, source_edge_idx, source_free_v);
    // For local edge 0, the free vertex is index 2
    const int test_free_vertex_idx = 2;
    const int source_free_vertex_idx = 2;

    // High-order quadrature for accuracy
    const int rule = 20;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpq{}, ref_matrix);
    const complex__float ref_val = ref_matrix[test_edge_idx][source_edge_idx];

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpqRegular{}, reg_matrix);
    const complex__float reg_val = reg_matrix[test_edge_idx][source_edge_idx];

    // C. Singular (Coulomb) part using semi-analytic method
    TKSingularKernel tk_singular_kernel;
    tk_singular_kernel.source_tri = source_tri_v;
    tk_singular_kernel.source_free_vertex_idx = source_free_vertex_idx;
    tk_singular_kernel.k = k;

    float singular_integral = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx,
        quad_points.data(), quad_weights.data(), order_num,
        tk_singular_kernel);
    
    // The formula for the Coulomb part of TK is scaled by emj*eni/An.
    // The 1/(4*pi) factor is already included inside calculate_Upsilon_coul.
    const float coul_scaling = 1.f / (4.f * M_PI) / source_area*emj*eni;
    const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

    // D. Combine regular and singular parts
    const complex__float combined_val = complexAdd(reg_val, singular_val);

    // 3. --- VALIDATION ---
    INFO("Reference: (" << ref_val.x << ", " << ref_val.y << ")");
    INFO("Combined:  (" << combined_val.x << ", " << combined_val.y << ")");
    REQUIRE(combined_val.x == Approx(ref_val.x).epsilon(1e-4));
    REQUIRE(combined_val.y == Approx(ref_val.y).epsilon(1e-4));
}

TEST_CASE("TK Operator: Standard Double Quadrature vs. Singularity Subtraction (Non-Coplanar))", "[Quadrature][TK][SingularitySubtraction][NonCoplanar]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close but not touching.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};

    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // Choose basis functions (e.g., edge 0 on both triangles)
    const int test_edge_idx = 0;
    const int source_edge_idx = 0;

    // Get free vertex indices corresponding to the chosen edges
    vec3_t test_free_v, source_free_v;
    const float emj = getEdgeInfo(test_tri_indices, test_tri_v, test_edge_idx, test_free_v);
    const float eni = getEdgeInfo(source_tri_indices, source_tri_v, source_edge_idx, source_free_v);
    // For local edge 0, the free vertex is index 2
    const int test_free_vertex_idx = 2;
    const int source_free_vertex_idx = 2;

    // High-order quadrature for accuracy
    const int rule = 20;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpq{}, ref_matrix);
    const complex__float ref_val = ref_matrix[test_edge_idx][source_edge_idx];

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpqRegular{}, reg_matrix);
    const complex__float reg_val = reg_matrix[test_edge_idx][source_edge_idx];

    // C. Singular (Coulomb) part using semi-analytic method
    TKSingularKernel tk_singular_kernel;
    tk_singular_kernel.source_tri = source_tri_v;
    tk_singular_kernel.source_free_vertex_idx = source_free_vertex_idx;
    tk_singular_kernel.k = k;

    float singular_integral = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx,
        quad_points.data(), quad_weights.data(), order_num,
        tk_singular_kernel);
    
    // The formula for the Coulomb part of TK is scaled by emj*eni/An.
    // The 1/(4*pi) factor is already included inside calculate_Upsilon_coul.
    const float coul_scaling = 1.f / (4.f * M_PI) / source_area*emj*eni;
    const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

    // D. Combine regular and singular parts
    const complex__float combined_val = complexAdd(reg_val, singular_val);

    // 3. --- VALIDATION ---
    INFO("Reference: (" << ref_val.x << ", " << ref_val.y << ")");
    INFO("Combined:  (" << combined_val.x << ", " << combined_val.y << ")");
    REQUIRE(combined_val.x == Approx(ref_val.x).epsilon(1e-4));
    REQUIRE(combined_val.y == Approx(ref_val.y).epsilon(1e-4));
}

TEST_CASE("TK Operator Full Matrix: Standard Double Quadrature vs. Singularity Subtraction (Non-Coplanar)", "[Quadrature][TL][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for the TL operator:
    // Integral(G) == Integral(G_reg) + Integral(G_coul)
    // for two close, coplanar/non-coplanar, non-touching triangles.

    // 1. --- SETUP ---
    // Two triangles, close and non-coplanar.
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};
    
    const float test_area = 0.5f;
    const float source_area = 0.5f;
    const float k = 1.5f;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATIONS ---

    // A. Reference value using standard double quadrature
    complex__float ref_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpq{}, ref_matrix);

    // B. Regular part using double quadrature with regularized Green's function
    complex__float reg_matrix[3][3];
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k,
                              TKModifier{}, TKDpqRegular{}, reg_matrix);

    // C. Assemble the full 3x3 singular (Coulomb) matrix and add it to the regular matrix
    complex__float combined_matrix[3][3];
    TKSingularKernel tk_singular_kernel;
    tk_singular_kernel.source_tri = source_tri_v;
    tk_singular_kernel.k = k;

    for (int j = 0; j < 3; ++j)
    {
        int test_free_v_idx;
        const float emj = getEdgeInfo(test_tri_indices, test_tri_v, j, test_free_v_idx);
        for (int i = 0; i < 3; ++i)
        {
            int source_free_v_idx;
            const float eni = getEdgeInfo(source_tri_indices, source_tri_v, i, source_free_v_idx);
            tk_singular_kernel.source_free_vertex_idx = source_free_v_idx;

            // Calculate the raw singular integral for the (j,i) interaction
            float singular_integral = calculateSingularPartQuadrature(
                test_tri_v, test_free_v_idx,
                quad_points.data(), quad_weights.data(), order_num,
                tk_singular_kernel);

            // The formula for the Coulomb part has a 1/(4*pi) scaling factor
            const float coul_scaling = 1.f / (4.f * M_PI)/source_area*emj*eni;
            const complex__float singular_val = make_complex(singular_integral * coul_scaling, 0.f);

            // Combine with the regular part for this matrix element
            combined_matrix[j][i] = complexAdd(reg_matrix[j][i], singular_val);
        }
    }

    // 3. --- VALIDATION ---
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("Matrix element (" << j << ", " << i << ")");
            REQUIRE(combined_matrix[j][i].x == Approx(ref_matrix[j][i].x).epsilon(1e-4));
            REQUIRE(combined_matrix[j][i].y == Approx(ref_matrix[j][i].y).epsilon(1e-4));
        }
    }
}

TEST_CASE("Compare Singular Quadrature Overloads for a Single Interaction", "[Quadrature][SingularitySubtraction][Debug]")
{
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};
    const float source_area = getTriangleArea(source_tri_v);

    const float k1 = 1.5f;
    const float k2 = 2.5f;

    const int test_edge_idx = 0;
    const int source_edge_idx = 1;
    int test_free_vertex_idx, source_free_vertex_idx;
    const float emj = getEdgeInfo(test_tri_indices, test_tri_v, test_edge_idx, test_free_vertex_idx);
    const float eni = getEdgeInfo(source_tri_indices, source_tri_v, source_edge_idx, source_free_vertex_idx);

    const int rule = 19;
    const int order_num = dunavant_order_num(rule);
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    {
        std::vector<double> xy_double(2 * order_num);
        std::vector<double> w_double(order_num);
        dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
        for (int i = 0; i < order_num; ++i)
        {
            quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
            quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
            quad_weights[i] = static_cast<float>(w_double[i]);
        }
    }

    TLSingularKernel tl_kernel_k1;
    tl_kernel_k1.source_tri = source_tri_v;
    tl_kernel_k1.source_free_vertex_idx = source_free_vertex_idx;
    tl_kernel_k1.k_inv_sq = 1.f / (k1 * k1);
    const float ref_tl1_raw = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx, quad_points.data(), quad_weights.data(), order_num, tl_kernel_k1);

    TLSingularKernel tl_kernel_k2 = tl_kernel_k1;
    tl_kernel_k2.k_inv_sq = 1.f / (k2 * k2);
    const float ref_tl2_raw = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx, quad_points.data(), quad_weights.data(), order_num, tl_kernel_k2);

    TKSingularKernel tk_kernel_k1;
    tk_kernel_k1.source_tri = source_tri_v;
    tk_kernel_k1.source_free_vertex_idx = source_free_vertex_idx;
    tk_kernel_k1.k = k1;
    const float ref_tk1_raw = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx, quad_points.data(), quad_weights.data(), order_num, tk_kernel_k1);

    TKSingularKernel tk_kernel_k2 = tk_kernel_k1;
    tk_kernel_k2.k = k2;
    const float ref_tk2_raw = calculateSingularPartQuadrature(
        test_tri_v, test_free_vertex_idx, quad_points.data(), quad_weights.data(), order_num, tk_kernel_k2);

    const float global_scaling = emj * eni / (4.f * M_PI * source_area);
    const float ref_tl1 = ref_tl1_raw * global_scaling;
    const float ref_tl2 = ref_tl2_raw * global_scaling;
    const float ref_tk1 = ref_tk1_raw * global_scaling;
    const float ref_tk2 = ref_tk2_raw * global_scaling;

    complex__float matrix_tl_k1[3][3];
    complex__float matrix_tl_k2[3][3];
    complex__float matrix_tk_k1[3][3];
    complex__float matrix_tk_k2[3][3];
    for (int j = 0; j < 3; ++j)
    {
        for (int i = 0; i < 3; ++i)
        {
            matrix_tl_k1[j][i] = make_complex(0.f, 0.f);
            matrix_tl_k2[j][i] = make_complex(0.f, 0.f);
            matrix_tk_k1[j][i] = make_complex(0.f, 0.f);
            matrix_tk_k2[j][i] = make_complex(0.f, 0.f);
        }
    }

    calculateSingularPartQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points.data(), quad_weights.data(), order_num,
                                    k1, k2,
                                    matrix_tl_k1, matrix_tl_k2, matrix_tk_k1, matrix_tk_k2);

    const complex__float tl1 = matrix_tl_k1[test_edge_idx][source_edge_idx];
    const complex__float tl2 = matrix_tl_k2[test_edge_idx][source_edge_idx];
    const complex__float tk1 = matrix_tk_k1[test_edge_idx][source_edge_idx];
    const complex__float tk2 = matrix_tk_k2[test_edge_idx][source_edge_idx];

    REQUIRE(tl1.x == Approx(ref_tl1).epsilon(1e-6f));
    REQUIRE(tl1.y == Approx(0.f));
    REQUIRE(tl2.x == Approx(ref_tl2).epsilon(1e-6f));
    REQUIRE(tl2.y == Approx(0.f));
    REQUIRE(tk1.x == Approx(ref_tk1).epsilon(1e-6f));
    REQUIRE(tk1.y == Approx(0.f));
    REQUIRE(tk2.x == Approx(ref_tk2).epsilon(1e-6f));
    REQUIRE(tk2.y == Approx(0.f));
}


TEST_CASE("All Operators Full Matrix: Standard vs. Singularity Subtraction (Non-Coplanar)", "[Quadrature][AllOps][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for all four operators (TL/TK at k1/k2)
    // using the optimized, combined quadrature functions.
    // It computes Integral(G) and compares it against Integral(G_reg) + Integral(G_coul).

    // 1. --- SETUP ---
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};
    
    const float k1 = 1.5f;
    const float k2 = 2.5f;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    {
        std::vector<double> xy_double(2 * order_num);
        std::vector<double> w_double(order_num);
        dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
        for (int i = 0; i < order_num; ++i) {
            quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
            quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
            quad_weights[i] = static_cast<float>(w_double[i]);
        }
    }

    // 2. --- CALCULATIONS ---

    // --- A. Reference values using the correct full-integral kernels ---
    complex__float ref_matrix_tl_k1[3][3], ref_matrix_tl_k2[3][3];
    complex__float ref_matrix_tk_k1[3][3], ref_matrix_tk_k2[3][3];

    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k1,
                              TLModifier{}, TLDpq{}, ref_matrix_tl_k1);
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k2,
                              TLModifier{}, TLDpq{}, ref_matrix_tl_k2);
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k1,
                              TKModifier{}, TKDpq{}, ref_matrix_tk_k1);
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num, k2,
                              TKModifier{}, TKDpq{}, ref_matrix_tk_k2);

    // --- B. Test values using the combined singularity subtraction method ---
    complex__float combined_tl_k1[3][3], combined_tl_k2[3][3];
    complex__float combined_tk_k1[3][3], combined_tk_k2[3][3];

    // B1. Calculate the regular part for all operators at once
    calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                              quad_points.data(), quad_weights.data(), order_num,
                              k1, k2, true, // is_singular = true -> use regular kernel
                              combined_tl_k1, combined_tl_k2, combined_tk_k1, combined_tk_k2);

    // B2. Calculate the singular (Coulomb) part into SEPARATE matrices
    complex__float singular_tl_k1[3][3], singular_tl_k2[3][3];
    complex__float singular_tk_k1[3][3], singular_tk_k2[3][3];

    calculateSingularPartQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                   quad_points.data(), quad_weights.data(), order_num,
                                   k1, k2,
                                   singular_tl_k1, singular_tl_k2, singular_tk_k1, singular_tk_k2);

    // 3. --- VALIDATION ---
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("Matrix element (" << j << ", " << i << ")");
            // Manually add the regular and singular parts for validation
            
            // TL k1
            complex__float final_tl_k1 = complexAdd(combined_tl_k1[j][i], singular_tl_k1[j][i]);
            REQUIRE(final_tl_k1.x == Approx(ref_matrix_tl_k1[j][i].x).epsilon(1e-4));
            REQUIRE(final_tl_k1.y == Approx(ref_matrix_tl_k1[j][i].y).epsilon(1e-4));
            
            // TL k2
            complex__float final_tl_k2 = complexAdd(combined_tl_k2[j][i], singular_tl_k2[j][i]);
            REQUIRE(final_tl_k2.x == Approx(ref_matrix_tl_k2[j][i].x).epsilon(1e-4));
            REQUIRE(final_tl_k2.y == Approx(ref_matrix_tl_k2[j][i].y).epsilon(1e-4));

            // TK k1
            complex__float final_tk_k1 = complexAdd(combined_tk_k1[j][i], singular_tk_k1[j][i]);
            REQUIRE(final_tk_k1.x == Approx(ref_matrix_tk_k1[j][i].x).epsilon(1e-4));
            REQUIRE(final_tk_k1.y == Approx(ref_matrix_tk_k1[j][i].y).epsilon(1e-4));

            // TK k2
            complex__float final_tk_k2 = complexAdd(combined_tk_k2[j][i], singular_tk_k2[j][i]);
            REQUIRE(final_tk_k2.x == Approx(ref_matrix_tk_k2[j][i].x).epsilon(1e-4));
            REQUIRE(final_tk_k2.y == Approx(ref_matrix_tk_k2[j][i].y).epsilon(1e-4));
        }
    }
}

TEST_CASE("All Operators Full Matrix: Standard vs. Singularity Subtraction nTL/nTK (Non-Coplanar)", "[Quadrature][nTL][nTK][SingularitySubtraction]")
{
    // This test validates the singularity subtraction identity for the nTL and nTK operators.
    // It cleanly computes the raw DoubleQuadrature(is_singular=false) and compares it against 
    // DoubleQuadrature(is_singular=true) + calculateSingularPartQuadrature_nTL_nTK().

    // 1. --- SETUP ---
    // Use disjoint, non-coplanar triangles so the numerical "truth" integral is completely stable,
    // and the PV skipping logic is not triggered (allowing nTK to be fully evaluated).
    const vec3_t test_tri_v[3] = { {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f} };
    const vec3_t source_tri_v[3] = { {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f} };
    const int test_tri_indices[3] = {0, 1, 2};
    const int source_tri_indices[3] = {3, 4, 5};
    
    const float k1 = 1.5f;
    const float k2 = 2.5f;

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    {
        std::vector<double> xy_double(2 * order_num);
        std::vector<double> w_double(order_num);
        dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
        for (int i = 0; i < order_num; ++i) {
            quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
            quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
            quad_weights[i] = static_cast<float>(w_double[i]);
        }
    }

    // 2. --- CALCULATIONS ---

    // --- A. Reference values using the RAW standard double quadrature (no extraction) ---
    complex__float ref_ntl_k1[3][3], ref_ntl_k2[3][3];
    complex__float ref_ntk_k1[3][3], ref_ntk_k2[3][3];

    calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                      quad_points.data(), quad_weights.data(), order_num,
                                      k1, k2, false, // is_singular = false -> raw G and nabla G
                                      ref_ntl_k1, ref_ntl_k2, ref_ntk_k1, ref_ntk_k2);

    // --- B. Test values using the combined singularity subtraction method ---
    complex__float reg_ntl_k1[3][3], reg_ntl_k2[3][3];
    complex__float reg_ntk_k1[3][3], reg_ntk_k2[3][3];

    // B1. Calculate the regular part
    calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                      quad_points.data(), quad_weights.data(), order_num,
                                      k1, k2, true, // is_singular = true -> G_reg and nabla G_reg
                                      reg_ntl_k1, reg_ntl_k2, reg_ntk_k1, reg_ntk_k2);

    // B2. Calculate the singular (Coulomb) part analytically
    complex__float sing_ntl_k1[3][3], sing_ntl_k2[3][3];
    complex__float sing_ntk_k1[3][3], sing_ntk_k2[3][3];

    calculateSingularPartQuadrature_nTL_nTK(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                           quad_points.data(), quad_weights.data(), order_num,
                                           k1, k2,
                                           sing_ntl_k1, sing_ntl_k2, sing_ntk_k1, sing_ntk_k2);

    // 3. --- VALIDATION ---
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("Matrix element (" << j << ", " << i << ")");
            
            // nTL k1
            complex__float final_ntl_k1 = complexAdd(reg_ntl_k1[j][i], sing_ntl_k1[j][i]);
            REQUIRE(final_ntl_k1.x == Approx(ref_ntl_k1[j][i].x).margin(1e-5));
            REQUIRE(final_ntl_k1.y == Approx(ref_ntl_k1[j][i].y).margin(1e-5));
            
            // nTL k2
            complex__float final_ntl_k2 = complexAdd(reg_ntl_k2[j][i], sing_ntl_k2[j][i]);
            REQUIRE(final_ntl_k2.x == Approx(ref_ntl_k2[j][i].x).margin(1e-5));
            REQUIRE(final_ntl_k2.y == Approx(ref_ntl_k2[j][i].y).margin(1e-5));

            // nTK k1
            complex__float final_ntk_k1 = complexAdd(reg_ntk_k1[j][i], sing_ntk_k1[j][i]);
            REQUIRE(final_ntk_k1.x == Approx(ref_ntk_k1[j][i].x).margin(1e-5));
            REQUIRE(final_ntk_k1.y == Approx(ref_ntk_k1[j][i].y).margin(1e-5));

            // nTK k2
            complex__float final_ntk_k2 = complexAdd(reg_ntk_k2[j][i], sing_ntk_k2[j][i]);
            REQUIRE(final_ntk_k2.x == Approx(ref_ntk_k2[j][i].x).margin(1e-5));
            REQUIRE(final_ntk_k2.y == Approx(ref_ntk_k2[j][i].y).margin(1e-5));
        }
    }
}

TEST_CASE("Form A nTL and nTK: raw quadrature equals regular plus singular",
          "[Quadrature][nTL][nTK][FormASingularitySubtraction]")
{
    // Disjoint noncoplanar triangles keep the raw G integral well defined.
    const vec3_t test_tri[3] = {
        {0.f, 0.f, 0.1f}, {1.f, 0.f, 0.1f}, {0.f, 1.f, 0.1f}
    };
    const vec3_t source_tri[3] = {
        {1.2f, 0.f, 0.f}, {2.2f, 0.f, 0.f}, {1.2f, 1.f, 0.f}
    };
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {3, 4, 5};
    constexpr real_t k1 = real_t(1.5);
    constexpr real_t k2 = real_t(2.5);

    const int triangle_rule = 13;
    const int triangle_points = dunavant_order_num(triangle_rule);
    std::vector<double> triangle_nodes_double(2 * triangle_points);
    std::vector<double> triangle_weights_double(triangle_points);
    dunavant_rule(triangle_rule, triangle_points,
                  triangle_nodes_double.data(), triangle_weights_double.data());
    std::vector<float> triangle_nodes(triangle_nodes_double.begin(), triangle_nodes_double.end());
    std::vector<float> triangle_weights(triangle_weights_double.begin(), triangle_weights_double.end());

    const int line_points = 20;
    std::vector<double> line_nodes_double(line_points);
    std::vector<double> line_weights_double(line_points);
    gauss_legendre_rule(line_points, line_nodes_double.data(), line_weights_double.data());
    std::vector<float> line_nodes(line_nodes_double.begin(), line_nodes_double.end());
    std::vector<float> line_weights(line_weights_double.begin(), line_weights_double.end());

    complex__float raw_ntl_k1[3][3], raw_ntl_k2[3][3];
    complex__float raw_ntk_k1[3][3], raw_ntk_k2[3][3];
    calculateFormAQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, false, raw_ntl_k1, raw_ntl_k2, raw_ntk_k1, raw_ntk_k2);

    complex__float reg_ntl_k1[3][3], reg_ntl_k2[3][3];
    complex__float reg_ntk_k1[3][3], reg_ntk_k2[3][3];
    calculateFormAQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, true, reg_ntl_k1, reg_ntl_k2, reg_ntk_k1, reg_ntk_k2);

    complex__float sing_ntl_k1[3][3], sing_ntl_k2[3][3];
    complex__float sing_ntk_k1[3][3], sing_ntk_k2[3][3];
    calculateFormASingularPartQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, sing_ntl_k1, sing_ntl_k2, sing_ntk_k1, sing_ntk_k2);

    // nTK has the same analytical surface contribution in both formulations.
    complex__float old_sing_ntl_k1[3][3], old_sing_ntl_k2[3][3];
    complex__float old_sing_ntk_k1[3][3], old_sing_ntk_k2[3][3];
    calculateSingularPartQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        k1, k2, old_sing_ntl_k1, old_sing_ntl_k2,
        old_sing_ntk_k1, old_sing_ntk_k2);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const complex__float raw[2][2] = {
                {raw_ntl_k1[j][i], raw_ntl_k2[j][i]},
                {raw_ntk_k1[j][i], raw_ntk_k2[j][i]}
            };
            const complex__float reg[2][2] = {
                {reg_ntl_k1[j][i], reg_ntl_k2[j][i]},
                {reg_ntk_k1[j][i], reg_ntk_k2[j][i]}
            };
            const complex__float sing[2][2] = {
                {sing_ntl_k1[j][i], sing_ntl_k2[j][i]},
                {sing_ntk_k1[j][i], sing_ntk_k2[j][i]}
            };
            INFO("singular nTK row=" << j << ", col=" << i);
            CHECK(sing_ntk_k1[j][i].x == Approx(old_sing_ntk_k1[j][i].x).margin(1e-6));
            CHECK(sing_ntk_k1[j][i].y == Approx(old_sing_ntk_k1[j][i].y).margin(1e-6));
            CHECK(sing_ntk_k2[j][i].x == Approx(old_sing_ntk_k2[j][i].x).margin(1e-6));
            CHECK(sing_ntk_k2[j][i].y == Approx(old_sing_ntk_k2[j][i].y).margin(1e-6));
            for (int op = 0; op < 2; ++op) {
                for (int medium = 0; medium < 2; ++medium) {
                    const complex__float reconstructed =
                        complexAdd(reg[op][medium], sing[op][medium]);
                    INFO("operator=" << (op == 0 ? "nTL" : "nTK")
                         << ", row=" << j << ", col=" << i << ", medium=" << medium);
                    CHECK(reconstructed.x == Approx(raw[op][medium].x).margin(1e-5));
                    CHECK(reconstructed.y == Approx(raw[op][medium].y).margin(1e-5));
                }
            }
        }
    }
}

TEST_CASE("Scattered Field: Standard Quadrature vs. Singularity Subtraction", "[Quadrature][Scattering][SingularitySubtraction]")
{
    // 1. --- SETUP ---
    // Single triangle
    const vec3_t source_tri_v[3] = { {0.f, 0.f, 0.0f}, {1.f, 0.f, 0.0f}, {0.f, 1.f, 0.0f} };
    const int source_tri_indices[3] = {0, 1, 2};
    const float k_medium = 1.5f;

    // Observation point near the triangle but not exactly on it
    const vec3_t obs_point = {0.33f, 0.33f, 2.0f}; 

    // High-order quadrature for accuracy
    const int rule = 13;
    const int order_num = dunavant_order_num(rule);
    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());
    
    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int i = 0; i < order_num; ++i) {
        quad_points[2 * i] = static_cast<float>(xy_double[2 * i]);
        quad_points[2 * i + 1] = static_cast<float>(xy_double[2 * i + 1]);
        quad_weights[i] = static_cast<float>(w_double[i]);
    }

    // 2. --- CALCULATE STANDARD (UN-REGULARIZED) QUADRATURE ---
    cvec3_t L_standard[3];
    cvec3_t K_standard[3];
    calculateScatterFieldQuadrature(
        obs_point, source_tri_v, source_tri_indices,
        quad_points.data(), quad_weights.data(), order_num,
        k_medium, false, L_standard, K_standard
    );

    // 3. --- CALCULATE REGULARIZED QUADRATURE + ANALYTIC SINGULARITY ---
    cvec3_t L_reg[3];
    cvec3_t K_reg[3];
    calculateScatterFieldQuadrature(
        obs_point, source_tri_v, source_tri_indices,
        quad_points.data(), quad_weights.data(), order_num,
        k_medium, true, L_reg, K_reg
    );

    cvec3_t L_sing[3];
    cvec3_t K_sing[3];
    calculateScatterFieldSingular(
        obs_point, source_tri_v, source_tri_indices, 
        k_medium, L_sing, K_sing
    );

    // 4. --- VERIFY MATCH ---
    for (int i = 0; i < 3; ++i) {
        INFO("Checking edge " << i);
        
        // Sum regular + singular
        cvec3_t L_extracted = L_reg[i] + L_sing[i];
        cvec3_t K_extracted = K_reg[i] + K_sing[i];

        // We expect standard numerical quadrature to be reasonably close to the analytical extraction
        // (for a distance of 0.05, rule 13 should integrate the 1/R standard kernel ok-ish,
        // but we'll use a slightly looser tolerance for the purely numerical standard quadrature)
        const float tol = 1e-3f;

        CHECK(L_standard[i].x.x == Approx(L_extracted.x.x).margin(tol));
        CHECK(L_standard[i].y.x == Approx(L_extracted.y.x).margin(tol));
        CHECK(L_standard[i].z.x == Approx(L_extracted.z.x).margin(tol));

        CHECK(L_standard[i].x.y == Approx(L_extracted.x.y).margin(tol));
        CHECK(L_standard[i].y.y == Approx(L_extracted.y.y).margin(tol));
        CHECK(L_standard[i].z.y == Approx(L_extracted.z.y).margin(tol));

        CHECK(K_standard[i].x.x == Approx(K_extracted.x.x).margin(tol));
        CHECK(K_standard[i].y.x == Approx(K_extracted.y.x).margin(tol));
        CHECK(K_standard[i].z.x == Approx(K_extracted.z.x).margin(tol));

        CHECK(K_standard[i].x.y == Approx(K_extracted.x.y).margin(tol));
        CHECK(K_standard[i].y.y == Approx(K_extracted.y.y).margin(tol));
        CHECK(K_standard[i].z.y == Approx(K_extracted.z.y).margin(tol));
    }
}
