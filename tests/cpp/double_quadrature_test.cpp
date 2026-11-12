#include <iostream>
#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/green_functions_utils.cuh"
#include "sie_mom/cuda/rwg_basis_utils.cuh"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/quadrature/gauss_legendre_rule.hpp"

namespace
{
template <typename ModifierKernel, typename OperatorKernel>
void referenceDoubleQuadrature(const vec3_t test_tri[3],
                               const int test_tri_indices[3],
                               const vec3_t source_tri[3],
                               const int source_tri_indices[3],
                               const float* quad_points,
                               const float* quad_weights,
                               int num_quad_points,
                               real_t k,
                               const ModifierKernel& modifier,
                               const OperatorKernel& op,
                               complex__float out_matrix[3][3])
{
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            out_matrix[row][col] = make_complex(0.f, 0.f);
        }
    }

    for (int p = 0; p < num_quad_points; ++p)
    {
        const real_t xi_p = static_cast<real_t>(quad_points[2 * p + 0]);
        const real_t eta_p = static_cast<real_t>(quad_points[2 * p + 1]);
        const real_t w_p = static_cast<real_t>(quad_weights[p]);

        vec3_t rm_p;
        mapTriQuadPoint(test_tri, xi_p, eta_p, rm_p);

        for (int q = 0; q < num_quad_points; ++q)
        {
            const real_t xi_q = static_cast<real_t>(quad_points[2 * q + 0]);
            const real_t eta_q = static_cast<real_t>(quad_points[2 * q + 1]);
            const real_t w_q = static_cast<real_t>(quad_weights[q]);

            vec3_t rn_q;
            mapTriQuadPoint(source_tri, xi_q, eta_q, rn_q);

            const auto kernel_value = op(rm_p, rn_q, k);
            const real_t combined_weight = w_p * w_q;

            for (int j = 0; j < 3; ++j)
            {
                vec3_t rm_free;
                const real_t emj = getEdgeInfo(test_tri_indices, test_tri, j, rm_free);

                for (int i = 0; i < 3; ++i)
                {
                    vec3_t rn_free;
                    const real_t eni = getEdgeInfo(source_tri_indices, source_tri, i, rn_free);

                    const auto modifier_value = modifier(rm_p, rm_free, rn_q, rn_free, k);

                    const complex__float integrand = evaluateModifierOperator(modifier_value, kernel_value);
                    const complex__float weighted_integrand = complexScale(integrand, emj*eni*combined_weight);
                    complexAddInPlace(out_matrix[j][i], weighted_integrand);
                }
            }
        }
    }
}
} // namespace

TEST_CASE("calculateDoubleQuadrature matches reference CPU implementation", "[Quadrature][Double]")
{
    // Define simple non-degenerate testing and source triangles.
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(1.f, 0.f, 0.f),
        make_vec3(0.f, 1.f, 0.f)
    };
    const int test_tri_indices[3] = {0, 1, 2};

    const vec3_t source_tri[3] = {
        make_vec3(0.2f, 0.1f, 0.3f),
        make_vec3(0.9f, -0.2f, 0.4f),
        make_vec3(-0.3f, 0.6f, 0.5f)
    };
    const int source_tri_indices[3] = {3, 4, 1};

    const int rule = 3;
    const int order_num = dunavant_order_num(rule);

    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());

    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int idx = 0; idx < order_num; ++idx)
    {
        quad_points[2 * idx + 0] = static_cast<float>(xy_double[2 * idx + 0]);
        quad_points[2 * idx + 1] = static_cast<float>(xy_double[2 * idx + 1]);
        quad_weights[idx] = static_cast<float>(w_double[idx]);
    }

    constexpr real_t k = real_t(2.0);
    TLModifier modifier;
    TLDpq kernel;

    complex__float expected[3][3];
    referenceDoubleQuadrature(test_tri,
                              test_tri_indices,
                              source_tri,
                              source_tri_indices,
                              quad_points.data(),
                              quad_weights.data(),
                              order_num,
                              k,
                              modifier,
                              kernel,
                              expected);

    complex__float actual[3][3];
    calculateDoubleQuadrature(test_tri,
                              test_tri_indices,
                              source_tri,
                              source_tri_indices,
                              quad_points.data(),
                              quad_weights.data(),
                              order_num,
                              k,
                              modifier,
                              kernel,
                              actual);

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            INFO("row=" << row << ", col=" << col);
            CHECK(actual[row][col].x == Approx(expected[row][col].x).margin(1e-5f));
            CHECK(actual[row][col].y == Approx(expected[row][col].y).margin(1e-5f));
        }
    }
}

TEST_CASE("calculateDoubleQuadrature TK operator matches reference CPU implementation", "[Quadrature][Double][TK]")
{
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(1.f, 0.f, 0.f),
        make_vec3(0.f, 1.f, 0.f)
    };
    const int test_tri_indices[3] = {0, 1, 2};

    const vec3_t source_tri[3] = {
        make_vec3(-0.4f, 0.5f, 0.2f),
        make_vec3(0.6f, -0.3f, 0.7f),
        make_vec3(0.1f, 0.9f, -0.4f)
    };
    const int source_tri_indices[3] = {5, 2, 7};

    const int rule = 4;
    const int order_num = dunavant_order_num(rule);

    std::vector<double> xy_double(2 * order_num);
    std::vector<double> w_double(order_num);
    dunavant_rule(rule, order_num, xy_double.data(), w_double.data());

    std::vector<float> quad_points(2 * order_num);
    std::vector<float> quad_weights(order_num);
    for (int idx = 0; idx < order_num; ++idx)
    {
        quad_points[2 * idx + 0] = static_cast<float>(xy_double[2 * idx + 0]);
        quad_points[2 * idx + 1] = static_cast<float>(xy_double[2 * idx + 1]);
        quad_weights[idx] = static_cast<float>(w_double[idx]);
    }

    constexpr real_t k = real_t(1.2);
    TKModifier modifier;
    TKDpq kernel;

    complex__float expected[3][3];
    referenceDoubleQuadrature(test_tri,
                              test_tri_indices,
                              source_tri,
                              source_tri_indices,
                              quad_points.data(),
                              quad_weights.data(),
                              order_num,
                              k,
                              modifier,
                              kernel,
                              expected);

    complex__float actual[3][3];
    calculateDoubleQuadrature(test_tri,
                              test_tri_indices,
                              source_tri,
                              source_tri_indices,
                              quad_points.data(),
                              quad_weights.data(),
                              order_num,
                              k,
                              modifier,
                              kernel,
                              actual);

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            INFO("row=" << row << ", col=" << col);
            CHECK(actual[row][col].x == Approx(expected[row][col].x).margin(1e-5f));
            CHECK(actual[row][col].y == Approx(expected[row][col].y).margin(1e-5f));
        }
    }
}

namespace
{
void compareFormA_DirectVsExtracted(const char* case_name,
                                    const vec3_t test_tri[3],
                                    const int test_indices[3],
                                    const vec3_t source_tri[3],
                                    const int source_indices[3])
{
    const int triangle_rule = 20;
    const int triangle_points = dunavant_order_num(triangle_rule);
    const int line_points = 20;
    std::vector<double> triangle_nodes_double(2 * triangle_points);
    std::vector<double> triangle_weights_double(triangle_points);
    dunavant_rule(triangle_rule, triangle_points,
                  triangle_nodes_double.data(), triangle_weights_double.data());
    std::vector<float> triangle_nodes(triangle_nodes_double.begin(), triangle_nodes_double.end());
    std::vector<float> triangle_weights(triangle_weights_double.begin(), triangle_weights_double.end());

    std::vector<double> line_nodes_double(line_points);
    std::vector<double> line_weights_double(line_points);
    gauss_legendre_rule(line_points, line_nodes_double.data(), line_weights_double.data());
    std::vector<float> line_nodes(line_nodes_double.begin(), line_nodes_double.end());
    std::vector<float> line_weights(line_weights_double.begin(), line_weights_double.end());

    constexpr real_t k1 = real_t(1.5);
    constexpr real_t k2 = real_t(2.5);
    
    // Direct Numerical Evaluation
    complex__float direct_ntl_k1[3][3], direct_ntl_k2[3][3];
    complex__float direct_ntk_k1[3][3], direct_ntk_k2[3][3];
    calculateFormAQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, false,
        direct_ntl_k1, direct_ntl_k2, direct_ntk_k1, direct_ntk_k2);

    // Singularity Extracted Evaluation
    complex__float reg_ntl_k1[3][3], reg_ntl_k2[3][3];
    complex__float reg_ntk_k1[3][3], reg_ntk_k2[3][3];
    calculateFormAQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, true,
        reg_ntl_k1, reg_ntl_k2, reg_ntk_k1, reg_ntk_k2);

    complex__float sing_ntl_k1[3][3], sing_ntl_k2[3][3];
    complex__float sing_ntk_k1[3][3], sing_ntk_k2[3][3];
    calculateFormASingularPartQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        triangle_nodes.data(), triangle_weights.data(), triangle_points,
        line_nodes.data(), line_weights.data(), line_points,
        k1, k2, sing_ntl_k1, sing_ntl_k2, sing_ntk_k1, sing_ntk_k2);

    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            const complex__float direct_values[2][2] = {
                {direct_ntl_k1[row][col], direct_ntl_k2[row][col]},
                {direct_ntk_k1[row][col], direct_ntk_k2[row][col]}
            };
            const complex__float extracted_values[2][2] = {
                {complexAdd(reg_ntl_k1[row][col], sing_ntl_k1[row][col]), complexAdd(reg_ntl_k2[row][col], sing_ntl_k2[row][col])},
                {complexAdd(reg_ntk_k1[row][col], sing_ntk_k1[row][col]), complexAdd(reg_ntk_k2[row][col], sing_ntk_k2[row][col])}
            };
            for (int op = 0; op < 2; ++op)
            {
                for (int medium = 0; medium < 2; ++medium)
                {
                    INFO("case=" << case_name << ", operator=" << (op == 0 ? "nTL" : "nTK")
                         << ", row=" << row << ", col=" << col << ", medium=" << medium);
                    // Use Catch2's Approx for simple, readable comparisons that print values on failure
                    CHECK(direct_values[op][medium].x == Approx(extracted_values[op][medium].x).margin(1e-6).epsilon(1e-5));
                    CHECK(direct_values[op][medium].y == Approx(extracted_values[op][medium].y).margin(1e-6).epsilon(1e-5));
                }
            }
        }
    }
}
} // namespace

TEST_CASE("Form A direct matches Form A singularity extracted for separated triangles",
          "[Quadrature][nTL][nTK][FormA][Separated]")
{
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(1.f, 0.f, 0.f),
        make_vec3(0.f, 1.f, 0.f)
    };
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {3, 4, 5};

    const vec3_t coplanar_source[3] = {
        make_vec3(2.2f, 0.15f, 0.f),
        make_vec3(3.1f, 0.15f, 0.f),
        make_vec3(2.2f, 1.05f, 0.f)
    };
    compareFormA_DirectVsExtracted("separated coplanar", test_tri, test_indices, coplanar_source, source_indices);

    const vec3_t noncoplanar_source[3] = {
        make_vec3(0.2f, -0.1f, 1.2f),
        make_vec3(1.1f, 0.2f, 1.4f),
        make_vec3(-0.1f, 0.9f, 1.6f)
    };
    compareFormA_DirectVsExtracted("separated noncoplanar", test_tri, test_indices, noncoplanar_source, source_indices);

    const vec3_t reversed_test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(0.f, 1.f, 0.f),
        make_vec3(1.f, 0.f, 0.f)
    };
    compareFormA_DirectVsExtracted("reversed test winding", reversed_test_tri, test_indices, noncoplanar_source, source_indices);
}

TEST_CASE("Form A direct matches Form A singularity extracted for nearby triangles",
          "[Quadrature][nTL][nTK][FormA][NearSeparated]")
{
    const real_t triangle_side = real_t(0.04);
    const vec3_t source_offset = make_vec3(0.004f, 0.003f, 0.015f);
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(triangle_side, 0.f, 0.f),
        make_vec3(0.f, triangle_side, 0.f)
    };
    const vec3_t source_tri[3] = {
        test_tri[0] + source_offset,
        test_tri[1] + source_offset,
        test_tri[2] + source_offset
    };
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {3, 4, 5};
    compareFormA_DirectVsExtracted("parallel non-coplanar planes", test_tri, test_indices, source_tri, source_indices);
}

TEST_CASE("Form A direct matches Form A singularity extracted for nearly-touching triangles",
          "[Quadrature][nTL][nTK][FormA][NearSharedEdge]")
{
    const real_t triangle_side = real_t(0.04);
    // Move slightly away to avoid actual singularity which direct quadrature cannot handle
    const real_t h = real_t(0.002); 
    const real_t angle = real_t(20) * std::acos(real_t(-1)) / real_t(180);
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f),
        make_vec3(triangle_side, 0.f, 0.f),
        make_vec3(0.f, triangle_side, 0.f)
    };
    const vec3_t source_shift = make_vec3(0.f, -h, 0.f);
    const vec3_t source_tri[3] = {
        test_tri[1] + source_shift,
        test_tri[0] + source_shift,
        make_vec3(0.f, -triangle_side * std::cos(angle),
                  triangle_side * std::sin(angle)) + source_shift
    };
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {3, 4, 5};
    compareFormA_DirectVsExtracted("adjustable edge gap", test_tri, test_indices, source_tri, source_indices);
}
