#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/cuda/rwg_basis_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/quadrature/gauss_legendre_rule.hpp"

TEST_CASE("Form A constant-potential line integral vanishes for sphere shared-edge faces",
          "[Quadrature][nTL][FormA][LineGeometry]")
{
    // Faces 6 (test) and 50 (source) of sphere_r2lambda_lev4_5120f.obj.
    // The source is retained for the exact matrix prefactors. Replacing phi_0
    // by 1 removes its geometry from the line integral itself.
    const int test_indices[3] = {647, 646, 163};
    const vec3_t test_tri[3] = {
        make_vec3(-0.35669027, 0.47377408, 0.08838692),
        make_vec3(-0.38048416, 0.44956014, 0.11238131),
        make_vec3(-0.34251388, 0.47526052, 0.12772530)
    };
    const int source_indices[3] = {646, 669, 163};
    const vec3_t source_tri[3] = {
        make_vec3(-0.38048416, 0.44956014, 0.11238131),
        make_vec3(-0.36630195, 0.44934997, 0.15300183),
        make_vec3(-0.34251388, 0.47526052, 0.12772530)
    };

    constexpr int line_points = 20; // PEC Form A app default.
    std::vector<double> nodes_double(line_points), weights_double(line_points);
    gauss_legendre_rule(line_points, nodes_double.data(), weights_double.data());
    const std::vector<float> nodes(nodes_double.begin(), nodes_double.end());
    const std::vector<float> weights(weights_double.begin(), weights_double.end());

    const vec3_t test_cross = cross(test_tri[1] - test_tri[0],
                                    test_tri[2] - test_tri[0]);
    const real_t A_m = real_t(0.5) * length(test_cross);
    const real_t A_n = getTriangleArea(source_tri);
    const vec3_t n_m = test_cross / length(test_cross);
    const real_t k2 = real_t(2) * real_t(M_PI) * real_t(1e9) /
                      real_t(299792458);

    vec3_t test_free[3], source_free[3];
    real_t test_edge_lengths[3], source_edge_lengths[3];
    for (int j = 0; j < 3; ++j) {
        test_edge_lengths[j] = getEdgeInfo(test_indices, test_tri, j, test_free[j]);
        source_edge_lengths[j] = getEdgeInfo(source_indices, source_tri, j,
                                              source_free[j]);
    }

    // Identical to the production singular line loop, except phi_0(r_eq) = 1.
    std::array<real_t, 3> line_sum{};
    std::array<real_t, 3> absolute_contributions{};
    for (int edge = 0; edge < 3; ++edge) {
        const vec3_t vA = test_tri[edge];
        const vec3_t vB = test_tri[(edge + 1) % 3];
        const vec3_t edge_vec = vB - vA;
        const real_t L_e = length(edge_vec);
        const vec3_t nu_e = cross(edge_vec, n_m) / L_e;
        for (int q = 0; q < line_points; ++q) {
            vec3_t r_eq;
            mapEdgeQuadPoint(vA, vB, static_cast<real_t>(nodes[q]), r_eq);
            const real_t w_q = static_cast<real_t>(weights[q]);
            for (int j = 0; j < 3; ++j) {
                const vec3_t tau_mj = cross(rhoRwg(r_eq, test_free[j]), n_m);
                const real_t contribution = L_e * w_q * dot(tau_mj, nu_e);
                line_sum[j] += contribution;
                absolute_contributions[j] += std::abs(contribution);
            }
        }
    }

    // The solver uploads Gauss nodes and weights as float even in a double build.
    const real_t tolerance = std::max(
        real_t(1e-8), real_t(100) * std::numeric_limits<real_t>::epsilon());
    for (int j = 0; j < 3; ++j) {
        INFO("line_sum[" << j << "]=" << line_sum[j]
             << ", sum of absolute terms=" << absolute_contributions[j]);
        REQUIRE(absolute_contributions[j] > real_t(0));
        CHECK(std::abs(line_sum[j]) <= tolerance * absolute_contributions[j]);
    }

    // Same signed-edge and area prefactor as the production singular routine.
    const real_t line_scale = real_t(1) /
        (real_t(8) * real_t(M_PI) * A_m * A_n * k2 * k2);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const real_t matrix_entry = test_edge_lengths[j] *
                                        source_edge_lengths[i] *
                                        line_scale * line_sum[j];
            const real_t matrix_scale = std::abs(test_edge_lengths[j] *
                                                 source_edge_lengths[i] *
                                                 line_scale) * absolute_contributions[j];
            INFO("line_matrix[" << j << "][" << i << "]=" << matrix_entry);
            CHECK(std::abs(matrix_entry) <= tolerance * matrix_scale);
        }
    }
}

TEST_CASE("Form A phi_0 at Gauss point 10 on three shared-edge test-face edges",
          "[Quadrature][nTL][FormA][Phi0GaussPoint]")
{
    // Same real mesh pair as the line-geometry test: face 6 tests face 50.
    // Test vertex indices: (647, 646, 163); source: (646, 669, 163).
    const vec3_t test_tri[3] = {
        make_vec3(-0.35669027, 0.47377408, 0.08838692),
        make_vec3(-0.38048416, 0.44956014, 0.11238131),
        make_vec3(-0.34251388, 0.47526052, 0.12772530)
    };
    const vec3_t source_tri[3] = {
        make_vec3(-0.38048416, 0.44956014, 0.11238131),
        make_vec3(-0.36630195, 0.44934997, 0.15300183),
        make_vec3(-0.34251388, 0.47526052, 0.12772530)
    };

    constexpr int line_points = 20;
    constexpr int q = 10; // Zero-based: the 11th Gauss point, just above 1/2.
    std::vector<double> nodes_double(line_points), weights_double(line_points);
    gauss_legendre_rule(line_points, nodes_double.data(), weights_double.data());
    const real_t t_q = static_cast<real_t>(static_cast<float>(nodes_double[q]));

    for (int edge = 0; edge < 3; ++edge) {
        vec3_t r_eq;
        mapEdgeQuadPoint(test_tri[edge], test_tri[(edge + 1) % 3], t_q, r_eq);
        const double phi_0 = calculate_phi_0(r_eq, source_tri);
        CHECK(std::isfinite(phi_0));
        CHECK(phi_0 > 0.0);
    }
}
