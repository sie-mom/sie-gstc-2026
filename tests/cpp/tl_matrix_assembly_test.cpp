#include "catch.hpp"

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/rwg_basis_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/green_functions_utils.cuh"

#include <Eigen/Dense>


TEST_CASE("Assembly TL operator on spherical surface mesh", "[TLMatrixAssembly][SphericalSurface]")
{
    using sie_mom::mesh::SphericalSurface;
    using sie_mom::mesh::MeshData;

    // Create a spherical surface mesh with radius 1.0 and refinement level 2
    SphericalSurface sphere(/*radius=*/1.0f, /*refinement_level=*/1);
    const MeshData& mesh = sphere.getMeshData();

    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;
    const auto& edges = mesh.edges;
    const auto& faces_edges = mesh.faces_edges;

    // Get dunavant quadrature points and weights 
    const int rule = 7;
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

    const size_t num_edges = edges.size();
    // Define empty global Z matrix to hold results in row-major contiguous storage
    std::vector<complex__float> global_Z_matrix(num_edges * num_edges, make_complex(0.0f, 0.0f));
    auto matrix_index = [num_edges](size_t row, size_t col) {
        return row * num_edges + col;
    };

    // For a frequency of 1 GHz and a medium with mu_r=1, epsilon_r=2.2,
    // the wavenumber k = (2*pi*f/c) * sqrt(mu_r*epsilon_r) is calculated.
    const real_t k1 = real_t(31.086);
    const real_t k2 = real_t(26.086);

    // Began TL assembly process loop through faces pairs
    for (size_t m = 0; m < faces.size(); ++m) {
        const auto& face_m = faces[m];
        // Get test triangle indices
        const int test_tri_indices[3] = {face_m.x, face_m.y, face_m.z};
        // Get test triangle vertices
        vec3_t test_tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

        for (size_t n = 0; n < faces.size(); ++n) {
            const auto& face_n = faces[n];

            // Get source triangle indices
            const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};
            // Get source triangle vertices
            vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

            const auto& test_face_edges = faces_edges[m];
            const auto& source_face_edges = faces_edges[n];
            // Check if triangles are close enough to consider singularity subtraction,
            if (shareVertex(test_tri_indices, source_tri_indices)) {
                // The two triangles are adjacent, use singularity extraction.
                complex__float result_matrix_tl_k1[3][3];
                complex__float result_matrix_tl_k2[3][3];
                complex__float result_matrix_tk_k1[3][3];
                complex__float result_matrix_tk_k2[3][3];

                // 1. Calculate the regular part into the main result matrices
                calculateDoubleQuadrature(test_tri_v, test_tri_indices,
                                          source_tri_v, source_tri_indices,
                                          quad_points.data(), quad_weights.data(),
                                          order_num, k1, k2, true, // is_singular=true
                                          result_matrix_tl_k1, result_matrix_tl_k2,
                                          result_matrix_tk_k1, result_matrix_tk_k2);

                // 2. Calculate the singular part into temporary matrices
                complex__float singular_tl_k1[3][3], singular_tl_k2[3][3];
                complex__float singular_tk_k1[3][3], singular_tk_k2[3][3];
                calculateSingularPartQuadrature(test_tri_v, test_tri_indices,
                                                source_tri_v, source_tri_indices,
                                                quad_points.data(), quad_weights.data(),
                                                order_num, k1, k2,
                                                singular_tl_k1, singular_tl_k2,
                                                singular_tk_k1, singular_tk_k2);
                
                // 3. Add the singular part to the regular part
                for (int j = 0; j < 3; ++j) {
                    for (int i = 0; i < 3; ++i) {
                        complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
                        complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
                        complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
                        complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
                    }
                }



                for (int j = 0; j < 3; ++j) {
                    int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
                    for (int i = 0; i < 3; ++i) {
                        int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;
                        
                        const size_t row = static_cast<size_t>(test_edge_idx);
                        const size_t col = static_cast<size_t>(source_edge_idx);
                        const size_t idx = matrix_index(row, col);

                        // For now, we are only assembling the TL matrix for k1
                        complexAddInPlace(global_Z_matrix[idx], result_matrix_tl_k1[j][i]);
                    }
                }
            } else {
                // Triangles are not adjacent, use standard double quadrature.
                complex__float result_matrix_tl_k1[3][3];
                complex__float result_matrix_tl_k2[3][3];
                complex__float result_matrix_tk_k1[3][3];
                complex__float result_matrix_tk_k2[3][3];

                calculateDoubleQuadrature(test_tri_v, test_tri_indices,
                                          source_tri_v, source_tri_indices,
                                          quad_points.data(), quad_weights.data(),
                                          order_num, k1, k2, false, // k1, k2, is_singular=false
                                          result_matrix_tl_k1, result_matrix_tl_k2,
                                          result_matrix_tk_k1, result_matrix_tk_k2);

                for (int j = 0; j < 3; ++j) {
                    int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
                    for (int i = 0; i < 3; ++i) {
                        int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;

                        const size_t row = static_cast<size_t>(test_edge_idx);
                        const size_t col = static_cast<size_t>(source_edge_idx);
                        const size_t idx = matrix_index(row, col);
                        
                        // For now, we are only assembling the TL matrix for k1
                        complexAddInPlace(global_Z_matrix[idx], result_matrix_tl_k1[j][i]);
                    }
                }
            }

        }
    }


    SECTION("Check for TL matrix symmetry")
    {
        for (size_t i = 0; i < num_edges; ++i)
        {
            for (size_t j = i + 1; j < num_edges; ++j)
            {
                const complex__float& val_ij = global_Z_matrix[matrix_index(i, j)];
                const complex__float& val_ji = global_Z_matrix[matrix_index(j, i)];
                INFO("Checking Z[" << i << "][" << j << "] == Z[" << j << "][" << i << "]");
                CHECK(val_ij.x == Approx(val_ji.x).margin(1e-5));
                CHECK(val_ij.y == Approx(val_ji.y).margin(1e-5));
            }
        }
    }

    SECTION("Check for TL matrix diagonal dominance")
    {
        auto abs_complex = [](const complex__float& c) {
            return std::sqrt(c.x * c.x + c.y * c.y);
        };

        for (size_t i = 0; i < num_edges; ++i)
        {
            const real_t diag_magnitude = abs_complex(global_Z_matrix[matrix_index(i, i)]);
            for (size_t j = 0; j < num_edges; ++j)
            {
                if (i == j) continue;

                const real_t off_diag_magnitude = abs_complex(global_Z_matrix[matrix_index(i, j)]);
                INFO("Checking |Z[" << i << "][" << i << "]| > |Z[" << i << "][" << j << "]|");
                CHECK(diag_magnitude > off_diag_magnitude);
            }
        }
    }

    SECTION("Solve the linear system Z*I = V with Eigen and V=[1,0,0,...]")
    {
        // Map the std::vector to an Eigen matrix without copying
        using scalar_real = real_t;
        using scalar_complex = std::complex<scalar_real>;
        Eigen::Map<Eigen::Matrix<scalar_complex, Eigen::Dynamic, Eigen::Dynamic>> Z_eigen(
            reinterpret_cast<scalar_complex*>(global_Z_matrix.data()),
            num_edges,
            num_edges);

        // Create a placeholder excitation vector V = [1, 0, 0, ...]
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> V_eigen =
            Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1>::Zero(num_edges);
        if (num_edges > 0) {
            V_eigen(0) = scalar_complex(scalar_real(1), scalar_real(0));
        }

        // Solve the system Z * I = V using LU decomposition
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> I_eigen = Z_eigen.fullPivLu().solve(V_eigen);

        // Verification: Check if Z * I is close to V
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> V_check = Z_eigen * I_eigen;

        REQUIRE(V_check.size() == num_edges);
        if (num_edges > 0) {
            // Check the first element
            CHECK(V_check(0).real() == Approx(1.0).margin(1e-5));
            CHECK(V_check(0).imag() == Approx(0.0).margin(1e-5));

            // Check a few other elements to ensure they are close to zero
            for (int i = 1; i < std::min((long)num_edges, 10L); ++i) {
                CHECK(V_check(i).real() == Approx(0.0).margin(1e-5));
                CHECK(V_check(i).imag() == Approx(0.0).margin(1e-5));
            }
        }
    }
}