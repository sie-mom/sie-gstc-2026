#include "catch.hpp"

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/rwg_basis_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/green_functions_utils.cuh"
#include "sie_mom/excitation/source_functors.cuh"

#include <Eigen/Dense>
#include <cmath>


TEST_CASE("SIE on spherical surface mesh", "[MatrixAssembly][SphericalSurface][!mayfail][PMCHWT][.]")
{
    using sie_mom::mesh::SphericalSurface;
    using sie_mom::mesh::MeshData;

    // --- Physical Parameters ---
    const real_t c0 = real_t(299792458.0);
    const real_t mu0 = real_t(4.0) * real_t(M_PI) * real_t(1e-7);
    const real_t eps0 = real_t(1.0) / (mu0 * c0 * c0);

    const real_t freq = real_t(1e9); // 1 GHz
    const real_t omega = real_t(2.0) * real_t(M_PI) * freq;
    const real_t lambda0 = c0 / freq;
    const real_t radius = real_t(2.0) * lambda0; // Sphere radius can be set based on wavelength

    const real_t eps_r1 = real_t(2.2); // Dielectric inside
    const real_t mu_r1 = real_t(1.0);
    const real_t eps_r2 = real_t(1.0); // Air outside
    const real_t mu_r2 = real_t(1.0);

    const real_t k1 = omega * std::sqrt(eps_r1 * eps0 * mu_r1 * mu0);
    const real_t k2 = omega * std::sqrt(eps_r2 * eps0 * mu_r2 * mu0);
    const real_t eta1 = std::sqrt((mu_r1 * mu0) / (eps_r1 * eps0));
    const real_t eta2 = std::sqrt((mu_r2 * mu0) / (eps_r2 * eps0));


    // Create a spherical surface mesh with radius 1.0 and refinement level
    SphericalSurface sphere(/*radius=*/radius, /*refinement_level=*/2);
    const MeshData& mesh = sphere.getMeshData();

    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;
    const auto& edges = mesh.edges;
    const auto& faces_edges = mesh.faces_edges;

    // Get dunavant quadrature points and weights 
    // const int rule = 13; // with rule 13 give very good accuracy, however is computationally more expensive
    const int rule = 7; // with rule 7 give good accuracy and is computationally less expensive
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
    const size_t matrix_dim = 2 * num_edges;
    // Define empty global Z matrix to hold results in row-major contiguous storage
    std::vector<complex__float> global_Z_matrix(matrix_dim * matrix_dim, make_complex(0.0f, 0.0f));
    // Define empty global V vector
    std::vector<complex__float> global_V_vector(matrix_dim, make_complex(0.0f, 0.0f));

    auto matrix_index = [matrix_dim](size_t row, size_t col) {
        return row * matrix_dim + col;
    };

    
    // Began assembly process loop through faces pairs
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
            // we need to check the coindence of triangles, in that case TK=0
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
                        
                        // --- Combine local matrices into global Z blocks ---
                        // Z_JJ = j*omega*(mu1*TL1 + mu2*TL2)
                        complex__float z_jj = complexAdd(complexScale(result_matrix_tl_k1[j][i], mu_r1 * mu0),
                                                         complexScale(result_matrix_tl_k2[j][i], mu_r2 * mu0));
                        z_jj = complexMul(z_jj, make_complex(0.f, omega));

                        // Z_JM = (TK1 + TK2)
                        complex__float z_jm = complexAdd(result_matrix_tk_k1[j][i], result_matrix_tk_k2[j][i]);

                        // Z_MJ = -(TK1 + TK2)
                        complex__float z_mj = complexScale(z_jm, -1.f);

                        // Z_MM = j*omega*(eps1*TL1 + eps2*TL2)
                        complex__float z_mm = complexAdd(complexScale(result_matrix_tl_k1[j][i], eps_r1 * eps0),
                                                         complexScale(result_matrix_tl_k2[j][i], eps_r2 * eps0));
                        z_mm = complexMul(z_mm, make_complex(0.f, omega));

                        // --- Add to global matrix ---
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx, source_edge_idx)], z_jj);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx, source_edge_idx + num_edges)], z_jm);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx + num_edges, source_edge_idx)], z_mj);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx + num_edges, source_edge_idx + num_edges)], z_mm);
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

                        // --- Combine local matrices into global Z blocks ---
                        // Z_JJ = j*omega*(mu1*TL1 + mu2*TL2)
                        complex__float z_jj = complexAdd(complexScale(result_matrix_tl_k1[j][i], mu_r1 * mu0),
                                                         complexScale(result_matrix_tl_k2[j][i], mu_r2 * mu0));
                        z_jj = complexMul(z_jj, make_complex(0.f, omega));

                        // Z_JM = (TK1 + TK2)
                        complex__float z_jm = complexAdd(result_matrix_tk_k1[j][i], result_matrix_tk_k2[j][i]);

                        // Z_MJ = -(TK1 + TK2)
                        complex__float z_mj = complexScale(z_jm, -1.f);

                        // Z_MM = j*omega*(eps1*TL1 + eps2*TL2)
                        complex__float z_mm = complexAdd(complexScale(result_matrix_tl_k1[j][i], eps_r1 * eps0),
                                                         complexScale(result_matrix_tl_k2[j][i], eps_r2 * eps0));
                        z_mm = complexMul(z_mm, make_complex(0.f, omega));

                        // --- Add to global matrix ---
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx, source_edge_idx)], z_jj);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx, source_edge_idx + num_edges)], z_jm);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx + num_edges, source_edge_idx)], z_mj);
                        complexAddInPlace(global_Z_matrix[matrix_index(test_edge_idx + num_edges, source_edge_idx + num_edges)], z_mm);
                    }
                }
            }

        }
    }

    // --- Assemble Excitation Vector V ---
    // Define the incident plane wave
    sie_mom::cuda::PlaneWave plane_wave;
    plane_wave.E0 = make_vec3(1.f, 0.f, 0.f); // E-field polarized along x-axis
    plane_wave.propagation_k = make_vec3(0.f, 0.f, 1.f); // Propagating along z-axis
    plane_wave.k = k2; // Wavenumber in region 2 (outside)
    plane_wave.eta = eta2; // Impedance in region 2

    // Loop over faces to compute local contributions to V
    for (size_t m = 0; m < faces.size(); ++m) {
        const auto& face_m = faces[m];
        const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
        vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

        complex__float V_E_local[3];
        complex__float V_H_local[3];

        calculateExcitationQuadrature(tri_v, tri_indices,
                                      quad_points.data(), quad_weights.data(),
                                      order_num, plane_wave,
                                      V_E_local, V_H_local);

        const auto& face_edges = faces_edges[m];
        const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

        for (int j = 0; j < 3; ++j) {
            const int global_edge_idx = local_edge_indices[j];

            // The final excitation vector V is composed of [V_E, V_H]
            // V_E = integral(f_m . E_inc_tan)
            // V_H = integral(f_m . H_inc_tan)
            // Note: The tangential projection is implicitly handled by the dot product
            // with the tangential basis function f_m.

            complexAddInPlace(global_V_vector[global_edge_idx], V_E_local[j]);
            complexAddInPlace(global_V_vector[global_edge_idx + num_edges], V_H_local[j]);
        }
    }

    SECTION("Check left upper block matrix symmetry")
    {
        for (size_t i = 0; i < num_edges; ++i)
        {
            for (size_t j = i + 1; j < num_edges; ++j)
            {
                const complex__float& val_ij = global_Z_matrix[matrix_index(i, j)];
                const complex__float& val_ji = global_Z_matrix[matrix_index(j, i)];
                INFO("Checking Z[" << i << "][" << j << "] == Z[" << j << "][" << i << "]");
                CHECK(val_ij.x == Approx(val_ji.x).margin(1e-3));
                CHECK(val_ij.y == Approx(val_ji.y).margin(1e-3));
            }
        }
    }


    SECTION("Solve the linear system Z*I = V for plane wave excitation")
    {
        using scalar_real = real_t;
        using scalar_complex = std::complex<scalar_real>;

        // Build Eigen copies explicitly to avoid relying on binary layout equivalence.
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, Eigen::Dynamic> Z_eigen(matrix_dim, matrix_dim);
        for (size_t r = 0; r < matrix_dim; ++r) {
            for (size_t c = 0; c < matrix_dim; ++c) {
                const complex__float& z = global_Z_matrix[matrix_index(r, c)];
                Z_eigen(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = scalar_complex(z.x, z.y);
            }
        }

        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> V_eigen(matrix_dim);
        for (size_t i = 0; i < matrix_dim; ++i) {
            const complex__float& v = global_V_vector[i];
            V_eigen(static_cast<Eigen::Index>(i)) = scalar_complex(v.x, v.y);
        }

        // Solve the system Z * I = V using QR decomposition
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> I_eigen = Z_eigen.colPivHouseholderQr().solve(V_eigen);

        // Verification: Check if Z * I is close to V
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> V_check = Z_eigen * I_eigen;

        REQUIRE(V_check.size() == matrix_dim);
        for (int i = 0; i < matrix_dim; ++i) {
            INFO("Checking V_check(" << i << ")");
            CHECK(V_check(i).real() == Approx(V_eigen(i).real()).margin(1e-5));
            CHECK(V_check(i).imag() == Approx(V_eigen(i).imag()).margin(1e-5));
        }
    }

    SECTION("Testing extinction theorem - scattered field from interior/exterior currents")
    {
        using scalar_real = real_t;
        using scalar_complex = std::complex<scalar_real>;

        // Rebuild Eigen copies explicitly to avoid relying on binary layout equivalence.
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, Eigen::Dynamic> Z_eigen(matrix_dim, matrix_dim);
        for (size_t r = 0; r < matrix_dim; ++r) {
            for (size_t c = 0; c < matrix_dim; ++c) {
                const complex__float& z = global_Z_matrix[matrix_index(r, c)];
                Z_eigen(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = scalar_complex(z.x, z.y);
            }
        }

        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> V_eigen(matrix_dim);
        for (size_t i = 0; i < matrix_dim; ++i) {
            const complex__float& v = global_V_vector[i];
            V_eigen(static_cast<Eigen::Index>(i)) = scalar_complex(v.x, v.y);
        }
        Eigen::Matrix<scalar_complex, Eigen::Dynamic, 1> I_eigen = Z_eigen.colPivHouseholderQr().solve(V_eigen);

        // I_eigen contains: [J2 (indices 0 to N-1), M2 (indices N to 2N-1)]
        // Interior currents: J1 = -J2, M1 = -M2

        // Helper lambda to compute E_scattered at an observation point
        auto compute_E_scattered = [&](const vec3_t& obs_point, 
                                       const std::vector<scalar_complex>& alpha_J,
                                       const std::vector<scalar_complex>& alpha_M,
                                       scalar_real k_medium) -> scalar_complex {
            // Initialize result (we'll just return x-component for simplicity)
            cvec3_t E_scat_total = make_cvec3(
                make_complex(0.f, 0.f), 
                make_complex(0.f, 0.f), 
                make_complex(0.f, 0.f));

            // Loop over all faces
            for (size_t n = 0; n < faces.size(); ++n) {
                const auto& face_n = faces[n];
                const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};
                vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

                // Compute L and K for the 3 half-basis functions on this triangle
                cvec3_t L_results[3];
                cvec3_t K_results[3];
                
                // Distance branch logic for singularity extraction
                const vec3_t source_centroid = make_vec3(
                    (source_tri_v[0].x + source_tri_v[1].x + source_tri_v[2].x) / 3.f,
                    (source_tri_v[0].y + source_tri_v[1].y + source_tri_v[2].y) / 3.f,
                    (source_tri_v[0].z + source_tri_v[1].z + source_tri_v[2].z) / 3.f
                );
                const float dist = length(obs_point - source_centroid);
                const float lambda_w = 2.0f * static_cast<float>(M_PI) / k_medium;
                const float singularity_threshold = 0.1f * lambda_w;

                if (dist < singularity_threshold) {
                    calculateScatterFieldQuadrature(
                        obs_point,
                        source_tri_v,
                        source_tri_indices,
                        quad_points.data(),
                        quad_weights.data(),
                        order_num,
                        k_medium,
                        true, // is_regular
                        L_results,
                        K_results);
                        
                    cvec3_t L_singular[3];
                    cvec3_t K_singular[3];
                    calculateScatterFieldSingular(
                        obs_point, source_tri_v, source_tri_indices, k_medium,
                        L_singular, K_singular);
                        
                    for (int i = 0; i < 3; ++i) {
                        L_results[i] = L_results[i] + L_singular[i];
                        K_results[i] = K_results[i] + K_singular[i];
                    }
                } else {
                    calculateScatterFieldQuadrature(
                        obs_point,
                        source_tri_v,
                        source_tri_indices,
                        quad_points.data(),
                        quad_weights.data(),
                        order_num,
                        k_medium,
                        false, // is_regular
                        L_results,
                        K_results);
                }

                // Get the face's edge indices
                const auto& face_edges = faces_edges[n];
                const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

                // Accumulate contributions from each edge on this face
                for (int i = 0; i < 3; ++i) {
                    int global_edge_idx = local_edge_indices[i];
                    
                    // Get current coefficients for this edge
                    scalar_complex I_J = alpha_J[global_edge_idx];
                    scalar_complex I_M = alpha_M[global_edge_idx];
                    
                    complex__float I_J_native = make_complex(I_J.real(), I_J.imag());
                    complex__float I_M_native = make_complex(I_M.real(), I_M.imag());

                    // E_scat = -jωμ₂ * L(J) - K(M)
                    // Contribution from J current: -jωμ₂ * I_J * L
                    complex__float jw_mu2 = make_complex(0.f, -omega * mu_r2 * mu0);
                    complex__float coeff_J = complexMul(jw_mu2, I_J_native);
                    cvec3_t E_from_J = coeff_J * L_results[i];
                    
                    // Contribution from M current: -I_M * K
                    complex__float neg_I_M = complexScale(I_M_native, -1.f);
                    cvec3_t E_from_M = neg_I_M * K_results[i];
                    
                    E_scat_total = E_scat_total + E_from_J + E_from_M;
                }
            }
            
            // Return x-component for checking (convert to std::complex)
            return scalar_complex(E_scat_total.x.x, E_scat_total.x.y);
        };

        // --- Define observation points ---
        const real_t r_exterior = real_t(1.5) * radius; // Outside the sphere
        const real_t r_interior = real_t(0.5) * radius; // Inside the sphere
        
        std::vector<vec3_t> obs_points_exterior = {
            make_vec3(r_exterior, 0.f, 0.f),  // +x axis
            make_vec3(0.f, r_exterior, 0.f),  // +y axis
            make_vec3(0.f, 0.f, r_exterior)   // +z axis
        };
        
        std::vector<vec3_t> obs_points_interior = {
            make_vec3(r_interior, 0.f, 0.f),  // +x axis
            make_vec3(0.f, r_interior, 0.f),  // +y axis
            make_vec3(0.f, 0.f, r_interior)   // +z axis
        };

        // Extract J2 and M2 from the solution
        std::vector<scalar_complex> J2(num_edges);
        std::vector<scalar_complex> M2(num_edges);
        for (size_t i = 0; i < num_edges; ++i) {
            J2[i] = I_eigen(i);
            M2[i] = I_eigen(i + num_edges);
        }

        // --- Test A: Interior currents (J1=-J2, M1=-M2) should produce zero field outside ---
        INFO("Test A: Interior currents should produce zero field outside the sphere");
        std::vector<scalar_complex> J1(num_edges);
        std::vector<scalar_complex> M1(num_edges);
        for (size_t i = 0; i < num_edges; ++i) {
            J1[i] = -J2[i];
            M1[i] = -M2[i];
        }

        for (size_t pt = 0; pt < obs_points_exterior.size(); ++pt) {
            scalar_complex E_scat_x = compute_E_scattered(obs_points_exterior[pt], J1, M1, k1);
            INFO("Exterior point " << pt << " (using interior currents): E_scat.x = " 
                 << E_scat_x.real() << " + j*" << E_scat_x.imag());
            CHECK(std::abs(E_scat_x.real()) < 1e-2);
            CHECK(std::abs(E_scat_x.imag()) < 1e-2);
        }

        // --- Test B: Exterior currents (J2, M2) should produce field that cancels incident inside ---
        INFO("Test B: Exterior currents should produce total field = 0 inside the sphere");
        for (size_t pt = 0; pt < obs_points_interior.size(); ++pt) {
            scalar_complex E_scat_x = compute_E_scattered(obs_points_interior[pt], J2, M2, k2);
            
            // Compute incident field at this point
            cvec3_t E_inc, H_inc;
            plane_wave(obs_points_interior[pt], E_inc, H_inc);
            
            // Total field = scattered + incident
            scalar_complex E_total_x(E_scat_x.real() + E_inc.x.x,
                                     E_scat_x.imag() + E_inc.x.y);
            
            INFO("Interior point " << pt << " (using exterior currents): E_total.x = " 
                 << E_total_x.real() << " + j*" << E_total_x.imag());
            CHECK(std::abs(E_total_x.real()) < 1e-2);
            CHECK(std::abs(E_total_x.imag()) < 1e-2);
        }
    }
}