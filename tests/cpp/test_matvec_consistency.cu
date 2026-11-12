#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

#include "sie_mom/cuda/platform.cuh"
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/sie_kernels_matvec.cuh"
#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

// Define complex__float alias for convenience if not already available
// platform.cuh defines it as cuComplex if __CUDACC__ is defined.

#define CHECK_LAST_CUDA(msg) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Error after %s: %s at %s:%d\n", msg, cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__host__ __device__ inline size_t columnMajorIndex(size_t row, size_t col, size_t leading_dim) {
    return col * leading_dim + row;
}

// Helper to copy data to GPU
template <typename T>
T* to_gpu(const std::vector<T>& h_data) {
    if(h_data.empty()) return nullptr;
    T* d_data = nullptr;
    size_t size_bytes = h_data.size() * sizeof(T);
    cudaMalloc(&d_data, size_bytes);
    cudaMemcpy(d_data, h_data.data(), size_bytes, cudaMemcpyHostToDevice);
    return d_data;
}

// --------------------------------------------------------
// Copy of Assemble Kernels from cloaking_example.cu
// --------------------------------------------------------

__global__ void assemble_Z_calderon_kernel_test(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    float k1, float k2,
    float mu_r1, float mu_r2,
    float eps_r1, float eps_r2,
    float mu0, float eps0,
    float omega,
    float singularity_threshold,
    complex__float* global_Z_matrix)
{
    size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces || n >= num_faces) {
        return;
    }

    // --- Fetch data for face m (test) and face n (source) ---
    const auto& face_m = faces[m];
    const int test_tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t test_tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    const auto& face_n = faces[n];
    const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};
    vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

    const auto& test_face_edges = faces_edges[m];
    const auto& source_face_edges = faces_edges[n];

    complex__float result_matrix_tl_k1[3][3], result_matrix_tl_k2[3][3];
    complex__float result_matrix_tk_k1[3][3], result_matrix_tk_k2[3][3];

    // Centroid distance check for singularity
    vec3_t test_centroid = (test_tri_v[0] + test_tri_v[1] + test_tri_v[2]) / 3.f;
    vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points, quad_weights, order_num, k1, k2, true,
                                    result_matrix_tl_k1, result_matrix_tl_k2,
                                    result_matrix_tk_k1, result_matrix_tk_k2);

        complex__float singular_tl_k1[3][3], singular_tl_k2[3][3];
        complex__float singular_tk_k1[3][3], singular_tk_k2[3][3];
        calculateSingularPartQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                        quad_points, quad_weights, order_num, k1, k2,
                                        singular_tl_k1, singular_tl_k2,
                                        singular_tk_k1, singular_tk_k2);
        
        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
            complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
            complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
            complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
        }
    } else {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points, quad_weights, order_num, k1, k2, false,
                                    result_matrix_tl_k1, result_matrix_tl_k2,
                                    result_matrix_tk_k1, result_matrix_tk_k2);
    }

    // --- Atomically add results to the global Z matrix ---
    for (int j = 0; j < 3; ++j) {
        int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
        for (int i = 0; i < 3; ++i) {
            int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;
            
            // --- GSTC 4x4 Matrix Assembly (Calderon Part A_C) ---
            const complex__float j_omega = make_complex(0.f, omega);
            const float mu1 = mu_r1 * mu0;
            const float mu2 = mu_r2 * mu0;
            const float eps1 = eps_r1 * eps0;
            const float eps2 = eps_r2 * eps0;

            // Precompute terms
            complex__float term_TL1_mu = complexMul(j_omega, complexScale(result_matrix_tl_k1[j][i], mu1));
            complex__float term_TL1_eps = complexMul(j_omega, complexScale(result_matrix_tl_k1[j][i], eps1));
            complex__float term_TK1 = result_matrix_tk_k1[j][i];

            complex__float term_TL2_mu = complexMul(j_omega, complexScale(result_matrix_tl_k2[j][i], mu2));
            complex__float term_TL2_eps = complexMul(j_omega, complexScale(result_matrix_tl_k2[j][i], eps2));
            complex__float term_TK2 = result_matrix_tk_k2[j][i];

            // Offsets for 4x4 blocks
            size_t off_J1 = 0;
            size_t off_M1 = num_edges;
            size_t off_J2 = 2 * num_edges;
            size_t off_M2 = 3 * num_edges;

            // Helper lambda to add to global matrix. This will be inlined by the compiler.
            auto add_to_global = [&](size_t row_offset, size_t col_offset, complex__float val) {
                size_t row = static_cast<size_t>(test_edge_idx) + row_offset;
                size_t col = static_cast<size_t>(source_edge_idx) + col_offset;
                size_t idx = columnMajorIndex(row, col, matrix_dim);
                atomicAddComplex(&global_Z_matrix[idx], val);
            };

            // Row 1 (J1 eq)
            add_to_global(off_J1, off_J1, term_TL1_mu); // +jωμ₁TL₁
            add_to_global(off_J1, off_M1, term_TK1); // +TK₁

            // Row 2 (M1 eq)
            add_to_global(off_M1, off_J1, complexNeg(term_TK1)); // -TK₁
            add_to_global(off_M1, off_M1, term_TL1_eps); // +jωε₁TL₁

            // Row 3 (J2 eq)
            add_to_global(off_J2, off_J2, term_TL2_mu); // +jωμ₂TL₂
            add_to_global(off_J2, off_M2, term_TK2); // +TK₂

            // Row 4 (M2 eq)
            add_to_global(off_M2, off_J2, complexNeg(term_TK2)); // -TK₂
            add_to_global(off_M2, off_M2, term_TL2_eps); // +jωε₂TL₂
        }
    }
}

__global__ void assemble_Z_mass_kernel_test(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    const complex__float* uniform_gstc_tensors, // Pointer to the 32 complex floats (Y_tot and Z_tot)
    bool is_uniform, // If true, use uniform_gstc_tensors[0..31]. If false, use [face_idx * 32 ..]
    complex__float* global_Z_matrix)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    // 1. Compute Basis Vectors and Rotation Matrix R at centroid
    vec3_t u_hat, v_hat;
    get_spherical_basis_vectors(tri_v[0], tri_v[1], tri_v[2], u_hat, v_hat);

    // 2. Load Local 4x4 Tensors (Y_local, Z_local)
    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];

    auto extract_and_rotate_2x2 = [&](const complex__float* matrix_4x4_ptr, int row_block, int col_block, complex__float T_glob[3][3]) {
        // Indices in 4x4 matrix
        int r0 = row_block * 2;
        int c0 = col_block * 2;
        
        // Load the 2x2 local block
        complex__float t00 = matrix_4x4_ptr[r0 * 4 + c0];
        complex__float t01 = matrix_4x4_ptr[r0 * 4 + c0 + 1];
        complex__float t10 = matrix_4x4_ptr[(r0 + 1) * 4 + c0];
        complex__float t11 = matrix_4x4_ptr[(r0 + 1) * 4 + c0 + 1];

        // Rotate to 3x3 Global
        rotate_tensor_2x2_to_3x3(t00, t01, t10, t11, u_hat, v_hat, T_glob);
    };

    const complex__float* Y_ptr = tensors_ptr;      // First 16 elements
    const complex__float* Z_ptr = tensors_ptr + 16; // Next 16 elements

    // 3. Compute the 8 necessary 3x3 global matrices
    complex__float Y_ee_3D[3][3], Y_em_3D[3][3], Y_me_3D[3][3], Y_mm_3D[3][3];
    complex__float Z_ee_3D[3][3], Z_em_3D[3][3], Z_me_3D[3][3], Z_mm_3D[3][3];

    extract_and_rotate_2x2(Y_ptr, 0, 0, Y_ee_3D);
    extract_and_rotate_2x2(Y_ptr, 0, 1, Y_em_3D);
    extract_and_rotate_2x2(Y_ptr, 1, 0, Y_me_3D);
    extract_and_rotate_2x2(Y_ptr, 1, 1, Y_mm_3D);

    extract_and_rotate_2x2(Z_ptr, 0, 0, Z_ee_3D);
    extract_and_rotate_2x2(Z_ptr, 0, 1, Z_em_3D);
    extract_and_rotate_2x2(Z_ptr, 1, 0, Z_me_3D);
    extract_and_rotate_2x2(Z_ptr, 1, 1, Z_mm_3D);

    // 4. Compute standard mass matrix integrals with Batched Quadrature
    complex__float tensors_batch[8][3][3];
    complex__float results_batch[8][3][3];

    // Compute Y_perp = N * Y * N
    vec3_t e1 = tri_v[1] - tri_v[0];
    vec3_t e2 = tri_v[2] - tri_v[0];
    vec3_t n_vec = normalize(cross(e1, e2));

    // Pack Y tensors (transformed to perp: Y_perp = - [n]x Y [n]x)
    sie_mom::apply_perp_transform(Y_ee_3D, n_vec, tensors_batch[0]);
    sie_mom::apply_perp_transform(Y_em_3D, n_vec, tensors_batch[1]);
    sie_mom::apply_perp_transform(Y_me_3D, n_vec, tensors_batch[2]);
    sie_mom::apply_perp_transform(Y_mm_3D, n_vec, tensors_batch[3]);

    // Pack Z tensors (copy directly)
    sie_mom::copy_tensor_3x3(Z_ee_3D, tensors_batch[4]);
    sie_mom::copy_tensor_3x3(Z_em_3D, tensors_batch[5]);
    sie_mom::copy_tensor_3x3(Z_me_3D, tensors_batch[6]);
    sie_mom::copy_tensor_3x3(Z_mm_3D, tensors_batch[7]);

    calculateMassMatrixQuadrature(tri_v, tri_indices, quad_points, quad_weights, order_num, tensors_batch, results_batch);

    const auto& face_edges = faces_edges[m];
    const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

    size_t off_J1 = 0, off_M1 = num_edges, off_J2 = 2 * num_edges, off_M2 = 3 * num_edges;

    for (int j = 0; j < 3; ++j) {
        const int global_test_edge_idx = local_edge_indices[j];
        for (int i = 0; i < 3; ++i) {
            const int global_source_edge_idx = local_edge_indices[i];

            auto add_val = [&](size_t row_off, size_t col_off, complex__float val) {
                size_t row = static_cast<size_t>(global_test_edge_idx) + row_off;
                size_t col = static_cast<size_t>(global_source_edge_idx) + col_off;
                size_t idx = columnMajorIndex(row, col, matrix_dim);
                atomicAddComplex(&global_Z_matrix[idx], val);
            };

            const float f = 0.5f;
            const float f_q = 0.5f * 0.25f;

            complex__float I_Zee = results_batch[4][j][i];
            complex__float I_Zem = results_batch[5][j][i];
            complex__float I_Zme = results_batch[6][j][i];
            complex__float I_Zmm = results_batch[7][j][i];
            
            complex__float I_Yee = results_batch[0][j][i];
            complex__float I_Yem = results_batch[1][j][i];
            complex__float I_Yme = results_batch[2][j][i];
            complex__float I_Ymm = results_batch[3][j][i];

            // Row 1
            add_val(off_J1, off_J1, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M1, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J1, off_J2, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M2, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            // Row 2
            add_val(off_M1, off_J1, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M1, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M1, off_J2, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M2, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));

            // Row 3
            add_val(off_J2, off_J1, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M1, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J2, off_J2, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M2, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            // Row 4
            add_val(off_M2, off_J1, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M1, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M2, off_J2, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M2, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
        }
    }
}

#include "catch.hpp"

// --------------------------------------------------------
// Test Code
// --------------------------------------------------------

sie_mom::mesh::SphericalSurface generate_sphere() {
    return sie_mom::mesh::SphericalSurface(1.0f, 2); // Coarse mesh (refinement 0 or 1)
}

std::vector<complex__float> generate_uniform_tensors() {
    std::vector<complex__float> tensors(32);
    // Fill with identity-like or random values
    // For simplicity: Scalar identity
    // Y_ee = I --> (1,0,0,1) * factor
    // rest 0
    for(auto& v : tensors) v = make_complex(0.f, 0.f);

    // Set Y_ee diagonal to 1.0 (indices 0 and 3 in the first block)
    // 4x4 matrix, Y_local is top left 2x2 of the 32 numbers?
    // Wait, the kernel expects [32] floats: [Y (16), Z (16)]
    // Y is 4x4, Z is 4x4.
    // Layout: 4 rows.
    
    // Just set everything to 1.0 everywhere to ensure non-zero.
    for(auto& v : tensors) v = make_complex(1.0f, 0.f);
    
    return tensors;
}

TEST_CASE("MatVec Consistency", "[gpu][matvec]") {
    std::cout << "Running MatVec Consistency Test..." << std::endl;

    // 1. Setup Mesh
    auto mesh_obj = generate_sphere();
    auto mesh = mesh_obj.getMeshData();
    size_t num_faces = mesh.faces.size();
    size_t num_edges = mesh.edges.size();
    size_t num_unknowns = 4 * num_edges;
    
    std::cout << "Mesh: " << num_faces << " faces, " << num_edges << " edges, " << num_unknowns << " unknowns." << std::endl;

    // 2. Setup GPU Data
    vec3_t* d_vertices = to_gpu(mesh.vertices);
    int3* d_faces = to_gpu(mesh.faces);
    int3* d_faces_edges = to_gpu(mesh.faces_edges);

    // 3. Setup Quadrature
    int rule = 13; 
    int order_num = dunavant_order_num(rule);
    std::vector<double> h_qp_d(2 * order_num);
    std::vector<double> h_qw_d(order_num);
    dunavant_rule(rule, order_num, h_qp_d.data(), h_qw_d.data());
    
    std::vector<float> h_qp(h_qp_d.begin(), h_qp_d.end());
    std::vector<float> h_qw(h_qw_d.begin(), h_qw_d.end());

    float* d_qp = to_gpu(h_qp);
    float* d_qw = to_gpu(h_qw);

    // 4. Params
    float k1 = 1.0f, k2 = 1.0f;
    float mu_r1 = 1.0f, mu_r2 = 1.0f;
    float eps_r1 = 1.0f, eps_r2 = 1.0f;
    float mu0 = 1.0f, eps0 = 1.0f;
    float omega = 1.0f;
    float singularity_threshold = 0.1f;

    // 5. Build DENSE Matrix
    std::vector<complex__float> h_Z(num_unknowns * num_unknowns, make_complex(0.f, 0.f));
    complex__float* d_Z = to_gpu(h_Z);

    dim3 block(16, 16);
     dim3 grid((num_faces + block.x - 1) / block.x, (num_faces + block.y - 1) / block.y);

    assemble_Z_calderon_kernel_test<<<grid, block>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges, num_unknowns,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega, singularity_threshold,
        d_Z
    );
    CHECK_LAST_CUDA("assemble_Z_calderon_kernel_test");

    // Tensors
    auto h_tensors = generate_uniform_tensors();
    complex__float* d_tensors = to_gpu(h_tensors);
    
    dim3 block1D(256);
    dim3 grid1D((num_faces + block1D.x - 1) / block1D.x);

    assemble_Z_mass_kernel_test<<<grid1D, block1D>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges, num_unknowns,
        d_tensors, true, d_Z
    );
    CHECK_LAST_CUDA("assemble_Z_mass_kernel_test");

    // Copy dense matrix back
    cudaMemcpy(h_Z.data(), d_Z, h_Z.size() * sizeof(complex__float), cudaMemcpyDeviceToHost);

    // 6. Test Vector X
    std::vector<complex__float> h_X(num_unknowns);
    for(size_t i=0; i<num_unknowns; ++i) h_X[i] = make_complex(1.0f, 0.0f); // All ones
    complex__float* d_X = to_gpu(h_X);
    complex__float* d_Y_dense = nullptr;
    cudaMalloc(&d_Y_dense, num_unknowns * sizeof(complex__float));
    cudaMemset(d_Y_dense, 0, num_unknowns * sizeof(complex__float));

    // 7. Dense MatVec: Y = Z * X
    // (Simulated on CPU for simplicity)
    std::vector<complex__float> h_Y_dense(num_unknowns, make_complex(0.f, 0.f));
    for (size_t r = 0; r < num_unknowns; ++r) {
        for (size_t c = 0; c < num_unknowns; ++c) {
             h_Y_dense[r] = complexAdd(h_Y_dense[r], complexMul(h_Z[columnMajorIndex(r, c, num_unknowns)], h_X[c]));
        }
    }

    // 8. Matrix Free MatVec
    complex__float* d_Y_mf = nullptr;
    cudaMalloc(&d_Y_mf, num_unknowns * sizeof(complex__float));
    cudaMemset(d_Y_mf, 0, num_unknowns * sizeof(complex__float));

    // Calderon Part
    matvec_Z_calderon_kernel<<<grid, block>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega, singularity_threshold,
        d_X, d_Y_mf
    );
    CHECK_LAST_CUDA("matvec_Z_calderon_kernel");

    // Mass Part
    matvec_Z_mass_kernel<<<grid1D, block1D>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges,
        d_tensors, true,
        d_X, d_Y_mf
    );
     CHECK_LAST_CUDA("matvec_Z_mass_kernel");

    // Copy back
    std::vector<complex__float> h_Y_mf(num_unknowns);
    cudaMemcpy(h_Y_mf.data(), d_Y_mf, num_unknowns * sizeof(complex__float), cudaMemcpyDeviceToHost);

    // 9. Compare
    double error_norm = 0.0;
    double ref_norm = 0.0;
    for(size_t i=0; i<num_unknowns; ++i) {
        float diff_r = h_Y_dense[i].x - h_Y_mf[i].x;
        float diff_i = h_Y_dense[i].y - h_Y_mf[i].y;
        error_norm += diff_r*diff_r + diff_i*diff_i;
        ref_norm += h_Y_dense[i].x*h_Y_dense[i].x + h_Y_dense[i].y*h_Y_dense[i].y;
    }
    error_norm = sqrt(error_norm);
    ref_norm = sqrt(ref_norm);
    
    std::cout << "Dense Norm: " << ref_norm << std::endl;
    // MF Norm
    double mf_norm = 0.0;
    for(size_t i=0;i<num_unknowns;++i) mf_norm += h_Y_mf[i].x*h_Y_mf[i].x + h_Y_mf[i].y*h_Y_mf[i].y;
    std::cout << "MF Norm: " << sqrt(mf_norm) << std::endl;

    double rel_error = error_norm / (ref_norm + 1e-10);
    std::cout << "Relative Error: " << rel_error << std::endl;

    if(rel_error < 1e-4) {
        std::cout << "SUCCESS: Matrix-Free matches Dense assembly." << std::endl;
    } else {
        std::cout << "FAILURE: Mismatch detected." << std::endl;
    }
    REQUIRE(rel_error < 1e-4);
}
