// Eigen is used only for the CPU reference solve in this test.
// Avoid Eigen/MSVC complex-operator conflicts under NVCC.
// TODO: This test is a candidate for future deprecation. Remove this local
// workaround when the test is retired; revisit it if Eigen is used in kernels.
#if defined(_MSC_VER) && defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_NO_CUDA
#endif

#include "catch.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>

#include "sie_mom/cuda/platform.cuh"
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/sie_kernels_matvec.cuh"
#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/gmres/gmres_solver.cuh"
#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

// --- Reuse Helpers from test_matvec_consistency.cu ---

#define CHECK_LAST_CUDA(msg) do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Error after %s: %s at %s:%d\n", msg, cudaGetErrorString(err), __FILE__, __LINE__); \
        FAIL("CUDA Error"); \
    } \
} while (0)

__host__ __device__ inline size_t columnMajorIndex(size_t row, size_t col, size_t leading_dim) {
    return col * leading_dim + row;
}

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
// Dense Assembly Kernel (Locally defined for validation)
// --------------------------------------------------------
// Note: This matches the one in test_matvec_consistency.cu
__global__ void assemble_Z_calderon_valid_kernel(
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
    if (m >= num_faces || n >= num_faces) return;

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

    vec3_t test_centroid = (test_tri_v[0] + test_tri_v[1] + test_tri_v[2]) / 3.f;
    vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points, quad_weights, order_num, k1, k2, true,
                                    result_matrix_tl_k1, result_matrix_tl_k2, result_matrix_tk_k1, result_matrix_tk_k2);
        complex__float singular_tl_k1[3][3], singular_tl_k2[3][3], singular_tk_k1[3][3], singular_tk_k2[3][3];
        calculateSingularPartQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                        quad_points, quad_weights, order_num, k1, k2,
                                        singular_tl_k1, singular_tl_k2, singular_tk_k1, singular_tk_k2);
        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
            complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
            complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
            complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
        }
    } else {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points, quad_weights, order_num, k1, k2, false,
                                    result_matrix_tl_k1, result_matrix_tl_k2, result_matrix_tk_k1, result_matrix_tk_k2);
    }

    for (int j = 0; j < 3; ++j) {
        int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
        for (int i = 0; i < 3; ++i) {
            int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;
            
            complex__float j_omega = make_complex(0.f, omega);
            complex__float term_TL1_mu = complexMul(j_omega, complexScale(result_matrix_tl_k1[j][i], mu_r1 * mu0));
            complex__float term_TL1_eps = complexMul(j_omega, complexScale(result_matrix_tl_k1[j][i], eps_r1 * eps0));
            complex__float term_TK1 = result_matrix_tk_k1[j][i];
            complex__float term_TL2_mu = complexMul(j_omega, complexScale(result_matrix_tl_k2[j][i], mu_r2 * mu0));
            complex__float term_TL2_eps = complexMul(j_omega, complexScale(result_matrix_tl_k2[j][i], eps_r2 * eps0));
            complex__float term_TK2 = result_matrix_tk_k2[j][i];

            size_t off_J1 = 0, off_M1 = num_edges, off_J2 = 2 * num_edges, off_M2 = 3 * num_edges;
            auto add_to_global = [&](size_t row_offset, size_t col_offset, complex__float val) {
                size_t row = static_cast<size_t>(test_edge_idx) + row_offset;
                size_t col = static_cast<size_t>(source_edge_idx) + col_offset;
                size_t idx = columnMajorIndex(row, col, matrix_dim);
                atomicAddComplex(&global_Z_matrix[idx], val);
            };

            add_to_global(off_J1, off_J1, term_TL1_mu);
            add_to_global(off_J1, off_M1, term_TK1);
            add_to_global(off_M1, off_J1, complexNeg(term_TK1));
            add_to_global(off_M1, off_M1, term_TL1_eps);
            add_to_global(off_J2, off_J2, term_TL2_mu);
            add_to_global(off_J2, off_M2, term_TK2);
            add_to_global(off_M2, off_J2, complexNeg(term_TK2));
            add_to_global(off_M2, off_M2, term_TL2_eps);
        }
    }
}

__global__ void assemble_Z_mass_valid_kernel(
    const vec3_t* vertices, const int3* faces, const int3* faces_edges,
    const float* quad_points, const float* quad_weights, int order_num,
    size_t num_faces, size_t num_edges, size_t matrix_dim,
    const complex__float* uniform_gstc_tensors, bool is_uniform,
    complex__float* global_Z_matrix)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_faces) return;

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    vec3_t u_hat, v_hat;
    get_spherical_basis_vectors(tri_v[0], tri_v[1], tri_v[2], u_hat, v_hat);

    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];
    auto extract_and_rotate_2x2 = [&](const complex__float* matrix_4x4_ptr, int row_block, int col_block, complex__float T_glob[3][3]) {
        int r0 = row_block * 2; int c0 = col_block * 2;
        complex__float t00 = matrix_4x4_ptr[r0 * 4 + c0];
        complex__float t01 = matrix_4x4_ptr[r0 * 4 + c0 + 1];
        complex__float t10 = matrix_4x4_ptr[(r0 + 1) * 4 + c0];
        complex__float t11 = matrix_4x4_ptr[(r0 + 1) * 4 + c0 + 1];
        rotate_tensor_2x2_to_3x3(t00, t01, t10, t11, u_hat, v_hat, T_glob);
    };

    const complex__float* Y_ptr = tensors_ptr;
    const complex__float* Z_ptr = tensors_ptr + 16;

    complex__float Y_ee_3D[3][3], Y_em_3D[3][3], Y_me_3D[3][3], Y_mm_3D[3][3];
    complex__float Z_ee_3D[3][3], Z_em_3D[3][3], Z_me_3D[3][3], Z_mm_3D[3][3];
    extract_and_rotate_2x2(Y_ptr, 0, 0, Y_ee_3D); extract_and_rotate_2x2(Y_ptr, 0, 1, Y_em_3D);
    extract_and_rotate_2x2(Y_ptr, 1, 0, Y_me_3D); extract_and_rotate_2x2(Y_ptr, 1, 1, Y_mm_3D);
    extract_and_rotate_2x2(Z_ptr, 0, 0, Z_ee_3D); extract_and_rotate_2x2(Z_ptr, 0, 1, Z_em_3D);
    extract_and_rotate_2x2(Z_ptr, 1, 0, Z_me_3D); extract_and_rotate_2x2(Z_ptr, 1, 1, Z_mm_3D);

    complex__float tensors_batch[8][3][3], results_batch[8][3][3];
    
    // Y_perp calculation
    vec3_t e1 = tri_v[1] - tri_v[0]; vec3_t e2 = tri_v[2] - tri_v[0];
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
            const float f = 0.5f, f_q = 0.125f;
            complex__float I_Zee = results_batch[4][j][i], I_Zem = results_batch[5][j][i];
            complex__float I_Zme = results_batch[6][j][i], I_Zmm = results_batch[7][j][i];
            complex__float I_Yee = results_batch[0][j][i], I_Yem = results_batch[1][j][i];
            complex__float I_Yme = results_batch[2][j][i], I_Ymm = results_batch[3][j][i];

            add_val(off_J1, off_J1, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M1, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J1, off_J2, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M2, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            add_val(off_M1, off_J1, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M1, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M1, off_J2, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M2, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));

            add_val(off_J2, off_J1, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M1, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J2, off_J2, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M2, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            add_val(off_M2, off_J1, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M1, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M2, off_J2, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M2, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
        }
    }
}

// --------------------------------------------------------
// Function Object for GMRES
// --------------------------------------------------------

struct PhysicsMatVec {
    // Parameters needed for kernels
    const vec3_t* d_vertices;
    const int3* d_faces;
    const int3* d_faces_edges;
    const float* d_qp;
    const float* d_qw;
    int order_num;
    size_t num_faces;
    size_t num_edges;
    float k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega, singularity_threshold;
    const complex__float* d_tensors;
    bool is_uniform;

    void operator()(const complex__float* x, complex__float* y) {
        // Zero output Y
        size_t num_unknowns = 4 * num_edges;
        cudaMemset(y, 0, num_unknowns * sizeof(complex__float));

        dim3 block(16, 16);
        dim3 grid((num_faces + block.x - 1) / block.x, (num_faces + block.y - 1) / block.y);

        matvec_Z_calderon_kernel<<<grid, block>>>(
            d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
            num_faces, num_edges,
            k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega, singularity_threshold,
            x, y
        );
        CHECK_LAST_CUDA("matvec_Z_calderon_kernel in GMRES");

        dim3 block1D(256);
        dim3 grid1D((num_faces + block1D.x - 1) / block1D.x);
        matvec_Z_mass_kernel<<<grid1D, block1D>>>(
            d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
            num_faces, num_edges,
            d_tensors, is_uniform,
            x, y
        );
        CHECK_LAST_CUDA("matvec_Z_mass_kernel in GMRES");
    }
};

// --------------------------------------------------------
// Test Case
// --------------------------------------------------------

TEST_CASE("GMRES Solver with Physics Matrix", "[gpu][gmres][physics]") {
    std::cout << "Starting GMRES Physics Test..." << std::endl;

    // 1. Setup Mesh (Coarse Sphere)
    sie_mom::mesh::SphericalSurface mesh_obj(1.0f, 0); // Keep runtime manageable for iterative GMRES checks
    auto mesh = mesh_obj.getMeshData();
    size_t num_faces = mesh.faces.size();
    size_t num_edges = mesh.edges.size();
    size_t num_unknowns = 4 * num_edges;
    
    // 2. GPU Data Setup
    vec3_t* d_vertices = to_gpu(mesh.vertices);
    int3* d_faces = to_gpu(mesh.faces);
    int3* d_faces_edges = to_gpu(mesh.faces_edges);

    int rule = 13; 
    int order_num = dunavant_order_num(rule);
    std::vector<double> h_qp_d(2 * order_num), h_qw_d(order_num);
    dunavant_rule(rule, order_num, h_qp_d.data(), h_qw_d.data());
    std::vector<float> h_qp(h_qp_d.begin(), h_qp_d.end()), h_qw(h_qw_d.begin(), h_qw_d.end());
    float* d_qp = to_gpu(h_qp);
    float* d_qw = to_gpu(h_qw);

    // 3. Parameters
    float k1 = 1.0f, k2 = 1.0f;
    float mu_r1 = 1.0f, mu_r2 = 1.0f;
    float eps_r1 = 1.0f, eps_r2 = 1.0f;
    float mu0 = 1.0f, eps0 = 1.0f;
    float omega = 1.0f;
    float singularity_threshold = 0.1f;

    std::vector<complex__float> h_tensors(32);
    for(auto& v : h_tensors) v = make_complex(1.0f, 0.f); // Identity-like
    complex__float* d_tensors = to_gpu(h_tensors);

    // 4. Assemble Dense Matrix (for Reference)
    std::cout << "Assembling Dense Matrix..." << std::endl;
    std::vector<complex__float> h_Z(num_unknowns * num_unknowns, make_complex(0.f, 0.f));
    complex__float* d_Z = to_gpu(h_Z);

    dim3 block(16, 16);
    dim3 grid((num_faces + block.x - 1) / block.x, (num_faces + block.y - 1) / block.y);
    assemble_Z_calderon_valid_kernel<<<grid, block>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges, num_unknowns,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega, singularity_threshold,
        d_Z
    );
    
    dim3 block1D(256);
    dim3 grid1D((num_faces + block1D.x - 1) / block1D.x);
    assemble_Z_mass_valid_kernel<<<grid1D, block1D>>>(
        d_vertices, d_faces, d_faces_edges, d_qp, d_qw, order_num,
        num_faces, num_edges, num_unknowns,
        d_tensors, true, d_Z
    );
    CHECK_LAST_CUDA("Assembly Valid");

    // Copy to Host for Eigen Solve
    cudaMemcpy(h_Z.data(), d_Z, h_Z.size() * sizeof(complex__float), cudaMemcpyDeviceToHost);

    // 5. Setup RHS
    std::vector<complex__float> h_b(num_unknowns);
    for(size_t i=0; i<num_unknowns; ++i) h_b[i] = make_complex(1.0f, 0.0f); // Constant RHS
    complex__float* d_b = to_gpu(h_b);

    // 6. Direct Solve (CPU)
    std::cout << "Direct Solve (CPU)..." << std::endl;
    Eigen::MatrixXcf mat_eigen(num_unknowns, num_unknowns);
    Eigen::VectorXcf rhs_eigen(num_unknowns);
    for(size_t c = 0; c < num_unknowns; ++c) {
        for(size_t r = 0; r < num_unknowns; ++r) {
            complex__float val = h_Z[columnMajorIndex(r, c, num_unknowns)];
            mat_eigen(r, c) = std::complex<float>(val.x, val.y);
        }
        rhs_eigen(c) = std::complex<float>(h_b[c].x, h_b[c].y);
    }

    Eigen::VectorXcf x_ref_eigen = mat_eigen.partialPivLu().solve(rhs_eigen);
    double ref_norm = x_ref_eigen.norm();
    std::cout << "Reference Norm: " << ref_norm << std::endl;

    // 7. GMRES Solve (GPU)
    std::cout << "GMRES Solve (GPU)..." << std::endl;
    
    PhysicsMatVec matvec_op;
    matvec_op.d_vertices = d_vertices;
    matvec_op.d_faces = d_faces;
    matvec_op.d_faces_edges = d_faces_edges;
    matvec_op.d_qp = d_qp;
    matvec_op.d_qw = d_qw;
    matvec_op.order_num = order_num;
    matvec_op.num_faces = num_faces;
    matvec_op.num_edges = num_edges;
    matvec_op.k1 = k1; matvec_op.k2 = k2;
    matvec_op.mu_r1 = mu_r1; matvec_op.mu_r2 = mu_r2;
    matvec_op.eps_r1 = eps_r1; matvec_op.eps_r2 = eps_r2;
    matvec_op.mu0 = mu0; matvec_op.eps0 = eps0;
    matvec_op.omega = omega;
    matvec_op.singularity_threshold = singularity_threshold;
    matvec_op.d_tensors = d_tensors;
    matvec_op.is_uniform = true;

    cublasHandle_t handle;
    cublasCreate(&handle);

    GMRESSolver<PhysicsMatVec> gmres(handle, num_unknowns, matvec_op);
    
    complex__float* d_x_gmres = nullptr;
    cudaMalloc(&d_x_gmres, num_unknowns * sizeof(complex__float));
    cudaMemset(d_x_gmres, 0, num_unknowns * sizeof(complex__float)); // Initial guess 0

    // Solver Params
    int m_restart = 25;
    int max_iter = 300; // Total iters
    float tol = 1e-5f;

    gmres.solve(d_b, d_x_gmres, m_restart, max_iter, tol);

    // 8. Compare
    std::vector<complex__float> h_x_gmres(num_unknowns);
    cudaMemcpy(h_x_gmres.data(), d_x_gmres, num_unknowns * sizeof(complex__float), cudaMemcpyDeviceToHost);

    double diff_norm = 0.0;
    for(size_t i=0; i<num_unknowns; ++i) {
        std::complex<float> val_ref = x_ref_eigen(i);
        std::complex<float> val_gmres(h_x_gmres[i].x, h_x_gmres[i].y);
        diff_norm += std::norm(val_ref - val_gmres);
    }
    diff_norm = std::sqrt(diff_norm);

    std::cout << "GMRES vs Reference Difference Norm: " << diff_norm << std::endl;
    double rel_error = diff_norm / (ref_norm + 1e-12);
    std::cout << "Relative Error: " << rel_error << std::endl;

    REQUIRE(rel_error < 1e-3);

    // Cleanup
    cudaFree(d_vertices); cudaFree(d_faces); cudaFree(d_faces_edges);
    cudaFree(d_qp); cudaFree(d_qw); cudaFree(d_tensors);
    cudaFree(d_Z); cudaFree(d_b); cudaFree(d_x_gmres);
    cublasDestroy(handle);
}
