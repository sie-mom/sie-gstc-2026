#pragma once

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/gstc_tensor_utils.cuh"

static __global__ void matvec_Z_calderon_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    float k1, float k2,
    float mu_r1, float mu_r2,
    float eps_r1, float eps_r2,
    float mu0, float eps0,
    float omega,
    float singularity_threshold,
    const complex__float* x_vector, // Input vector (size: 4 * num_edges)
    complex__float* y_vector        // Output vector (accumulates results)
) {
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

    // Centroid distance check for singularity and integration
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
        // Note: Assuming calculateSingularPartQuadrature allows null/dummy args if needed or works as is
        // We replicate the exact call structure from the original kernel
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

    // Offsets for the vector x and y
    size_t off_J1 = 0;
    size_t off_M1 = num_edges;
    size_t off_J2 = 2 * num_edges;
    size_t off_M2 = 3 * num_edges;

    const complex__float j_omega = make_complex(0.f, omega);
    const float mu1 = mu_r1 * mu0;
    const float mu2 = mu_r2 * mu0;
    const float eps1 = eps_r1 * eps0;
    const float eps2 = eps_r2 * eps0;

    for (int j = 0; j < 3; ++j) { // Test edges
        int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
        
        for (int i = 0; i < 3; ++i) { // Source edges
            int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;
            
            // --- Precompute terms ---
            complex__float term_TL1_mu = j_omega * (mu1 * result_matrix_tl_k1[j][i]);
            complex__float term_TL1_eps = j_omega * (eps1 * result_matrix_tl_k1[j][i]);
            complex__float term_TK1 = result_matrix_tk_k1[j][i];

            complex__float term_TL2_mu = j_omega * (mu2 * result_matrix_tl_k2[j][i]);
            complex__float term_TL2_eps = j_omega * (eps2 * result_matrix_tl_k2[j][i]);
            complex__float term_TK2 = result_matrix_tk_k2[j][i];

            // --- Load components of x_vector ---
            // x = [J1, M1, J2, M2]
            complex__float x_J1 = x_vector[off_J1 + source_edge_idx];
            complex__float x_M1 = x_vector[off_M1 + source_edge_idx];
            complex__float x_J2 = x_vector[off_J2 + source_edge_idx];
            complex__float x_M2 = x_vector[off_M2 + source_edge_idx];

            // --- Apply Matrix-Vector Product Logic ---
            
            // Row 1 (J1 eq): (+jωμ₁TL₁)*J1 + (+TK₁)*M1
            complex__float y_contribution;
            
            // Target: y[off_J1 + test_edge_idx]
            y_contribution = complexAdd( complexMul(term_TL1_mu, x_J1), complexMul(term_TK1, x_M1) );
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], y_contribution);

            // Row 2 (M1 eq): (-TK₁)*J1 + (+jωε₁TL₁)*M1
            // Target: y[off_M1 + test_edge_idx]
            y_contribution = complexAdd( complexMul(complexNeg(term_TK1), x_J1), complexMul(term_TL1_eps, x_M1) );
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], y_contribution);

            // Row 3 (J2 eq): (+jωμ₂TL₂)*J2 + (+TK₂)*M2
            // Target: y[off_J2 + test_edge_idx]
            y_contribution = complexAdd( complexMul(term_TL2_mu, x_J2), complexMul(term_TK2, x_M2) );
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], y_contribution);

            // Row 4 (M2 eq): (-TK₂)*J2 + (+jωε₂TL₂)*M2
            // Target: y[off_M2 + test_edge_idx]
            y_contribution = complexAdd( complexMul(complexNeg(term_TK2), x_J2), complexMul(term_TL2_eps, x_M2) );
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], y_contribution);
        }
    }
}

static __global__ void matvec_Z_mass_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    const complex__float* uniform_gstc_tensors,
    bool is_uniform,
    const complex__float* x_vector, // Input vector
    complex__float* y_vector,       // Output vector
    sie_mom::GstcBasisType basis_type = sie_mom::GstcBasisType::spherical
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    // 1. Compute Local Tangent Basis Vectors and Rotation Matrix R = [u_hat, v_hat]
    vec3_t u_hat, v_hat;
    get_local_basis_vectors(tri_v[0], tri_v[1], tri_v[2], basis_type, u_hat, v_hat);

    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];

    // Helper to rotate 2x2 to 3x3
    auto extract_and_rotate_2x2 = [&](const complex__float* matrix_4x4_ptr, int row_block, int col_block, complex__float T_glob[3][3]) {
        int r0 = row_block * 2;
        int c0 = col_block * 2;
        complex__float t00 = matrix_4x4_ptr[r0 * 4 + c0];
        complex__float t01 = matrix_4x4_ptr[r0 * 4 + c0 + 1];
        complex__float t10 = matrix_4x4_ptr[(r0 + 1) * 4 + c0];
        complex__float t11 = matrix_4x4_ptr[(r0 + 1) * 4 + c0 + 1];
        rotate_tensor_2x2_to_3x3(t00, t01, t10, t11, u_hat, v_hat, T_glob);
    };

    const complex__float* Y_ptr = tensors_ptr;      // First 16 elements
    const complex__float* Z_ptr = tensors_ptr + 16; // Next 16 elements

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

    complex__float tensors_batch[8][3][3];
    complex__float results_batch[8][3][3];

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

    const auto& face_edges_idxs = faces_edges[m];
    const int local_edge_indices[3] = {face_edges_idxs.x, face_edges_idxs.y, face_edges_idxs.z};

    size_t off_J1 = 0, off_M1 = num_edges, off_J2 = 2 * num_edges, off_M2 = 3 * num_edges;
    const float f = 0.5f;
    const float f_q = 0.5f * 0.25f; // 1/8

    for (int j = 0; j < 3; ++j) {
        int test_edge_idx = local_edge_indices[j];
        
        for (int i = 0; i < 3; ++i) {
            int source_edge_idx = local_edge_indices[i];
            
            // Fetch input vector x values
            complex__float x_J1 = x_vector[off_J1 + source_edge_idx];
            complex__float x_M1 = x_vector[off_M1 + source_edge_idx];
            complex__float x_J2 = x_vector[off_J2 + source_edge_idx];
            complex__float x_M2 = x_vector[off_M2 + source_edge_idx];

            // Integrals
            complex__float I_Zee = results_batch[4][j][i];
            complex__float I_Zem = results_batch[5][j][i];
            complex__float I_Zme = results_batch[6][j][i];
            complex__float I_Zmm = results_batch[7][j][i];
            
            complex__float I_Yee = results_batch[0][j][i];
            complex__float I_Yem = results_batch[1][j][i];
            complex__float I_Yme = results_batch[2][j][i];
            complex__float I_Ymm = results_batch[3][j][i];

            complex__float val;

            // -- Contribution to J1 Row (Test J1)
            // Block 11 (J1-J1): Zee + 0.25 Ymm
            val = complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(val, x_J1));

            // Block 12 (J1-M1): Zem - 0.25 Yme
            val = complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(val, x_M1));

            // Block 13 (J1-J2): Zee - 0.25 Ymm
            val = complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(val, x_J2));

            // Block 14 (J1-M2): Zem + 0.25 Yme
            val = complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(val, x_M2));


            // -- Contribution to M1 Row (Test M1)
            // Block 21 (M1-J1): Zme - 0.25 Yem
            val = complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(val, x_J1));

            // Block 22 (M1-M1): Zmm + 0.25 Yee
            val = complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(val, x_M1));

            // Block 23 (M1-J2): Zme + 0.25 Yem
            val = complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(val, x_J2));

            // Block 24 (M1-M2): Zmm - 0.25 Yee
            val = complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(val, x_M2));


            // -- Contribution to J2 Row (Test J2)
            // Block 31 (J2-J1): (Zee - 0.25 Ymm)
            val = complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(val, x_J1));

            // Block 32 (J2-M1): (Zem + 0.25 Yme)
            val = complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(val, x_M1));

            // Block 33 (J2-J2): (Zee + 0.25 Ymm)
            val = complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(val, x_J2));

            // Block 34 (J2-M2): (Zem - 0.25 Yme)
            val = complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(val, x_M2));

            
            // -- Contribution to M2 Row (Test M2)
            // Block 41 (M2-J1): (Zme + 0.25 Yem)
            val = complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(val, x_J1));

            // Block 42 (M2-M1): (Zmm - 0.25 Yee)
            val = complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(val, x_M1));

            // Block 43 (M2-J2): (Zme - 0.25 Yem)
            val = complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(val, x_J2));

            // Block 44 (M2-M2): (Zmm + 0.25 Yee)
            val = complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(val, x_M2));

        }
    }
}



// --- Approximate Inverse Mass Matrix Application Kernel ---
/**
 * @brief Applies Approximated Inverse Mass Matrix Preconditioner (y += P*x).
 * 
 * This kernel assumes the preconditioner P is an accumulation of 
 * inverted local face element interactions:
 *      y_i += sum_faces ( (1 / local_element_ij) * x_j )
 * 
 * This is a matrix-free operation that should be called during the iterative solver step.
 * It replaces the previous diagonal extraction strategy.
 * 
 * @param x_vector Input vector (e.g., Residual or Vector to invoke P on)
 * @param y_vector Output vector (Accumulates result, e.g., P * x)
 */
static __global__ void matvec_Z_mass_precond_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    const complex__float* uniform_gstc_tensors,
    bool is_uniform,
    const complex__float* x_vector, // Input vector
    complex__float* y_vector,       // Output vector
    sie_mom::GstcBasisType basis_type = sie_mom::GstcBasisType::spherical
) {
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    // 1. Compute Local Tangent Basis Vectors and Rotation Matrix R = [u_hat, v_hat]
    vec3_t u_hat, v_hat;
    get_local_basis_vectors(tri_v[0], tri_v[1], tri_v[2], basis_type, u_hat, v_hat);

    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];

    // Helper to rotate 2x2 to 3x3
    auto extract_and_rotate_2x2 = [&](const complex__float* matrix_4x4_ptr, int row_block, int col_block, complex__float T_glob[3][3]) {
        int r0 = row_block * 2;
        int c0 = col_block * 2;
        complex__float t00 = matrix_4x4_ptr[r0 * 4 + c0];
        complex__float t01 = matrix_4x4_ptr[r0 * 4 + c0 + 1];
        complex__float t10 = matrix_4x4_ptr[(r0 + 1) * 4 + c0];
        complex__float t11 = matrix_4x4_ptr[(r0 + 1) * 4 + c0 + 1];
        rotate_tensor_2x2_to_3x3(t00, t01, t10, t11, u_hat, v_hat, T_glob);
    };

    const complex__float* Y_ptr = tensors_ptr;      // First 16 elements
    const complex__float* Z_ptr = tensors_ptr + 16; // Next 16 elements

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

    complex__float tensors_batch[8][3][3];
    complex__float results_batch[8][3][3];

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

    const auto& face_edges_idxs = faces_edges[m];
    const int local_edge_indices[3] = {face_edges_idxs.x, face_edges_idxs.y, face_edges_idxs.z};

    size_t off_J1 = 0, off_M1 = num_edges, off_J2 = 2 * num_edges, off_M2 = 3 * num_edges;
    const float f = 0.5f;
    const float f_q = 0.5f * 0.25f; // 1/8

    for (int j = 0; j < 3; ++j) {
        int test_edge_idx = local_edge_indices[j];
        
        for (int i = 0; i < 3; ++i) {
            int source_edge_idx = local_edge_indices[i];
            
            // Fetch input vector x values
            complex__float x_J1 = x_vector[off_J1 + source_edge_idx];
            complex__float x_M1 = x_vector[off_M1 + source_edge_idx];
            complex__float x_J2 = x_vector[off_J2 + source_edge_idx];
            complex__float x_M2 = x_vector[off_M2 + source_edge_idx];

            // Integrals
            complex__float I_Zee = results_batch[4][j][i];
            complex__float I_Zem = results_batch[5][j][i];
            complex__float I_Zme = results_batch[6][j][i];
            complex__float I_Zmm = results_batch[7][j][i];
            
            complex__float I_Yee = results_batch[0][j][i];
            complex__float I_Yem = results_batch[1][j][i];
            complex__float I_Yme = results_batch[2][j][i];
            complex__float I_Ymm = results_batch[3][j][i];

            complex__float val;

            // -- Contribution to J1 Row (Test J1)
            // Block 11 (J1-J1): Zee + 0.25 Ymm
            val = complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(complexInverse(val), x_J1));

            // Block 12 (J1-M1): Zem - 0.25 Yme
            val = complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(complexInverse(val), x_M1));

            // Block 13 (J1-J2): Zee - 0.25 Ymm
            val = complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(complexInverse(val), x_J2));

            // Block 14 (J1-M2): Zem + 0.25 Yme
            val = complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], complexMul(complexInverse(val), x_M2));


            // -- Contribution to M1 Row (Test M1)
            // Block 21 (M1-J1): Zme - 0.25 Yem
            val = complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(complexInverse(val), x_J1));

            // Block 22 (M1-M1): Zmm + 0.25 Yee
            val = complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(complexInverse(val), x_M1));

            // Block 23 (M1-J2): Zme + 0.25 Yem
            val = complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(complexInverse(val), x_J2));

            // Block 24 (M1-M2): Zmm - 0.25 Yee
            val = complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], complexMul(complexInverse(val), x_M2));


            // -- Contribution to J2 Row (Test J2)
            // Block 31 (J2-J1): (Zee - 0.25 Ymm)
            val = complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(complexInverse(val), x_J1));

            // Block 32 (J2-M1): (Zem + 0.25 Yme)
            val = complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(complexInverse(val), x_M1));

            // Block 33 (J2-J2): (Zee + 0.25 Ymm)
            val = complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(complexInverse(val), x_J2));

            // Block 34 (J2-M2): (Zem - 0.25 Yme)
            val = complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q));
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], complexMul(complexInverse(val), x_M2));

            
            // -- Contribution to M2 Row (Test M2)
            // Block 41 (M2-J1): (Zme + 0.25 Yem)
            val = complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(complexInverse(val), x_J1));

            // Block 42 (M2-M1): (Zmm - 0.25 Yee)
            val = complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(complexInverse(val), x_M1));

            // Block 43 (M2-J2): (Zme - 0.25 Yem)
            val = complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(complexInverse(val), x_J2));

            // Block 44 (M2-M2): (Zmm + 0.25 Yee)
            val = complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q));
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], complexMul(complexInverse(val), x_M2));

        }
    }
}

// --- Invert Diagonal Kernel ---
// NOTE: Now used only to finalize or fix NaNs, but if extract_diagonal already inverts, this should technically be identity.
// However, to avoid changing host code, we just make this kernel logic effectively a no-op if the values are valid.
static __global__ void invert_diagonal_kernel(
    complex__float* diag_vec,
    size_t vector_size
) {
    // MODIFIED: The preconditioner is already built as the INVERSE in extract_diagonal_kernel.
    // So this kernel should do NOTHING (or maybe scaling check).
    // We just return.
    return; 
}

// --- Element-wise Complex Multiply (Apply Preconditioner) ---
static __global__ void apply_preconditioner_kernel(
    const complex__float* diag_inv, // P
    const complex__float* vec_in,   // v
    complex__float* vec_out,        // P * v
    size_t size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    vec_out[idx] = complexMul(diag_inv[idx], vec_in[idx]);
}

// Adaptive Quadrature Version of matvec_Z_calderon_kernel
static __global__ void matvec_Z_calderon_adaptive_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    
    // Adaptive Quadrature Sets (Low, Medium, High)
    const float* quad_points_low, const float* quad_weights_low, int order_low,
    const float* quad_points_med, const float* quad_weights_med, int order_med,
    const float* quad_points_high, const float* quad_weights_high, int order_high,
    
    size_t num_faces,
    size_t num_edges,
    float k1, float k2,
    float mu_r1, float mu_r2,
    float eps_r1, float eps_r2,
    float mu0, float eps0,
    float omega,
    float singularity_threshold, // Usually 0.1 lambda
    float far_threshold,         // Usually 0.5 lambda or 1.0 lambda
    
    const complex__float* x_vector, // Input vector (size: 4 * num_edges)
    complex__float* y_vector        // Output vector (accumulates results)
) {
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

    // Centroid distance check for singularity and integration
    vec3_t test_centroid = (test_tri_v[0] + test_tri_v[1] + test_tri_v[2]) / 3.f;
    vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        // --- SINGULAR / TOUCHING (High Precision + Extraction) ---
        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    quad_points_high, quad_weights_high, order_high, k1, k2, true,
                                    result_matrix_tl_k1, result_matrix_tl_k2,
                                    result_matrix_tk_k1, result_matrix_tk_k2);

        complex__float singular_tl_k1[3][3], singular_tl_k2[3][3];
        complex__float singular_tk_k1[3][3], singular_tk_k2[3][3];
        
        calculateSingularPartQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                        quad_points_high, quad_weights_high, order_high, k1, k2,
                                        singular_tl_k1, singular_tl_k2,
                                        singular_tk_k1, singular_tk_k2);
        
        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
            complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
            complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
            complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
        }
    } else {
        // --- NON-SINGULAR (Adaptive) ---
        const float* q_pts;
        const float* q_wts;
        int q_order;

        if (dist < far_threshold) {
            // Near Field (Medium Precision)
            q_pts = quad_points_med;
            q_wts = quad_weights_med;
            q_order = order_med;
        } else {
            // Far Field (Low Precision)
            q_pts = quad_points_low;
            q_wts = quad_weights_low;
            q_order = order_low;
        }

        calculateDoubleQuadrature(test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                                    q_pts, q_wts, q_order, k1, k2, false,
                                    result_matrix_tl_k1, result_matrix_tl_k2,
                                    result_matrix_tk_k1, result_matrix_tk_k2);
    }

    // Rest of result accumulation logic (Same as original kernel)
    // Offsets for the vector x and y
    size_t off_J1 = 0;
    size_t off_M1 = num_edges;
    size_t off_J2 = 2 * num_edges;
    size_t off_M2 = 3 * num_edges;

    const complex__float j_omega = make_complex(0.f, omega);
    const float mu1 = mu_r1 * mu0;
    const float mu2 = mu_r2 * mu0;
    const float eps1 = eps_r1 * eps0;
    const float eps2 = eps_r2 * eps0;

    for (int j = 0; j < 3; ++j) { // Test edges
        int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
        
        for (int i = 0; i < 3; ++i) { // Source edges
            int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;
            
            // --- Precompute terms ---
            complex__float term_TL1_mu = j_omega * (mu1 * result_matrix_tl_k1[j][i]);
            complex__float term_TL1_eps = j_omega * (eps1 * result_matrix_tl_k1[j][i]);
            complex__float term_TK1 = result_matrix_tk_k1[j][i];

            complex__float term_TL2_mu = j_omega * (mu2 * result_matrix_tl_k2[j][i]);
            complex__float term_TL2_eps = j_omega * (eps2 * result_matrix_tl_k2[j][i]);
            complex__float term_TK2 = result_matrix_tk_k2[j][i];

            // --- Load components of x_vector ---
            // x = [J1, M1, J2, M2]
            complex__float x_J1 = x_vector[off_J1 + source_edge_idx];
            complex__float x_M1 = x_vector[off_M1 + source_edge_idx];
            complex__float x_J2 = x_vector[off_J2 + source_edge_idx];
            complex__float x_M2 = x_vector[off_M2 + source_edge_idx];

            // --- Apply Matrix-Vector Product Logic ---
            
            // Row 1 (J1 eq): (+jωμ₁TL₁)*J1 + (+TK₁)*M1
            complex__float y_contribution;
            
            // Target: y[off_J1 + test_edge_idx]
            y_contribution = complexAdd( complexMul(term_TL1_mu, x_J1), complexMul(term_TK1, x_M1) );
            atomicAddComplex(&y_vector[off_J1 + test_edge_idx], y_contribution);

            // Row 2 (M1 eq): (-TK₁)*J1 + (+jωε₁TL₁)*M1
            // Target: y[off_M1 + test_edge_idx]
            y_contribution = complexAdd( complexMul(complexNeg(term_TK1), x_J1), complexMul(term_TL1_eps, x_M1) );
            atomicAddComplex(&y_vector[off_M1 + test_edge_idx], y_contribution);

            // Row 3 (J2 eq): (+jωμ₂TL₂)*J2 + (+TK₂)*M2
            // Target: y[off_J2 + test_edge_idx]
            y_contribution = complexAdd( complexMul(term_TL2_mu, x_J2), complexMul(term_TK2, x_M2) );
            atomicAddComplex(&y_vector[off_J2 + test_edge_idx], y_contribution);

            // Row 4 (M2 eq): (-TK₂)*J2 + (+jωε₂TL₂)*M2
            // Target: y[off_M2 + test_edge_idx]
            y_contribution = complexAdd( complexMul(complexNeg(term_TK2), x_J2), complexMul(term_TL2_eps, x_M2) );
            atomicAddComplex(&y_vector[off_M2 + test_edge_idx], y_contribution);
        }
    }
}
