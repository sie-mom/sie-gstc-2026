#pragma once

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/cuda/sie_scaling.cuh"

// Header-defined kernels have internal linkage so multiple solvers can share them.

__host__ __device__ inline size_t columnMajorIndex(size_t row, size_t col, size_t leading_dim) {
    return col * leading_dim + row;
}

/**
 * @brief Assembles the Calderón operator block (A_C) for the SIE-GSTC MoM matrix.
 *
 * Computes the required numerical integrations (using standard or singular double
 * quadrature) for pairs of test and source triangles. It scales the results with the
 * appropriate physical constants and atomically adds the corresponding 4x4 local block 
 * contributions (J1, M1, J2, M2 equations) to the global Z matrix.
 */
static __global__ void assemble_Z_calderon_kernel(
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
            complex__float term_TL1_mu = j_omega * (mu1 * result_matrix_tl_k1[j][i]);
            complex__float term_TL1_eps = j_omega * (eps1 * result_matrix_tl_k1[j][i]);
            complex__float term_TK1 = result_matrix_tk_k1[j][i];

            complex__float term_TL2_mu = j_omega * (mu2 * result_matrix_tl_k2[j][i]);
            complex__float term_TL2_eps = j_omega * (eps2 * result_matrix_tl_k2[j][i]);
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

/**
 * @brief Assembles an eta0-scaled Calderón operator block for improved conditioning.
 *
 * This kernel keeps the same quadrature path as assemble_Z_calderon_kernel but applies
 * the row/column scaling associated with M -> M / eta0, using eta0 = sqrt(mu0 / eps0).
 */
static __global__ void assemble_Z_calderon_eta0_scaled_kernel(
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

    const real_t eta0 = sqrt(mu0 / eps0);
    const auto block1_factors = sie_mom::cuda::make_eta0_scaled_calderon_factors(
        omega, mu_r1 * mu0, eps_r1 * eps0, eta0
    );
    const auto block2_factors = sie_mom::cuda::make_eta0_scaled_calderon_factors(
        omega, mu_r2 * mu0, eps_r2 * eps0, eta0
    );

    for (int j = 0; j < 3; ++j) {
        int test_edge_idx = (j == 0) ? test_face_edges.x : (j == 1) ? test_face_edges.y : test_face_edges.z;
        for (int i = 0; i < 3; ++i) {
            int source_edge_idx = (i == 0) ? source_face_edges.x : (i == 1) ? source_face_edges.y : source_face_edges.z;

            const auto block1 = sie_mom::cuda::make_eta0_scaled_calderon_block(
                result_matrix_tl_k1[j][i], result_matrix_tk_k1[j][i],
                block1_factors
            );
            const auto block2 = sie_mom::cuda::make_eta0_scaled_calderon_block(
                result_matrix_tl_k2[j][i], result_matrix_tk_k2[j][i],
                block2_factors
            );

            size_t off_J1 = 0;
            size_t off_M1 = num_edges;
            size_t off_J2 = 2 * num_edges;
            size_t off_M2 = 3 * num_edges;

            auto add_to_global = [&](size_t row_offset, size_t col_offset, complex__float val) {
                size_t row = static_cast<size_t>(test_edge_idx) + row_offset;
                size_t col = static_cast<size_t>(source_edge_idx) + col_offset;
                size_t idx = columnMajorIndex(row, col, matrix_dim);
                atomicAddComplex(&global_Z_matrix[idx], val);
            };

            add_to_global(off_J1, off_J1, block1.jj);
            add_to_global(off_J1, off_M1, block1.jm);

            add_to_global(off_M1, off_J1, block1.mj);
            add_to_global(off_M1, off_M1, block1.mm);

            add_to_global(off_J2, off_J2, block2.jj);
            add_to_global(off_J2, off_M2, block2.jm);

            add_to_global(off_M2, off_J2, block2.mj);
            add_to_global(off_M2, off_M2, block2.mm);
        }
    }
}

/**
 * @brief Assembles the sparse mass matrix operator block (A_M) for the SIE-GSTC MoM matrix.
 *
 * Integrates the GSTC material properties (Y and Z tensors) over each face using numerical 
 * quadrature. It handles the local-to-global rotation of the macroscopic susceptibilities, 
 * applies the necessary perpendicular transformations, and atomically accumulates the 
 * corresponding 4x4 local block contributions (J1, M1, J2, M2 equations) into the global Z matrix.
 */
static __global__ void assemble_Z_mass_kernel(
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
    complex__float* global_Z_matrix,
    sie_mom::GstcBasisType basis_type = sie_mom::GstcBasisType::spherical)
{
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

    // 2. Load Local 4x4 Tensors (Y_local, Z_local)
    // Memory layout: [Y_row0, Y_row1, ..., Y_row3, Z_row0, ... Z_row3]
    // Each row has 4 complex floats.
    // Total 32 complex floats per block.
    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];

    // Helper to extract a 2x2 block from the 4x4 array
    // Matrix is 4x4 row-major.
    // row_block 0,1. col_block 0,1.
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
    // We compute 8 tensors simultaneously to minimize geometric overhead.
    // Index mapping:
    // [0..3]: Y_perp terms (Y_ee, Y_em, Y_me, Y_mm)
    // [4..7]: Z terms      (Z_ee, Z_em, Z_me, Z_mm)
    
    // WARNING regarding array dimensions being coincidentally equal [3][3]:
    // tensors_batch: [8][3][3] -> [TensorID][Cartesian_Row(x,y,z)][Cartesian_Col(x,y,z)]
    //                Represents the physical 3D tensor constant on this face.
    complex__float tensors_batch[8][3][3];

    // results_batch: [8][3][3] -> [TensorID][Test_Edge_Local_Idx(0,1,2)][Source_Edge_Local_Idx(0,1,2)]
    //                Represents the Mass Matrix integral <f_test, Tensor * f_source> on this face.
    //                These are PARTIAL contributions to the global edges (atomicAdd will sum contributions from adjacent faces).
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

    // Perform batched quadrature (1 call instead of 8)
    calculateMassMatrixQuadrature(tri_v, tri_indices, quad_points, quad_weights, order_num, tensors_batch, results_batch);

    const auto& face_edges = faces_edges[m];
    const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

    // Offsets for 4x4 blocks
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

            // Factor 0.5f from documentation
            const float f = 0.5f;
            const float f_q = 0.5f * 0.25f; // 1/8

            // Fetch computed integrals from batched results
            // indices: 4..7 correspond to Z (ee, em, me, mm)
            complex__float I_Zee = results_batch[4][j][i];
            complex__float I_Zem = results_batch[5][j][i];
            complex__float I_Zme = results_batch[6][j][i];
            complex__float I_Zmm = results_batch[7][j][i];
            
            // indices: 0..3 correspond to Y (ee, em, me, mm)
            complex__float I_Yee = results_batch[0][j][i];
            complex__float I_Yem = results_batch[1][j][i];
            complex__float I_Yme = results_batch[2][j][i];
            complex__float I_Ymm = results_batch[3][j][i];

            // Row 1 of A_M''
            // (Zee + 1/4 Ymm) | (Zem - 1/4 Yme) | (Zee - 1/4 Ymm) | (Zem + 1/4 Yme)
            add_val(off_J1, off_J1, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M1, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J1, off_J2, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J1, off_M2, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            // Row 2 of A_M''
            // (Zme - 1/4 Yem) | (Zmm + 1/4 Yee) | (Zme + 1/4 Yem) | (Zmm - 1/4 Yee)
            add_val(off_M1, off_J1, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M1, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M1, off_J2, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M1, off_M2, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));

            // Row 3 of A_M''
            // (Zee - 1/4 Ymm) | (Zem + 1/4 Yme) | (Zee + 1/4 Ymm) | (Zem - 1/4 Yme)
            add_val(off_J2, off_J1, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M1, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));
            add_val(off_J2, off_J2, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)));
            add_val(off_J2, off_M2, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)));

            // Row 4 of A_M''
            // (Zme + 1/4 Yem) | (Zmm - 1/4 Yee) | (Zme - 1/4 Yem) | (Zmm + 1/4 Yee)
            add_val(off_M2, off_J1, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M1, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
            add_val(off_M2, off_J2, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)));
            add_val(off_M2, off_M2, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)));
        }
    }
}

/**
 * @brief Assembles an eta0-scaled sparse mass matrix block for improved conditioning.
 *
 * This kernel matches assemble_Z_mass_kernel except for the final row/column scaling with
 * D = diag(1, eta0, 1, eta0), using eta0 = sqrt(mu0 / eps0).
 */
static __global__ void assemble_Z_mass_eta0_scaled_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    const complex__float* uniform_gstc_tensors,
    bool is_uniform,
    float mu0,
    float eps0,
    complex__float* global_Z_matrix,
    sie_mom::GstcBasisType basis_type = sie_mom::GstcBasisType::spherical)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    vec3_t u_hat, v_hat;
    get_local_basis_vectors(tri_v[0], tri_v[1], tri_v[2], basis_type, u_hat, v_hat);

    const complex__float* tensors_ptr = is_uniform ? uniform_gstc_tensors : &uniform_gstc_tensors[m * 32];

    auto extract_and_rotate_2x2 = [&](const complex__float* matrix_4x4_ptr, int row_block, int col_block, complex__float T_glob[3][3]) {
        int r0 = row_block * 2;
        int c0 = col_block * 2;

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

    const auto& face_edges = faces_edges[m];
    const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

    size_t off_J1 = 0, off_M1 = num_edges, off_J2 = 2 * num_edges, off_M2 = 3 * num_edges;
    const float eta0 = sqrtf(mu0 / eps0);
    const float eta0_sq = eta0 * eta0;

    for (int j = 0; j < 3; ++j) {
        const int global_test_edge_idx = local_edge_indices[j];
        for (int i = 0; i < 3; ++i) {
            const int global_source_edge_idx = local_edge_indices[i];

            auto add_val = [&](size_t row_off, size_t col_off, complex__float val, float scale) {
                size_t row = static_cast<size_t>(global_test_edge_idx) + row_off;
                size_t col = static_cast<size_t>(global_source_edge_idx) + col_off;
                size_t idx = columnMajorIndex(row, col, matrix_dim);
                atomicAddComplex(&global_Z_matrix[idx], complexScale(val, scale));
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

            add_val(off_J1, off_J1, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)), 1.f);
            add_val(off_J1, off_M1, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)), eta0);
            add_val(off_J1, off_J2, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)), 1.f);
            add_val(off_J1, off_M2, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)), eta0);

            add_val(off_M1, off_J1, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)), eta0);
            add_val(off_M1, off_M1, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)), eta0_sq);
            add_val(off_M1, off_J2, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)), eta0);
            add_val(off_M1, off_M2, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)), eta0_sq);

            add_val(off_J2, off_J1, complexSub(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)), 1.f);
            add_val(off_J2, off_M1, complexAdd(complexScale(I_Zem, f), complexScale(I_Yme, f_q)), eta0);
            add_val(off_J2, off_J2, complexAdd(complexScale(I_Zee, f), complexScale(I_Ymm, f_q)), 1.f);
            add_val(off_J2, off_M2, complexSub(complexScale(I_Zem, f), complexScale(I_Yme, f_q)), eta0);

            add_val(off_M2, off_J1, complexAdd(complexScale(I_Zme, f), complexScale(I_Yem, f_q)), eta0);
            add_val(off_M2, off_M1, complexSub(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)), eta0_sq);
            add_val(off_M2, off_J2, complexSub(complexScale(I_Zme, f), complexScale(I_Yem, f_q)), eta0);
            add_val(off_M2, off_M2, complexAdd(complexScale(I_Zmm, f), complexScale(I_Yee, f_q)), eta0_sq);
        }
    }
}





////////////////////////////////////////////////////////////////////////////
//////// EFIE Assembly with L
////////////////////////////////////////////////////////////////////////////

/**
 * @brief Assemble PEC EFIE dense operator on the exterior region only.
 *
 * The assembled matrix block is N x N (electric-current unknowns only):
 * Z = j*w*mu2*TL2
 */
static __global__ void assemble_pec_efie_operator_kernel(
    const vec3_t* __restrict__ vertices,
    const int3* __restrict__ faces,
    const int3* __restrict__ faces_edges,
    const float* __restrict__ quad_points,
    const float* __restrict__ quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    float k1,
    float k2,
    float mu_r2,
    float mu0,
    float omega,
    float singularity_threshold,
    complex__float* __restrict__ global_Z_matrix)
{
    const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces || n >= num_faces) {
        return;
    }

    const int3 face_m = faces[m];
    const int3 face_n = faces[n];

    const int test_tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};

    vec3_t test_tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

    const int3 tfe = faces_edges[m];
    const int3 sfe = faces_edges[n];
    const int test_edges[3] = {tfe.x, tfe.y, tfe.z};
    const int source_edges[3] = {sfe.x, sfe.y, sfe.z};

    complex__float result_matrix_tl_k1[3][3], result_matrix_tl_k2[3][3];
    complex__float result_matrix_tk_k1[3][3], result_matrix_tk_k2[3][3];

    const vec3_t test_centroid = (test_tri_v[0] + test_tri_v[1] + test_tri_v[2]) / 3.f;
    const vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    const float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices,
                                  source_tri_v, source_tri_indices,
                                  quad_points, quad_weights,
                                  order_num, k1, k2, true,
                                  result_matrix_tl_k1, result_matrix_tl_k2,
                                  result_matrix_tk_k1, result_matrix_tk_k2);

        complex__float singular_tl_k1[3][3], singular_tl_k2[3][3];
        complex__float singular_tk_k1[3][3], singular_tk_k2[3][3];
        calculateSingularPartQuadrature(test_tri_v, test_tri_indices,
                                        source_tri_v, source_tri_indices,
                                        quad_points, quad_weights,
                                        order_num, k1, k2,
                                        singular_tl_k1, singular_tl_k2,
                                        singular_tk_k1, singular_tk_k2);

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
                complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
                complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
                complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
            }
        }
    } else {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices,
                                  source_tri_v, source_tri_indices,
                                  quad_points, quad_weights,
                                  order_num, k1, k2, false,
                                  result_matrix_tl_k1, result_matrix_tl_k2,
                                  result_matrix_tk_k1, result_matrix_tk_k2);
    }

    const complex__float c_efie = make_complex(0.f, omega * mu_r2 * mu0);

    for (int j = 0; j < 3; ++j) {
        const int te = test_edges[j];
        for (int i = 0; i < 3; ++i) {
            const int se = source_edges[i];

            const complex__float tl2_term = c_efie * result_matrix_tl_k2[j][i];
            const complex__float z_val = tl2_term;

            atomicAddComplex(&global_Z_matrix[columnMajorIndex(
                                 static_cast<size_t>(te), static_cast<size_t>(se), matrix_dim)],
                             z_val);
        }
    }
}

/**
 * @brief Assemble the tested cross-identity matrix for PEC formulations.
 *
 * Computes I_cross[mj,ni] = <f_mj, n x f_ni> locally per face (m == n), and
 * adds scaled contributions into the N x N global matrix.
 *
 * WARNING: This operator is formulation-specific and can be numerically fragile.
 * Use only when the governing equation explicitly requires the tested n x term.
 * If unused in your active solver path, remove it to avoid future confusion.
 */
static __global__ void assemble_tested_cross_identity_kernel(
    const vec3_t* __restrict__ vertices,
    const int3* __restrict__ faces,
    const int3* __restrict__ faces_edges,
    const float* __restrict__ quad_points,
    const float* __restrict__ quad_weights,
    int order_num,
    size_t num_faces,
    size_t matrix_dim,
    float identity_scale,
    complex__float* __restrict__ global_Z_matrix)
{
    const size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_faces) {
        return;
    }

    if (identity_scale == 0.f) {
        return;
    }

    const int3 face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

    complex__float identity_matrix[3][3];
    calculateCrossIdentityQuadrature(tri_v, tri_indices,
                                     quad_points, quad_weights,
                                     order_num, identity_matrix);

    const int3 face_edge_ids = faces_edges[m];
    const int local_edge_indices[3] = {face_edge_ids.x, face_edge_ids.y, face_edge_ids.z};

    for (int j = 0; j < 3; ++j) {
        const int ge_test = local_edge_indices[j];
        for (int i = 0; i < 3; ++i) {
            const int ge_src = local_edge_indices[i];
            const complex__float contrib = complexScale(identity_matrix[j][i], identity_scale);
            atomicAddComplex(&global_Z_matrix[columnMajorIndex(
                                 static_cast<size_t>(ge_test),
                                 static_cast<size_t>(ge_src),
                                 matrix_dim)],
                             contrib);
        }
    }
}
