#include "sie_mom/solvers/dielectric/dielectric_kernels.cuh"
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/sie_scaling.cuh"
#include "sie_mom/cuda/sie_kernels_utils.cuh"

namespace sie_mom {
namespace solvers {
namespace dielectric {

__host__ __device__ static inline size_t columnMajorIndex(size_t row, size_t col, size_t leading_dim) {
    return col * leading_dim + row;
}

////////////////////////////////////////////////////////////////////////////
//////// T-PMCHWT (PMCHWT-II) Kernel Assembly
////////////////////////////////////////////////////////////////////////////
/**
 * @brief Assemble the PMCHWT kernel with standard TL and TK operators.
 */
static __global__ void t_pmchwt_kernel(
    const vec3_t* __restrict__ vertices,
    const int3*   __restrict__ faces,
    const int3*   __restrict__ faces_edges,
    const float*  __restrict__ quad_points,
    const float*  __restrict__ quad_weights,
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
    complex__float* __restrict__ global_Z_matrix)
{
    const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces || n >= num_faces) return;

    // --- Fetch face indices ---
    const int3 face_m = faces[m];
    const int3 face_n = faces[n];

    const int test_tri_indices[3]   = {face_m.x, face_m.y, face_m.z};
    const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};

    vec3_t test_tri_v[3]   = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

    // --- Build local edge index arrays once ---
    const int3 tfe = faces_edges[m];
    const int3 sfe = faces_edges[n];
    const int test_edges[3]   = {tfe.x, tfe.y, tfe.z};
    const int source_edges[3] = {sfe.x, sfe.y, sfe.z};

    // --- Quadrature results ---
    complex__float result_matrix_tl_k1[3][3], result_matrix_tl_k2[3][3];
    complex__float result_matrix_tk_k1[3][3], result_matrix_tk_k2[3][3];

    // Centroid distance
    const vec3_t test_centroid   = (test_tri_v[0]   + test_tri_v[1]   + test_tri_v[2])   / 3.f;
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

        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_tl_k1[j][i], singular_tl_k1[j][i]);
            complexAddInPlace(result_matrix_tl_k2[j][i], singular_tl_k2[j][i]);
            complexAddInPlace(result_matrix_tk_k1[j][i], singular_tk_k1[j][i]);
            complexAddInPlace(result_matrix_tk_k2[j][i], singular_tk_k2[j][i]);
        }
    } else {
        calculateDoubleQuadrature(test_tri_v, test_tri_indices,
                                  source_tri_v, source_tri_indices,
                                  quad_points, quad_weights,
                                  order_num, k1, k2, false,
                                  result_matrix_tl_k1, result_matrix_tl_k2,
                                  result_matrix_tk_k1, result_matrix_tk_k2);
    }

    // --- Hoist invariant physics scalars ---
    const complex__float j_omega = make_complex(0.f, omega);
    const float mu1  = mu_r1 * mu0;
    const float mu2  = mu_r2 * mu0;
    const float eps1 = eps_r1 * eps0;
    const float eps2 = eps_r2 * eps0;

    const float eta0 = sqrtf(mu0 / eps0);

    const complex__float c_mu1  = j_omega * mu1;
    const complex__float c_mu2  = j_omega * mu2;
    const complex__float c_eps1 = j_omega * eps1;
    const complex__float c_eps2 = j_omega * eps2;

    auto add_to_global = [&](size_t row, size_t col, complex__float val) {
        atomicAddComplex(&global_Z_matrix[columnMajorIndex(row, col, matrix_dim)], val);
    };

    // --- Atomically add results: PMCHWT 2x2 block structure ---
    for (int j = 0; j < 3; ++j) {
        const int te = test_edges[j];
        for (int i = 0; i < 3; ++i) {
            const int se = source_edges[i];

            const complex__float tl1 = result_matrix_tl_k1[j][i];
            const complex__float tl2 = result_matrix_tl_k2[j][i];
            const complex__float tk  = result_matrix_tk_k1[j][i] + result_matrix_tk_k2[j][i];

            const complex__float z_jj = c_mu1  * tl1 + c_mu2  * tl2;
            const complex__float z_mm = c_eps1 * tl1 + c_eps2 * tl2;
            const complex__float z_mj = -1.f * tk;
            const complex__float z_jm =  1.f * tk;  // = -z_mj

            // --- Apply eta0 scaling before assembly ---
            const complex__float z_jm_scaled = complexScale(z_jm, eta0);
            const complex__float z_mj_scaled = complexScale(z_mj, eta0);
            const complex__float z_mm_scaled = complexScale(z_mm, eta0 * eta0);

            add_to_global(te,             se,             z_jj);
            add_to_global(te,             se + num_edges, z_jm_scaled);
            add_to_global(te + num_edges, se,             z_mj_scaled);
            add_to_global(te + num_edges, se + num_edges, z_mm_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////
//////// N-PMCHWT Kernel Assembly
////////////////////////////////////////////////////////////////////////////
/**
 * @brief Assemble the N-PMCHWT kernel with nTL and nTK operators.
 */
static __global__ void n_pmchwt_kernel(
    const vec3_t* __restrict__ vertices,
    const int3*   __restrict__ faces,
    const int3*   __restrict__ faces_edges,
    const float*  __restrict__ quad_points,
    const float*  __restrict__ quad_weights,
    int order_num,
    const float*  __restrict__ line_quad_points,
    const float*  __restrict__ line_quad_weights,
    int line_order_num,
    bool use_form_a,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    float k1, float k2,
    float mu_r1, float mu_r2,
    float eps_r1, float eps_r2,
    float mu0, float eps0,
    float omega,
    float singularity_threshold,
    complex__float* __restrict__ global_Z_matrix)
{
    const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces || n >= num_faces) return;

    // --- Fetch face indices ---
    const int3 face_m = faces[m];
    const int3 face_n = faces[n];

    const int test_tri_indices[3]   = {face_m.x, face_m.y, face_m.z};
    const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};

    vec3_t test_tri_v[3]   = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

    // --- Build local edge index arrays once ---
    const int3 tfe = faces_edges[m];
    const int3 sfe = faces_edges[n];
    const int test_edges[3]   = {tfe.x, tfe.y, tfe.z};
    const int source_edges[3] = {sfe.x, sfe.y, sfe.z};

    // --- Quadrature results ---
    complex__float result_matrix_ntl_k1[3][3], result_matrix_ntl_k2[3][3];
    complex__float result_matrix_ntk_k1[3][3], result_matrix_ntk_k2[3][3];

    // Centroid distance
    const vec3_t test_centroid   = (test_tri_v[0]   + test_tri_v[1]   + test_tri_v[2])   / 3.f;
    const vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    const float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        if (use_form_a) {
            calculateFormAQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                             source_tri_v, source_tri_indices,
                                             quad_points, quad_weights, order_num,
                                             line_quad_points, line_quad_weights, line_order_num,
                                             k1, k2, true,
                                             result_matrix_ntl_k1, result_matrix_ntl_k2,
                                             result_matrix_ntk_k1, result_matrix_ntk_k2);
        } else {
            calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                              source_tri_v, source_tri_indices,
                                              quad_points, quad_weights,
                                              order_num, k1, k2, true,
                                              result_matrix_ntl_k1, result_matrix_ntl_k2,
                                              result_matrix_ntk_k1, result_matrix_ntk_k2);
        }

        complex__float singular_ntl_k1[3][3], singular_ntl_k2[3][3];
        complex__float singular_ntk_k1[3][3], singular_ntk_k2[3][3];
        if (use_form_a) {
            calculateFormASingularPartQuadrature_nTL_nTK(
                test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                quad_points, quad_weights, order_num,
                line_quad_points, line_quad_weights, line_order_num,
                k1, k2, singular_ntl_k1, singular_ntl_k2,
                singular_ntk_k1, singular_ntk_k2);
        } else {
            calculateSingularPartQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                                    source_tri_v, source_tri_indices,
                                                    quad_points, quad_weights,
                                                    order_num, k1, k2,
                                                    singular_ntl_k1, singular_ntl_k2,
                                                    singular_ntk_k1, singular_ntk_k2);
        }

        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_ntl_k1[j][i], singular_ntl_k1[j][i]);
            complexAddInPlace(result_matrix_ntl_k2[j][i], singular_ntl_k2[j][i]);
            complexAddInPlace(result_matrix_ntk_k1[j][i], singular_ntk_k1[j][i]);
            complexAddInPlace(result_matrix_ntk_k2[j][i], singular_ntk_k2[j][i]);
        }
    } else {
        if (use_form_a) {
            calculateFormAQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                             source_tri_v, source_tri_indices,
                                             quad_points, quad_weights, order_num,
                                             line_quad_points, line_quad_weights, line_order_num,
                                             k1, k2, false,
                                             result_matrix_ntl_k1, result_matrix_ntl_k2,
                                             result_matrix_ntk_k1, result_matrix_ntk_k2);
        } else {
            calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                              source_tri_v, source_tri_indices,
                                              quad_points, quad_weights,
                                              order_num, k1, k2, false,
                                              result_matrix_ntl_k1, result_matrix_ntl_k2,
                                              result_matrix_ntk_k1, result_matrix_ntk_k2);
        }
    }

    // --- Hoist invariant physics scalars ---
    const complex__float j_omega = make_complex(0.f, omega);
    const float mu1  = mu_r1 * mu0;
    const float mu2  = mu_r2 * mu0;
    const float eps1 = eps_r1 * eps0;
    const float eps2 = eps_r2 * eps0;

    const float eta0 = sqrtf(mu0 / eps0);

    const complex__float c_mu1  = j_omega * mu1;
    const complex__float c_mu2  = j_omega * mu2;
    const complex__float c_eps1 = j_omega * eps1;
    const complex__float c_eps2 = j_omega * eps2;

    auto add_to_global = [&](size_t row, size_t col, complex__float val) {
        atomicAddComplex(&global_Z_matrix[columnMajorIndex(row, col, matrix_dim)], val);
    };

    // --- Atomically add results: PMCHWT 2x2 block structure using nTL/nTK ---
    for (int j = 0; j < 3; ++j) {
        const int te = test_edges[j];
        for (int i = 0; i < 3; ++i) {
            const int se = source_edges[i];

            const complex__float ntl1 = result_matrix_ntl_k1[j][i];
            const complex__float ntl2 = result_matrix_ntl_k2[j][i];
            const complex__float ntk  = result_matrix_ntk_k1[j][i] + result_matrix_ntk_k2[j][i];

            const complex__float z_jj = c_mu1  * ntl1 + c_mu2  * ntl2;
            const complex__float z_mm = c_eps1 * ntl1 + c_eps2 * ntl2;
            const complex__float z_mj = -1.f * ntk;
            const complex__float z_jm =  1.f * ntk;  // = -z_mj

            // --- Apply eta0 scaling before assembly ---
            const complex__float z_jm_scaled = complexScale(z_jm, eta0);
            const complex__float z_mj_scaled = complexScale(z_mj, eta0);
            const complex__float z_mm_scaled = complexScale(z_mm, eta0 * eta0);

            add_to_global(te,             se,             z_jj);
            add_to_global(te,             se + num_edges, z_jm_scaled);
            add_to_global(te + num_edges, se,             z_mj_scaled);
            add_to_global(te + num_edges, se + num_edges, z_mm_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////
//////// N-Müller Kernel Assembly
////////////////////////////////////////////////////////////////////////////
/**
 * @brief Assemble the N-Müller kernel with the same line and surface quadrature arguments.
 */
static __global__ void n_muller_kernel(
    const vec3_t* __restrict__ vertices,
    const int3*   __restrict__ faces,
    const int3*   __restrict__ faces_edges,
    const float*  __restrict__ quad_points,
    const float*  __restrict__ quad_weights,
    int order_num,
    const float*  __restrict__ line_quad_points,
    const float*  __restrict__ line_quad_weights,
    int line_order_num,
    bool use_form_a,
    size_t num_faces,
    size_t num_edges,
    size_t matrix_dim,
    float k1, float k2,
    float mu_r1, float mu_r2,
    float eps_r1, float eps_r2,
    float mu0, float eps0,
    float omega,
    float singularity_threshold,
    complex__float* __restrict__ global_Z_matrix)
{
    const size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces || n >= num_faces) return;

    // --- Fetch face indices ---
    const int3 face_m = faces[m];
    const int3 face_n = faces[n];

    const int test_tri_indices[3]   = {face_m.x, face_m.y, face_m.z};
    const int source_tri_indices[3] = {face_n.x, face_n.y, face_n.z};

    vec3_t test_tri_v[3]   = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    vec3_t source_tri_v[3] = {vertices[face_n.x], vertices[face_n.y], vertices[face_n.z]};

    // --- Build local edge index arrays once ---
    const int3 tfe = faces_edges[m];
    const int3 sfe = faces_edges[n];
    const int test_edges[3]   = {tfe.x, tfe.y, tfe.z};
    const int source_edges[3] = {sfe.x, sfe.y, sfe.z};

    // --- Quadrature results ---
    complex__float result_matrix_ntl_k1[3][3], result_matrix_ntl_k2[3][3];
    complex__float result_matrix_ntk_k1[3][3], result_matrix_ntk_k2[3][3];

    // Centroid distance
    const vec3_t test_centroid   = (test_tri_v[0]   + test_tri_v[1]   + test_tri_v[2])   / 3.f;
    const vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
    const float dist = length(test_centroid - source_centroid);

    if (dist < singularity_threshold) {
        if (use_form_a) {
            calculateFormAQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                             source_tri_v, source_tri_indices,
                                             quad_points, quad_weights, order_num,
                                             line_quad_points, line_quad_weights, line_order_num,
                                             k1, k2, true,
                                             result_matrix_ntl_k1, result_matrix_ntl_k2,
                                             result_matrix_ntk_k1, result_matrix_ntk_k2);
        } else {
            calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                              source_tri_v, source_tri_indices,
                                              quad_points, quad_weights,
                                              order_num, k1, k2, true,
                                              result_matrix_ntl_k1, result_matrix_ntl_k2,
                                              result_matrix_ntk_k1, result_matrix_ntk_k2);
        }

        complex__float singular_ntl_k1[3][3], singular_ntl_k2[3][3];
        complex__float singular_ntk_k1[3][3], singular_ntk_k2[3][3];
        if (use_form_a) {
            calculateFormASingularPartQuadrature_nTL_nTK(
                test_tri_v, test_tri_indices, source_tri_v, source_tri_indices,
                quad_points, quad_weights, order_num,
                line_quad_points, line_quad_weights, line_order_num,
                k1, k2, singular_ntl_k1, singular_ntl_k2,
                singular_ntk_k1, singular_ntk_k2);
        } else {
            calculateSingularPartQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                                    source_tri_v, source_tri_indices,
                                                    quad_points, quad_weights,
                                                    order_num, k1, k2,
                                                    singular_ntl_k1, singular_ntl_k2,
                                                    singular_ntk_k1, singular_ntk_k2);
        }

        for (int j = 0; j < 3; ++j) for (int i = 0; i < 3; ++i) {
            complexAddInPlace(result_matrix_ntl_k1[j][i], singular_ntl_k1[j][i]);
            complexAddInPlace(result_matrix_ntl_k2[j][i], singular_ntl_k2[j][i]);
            complexAddInPlace(result_matrix_ntk_k1[j][i], singular_ntk_k1[j][i]);
            complexAddInPlace(result_matrix_ntk_k2[j][i], singular_ntk_k2[j][i]);
        }
    } else {
        if (use_form_a) {
            calculateFormAQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                             source_tri_v, source_tri_indices,
                                             quad_points, quad_weights, order_num,
                                             line_quad_points, line_quad_weights, line_order_num,
                                             k1, k2, false,
                                             result_matrix_ntl_k1, result_matrix_ntl_k2,
                                             result_matrix_ntk_k1, result_matrix_ntk_k2);
        } else {
            calculateDoubleQuadrature_nTL_nTK(test_tri_v, test_tri_indices,
                                              source_tri_v, source_tri_indices,
                                              quad_points, quad_weights,
                                              order_num, k1, k2, false,
                                              result_matrix_ntl_k1, result_matrix_ntl_k2,
                                              result_matrix_ntk_k1, result_matrix_ntk_k2);
        }
    }

    // --- Mass Matrix for Jump Terms ---
    complex__float mass_matrix[3][3];
    if (m == n) {
        calculateMassMatrixQuadrature(test_tri_v, test_tri_indices, quad_points, quad_weights, order_num, mass_matrix);
    }

    // --- Hoist invariant physics scalars ---
    const complex__float j_omega = make_complex(0.f, omega);
    const float mu1  = mu_r1 * mu0;
    const float mu2  = mu_r2 * mu0;
    const float eps1 = eps_r1 * eps0;
    const float eps2 = eps_r2 * eps0;

    const float eta0 = sqrtf(mu0 / eps0);

    const complex__float c_eps1 = j_omega * eps1;
    const complex__float c_eps2 = j_omega * eps2;

    auto add_to_global = [&](size_t row, size_t col, complex__float val) {
        atomicAddComplex(&global_Z_matrix[columnMajorIndex(row, col, matrix_dim)], val);
    };

    // --- Atomically add results: N-Müller 2x2 block structure using nTL/nTK and Gram matrices ---
    for (int j = 0; j < 3; ++j) {
        const int te = test_edges[j];
        for (int i = 0; i < 3; ++i) {
            const int se = source_edges[i];

            const complex__float ntl1 = result_matrix_ntl_k1[j][i];
            const complex__float ntl2 = result_matrix_ntl_k2[j][i];
            const complex__float ntk1 = result_matrix_ntk_k1[j][i];
            const complex__float ntk2 = result_matrix_ntk_k2[j][i];

            // Z_JJ block: j*omega*eps2*mu2*nTL2 - j*omega*eps1*mu1*nTL1
            const complex__float z_jj = (c_eps2 * mu2) * ntl2 - (c_eps1 * mu1) * ntl1;
            
            // Z_MM block: -(j*omega*eps2*mu2*nTL2 - j*omega*eps1*mu1*nTL1) = -Z_JJ
            complex__float z_mm = complexScale(-1.0f, z_jj);
            
            // Z_JM block: eps2*nTK2 - eps1*nTK1
            complex__float z_jm = eps2 * ntk2 - eps1 * ntk1;
            
            // Z_MJ block: mu2*nTK2 - mu1*nTK1
            complex__float z_mj = mu2 * ntk2 - mu1 * ntk1;

            if (m == n) {
                const complex__float g = mass_matrix[j][i];
                const float eps_factor = (eps1 + eps2) / 2.0f;
                const float mu_factor  = (mu1 + mu2) / 2.0f;
                
                // Subtract Gram matrix contribution (minus sign derived from n x (n x M) = -M testing)
                complex__float g_eps = complexScale(eps_factor, g);
                complex__float g_mu  = complexScale(mu_factor, g);
                
                z_jm = z_jm - g_eps;
                z_mj = z_mj - g_mu;
            }

            // --- Apply eta0 scaling before assembly ---
            const complex__float z_jm_scaled = complexScale(z_jm, eta0);
            const complex__float z_mj_scaled = complexScale(z_mj, eta0);
            const complex__float z_mm_scaled = complexScale(z_mm, eta0 * eta0);

            add_to_global(te,             se,             z_jj);
            add_to_global(te,             se + num_edges, z_jm_scaled);
            add_to_global(te + num_edges, se,             z_mj_scaled);
            add_to_global(te + num_edges, se + num_edges, z_mm_scaled);
        }
    }
}

void launch_t_pmchwt_kernel(
    dim3 blocks, dim3 threads,
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
    t_pmchwt_kernel<<<blocks, threads>>>(
        vertices, faces, faces_edges, quad_points, quad_weights, order_num,
        num_faces, num_edges, matrix_dim,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega,
        singularity_threshold, global_Z_matrix
    );
}

void launch_n_pmchwt_kernel(
    dim3 blocks, dim3 threads,
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    const float* line_quad_points,
    const float* line_quad_weights,
    int line_order_num,
    bool use_form_a,
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
    n_pmchwt_kernel<<<blocks, threads>>>(
        vertices, faces, faces_edges, quad_points, quad_weights, order_num,
        line_quad_points, line_quad_weights, line_order_num, use_form_a,
        num_faces, num_edges, matrix_dim,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega,
        singularity_threshold, global_Z_matrix
    );
}

void launch_n_muller_kernel(
    dim3 blocks, dim3 threads,
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    const float* line_quad_points,
    const float* line_quad_weights,
    int line_order_num,
    bool use_form_a,
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
    n_muller_kernel<<<blocks, threads>>>(
        vertices, faces, faces_edges, quad_points, quad_weights, order_num,
        line_quad_points, line_quad_weights, line_order_num, use_form_a,
        num_faces, num_edges, matrix_dim,
        k1, k2, mu_r1, mu_r2, eps_r1, eps_r2, mu0, eps0, omega,
        singularity_threshold, global_Z_matrix
    );
}

} // namespace dielectric
} // namespace solvers
} // namespace sie_mom
