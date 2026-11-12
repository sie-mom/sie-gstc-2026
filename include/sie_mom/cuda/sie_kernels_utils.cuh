#pragma once

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"

// Diagnostic helper: each of the three local results is a complex 3-vector.
static __device__ inline bool has_nonfinite_scatter_result(const cvec3_t results[3])
{
    for (int i = 0; i < 3; ++i) {
        const cvec3_t& v = results[i];
        if (!isfinite(v.x.x) || !isfinite(v.x.y) ||
            !isfinite(v.y.x) || !isfinite(v.y.y) ||
            !isfinite(v.z.x) || !isfinite(v.z.y))
            return true;
    }
    return false;
}

static __global__ void compute_scattered_fields_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    int num_faces,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    const vec3_t* obs_points,
    size_t num_obs_points,
    const complex__float* J_currents,
    const complex__float* M_currents,
    complex__float coeff_J,
    complex__float coeff_M,
    complex__float coeff_J_h,
    complex__float coeff_M_h,
    real_t k,
    cvec3_t* E_scat_out,
    cvec3_t* H_scat_out)
{
    // One block handles one observation point; its threads share the face work.
    size_t obs_idx = blockIdx.x;
    if (obs_idx >= num_obs_points) {
        return;
    }

    vec3_t obs_point = obs_points[obs_idx];
    const complex__float zero = make_complex(real_t(0), real_t(0));
    cvec3_t thread_E = make_cvec3(zero, zero, zero);
    cvec3_t thread_H = make_cvec3(zero, zero, zero);

    // Thread t visits faces t, t + blockDim.x, t + 2*blockDim.x, ...
    // Each thread keeps its field sum private until all faces are processed.
    for (int face_idx = threadIdx.x; face_idx < num_faces; face_idx += blockDim.x) {
        const int3 face = faces[face_idx];
        const int3 face_edges = faces_edges[face_idx];
        const int source_tri_indices[3] = {face.x, face.y, face.z};
        vec3_t source_tri_v[3] = {vertices[face.x], vertices[face.y], vertices[face.z]};

        const vec3_t source_centroid = (source_tri_v[0] + source_tri_v[1] + source_tri_v[2]) / 3.f;
        const float dist = length(obs_point - source_centroid);
        
        // This face independently selects direct quadrature or regular + singular
        // integration using its centroid distance from the observation point.
        // Use 0.1 * lambda as the threshold.
        const float lambda = 2.0f * static_cast<float>(M_PI) / k;
        const float singularity_threshold = 0.1f * lambda;

        cvec3_t L_results[3];
        cvec3_t K_results[3];
        
        if (dist < singularity_threshold) {
            calculateScatterFieldQuadrature(
                obs_point, source_tri_v, source_tri_indices,
                quad_points, quad_weights, order_num, k, true, // is_regular = true
                L_results, K_results);

            cvec3_t L_singular[3];
            cvec3_t K_singular[3];
            calculateScatterFieldSingular(
                obs_point, source_tri_v, source_tri_indices, k,
                L_singular, K_singular);

            for (int i = 0; i < 3; ++i) {
                L_results[i] += L_singular[i];
                K_results[i] += K_singular[i];
            }

        } else {
            calculateScatterFieldQuadrature(
                obs_point, source_tri_v, source_tri_indices,
                quad_points, quad_weights, order_num, k, false, // is_regular = false
                L_results, K_results);
        }

        const int edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};
        for (int i = 0; i < 3; ++i) {
            const int edge_idx = edge_indices[i];
            const complex__float J_edge = J_currents[edge_idx];
            const complex__float M_edge = M_currents[edge_idx];

            const complex__float weight_E_J = complexMul(coeff_J, J_edge);
            const complex__float weight_E_M = complexMul(coeff_M, M_edge);
            const complex__float weight_H_J = complexMul(coeff_J_h, J_edge);
            const complex__float weight_H_M = complexMul(coeff_M_h, M_edge);

            thread_E += weight_E_J * L_results[i];
            thread_E += weight_E_M * K_results[i];

            thread_H += weight_H_J * K_results[i];
            thread_H += weight_H_M * L_results[i];
        }
    }

    // Dynamic shared memory holds one E and one H partial sum per thread.
    // The first blockDim.x entries are E; the next blockDim.x entries are H.
    extern __shared__ cvec3_t shared_data[];
    cvec3_t* shared_E = shared_data;
    cvec3_t* shared_H = shared_data + blockDim.x;
    shared_E[threadIdx.x] = thread_E;
    shared_H[threadIdx.x] = thread_H;
    __syncthreads();

    // Halve the active partial sums each round until entry 0 holds the total.
    // All threads reach each barrier, including those not adding this round.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_E[threadIdx.x] += shared_E[threadIdx.x + stride];
            shared_H[threadIdx.x] += shared_H[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Only thread 0 writes this block's completed scattered field.
    if (threadIdx.x == 0) {
        E_scat_out[obs_idx] = shared_E[0];
        H_scat_out[obs_idx] = shared_H[0];
    }
}

template <TestingFunction TestFunc = TestingFunction::RWG, typename ExcitationSource>
__global__ void assemble_V_vector_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    ExcitationSource source,
    int source_region,
    complex__float* global_V_vector)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    const auto& face_edges_m = faces_edges[m];
    const int edge_indices[3] = {face_edges_m.x, face_edges_m.y, face_edges_m.z};

    // V_E and V_H results for the 3 basis functions on this face
    complex__float V_E[3], V_H[3];
    calculateExcitationQuadrature<TestFunc>(tri_v, tri_indices, quad_points, quad_weights, order_num, source, V_E, V_H);

    // Atomically add the contributions to the global V vector based on the source region
    for (int i = 0; i < 3; ++i) {
        int edge_idx = edge_indices[i];
        if (source_region == 1) {
            atomicAddComplex(&global_V_vector[edge_idx + 0 * num_edges], V_E[i]);
            atomicAddComplex(&global_V_vector[edge_idx + 1 * num_edges], V_H[i]);
        } else if (source_region == 2) {
            atomicAddComplex(&global_V_vector[edge_idx + 2 * num_edges], V_E[i]);
            atomicAddComplex(&global_V_vector[edge_idx + 3 * num_edges], V_H[i]);
        }
    }
}

template <typename ExcitationSource>
__global__ void assemble_V_vector_eta0_scaled_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    ExcitationSource source,
    int source_region,
    float mu0,
    float eps0,
    complex__float* global_V_vector)
{
    size_t m = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= num_faces) {
        return;
    }

    const auto& face_m = faces[m];
    const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
    vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};
    const auto& face_edges_m = faces_edges[m];
    const int edge_indices[3] = {face_edges_m.x, face_edges_m.y, face_edges_m.z};

    // V_E and V_H results for the 3 basis functions on this face
    complex__float V_E[3], V_H[3];
    calculateExcitationQuadrature(tri_v, tri_indices, quad_points, quad_weights, order_num, source, V_E, V_H);

    // Compute eta0 = sqrt(mu0 / eps0) for scaling V_H (M-row components)
    real_t eta0 = sqrt(static_cast<real_t>(mu0) / static_cast<real_t>(eps0));

    // Atomically add the contributions to the global V vector with row scaling
    // V_E (J rows): scale by 1
    // V_H (M rows): scale by eta0
    for (int i = 0; i < 3; ++i) {
        int edge_idx = edge_indices[i];
        if (source_region == 1) {
            atomicAddComplex(&global_V_vector[edge_idx + 0 * num_edges], V_E[i]);                           // J rows: unscaled
            atomicAddComplex(&global_V_vector[edge_idx + 1 * num_edges], complexScale(V_H[i], eta0));        // M rows: scaled by eta0
        } else if (source_region == 2) {
            atomicAddComplex(&global_V_vector[edge_idx + 2 * num_edges], V_E[i]);                           // J rows: unscaled
            atomicAddComplex(&global_V_vector[edge_idx + 3 * num_edges], complexScale(V_H[i], eta0));        // M rows: scaled by eta0
        }
    }
}

template <typename ExcitationSource>
__global__ void compute_incident_field_kernel(
    const vec3_t* obs_points,
    size_t num_obs_points,
    ExcitationSource source,
    cvec3_t* E_inc_out,
    cvec3_t* H_inc_out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_obs_points) {
        return;
    }

    cvec3_t E_inc, H_inc;
    source(obs_points[idx], E_inc, H_inc);
    E_inc_out[idx] = E_inc;
    H_inc_out[idx] = H_inc;
}
