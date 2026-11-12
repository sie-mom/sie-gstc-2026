#pragma once

#include "sie_mom/cuda/platform.cuh"
#include <cstddef>
#include <cuda_runtime.h>

namespace sie_mom {
namespace solvers {
namespace dielectric {

/**
 * @brief Host launcher for the Tangential PMCHWT (T-PMCHWT) assembly kernel.
 */
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
    complex__float* global_Z_matrix);

/**
 * @brief Host launcher for the Normal PMCHWT (N-PMCHWT) assembly kernel.
 */
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
    complex__float* global_Z_matrix);

/**
 * @brief Host launcher for the Normal Müller (N-Müller) assembly kernel.
 */
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
    complex__float* global_Z_matrix);

} // namespace dielectric
} // namespace solvers
} // namespace sie_mom
