#include "catch.hpp"

#include "sie_mom/tests/testing_utils.cuh"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/cuda/mesh_geometry.cuh"
#include <cuda_runtime.h>

// A simple global kernel to launch the test
__global__ void testSingularQuadratureKernel(float* result)
{
    // 1. Define a simple test triangle (e.g., with area 0.5)
    const vec3_t test_tri[3] = {
        make_vec3(0.f, 0.f, 0.f), // v0
        make_vec3(1.f, 0.f, 0.f), // v1
        make_vec3(0.f, 1.f, 0.f)  // v2
    };
    const float triangle_area = 0.5f;

    // 2. Define a simple quadrature rule (e.g., 1-point rule at the centroid)
    const float quad_xi_eta[] = {1.f/3.f, 1.f/3.f}; // (xi, eta)
    const float quad_w[] = {1.f};
    const int num_quad_points = 1;

    // 3. Define the testing basis function details
    // Let's test with the basis function associated with edge 0 (v0-v1),
    // so the free vertex is v2 (index 2).
    const int test_free_vertex_idx = 2;

    // 4. Define the lambda function for the operator kernel
    // This lambda will simulate a very simple operator.
    // Let's say the vector part is a constant <1, 0, 0> and the scalar part is 2.0.
    auto simple_kernel = [=] __device__ (const vec3_t& r_obs) -> IntegrandComponents {
        IntegrandComponents components;
        components.vector_part = make_vec3(1.f, 0.f, 0.f);
        components.scalar_part = 2.f;
        return components;
    };

    // 5. Call the function under test
    float integral_value = calculateSingularPartQuadrature(
        test_tri,
        test_free_vertex_idx,
        quad_xi_eta,
        quad_w,
        num_quad_points,
        simple_kernel
    );

    // The result is scaled by the triangle area in the full integral.
    *result = integral_value * triangle_area;
}


TEST_CASE("Test calculateSingularPartQuadrature with Lambda", "[CUDA][Quadrature]")
{
    // Allocate memory on the GPU for the result
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // Launch the kernel
    testSingularQuadratureKernel<<<1, 1>>>(d_result);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to the host
    float h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_result));

    // --- Calculate Expected Result ---
    // The test uses a 1-point quadrature rule at the centroid (1/3, 1/3, 1/3).
    // Quadrature point location: r_mp = (1/3)*v0 + (1/3)*v1 + (1/3)*v2 = (1/3, 1/3, 0)
    // Free vertex: r_free = v2 = (0, 1, 0)
    // Testing basis function: rho_mj = r_mp - r_free = (1/3, -2/3, 0)
    //
    // Integrand components from lambda:
    // vector_part = <1, 0, 0>
    // scalar_part = 2.0
    //
    // Dot product: dot(rho_mj, vector_part) = (1/3)*1 + (-2/3)*0 + 0*0 = 1/3
    // Full integrand value at quad point: dot(...) + scalar_part = 1/3 + 2.0 = 7/3
    //
    // Quadrature sum: Sum(w_p * integrand_value) = 1.0 * (7/3) = 7/3
    // Final result (scaled by area): (7/3) * 0.5 = 7/6
    const float expected_result = 7.f / 6.f;

    REQUIRE(h_result == Approx(expected_result));
}
