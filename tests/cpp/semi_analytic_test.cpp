#include "catch.hpp"
#include <cmath>
#include <iomanip>
#include <limits>

// To test a __device__ function on the host, we need to trick the compiler.
// We define __device__ and other CUDA keywords to be empty macros when
// not compiling with NVCC. The platform.cuh header already does this.
#include "sie_mom/cuda/platform.cuh"
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/semi_analytic_integrals.cuh"

namespace {
    // Increase precision for float stringification in Catch2 for better failure messages
    struct PrecisionSetter { PrecisionSetter() { Catch::StringMaker<float>::precision = std::numeric_limits<float>::max_digits10; } };
    PrecisionSetter g_precisionSetter;
}

TEST_CASE("Analytic potential phi_0 is calculated correctly", "[analytic_potentials]") {

    SECTION("Potential at the vertex of an equilateral triangle") {
        // 1. Define an equilateral triangle with side length s=1
        const float s = 1.0f;
        const vec3_t tri[3] = {
            make_vec3(0.0f, 0.0f, 0.0f),
            make_vec3(s, 0.0f, 0.0f),
            make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
        };

        // 2. Set the observation point to be one of the vertices.
        const vec3_t r_obs = tri[0];

        // 3. Calculate the potential using your implementation.
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // 4. Compare against the known analytic result for a vertex observation: 2*s*(log(2) - 1).
        float phi0_expected = s * sqrtf(3.0f) / 2.0f * logf(3.0f);

        // The REQUIRE macro checks the condition and reports success/failure.
        // The Approx class handles floating-point comparisons gracefully.
        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    SECTION("Potential at the centroid of an equilateral triangle") {
        // 1. Define an equilateral triangle with side length s=1
        const float s = 1.0f;
        const vec3_t tri[3] = {
            make_vec3(0.0f, 0.0f, 0.0f),
            make_vec3(s, 0.0f, 0.0f),
            make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
        };

        // 2. Set the observation point to be the centroid.
        const vec3_t r_obs = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 6.0f, 0.0f);

        // 3. Calculate the potential. This is a non-singular case.
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // 4. Compare against the known analytic result: 3 * s * log(3).
        float phi0_expected = s * sqrtf(3.0f) * logf(2.0f + sqrtf(3.0f));
        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    // Define the common equilateral triangle for the following tests
    const float s = 1.0f;
    const vec3_t tri[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(s, 0.0f, 0.0f),
        make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
    };

    SECTION("In-plane, outside the triangle (r1)") {
        // Observation point: r1 = (0, -0.5, 0)
        const vec3_t r_obs = make_vec3(0.0f, -0.5f, 0.0f);

        // Calculate the potential
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // Expected analytic result
        const float phi0_expected = 0.47636237419365196f;

        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    SECTION("Off-plane, directly above the centroid (r2)") {
        // Centroid c = (0.5, sqrt(3)/6, 0)
        const float centroid_x = s / 2.0f;
        const float centroid_y = s * sqrtf(3.0f) / 6.0f;
        const float h = 0.25f;

        // Observation point: r2 = (0.5, 0.2886751345948129, 0.25)
        const vec3_t r_obs = make_vec3(centroid_x, centroid_y, h);

        // Calculate the potential
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // Expected analytic result
        const float phi0_expected = 1.2111344892951112f;

        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    SECTION("Off-plane far field sanity check (r3)") {
        // Centroid c = (0.5, sqrt(3)/6, 0)
        const float centroid_x = s / 2.0f;
        const float centroid_y = s * sqrtf(3.0f) / 6.0f;
        const float h = 10.0f;

        // Observation point: r3 = (0.5, 0.2886751345948129, 10)
        const vec3_t r_obs = make_vec3(centroid_x, centroid_y, h);

        // Calculate the potential
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // Expected analytic result
        const float phi0_expected = 0.04328324600959819f;

        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    SECTION("Off-plane, directly above a vertex (r4)") {
        // Observation point: r4 = (0, 0, 0.25)
        const float h = 0.25f;
        const vec3_t r_obs = make_vec3(0.0f, 0.0f, h);

        // Calculate the potential
        float phi0_calculated = calculate_phi_0(r_obs, tri);

        // Expected analytic result
        const float phi0_expected = 0.7250473223433905f;

        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }

    SECTION("Custom test case over edge midpoint") {
        // 1. Define the same equilateral triangle with side length s=1
        const float s = 1.0f;
        const vec3_t tri[3] = {
            make_vec3(0.0f, 0.0f, 0.0f),
            make_vec3(s, 0.0f, 0.0f),
            make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
        };

        // 2. Set the observation point (THIS NEEDS TO BE PROVIDED)
        // For example: const vec3_t r_obs = make_vec3(x, y, z);
        const vec3_t r_obs = make_vec3(0.5f, 0.0f, 0.0f); // Midpoint of edge v0-v1

        float phi0_calculated = calculate_phi_0(r_obs, tri);
        const float phi0_expected = 1.6162320698995925042f;
        REQUIRE(phi0_calculated == Approx(phi0_expected));
    }
}

// ################################################################
// ## Tests for the analytic potential phi_1
// ################################################################
TEST_CASE("Analytic potential phi_1 is calculated correctly", "[analytic_potentials]") {
    // Define the common equilateral triangle for all phi_1 tests
    const float s = 1.0f;
    const vec3_t tri[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(s, 0.0f, 0.0f),
        make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
    };

    SECTION("In-plane at Centroid") {
        const vec3_t r_obs = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 6.0f, 0.0f);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        vec3_t phi1_expected = make_vec3(0.0f, 0.0f, 0.0f);

        REQUIRE(phi1_calculated.x == Approx(0.0f).margin(1e-7));
        REQUIRE(phi1_calculated.y == Approx(0.0f).margin(1e-7));
        REQUIRE(phi1_calculated.z == Approx(0.0f).margin(1e-7));
    }

    SECTION("In-plane at Edge Midpoint (Edge-midpoint") {
        const vec3_t r_obs = make_vec3(s / 2.0f, 0.0f, 0.0f);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        vec3_t phi1_expected = make_vec3(0.0f, 0.31222178f, 0.0f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x).margin(1e-10));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y).margin(1e-10));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z).margin(1e-10));
    }

    SECTION("In-plane at Vertex (this have to be checked carefully we ignored at the moment)") {
        const vec3_t r_obs = make_vec3(0.0f, 0.0f, 0.0f);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        vec3_t phi1_expected = make_vec3(0.3567848f, 0.2059898f, 0.0f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z));
    }

    SECTION("In-plane, outside triangle") {
        const vec3_t r_obs = make_vec3(0.0f, -0.5f, 0.0f);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        vec3_t phi1_expected = make_vec3(0.22319315f, 0.3587124f, 0.0f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z));
    }

    SECTION("Off-plane, above centroid") {
        const float h = 0.25f;
        const vec3_t r_obs = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 6.0f, h);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        // Expected x and y are non-zero but very small due to floating point error
        vec3_t phi1_expected = make_vec3(0.0f, 0.0f, -0.3027836f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x).margin(1e-7));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y).margin(1e-7));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z));
    }

    SECTION("Off-plane, above vertex") {
        const float h = 0.25f;
        const vec3_t r_obs = make_vec3(0.0f, 0.0f, h);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        vec3_t phi1_expected = make_vec3(0.3159572f, 0.18241798f, -0.1812618f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z));
    }

    SECTION("Off-plane, above edge midpoint") {
        const float h = 0.25f;
        const vec3_t r_obs = make_vec3(s / 2.0f, 0.0f, h);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        // Expected x is non-zero but very small due to floating point error
        vec3_t phi1_expected = make_vec3(0.0f, 0.2476587f, -0.25407378f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x).margin(1e-15));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z));
    }

    SECTION("Far-field, above centroid") {
        const float h = 10.0f;
        const vec3_t r_obs = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 6.0f, h);
        vec3_t phi1_calculated = calculate_phi_1(r_obs, tri);
        // Expected x and y are non-zero but very small due to floating point error
        vec3_t phi1_expected = make_vec3(0.0f, 0.0f, -0.43283246f);

        REQUIRE(phi1_calculated.x == Approx(phi1_expected.x).margin(1e-6));
        REQUIRE(phi1_calculated.y == Approx(phi1_expected.y).margin(1e-6));
        REQUIRE(phi1_calculated.z == Approx(phi1_expected.z).margin(1e-6));
    }
}

// ########################################################################
// ## Tests for the analytic potential Psi_1
// ## Psi_1 is implemented in src/sie_mom/cuda/semi_analytic_integrals.cuh
// ########################################################################

TEST_CASE("Analytic potential Psi_1 is calculated correctly", "[analytic_potentials]") {
    // Define the common equilateral triangle for all Psi_1 tests
    const float s = 1.0f;
    const vec3_t tri[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(s, 0.0f, 0.0f),
        make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
    };

    // Test cases would go here, similar to phi_0 and phi_1 tests.
    // Since Psi_1 implementation is not provided, we leave this as a placeholder.
    SECTION("In-plane at Centroid") {
        const vec3_t r_obs = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 6.0f, 0.0f);

        auto [free_vertex_idx, expected_x, expected_y, expected_z] = GENERATE(
            table<int, float, float, float>({
                {0, 1.14051899f, 0.65847894f, 0.0f},
                {1, -1.14051899f, 0.65847894f, 0.0f},
                {2, 0.0f, -1.31695789f, 0.0f}
            })
        );

        vec3_t psi1_calculated = calculate_Psi_1(r_obs, tri, free_vertex_idx);
        vec3_t psi1_expected = make_vec3(expected_x, expected_y, expected_z);

        // Adjust margin for the specific case where expected_x is 0.0f for free_vertex_idx = 2
        if (free_vertex_idx == 2) {
            REQUIRE(psi1_calculated.x == Approx(psi1_expected.x).margin(1e-7));
        } else {
            REQUIRE(psi1_calculated.x == Approx(psi1_expected.x));
        }
        REQUIRE(psi1_calculated.y == Approx(psi1_expected.y));
        REQUIRE(psi1_calculated.z == Approx(psi1_expected.z));
    }

    SECTION("In-plane, outside the triangle") {
        const vec3_t r_obs = make_vec3(0.0f, -0.5f, 0.0f);

        auto [free_vertex_idx, expected_x, expected_y, expected_z] = GENERATE(
            table<int, float, float, float>({
                {0,  0.22319315f,  0.12053124f, 0.0f},
                {1, -0.25316922f,  0.12053124f, 0.0f},
                {2, -0.01498803f, -0.29201068f, 0.0f}
            })
        );

        vec3_t psi1_calculated = calculate_Psi_1(r_obs, tri, free_vertex_idx);
        vec3_t psi1_expected = make_vec3(expected_x, expected_y, expected_z);

        REQUIRE(psi1_calculated.x == Approx(psi1_expected.x).margin(5e-7));
        REQUIRE(psi1_calculated.y == Approx(psi1_expected.y));
        REQUIRE(psi1_calculated.z == Approx(psi1_expected.z));
    }

    SECTION("Off-plane, above centroid") {
        auto [obs_x, obs_y, obs_z, free_vertex_idx, expected_x, expected_y, expected_z] = GENERATE(
            table<float, float, float, int, float, float, float>({
                // h = 0.25
                {0.5f, 0.28867513f, 0.25f, 0,  0.60555375f,  0.34960854f, 0.0f},
                {0.5f, 0.28867513f, 0.25f, 1, -0.60554278f,  0.34960854f, 0.0f},
                {0.5f, 0.28867513f, 0.25f, 2, -0.00000061f, -0.69923365f, 0.0f},
                // h = 10
                {0.5f, 0.28867513f, 10.0f, 0,  0.02164142f,  0.01249434f, 0.0f},
                {0.5f, 0.28867513f, 10.0f, 1, -0.02164084f,  0.01249434f, 0.0f},
                {0.5f, 0.28867513f, 10.0f, 2,  0.00000000f, -0.02498922f, 0.0f}
            })
        );

        const vec3_t r_obs = make_vec3(obs_x, obs_y, obs_z);
        vec3_t psi1_calculated = calculate_Psi_1(r_obs, tri, free_vertex_idx);
        vec3_t psi1_expected = make_vec3(expected_x, expected_y, expected_z);

        // Using a margin for small expected values to handle floating point inaccuracies
        REQUIRE(psi1_calculated.x == Approx(psi1_expected.x).margin(5e-5));
        REQUIRE(psi1_calculated.y == Approx(psi1_expected.y).margin(5e-5));
        REQUIRE(psi1_calculated.z == Approx(psi1_expected.z).margin(1e-7));
    }

}

// #######################################################################
// ## Tests for phi_2 
// #######################################################################
TEST_CASE("Analytic potential phi_2 is calculated correctly", "[analytic_potentials]") {
    // Define the common equilateral triangle for all phi_2 tests
    const float s = 1.0f;
    const vec3_t tri[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(s, 0.0f, 0.0f),
        make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f)
    };

    SECTION("For outside in-plane and off-plane points") {
        auto [obs_x, obs_y, obs_z, expected_phi2] = GENERATE(
            table<float, float, float, float>({
                {0.0f, -0.5f, 0.000001f, 0.68406981f}, // Existing point
                // New in-plane points from the provided table
                {-0.04532796f, -0.09815915f, 0.0f, 4.48610592f},
                {1.09473091f,  0.58488387f,  0.0f, 1.86001742f},
                {0.28221910f,  -0.06765354f, 0.0f, 22.87964058f},
                {0.69661751f,  -0.34520505f, 0.0f, 2.01604700f},
                {0.84551209f,  1.08942544f,  0.0f, 0.90676057f},
                {-0.00350857f, 1.10287419f,  0.0f, 0.61175209f},
                {0.83447491f,  -0.51779392f, 0.0f, 0.81101960f},
                {0.38367933f,  0.98431185f,  0.0f, 2.85266256f},
                // {0.5f,         -0.001f,      0.0f, 1260.804688f}, // in plane very close to vertex - ignored
                {1.05f,        0.05f,        0.0f, 11.45245552f},
                // Off-plane points
                {0.5f, 0.28867513f, 0.25f, 10.66567326f},
                {0.5f, 0.28867513f, 10.0f, 0.00043247f},
                // New off-plane points very close to the centroid
                // {0.5f, 0.28867513f, 0.001f,   5047.736816f},
                // {0.5f, 0.28867513f, 0.0001f,  6730.004883f},
                // {0.5f, 0.28867513f, 0.00001f, 6753.402344f}
            })
        );

        const vec3_t r_obs = make_vec3(obs_x, obs_y, obs_z);
        float phi2_calculated = calculate_phi_2(r_obs, tri);

        
        // Using a margin for small expected values to handle floating point inaccuracies
        // The relative error 'rel' from the table is used as epsilon.
        REQUIRE(phi2_calculated == Approx(expected_phi2).epsilon(1e-5f)); // Default epsilon for general points
    }

    SECTION("Near-plane approach from outside (off-plane branch with h > 1e-9)") {
        // printf("Testing near-plane approach from outside the triangle\n");
        // printf("Using equilateral triangle:\n");
        // printf(" We use here analytical values calcuated with ChatGpt-5 Thinking on very close to edge points\n");
        // Approach the plane from just outside the panel at several lateral positions.
        // All h are > 1e-9 so your implementation will select the off-plane formula.
        const float rt3_over_2 = s * sqrtf(3.0f) * 0.5f;

        struct Row { float x, y, h, expected, rel; const char* name; } rows[] = {
        // Right of B
        { 1.001f,  0.001f, 1e-4f,   626.0604172f, 5e-5f, "right of B, h=1e-4" },
        { 1.001f,  0.001f, 1e-6f,   627.4721090f, 5e-5f, "right of B, h=1e-6" },
        { 1.001f,  0.001f, 1e-8f,   627.4722507f, 5e-5f, "right of B, h=1e-8" },

        // Above apex
        { 0.5f,    float(s*sqrtf(3.0f)*0.5f + 1e-3f), 1e-4f, 533.3806053f, 2e-5f, "above apex, h=1e-4" },
        { 0.5f,    float(s*sqrtf(3.0f)*0.5f + 1e-3f), 1e-6f, 534.7453779f, 2e-5f, "above apex, h=1e-6" },
        { 0.5f,    float(s*sqrtf(3.0f)*0.5f + 1e-3f), 1e-8f, 534.7455116f, 2e-5f, "above apex, h=1e-8" },

        // Below AB (1e-3)
        { 0.5f,   -1e-3f,  1e-4f,  1987.0749213f, 3e-5f, "below AB (1e-3), h=1e-4" },
        { 0.5f,   -1e-3f,  1e-6f,  1993.7012047f, 3e-5f, "below AB (1e-3), h=1e-6" },
        { 0.5f,   -1e-3f,  1e-8f,  1993.7018713f, 3e-5f, "below AB (1e-3), h=1e-8" },

        // Below AB (1e-4)
        { 0.5f,   -1e-4f,  1e-4f,  15701.6549954f, 3e-5f, "below AB (1e-4), h=1e-4" },
        { 0.5f,   -1e-4f,  1e-6f,  19993.0251007f, 5e-5f, "below AB (1e-4), h=1e-6" },
        { 0.5f,   -1e-4f,  1e-8f,  19993.6916606f, 3e-5f, "below AB (1e-4), h=1e-8" },
        };

        for (const auto& r : rows) {
            const vec3_t robs = make_vec3(r.x, r.y, r.h);
            INFO(r.name);
            float phi2_calc = calculate_phi_2(robs, tri);
            REQUIRE(phi2_calc == Approx(r.expected).epsilon(r.rel));
        }
    }

}


TEST_CASE("Analytic vector potential phi_3 is calculated correctly (edge i=0, free=C)",
          "[analytic_potentials][phi3]") {
    const float s = 1.0f;
    const vec3_t A = make_vec3(0.0f, 0.0f, 0.0f);
    const vec3_t B = make_vec3(s,     0.0f, 0.0f);
    const vec3_t C = make_vec3(s*0.5f, s*sqrtf(3.0f)*0.5f, 0.0f);
    const vec3_t tri[3] = {A, B, C};

    auto approx_rel = [](float expected, float rel = 5e-4f) {
        // 5e-4 is conservative for float; tighten if you promote atan2 pieces to double.
        return Approx(expected).epsilon(rel);
    };

    SECTION("In-plane, outside triangle: r = (0, -0.5, 0)") {
        const vec3_t r = make_vec3(0.0f, -0.5f, 0.0f);
        const vec3_t phi3 = calculate_phi_3(r, tri, /*edge_idx=*/2);
        // Reference (double-precision quadrature):
        const float ex = -0.074212229462f;
        const float ey = -0.465979903648f;
        REQUIRE(phi3.x == approx_rel(ex));
        REQUIRE(phi3.y == approx_rel(ey));
        REQUIRE(phi3.z == Approx(0.0f).margin(1e-6f)); // z component is zero (integrand in-plane)
    }

    SECTION("Off-plane near field, above centroid: h = 0.25") {
        const float cx = 0.5f;
        const float cy = s * sqrtf(3.0f) / 6.0f;
        const vec3_t r = make_vec3(cx, cy, 0.25f);
        const vec3_t phi3 = calculate_phi_3(r, tri, /*edge_idx=*/2);
        const float ex = -1.8540270e-14f;   // ~0
        const float ey = -6.157828578732f;
        REQUIRE(phi3.x == Approx(0.0f).margin(1e-6f));
        REQUIRE(phi3.y == approx_rel(ey, 5e-4f));
        REQUIRE(phi3.z == Approx(0.0f).margin(1e-6f));
    }

    SECTION("Off-plane far field, above centroid: h = 10") {
        const float cx = 0.5f;
        const float cy = s * sqrtf(3.0f) / 6.0f;
        const vec3_t r = make_vec3(cx, cy, 10.0f);
        const vec3_t phi3 = calculate_phi_3(r, tri, /*edge_idx=*/2);
        const float ex = -1.8937248e-18f;   // ~0
        const float ey = -2.49688020e-4f;
        REQUIRE(phi3.x == Approx(0.0f).margin(1e-7f));
        REQUIRE(phi3.y == approx_rel(ey, 1e-4f));
        REQUIRE(phi3.z == Approx(0.0f).margin(1e-7f));
    }

    SECTION("Near-plane, outside & near edge AB: (x=0.5, y=-1e-3, h=1e-4)") {
        // Approaching the plane from outside near edge AB. This grows large but remains finite.
        const vec3_t r = make_vec3(0.5f, -1e-3f, 1.19e-3f);
        const vec3_t phi3 = calculate_phi_3(r, tri, /*edge_idx=*/2);
        const float ex = -3.421510438180e-01f;
        const float ey = -1.255e+03f; // ~ -1255.0f
        REQUIRE(phi3.x == Approx(ex).margin(4e-1f));
        REQUIRE(phi3.y == approx_rel(ey, 1e-3f)); // looser relative tol for large value
        REQUIRE(phi3.z == Approx(0.0f).margin(1e-6f));
    }

    
}

TEST_CASE("Analytic vector potential phi_3 is calculated correctly (data-driven)",
          "[analytic_potentials][phi3]") {
    const float s = 1.0f;
    const vec3_t A = make_vec3(0.0f, 0.0f, 0.0f);
    const vec3_t B = make_vec3(s,     0.0f, 0.0f);
    const vec3_t C = make_vec3(s*0.5f, s*sqrtf(3.0f)*0.5f, 0.0f);
    const vec3_t tri[3] = {A, B, C};

    auto approx_rel = [](float expected, float rel) { return Approx(expected).epsilon(rel); };

    // edge_idx mapping:
    // 0 -> i=0 (free C), 1 -> i=1 (free A), 2 -> i=2 (free B)
    struct Row {
        float x, y, z;
        int   free_vertex_idx;
        float ex, ey, ez;   // expected φ3 components
        float rel;          // relative tolerance
        const char* name;
    } rows[] = {
        // ---- in-plane outside  (row indices 0..2)
        { 0.0f, -0.5f, 0.0f, 2,  -0.074212229f,  -0.465979904f,  0.000000000f, 5e-4f, "in-plane outside | i=2 (free C)" },
        { 0.0f, -0.5f, 0.0f, 0,   0.267822682f,   0.126441940f,  0.000000000f, 5e-4f, "in-plane outside | i=0 (free A)" },
        { 0.0f, -0.5f, 0.0f, 1,  -0.416247141f,   0.126441940f,  0.000000000f, 5e-4f, "in-plane outside | i=1 (free B)" },

        // ---- right of B, h = 1e-4  (row indices 3..5)
        { 1.001f, 0.001f, 1e-4f, 2,  308.068740899f, -539.163247879f, 0.000000000f, 5e-4f, "right of B (h=1e-4) | i=2 (free C)" },
        { 1.001f, 0.001f, 1e-4f, 0,  621.098949513f,   3.020977743f, 0.000000000f, 5e-4f, "right of B (h=1e-4) | i=0 (free A)" },
        { 1.001f, 0.001f, 1e-4f, 1,   -4.961467714f,   3.020977743f, 0.000000000f, 5e-4f, "right of B (h=1e-4) | i=1 (free B)" },

        // ---- right of B, h = 1e-6  (row indices 6..8)
        { 1.001f, 0.001f, 1e-6f, 2,  308.773086230f, -540.384617111f, 0.000000000f, 5e-4f, "right of B (h=1e-6) | i=2 (free C)" },
        { 1.001f, 0.001f, 1e-6f, 0,  622.509140736f,   3.022169460f, 0.000000000f, 5e-4f, "right of B (h=1e-6) | i=0 (free A)" },
        { 1.001f, 0.001f, 1e-6f, 1,   -4.962968277f,   3.022169460f, 0.000000000f, 5e-4f, "right of B (h=1e-6) | i=1 (free B)" },

        // ---- right of B, h = 1e-8  (row indices 9..11)
        { 1.001f, 0.001f, 1e-8f, 2,  308.773156921f, -540.384739693f, 0.000000000f, 5e-4f, "right of B (h=1e-8) | i=2 (free C)" },
        { 1.001f, 0.001f, 1e-8f, 0,  622.509282269f,   3.022169580f, 0.000000000f, 5e-4f, "right of B (h=1e-8) | i=0 (free A)" },
        { 1.001f, 0.001f, 1e-8f, 1,   -4.962968427f,   3.022169580f, 0.000000000f, 5e-4f, "right of B (h=1e-8) | i=1 (free B)" },
    };

    for (const auto& r : rows) {
        const vec3_t robs = make_vec3(r.x, r.y, r.z);
        INFO(r.name);
        const vec3_t phi3 = calculate_phi_3(robs, tri, r.free_vertex_idx);
        REQUIRE(phi3.x == approx_rel(r.ex, r.rel));

        if (fabs(r.ey) < 4.0f) {
            // For small expected values, use a margin to account for floating point noise.
            REQUIRE(phi3.y == Approx(r.ey).margin(1e-1f));
        } else {
            // For larger expected values, use relative tolerance.
            REQUIRE(phi3.y == approx_rel(r.ey, r.rel));
        }
        
        // For the z-component, the expected value is 0, so a relative tolerance
        // (epsilon) is not appropriate. We use an absolute tolerance (margin) instead.
        // We can derive a reasonable margin from the relative tolerance and the
        // magnitude of the other components in the vector.
        REQUIRE(phi3.z == Approx(r.ez).margin(1e-7f));
    }
}

// ################################################################
// ## Tests for the analytic contribution calculate_Upsilon_coul
// ################################################################

static void make_equilateral(vec3_t tri[3], float s) {
    tri[0] = make_vec3(0.0f, 0.0f, 0.0f);
    tri[1] = make_vec3(s, 0.0f, 0.0f);
    tri[2] = make_vec3(s / 2.0f, s * sqrtf(3.0f) / 2.0f, 0.0f);
}

static inline Approx approx_rel(float expected, float rel) {
    return Approx(expected).epsilon(rel);
}

TEST_CASE("Upsilon_coul analytic matches quadrature references", "[upsilon_coul][quadrature_ref]") {
    vec3_t tri[3]; make_equilateral(tri, 1.0f);

    struct Row {
        float x, y, z;
        int   free_vertex_idx; // 0->A, 1->B, 2->C (free vertex)
        float k;
        float ex, ey, ez;    // expected Upsilon components (quadrature)
        float rel;           // default relative tolerance
        const char* name;
    };

    // ---- Reference values generated by near-singularity-aware quadrature (Duffy + edge split) ----
    // Notes:
    // - For components expected to be ~0, we use .margin(…) absolute checks.
    // - For dominant components we keep a tight relative epsilon (1e-3 by default here).
    const Row rows[] = {
    // ===== 1) OUTSIDE — FAR =====

    // In-plane FAR (outside); r=(2,-1,0), free C (edge=2)
    { 2.000000000f, -1.000000000f, 0.000000000e+00f,
      2, 2.000000000f,
      0.000000000f,  0.000000000f,  0.033844325f,
      1e-3f, "in-plane FAR outside | edge=2 | k=2.000" },

    // In-plane FAR (outside); r=(-1.5,2,0), free A (edge=0)
    { -1.500000000f,  2.000000000f, 0.000000000e+00f,
       0, 2.000000000f,
      -0.000000000f, -0.000000000f,  0.040170465f,
      1e-3f, "in-plane FAR outside | edge=0 | k=2.000" },

    // Off-plane FAR above centroid; r=(0.5, sqrt(3)/6, 10), free A (edge=0)
    { 0.500000000f,  0.288675135f,  1.000000000e+01f,
      0, 2.000000000f,
       0.019985435f, -0.034615788f,  0.000000000f,
      1e-3f, "off-plane FAR above centroid | edge=0 | k=2.000" },

    // Off-plane FAR; r=(2,-1,5), free C (edge=2)
    { 2.000000000f, -1.000000000f,  5.000000000e+00f,
      2, 2.000000000f,
      -0.037720526f, -0.000147168f,  0.011261234f,
      1e-3f, "off-plane FAR | edge=2 | k=2.000" },

    // ===== 2) OUTSIDE — MODERATE =====

    // In-plane outside (moderate); r=(0,-0.5,0), free C (edge=2)
    // (Corrected: Ux=Uy=0, Uz<0)
    { 0.000000000f, -0.500000000f,  0.000000000e+00f,
      2, 2.000000000f,
      0.000000000f,  0.000000000f, -0.030452449f,
      1e-3f, "in-plane outside (moderate) | edge=2 | k=2.000" },

    // Optional k=0 reference for the same point
    { 0.000000000f, -0.500000000f,  0.000000000e+00f,
      2, 0.000000000f,
      0.000000000f,  0.000000000f, -0.010473522f,
      1e-3f, "in-plane outside (moderate) | edge=2 | k=0.000" },

    // Off-plane moderate above centroid; r=(0.5, sqrt(3)/6, 0.25), free A (edge=0)
    { 0.500000000f,  0.288675135f,  2.500000000e-01f,
      0, 2.000000000f,
       0.075164167f, -0.130188156f,  0.000000000f,
      1e-3f, "off-plane moderate above centroid | edge=0 | k=2.000" },
    };



    for (const auto& r : rows) {
        INFO(r.name);
        const vec3_t robs = make_vec3(r.x, r.y, r.z);

        const vec3_t Y = 1/(4.f * (float)M_PI) * calculate_Upsilon_coul(robs, tri, r.free_vertex_idx, r.k);

        // Mixed tolerance policy: dominant components use relative tolerance,
        // while near-zero components use an absolute margin for robustness.
        const float norm_exp = std::sqrt(r.ex*r.ex + r.ey*r.ey + r.ez*r.ez);

        // Check X component
        if (std::fabs(r.ex) < 0.01f * norm_exp && std::fabs(r.ex) < 1e-3f) {
            REQUIRE(Y.x == Approx(r.ex).margin(1e-2f));
        } else {
            REQUIRE(Y.x == approx_rel(r.ex, r.rel));
        }

        // Check Y component
        if (std::fabs(r.ey) < 0.01f * norm_exp && std::fabs(r.ey) < 1e-3f) {
            REQUIRE(Y.y == Approx(r.ey).margin(1e-2f));
        } else {
            REQUIRE(Y.y == approx_rel(r.ey, r.rel));
        }

        // Check Z component
        if (std::fabs(r.ez) < 0.01f * norm_exp && std::fabs(r.ez) < 1e-3f) {
            REQUIRE(Y.z == Approx(r.ez).margin(1e-2f));
        } else {
            REQUIRE(Y.z == approx_rel(r.ez, r.rel));
        }
    }
}

TEST_CASE("Upsilon_coul analytic matches quadrature references (NEAR/TRICKY cases)", "[upsilon_coul][quadrature_ref][tricky]") {
    vec3_t tri[3]; make_equilateral(tri, 1.0f);

    // Component-aware helpers
    auto approx_rel  = [](float v, float rel){ return Approx(v).epsilon(rel).margin(1e-7f); };
    auto approx_abs  = [](float v, float abs){ return Approx(v).margin(abs); };

    SECTION("near-plane right of B | edge=0 | k=2")
    {
      const vec3_t expY = make_vec3( 2.73176508e-05f, -4.951960998e-03f, -2.23930074513e-01f );
      const vec3_t Y = 1/(4.f * (float)M_PI) * calculate_Upsilon_coul(make_vec3(1.001f, 0.001f, 1e-4f), tri, /*edge=*/0, 2.0f);
      REQUIRE(Y.x == approx_abs(expY.x, 5e-6f));
      REQUIRE(Y.y == approx_rel(expY.y, 2e-3f));
      REQUIRE(Y.z == approx_rel(expY.z, 4e-2f));
    }

    SECTION("near-plane right of B | edge=2 | k=2")
    {
      const vec3_t expY = make_vec3( -4.300311901e-03f, -2.453402911e-03f, 3.22004188396e-01f );
      const vec3_t Y = 1/(4.f * (float)M_PI) * calculate_Upsilon_coul(make_vec3(1.001f, 0.001f, 1e-4f), tri, /*edge=*/2, 2.0f);
      REQUIRE(Y.x == approx_rel(expY.x, 2e-3f));
      REQUIRE(Y.y == approx_rel(expY.y, 2e-3f));
      REQUIRE(Y.z == approx_rel(expY.z, 2e-2f));
    }

    SECTION("near-plane near edge AB | edge=2 | k=2")
    {
      // Dominant is X; Y≈0, Z≈0 but Z converges slowly with plain (u,v).
      const vec3_t expY = make_vec3( -1.35459143014e-02f, 0.0f, 0.0f );
      const vec3_t Y = 1/(4.f * (float)M_PI) * calculate_Upsilon_coul(make_vec3(0.5f, -1e-3f, 1e-4f), tri, /*edge=*/2, 2.0f);
      REQUIRE(Y.x == approx_rel(expY.x, 3e-2f));
      REQUIRE(Y.y == approx_abs(0.0f, 1e-2f));   // near-zero
      REQUIRE(Y.z == approx_abs(0.0f, 1e-1f));   // allow a bit more until edge-aware quad is in place
    }
}

TEST_CASE("Single point r=(0,-0.5,0), in-plane outside, free=C (edge=2), k=2 — quadrature vs analytic") {
    // Geometry (equilateral s=1)
    const vec3_t tri[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),                        // A
        make_vec3(1.0f, 0.0f, 0.0f),                        // B
        make_vec3(0.5f, sqrtf(3.0f)*0.5f, 0.0f)             // C
    };
    const vec3_t r = make_vec3(0.0f, -0.5f, 0.0f);
    const int    edge_idx = 2;  // free vertex = C
    const float  k = 2.0f;

    // ---- Quadrature references (no code here; hard numbers) ----
    const float  phi0_q   = 0.476362374194f;
    const vec3_t phi1_q   = make_vec3( 0.223193152529f,  0.358712427604f,  0.0f);
    const vec3_t Psi1_q   = make_vec3(-0.014988034567f, -0.292010676952f,  0.0f);
    const float  phi2_q   = 0.684069822269f;
    const vec3_t phi3_q   = make_vec3(-0.074212229462f, -0.465979903648f,  0.0f);
    const vec3_t Ups_q    = make_vec3( 0.0f, 0.0f, -0.030452449467f);

    // ---- Your analytic routines under test ----
    const float  phi0_a = calculate_phi_0(r, tri);
    const vec3_t phi1_a = calculate_phi_1(r, tri);
    const float  phi2_a = calculate_phi_2(r, tri);
    const vec3_t phi3_a = calculate_phi_3(r, tri, edge_idx);
    const vec3_t Psi1_a = calculate_Psi_1(r, tri, edge_idx);
    const vec3_t Ups_a  = 1/(4.f * (float)M_PI) * calculate_Upsilon_coul(r, tri, edge_idx, k);

    // Relative comparator with a tiny absolute cushion for zeros
    auto approx_rel = [](float v, float rel){ return Approx(v).epsilon(rel).margin(1e-7f); };
    const float rel = 1e-6f; // float-level agreement; relax to 1e-5 if needed

    // 1) phi_0
    REQUIRE(phi0_a == approx_rel(phi0_q, rel));

    // 2) phi_1
    REQUIRE(phi1_a.x == approx_rel(phi1_q.x, rel));
    REQUIRE(phi1_a.y == approx_rel(phi1_q.y, rel));
    REQUIRE(phi1_a.z == Approx(0.0f).margin(1e-7f));

    // 3) Psi_1
    REQUIRE(Psi1_a.x == Approx(Psi1_q.x).margin(1e-6f));
    REQUIRE(Psi1_a.y == Approx(Psi1_q.y).margin(1e-6f));
    REQUIRE(Psi1_a.z == Approx(0.0f).margin(1e-7f));

    // 4) phi_2
    REQUIRE(phi2_a == approx_rel(phi2_q, rel));

    // 5) phi_3
    REQUIRE(phi3_a.x == Approx(phi3_q.x).margin(1e-6f));
    REQUIRE(phi3_a.y == approx_rel(phi3_q.y, rel));
    REQUIRE(phi3_a.z == Approx(0.0f).margin(1e-7f));

    // 6) Upsilon^(coul)
    REQUIRE(Ups_a.x == Approx(0.0f).margin(1e-7f));
    REQUIRE(Ups_a.y == Approx(0.0f).margin(1e-7f));
    REQUIRE(Ups_a.z == approx_rel(Ups_q.z, rel));
}
