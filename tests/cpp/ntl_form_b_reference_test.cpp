#include "catch.hpp"

#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace {

using Complex = std::complex<double>;

struct Triangle {
    std::array<vec3_t, 3> vertices;
    std::array<int, 3> indices;
};

struct Rule {
    std::vector<float> points;
    std::vector<float> weights;
    int count;
};

struct DirectParts {
    Complex vector{};
    Complex scalar{};

    Complex total() const { return vector + scalar; }
};

Rule make_rule(int number) {
    const int count = dunavant_order_num(number);
    std::vector<double> points(2 * count), weights(count);
    dunavant_rule(number, count, points.data(), weights.data());
    return {
        std::vector<float>(points.begin(), points.end()),
        std::vector<float>(weights.begin(), weights.end()),
        count,
    };
}

double area(const Triangle& triangle) {
    const vec3_t a = triangle.vertices[1] - triangle.vertices[0];
    const vec3_t b = triangle.vertices[2] - triangle.vertices[0];
    return 0.5 * static_cast<double>(length(cross(a, b)));
}

Complex to_complex(complex__float value) {
    return {static_cast<double>(value.x), static_cast<double>(value.y)};
}

// Independent quadrature of the unintegrated L operator. The test basis is
// either the ordinary RWG half basis or that basis rotated as f x n.
// In particular, this reference never uses the production IntegrandKernels
// or the integrated-by-parts TL modifier.
DirectParts direct_gradient_tl(const Triangle& test, int j,
                               const Triangle& source, int i,
                               const Rule& rule, double k, bool rotated) {
    vec3_t test_free, source_free;
    const double test_edge = getEdgeInfo(
        test.indices.data(), test.vertices.data(), j, test_free);
    const double source_edge = getEdgeInfo(
        source.indices.data(), source.vertices.data(), i, source_free);
    const double test_area = area(test);
    const double source_area = area(source);
    const vec3_t normal = normalize(cross(
        test.vertices[1] - test.vertices[0],
        test.vertices[2] - test.vertices[0]));
    const double source_divergence = source_edge / source_area;
    const double pi = std::acos(-1.0);
    const Complex imaginary(0.0, 1.0);
    DirectParts result;

    for (int p = 0; p < rule.count; ++p) {
        vec3_t observation;
        mapTriQuadPoint(test.vertices.data(), rule.points[2 * p],
                        rule.points[2 * p + 1], observation);
        const vec3_t test_basis = static_cast<real_t>(test_edge / (2 * test_area))
            * (observation - test_free);
        const vec3_t testing = rotated ? cross(test_basis, normal) : test_basis;
        for (int q = 0; q < rule.count; ++q) {
            vec3_t source_point;
            mapTriQuadPoint(source.vertices.data(), rule.points[2 * q],
                            rule.points[2 * q + 1], source_point);
            const vec3_t source_basis =
                static_cast<real_t>(source_edge / (2 * source_area))
                * (source_point - source_free);
            const vec3_t separation = observation - source_point;
            const double distance = static_cast<double>(length(separation));
            const Complex green = std::exp(-imaginary * k * distance)
                / (4.0 * pi * distance);
            const double jacobian_weight = test_area * source_area
                * static_cast<double>(rule.weights[p])
                * static_cast<double>(rule.weights[q]);
            result.vector += jacobian_weight
                * static_cast<double>(dot(testing, source_basis)) * green;
            // Grad' G = (1/R + i k) G (r-r')/R, with the source derivative.
            const double projected_direction =
                static_cast<double>(dot(testing, separation)) / distance;
            result.scalar -= jacobian_weight * source_divergence / (k * k)
                * projected_direction * (1.0 / distance + imaginary * k) * green;
        }
    }
    return result;
}

const std::array<Triangle, 2> test_support{{
    {{{make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(1.0f, 0.0f, 0.0f),
       make_vec3(0.15f, 0.85f, 0.0f)}}, {{0, 1, 2}}},
    {{{make_vec3(1.0f, 0.0f, 0.0f),
       make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(0.25f, -0.75f, 0.0f)}}, {{1, 0, 3}}},
}};

const std::array<Triangle, 2> source_support{{
    {{{make_vec3(0.35f, -0.18f, 1.2f),
       make_vec3(1.35f, -0.18f, 1.2f),
       make_vec3(0.45f, 0.66f, 1.2f)}}, {{4, 5, 6}}},
    {{{make_vec3(1.35f, -0.18f, 1.2f),
       make_vec3(0.35f, -0.18f, 1.2f),
       make_vec3(0.62f, -0.89f, 1.2f)}}, {{5, 4, 7}}},
}};

struct EdgeComparison {
    Complex direct{};
    Complex integrated{};
    Complex direct_vector{};
    Complex direct_scalar{};
};

EdgeComparison compare_full_rwg_edges(int rule_number, real_t k) {
    const Rule rule = make_rule(rule_number);
    EdgeComparison result;
    // Local edge 0 is the shared edge on both triangles of each RWG support.
    for (const auto& test : test_support) {
        for (const auto& source : source_support) {
            const auto reference = direct_gradient_tl(
                test, 0, source, 0, rule, k, false);
            result.direct += reference.total();
            result.direct_vector += reference.vector;
            result.direct_scalar += reference.scalar;
            complex__float tl1[3][3], tl2[3][3], tk1[3][3], tk2[3][3];
            calculateDoubleQuadrature(
                test.vertices.data(), test.indices.data(),
                source.vertices.data(), source.indices.data(),
                rule.points.data(), rule.weights.data(), rule.count,
                k, k, false, tl1, tl2, tk1, tk2);
            result.integrated += to_complex(tl1[0][0]);
        }
    }
    return result;
}

} // namespace

TEST_CASE("Direct-gradient RWG TL converges to integrated-by-parts TL on full edges",
          "[Quadrature][TL][DirectGradient]") {
    constexpr real_t k = real_t(3.1);
    const auto coarse = compare_full_rwg_edges(3, k);
    const auto fine = compare_full_rwg_edges(20, k);
    const double coarse_error = std::abs(coarse.direct - coarse.integrated);
    const double fine_error = std::abs(fine.direct - fine.integrated);
    INFO("coarse_error=" << coarse_error << " fine_error=" << fine_error
         << " vector=" << fine.direct_vector << " scalar=" << fine.direct_scalar);
    CHECK(std::abs(fine.direct_scalar) > 1e-4);
    CHECK(fine_error < coarse_error);
    const double tolerance = sizeof(real_t) == sizeof(double) ? 2e-7 : 2e-5;
    CHECK(fine_error < tolerance);
}

TEST_CASE("Direct-gradient rotated RWG matches Form B nTL on separated faces",
          "[Quadrature][nTL][FormB][DirectGradient]") {
    constexpr real_t k = real_t(3.1);
    const Rule rule = make_rule(13);
    const auto& test = test_support[0];
    const auto& source = source_support[0];
    complex__float ntl1[3][3], ntl2[3][3], ntk1[3][3], ntk2[3][3];
    calculateDoubleQuadrature_nTL_nTK(
        test.vertices.data(), test.indices.data(),
        source.vertices.data(), source.indices.data(),
        rule.points.data(), rule.weights.data(), rule.count,
        k, k, false, ntl1, ntl2, ntk1, ntk2);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const auto reference = direct_gradient_tl(
                test, j, source, i, rule, k, true);
            const double error = std::abs(to_complex(ntl1[j][i]) - reference.total());
            const double tolerance = sizeof(real_t) == sizeof(double)
                ? 2e-9 + 2e-8 * std::abs(reference.total())
                : 2e-6 + 5e-5 * std::abs(reference.total());
            INFO("test_edge=" << j << " source_edge=" << i
                 << " vector=" << reference.vector
                 << " scalar=" << reference.scalar
                 << " production=" << to_complex(ntl1[j][i])
                 << " error=" << error);
            CHECK(error < tolerance);
        }
    }
}

namespace {

struct ExtractedParts {
    Complex regular{};
    double singular = 0.0;

    Complex total() const { return regular + singular; }
};

// Test-only Form B nTL with the testing vector changed from RWG x n to RWG.
// The regular kernel and Wilton add-back follow the production Form B split;
// no TK contribution or coincident-face shortcut is involved.
ExtractedParts extracted_direct_gradient_rwg_tl(
    const Triangle& test, int j, const Triangle& source, int i,
    const Rule& rule, real_t k) {
    vec3_t test_free, source_free;
    const real_t test_edge = getEdgeInfo(
        test.indices.data(), test.vertices.data(), j, test_free);
    const real_t source_edge = getEdgeInfo(
        source.indices.data(), source.vertices.data(), i, source_free);
    const real_t source_area = getTriangleArea(source.vertices.data());
    const vec3_t source_normal = normalize(cross(
        source.vertices[1] - source.vertices[0],
        source.vertices[2] - source.vertices[0]));
    const real_t inverse_k_squared = real_t(1) / (k * k);
    const real_t edge_factor = test_edge * source_edge;
    ExtractedParts result;

    for (int p = 0; p < rule.count; ++p) {
        vec3_t observation;
        mapTriQuadPoint(test.vertices.data(), rule.points[2 * p],
                        rule.points[2 * p + 1], observation);
        const vec3_t test_rho = rhoRwg(observation, test_free);
        const real_t test_weight = rule.weights[p];

        for (int q = 0; q < rule.count; ++q) {
            vec3_t source_point;
            mapTriQuadPoint(source.vertices.data(), rule.points[2 * q],
                            rule.points[2 * q + 1], source_point);
            const vec3_t source_rho = rhoRwg(source_point, source_free);
            const auto kernels = integrandKernelsRegular(
                observation, source_point, k, k);
            const real_t weight = edge_factor * test_weight * rule.weights[q];
            const Complex vector = real_t(0.25) * dot(test_rho, source_rho)
                * to_complex(kernels.tl_k1);
            const Complex scalar = -real_t(0.5) * inverse_k_squared
                * to_complex(dot(test_rho, kernels.tk_k1));
            result.regular += static_cast<double>(weight) * (vector + scalar);
        }

        const WiltonPotentials pots = calculateAllWiltonPotentials(
            observation, source.vertices.data());
        const vec3_t phi_1 = calculate_phi_1(pots, source_normal);
        const vec3_t extracted_gradient =
            pots.phi_3_base - (pots.d * pots.phi_2) * source_normal;
        const vec3_t source_free_offset = observation - source_free;
        const vec3_t singular_bracket =
            phi_1 + real_t(0.5) * source_free_offset * pots.phi_0
            + inverse_k_squared * extracted_gradient;
        const double scale = static_cast<double>(edge_factor * test_weight)
            / (8.0 * std::acos(-1.0) * static_cast<double>(source_area));
        result.singular += scale * static_cast<double>(
            dot(test_rho, singular_bracket));
    }
    return result;
}

struct ExtractedEdgeComparison {
    Complex extracted{};
    Complex regular{};
    double singular = 0.0;
    Complex raw_gradient{};
    Complex trusted_tl{};
};

ExtractedEdgeComparison compare_extracted_full_rwg_edges(int rule_number, real_t k) {
    const Rule rule = make_rule(rule_number);
    ExtractedEdgeComparison result;
    for (const auto& test : test_support) {
        for (const auto& source : source_support) {
            const auto extracted = extracted_direct_gradient_rwg_tl(
                test, 0, source, 0, rule, k);
            result.regular += extracted.regular;
            result.singular += extracted.singular;
            result.extracted += extracted.total();
            result.raw_gradient += direct_gradient_tl(
                test, 0, source, 0, rule, k, false).total();
            complex__float tl1[3][3], tl2[3][3], tk1[3][3], tk2[3][3];
            calculateDoubleQuadrature(
                test.vertices.data(), test.indices.data(),
                source.vertices.data(), source.indices.data(),
                rule.points.data(), rule.weights.data(), rule.count,
                k, k, false, tl1, tl2, tk1, tk2);
            result.trusted_tl += to_complex(tl1[0][0]);
        }
    }
    return result;
}

} // namespace

TEST_CASE("Extracted Form B gradient with RWG testing matches trusted TL on separated full edges",
          "[Quadrature][TL][FormB][SingularitySubtraction][DirectGradient]") {
    constexpr real_t k = real_t(3.1);
    const auto comparison = compare_extracted_full_rwg_edges(20, k);
    const double extraction_error =
        std::abs(comparison.extracted - comparison.raw_gradient);
    const double trusted_tl_error =
        std::abs(comparison.extracted - comparison.trusted_tl);
    INFO("regular=" << comparison.regular
         << " singular=" << comparison.singular
         << " extracted=" << comparison.extracted
         << " raw_gradient=" << comparison.raw_gradient
         << " trusted_TL=" << comparison.trusted_tl
         << " extraction_error=" << extraction_error
         << " trusted_TL_error=" << trusted_tl_error);
    CHECK(std::abs(comparison.singular) > 1e-4);
    const double tolerance = sizeof(real_t) == sizeof(double) ? 2e-7 : 2e-5;
    CHECK(extraction_error < tolerance);
    CHECK(trusted_tl_error < tolerance);
}

TEST_CASE("Production Form B regular plus singular nTL matches direct rotated reference",
          "[Quadrature][nTL][FormB][SingularitySubtraction][DirectGradient]") {
    constexpr real_t k = real_t(3.1);
    const Rule rule = make_rule(20);
    const auto& test = test_support[0];
    const auto& source = source_support[0];
    complex__float regular1[3][3], regular2[3][3];
    complex__float regular_tk1[3][3], regular_tk2[3][3];
    calculateDoubleQuadrature_nTL_nTK(
        test.vertices.data(), test.indices.data(),
        source.vertices.data(), source.indices.data(),
        rule.points.data(), rule.weights.data(), rule.count,
        k, k, true, regular1, regular2, regular_tk1, regular_tk2);
    complex__float singular1[3][3], singular2[3][3];
    complex__float singular_tk1[3][3], singular_tk2[3][3];
    calculateSingularPartQuadrature_nTL_nTK(
        test.vertices.data(), test.indices.data(),
        source.vertices.data(), source.indices.data(),
        rule.points.data(), rule.weights.data(), rule.count,
        k, k, singular1, singular2, singular_tk1, singular_tk2);

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const Complex reference = direct_gradient_tl(
                test, j, source, i, rule, k, true).total();
            const Complex combined = to_complex(regular1[j][i])
                + to_complex(singular1[j][i]);
            const double error = std::abs(combined - reference);
            const double tolerance = sizeof(real_t) == sizeof(double)
                ? 2e-7 : 2e-5;
            INFO("test_edge=" << j << " source_edge=" << i
                 << " regular=" << to_complex(regular1[j][i])
                 << " singular=" << to_complex(singular1[j][i])
                 << " combined=" << combined
                 << " reference=" << reference
                 << " error=" << error);
            CHECK(error < tolerance);
        }
    }
}

namespace {

// The two RWG supports have different edges and meet only at vertex 0.
// Every test/source face pair is distinct, so no self-face shortcut applies.
const std::array<Triangle, 2> vertex_touching_source_support{{
    {{{make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(0.32f, -0.14f, 0.72f),
       make_vec3(0.86f, 0.42f, 0.79f)}}, {{0, 4, 5}}},
    {{{make_vec3(0.32f, -0.14f, 0.72f),
       make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(-0.47f, 0.28f, 0.63f)}}, {{4, 0, 6}}},
}};

struct TouchingComparison {
    Complex form_b_rwg_regular{};
    double form_b_rwg_singular = 0.0;
    Complex tl_regular{};
    Complex tl_singular{};

    Complex form_b_rwg_total() const {
        return form_b_rwg_regular + form_b_rwg_singular;
    }
    Complex tl_total() const { return tl_regular + tl_singular; }
};

TouchingComparison compare_touching_edges(
    const std::array<Triangle, 2>& source_support, const Rule& rule, real_t k) {
    TouchingComparison result;
    for (const auto& test : test_support) {
        for (const auto& source : source_support) {
            const auto form_b = extracted_direct_gradient_rwg_tl(
                test, 0, source, 0, rule, k);
            result.form_b_rwg_regular += form_b.regular;
            result.form_b_rwg_singular += form_b.singular;

            complex__float tl_regular1[3][3], tl_regular2[3][3];
            complex__float tk_regular1[3][3], tk_regular2[3][3];
            calculateDoubleQuadrature(
                test.vertices.data(), test.indices.data(),
                source.vertices.data(), source.indices.data(),
                rule.points.data(), rule.weights.data(), rule.count,
                k, k, true,
                tl_regular1, tl_regular2, tk_regular1, tk_regular2);
            complex__float tl_singular1[3][3], tl_singular2[3][3];
            complex__float tk_singular1[3][3], tk_singular2[3][3];
            calculateSingularPartQuadrature(
                test.vertices.data(), test.indices.data(),
                source.vertices.data(), source.indices.data(),
                rule.points.data(), rule.weights.data(), rule.count,
                k, k,
                tl_singular1, tl_singular2, tk_singular1, tk_singular2);
            result.tl_regular += to_complex(tl_regular1[0][0]);
            result.tl_singular += to_complex(tl_singular1[0][0]);
        }
    }
    return result;
}

TouchingComparison compare_touching_edges(
    const std::array<Triangle, 2>& source_support, int rule_number, real_t k) {
    const Rule rule = make_rule(rule_number);
    return compare_touching_edges(source_support, rule, k);
}

Rule subdivide_rule(const Rule& base) {
    // Four equal-area children of the reference triangle; each keeps the
    // Dunavant [0,1] barycentric convention and receives one-quarter weight.
    const std::array<std::array<std::array<double, 2>, 3>, 4> children{{
        {{{0.0, 0.0}, {0.5, 0.0}, {0.0, 0.5}}},
        {{{0.5, 0.0}, {1.0, 0.0}, {0.5, 0.5}}},
        {{{0.0, 0.5}, {0.5, 0.5}, {0.0, 1.0}}},
        {{{0.5, 0.0}, {0.5, 0.5}, {0.0, 0.5}}},
    }};
    Rule refined;
    refined.count = 4 * base.count;
    refined.points.reserve(2 * refined.count);
    refined.weights.reserve(refined.count);
    for (const auto& child : children) {
        const auto& a = child[0];
        const auto& b = child[1];
        const auto& c = child[2];
        for (int point = 0; point < base.count; ++point) {
            const double xi = base.points[2 * point];
            const double eta = base.points[2 * point + 1];
            refined.points.push_back(static_cast<float>(
                a[0] + xi * (b[0] - a[0]) + eta * (c[0] - a[0])));
            refined.points.push_back(static_cast<float>(
                a[1] + xi * (b[1] - a[1]) + eta * (c[1] - a[1])));
            refined.weights.push_back(base.weights[point] * 0.25f);
        }
    }
    return refined;
}

} // namespace

TEST_CASE("RWG Form B extraction versus TL extraction on vertex-touching full edges",
          "[Quadrature][TL][FormB][SingularitySubtraction][VertexTouching]") {
    constexpr real_t k = real_t(3.1);
    const auto coarse = compare_touching_edges(vertex_touching_source_support, 7, k);
    const auto fine = compare_touching_edges(vertex_touching_source_support, 20, k);
    const double coarse_error = std::abs(
        coarse.form_b_rwg_total() - coarse.tl_total());
    const double fine_error = std::abs(
        fine.form_b_rwg_total() - fine.tl_total());
    INFO("coarse_error=" << coarse_error << " fine_error=" << fine_error
         << " FormB_regular=" << fine.form_b_rwg_regular
         << " FormB_singular=" << fine.form_b_rwg_singular
         << " TL_regular=" << fine.tl_regular
         << " TL_singular=" << fine.tl_singular
         << " FormB_total=" << fine.form_b_rwg_total()
         << " TL_total=" << fine.tl_total());
    CHECK(std::isfinite(coarse_error));
    CHECK(std::isfinite(fine_error));
    CHECK(fine_error < coarse_error);
    CHECK(fine_error / std::abs(fine.tl_total()) < 1e-3);
}

namespace {

// Test face {0,1,2} and source face {4,0,2} share edge {0,2}.
// The source RWG edge is {0,4}, so the two tested RWG edges are distinct.
// No test/source face pair is coincident.
const std::array<Triangle, 2> edge_touching_source_support{{
    {{{make_vec3(0.32f, -0.14f, 0.72f),
       make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(0.15f, 0.85f, 0.0f)}}, {{4, 0, 2}}},
    {{{make_vec3(0.0f, 0.0f, 0.0f),
       make_vec3(0.32f, -0.14f, 0.72f),
       make_vec3(-0.47f, 0.28f, 0.63f)}}, {{0, 4, 6}}},
}};

} // namespace

TEST_CASE("RWG Form B extraction versus TL extraction on edge-touching full edges",
          "[Quadrature][TL][FormB][SingularitySubtraction][EdgeTouching]") {
    constexpr real_t k = real_t(3.1);
    const auto coarse = compare_touching_edges(edge_touching_source_support, 7, k);
    const auto fine = compare_touching_edges(edge_touching_source_support, 20, k);
    const double coarse_error = std::abs(
        coarse.form_b_rwg_total() - coarse.tl_total());
    const double fine_error = std::abs(
        fine.form_b_rwg_total() - fine.tl_total());
    INFO("coarse_error=" << coarse_error << " fine_error=" << fine_error
         << " FormB_regular=" << fine.form_b_rwg_regular
         << " FormB_singular=" << fine.form_b_rwg_singular
         << " TL_regular=" << fine.tl_regular
         << " TL_singular=" << fine.tl_singular
         << " FormB_total=" << fine.form_b_rwg_total()
         << " TL_total=" << fine.tl_total());
    CHECK(std::isfinite(coarse_error));
    CHECK(std::isfinite(fine_error));
    CHECK(fine_error < coarse_error);
    CHECK(fine_error / std::abs(fine.tl_total()) < 1e-3);
}

TEST_CASE("RWG Form B versus TL extraction converges slowly on a coincident full edge",
          "[Quadrature][TL][FormB][SingularitySubtraction][Coincident]") {
    constexpr real_t k = real_t(3.1);
    const auto coarse = compare_touching_edges(test_support, 7, k);
    const auto fine = compare_touching_edges(test_support, 20, k);
    const auto refined = compare_touching_edges(
        test_support, subdivide_rule(make_rule(20)), k);
    const double coarse_error = std::abs(
        coarse.form_b_rwg_total() - coarse.tl_total());
    const double fine_error = std::abs(
        fine.form_b_rwg_total() - fine.tl_total());
    const double refined_error = std::abs(
        refined.form_b_rwg_total() - refined.tl_total());
    INFO("coarse_error=" << coarse_error << " fine_error=" << fine_error
         << " refined_error=" << refined_error
         << " FormB_regular=" << fine.form_b_rwg_regular
         << " FormB_singular=" << fine.form_b_rwg_singular
         << " TL_regular=" << fine.tl_regular
         << " TL_singular=" << fine.tl_singular
         << " FormB_total=" << fine.form_b_rwg_total()
         << " TL_total=" << fine.tl_total());
    CHECK(std::isfinite(coarse_error));
    CHECK(std::isfinite(fine_error));
    CHECK(std::isfinite(refined_error));
    CHECK(fine_error < coarse_error);
    CHECK(refined_error < fine_error);
}
