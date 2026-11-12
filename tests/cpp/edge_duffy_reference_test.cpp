#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include "sie_mom/quadrature/gauss_legendre_rule.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/cuda/dunavant_quadrature.cuh"

namespace {

struct Point3 {
    double x, y, z;
};

Point3 operator+(Point3 a, Point3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Point3 operator-(Point3 a, Point3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Point3 operator*(double c, Point3 a) {
    return {c * a.x, c * a.y, c * a.z};
}

double norm(Point3 a) {
    return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

double dot(Point3 a, Point3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Point3 cross(Point3 a, Point3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

double twice_area(Point3 edge_start, Point3 edge_end, Point3 apex) {
    const Point3 e = edge_end - edge_start;
    const Point3 d = apex - edge_start;
    return norm({e.y * d.z - e.z * d.y,
                 e.z * d.x - e.x * d.z,
                 e.x * d.y - e.y * d.x});
}

// u measures position along the shared edge; v measures position toward apex.
Point3 on_triangle(Point3 edge_start, Point3 edge_end, Point3 apex,
                   double u, double v) {
    return (1.0 - v) * ((1.0 - u) * edge_start + u * edge_end) + v * apex;
}

enum class Kernel { Constant, InverseDistance };

// CPU-only reference for a weakly singular shared-edge integral. This is a
// simple sector Duffy transform, not the full DIRECTFN algorithm or a PV rule.
//
// Split edge coordinates u and s into u >= s and s >= u. In each half, b is
// the smaller coordinate and d=(larger-smaller)/(1-b). The singular corner is
// (d,v,w)=(0,0,0). Split this cube into three sectors by its largest variable,
// writing that variable as rho and the other two as rho*alpha, rho*beta.
// The resulting sector Jacobian is rho^2.
template <typename Integrand>
auto shared_edge_average(Point3 edge_start, Point3 edge_end,
                         Point3 test_apex, Point3 source_apex,
                         int points, const Integrand& integrand)
    -> decltype(integrand(edge_start, edge_start)) {
    std::vector<double> nodes(points), weights(points);
    gauss_legendre_rule(points, nodes.data(), weights.data());

    using Value = decltype(integrand(edge_start, edge_start));
    Value result{};
    for (int ib = 0; ib < points; ++ib) {
        const double b = nodes[ib];
        for (int ir = 0; ir < points; ++ir) {
            const double rho = nodes[ir];
            for (int ia = 0; ia < points; ++ia) {
                const double alpha = nodes[ia];
                for (int ic = 0; ic < points; ++ic) {
                    const double beta = nodes[ic];
                    // Normalized triangle weights: no physical area factors.
                    const double weight = 4.0 * weights[ib] * weights[ir] *
                                          weights[ia] * weights[ic] *
                                          (1.0 - b) * rho * rho;

                    for (int half = 0; half < 2; ++half) {
                        for (int sector = 0; sector < 3; ++sector) {
                            double d, v, w;
                            if (sector == 0) {
                                d = rho;
                                v = rho * alpha;
                                w = rho * beta;
                            } else if (sector == 1) {
                                d = rho * alpha;
                                v = rho;
                                w = rho * beta;
                            } else {
                                d = rho * alpha;
                                v = rho * beta;
                                w = rho;
                            }

                            const double lower = b;
                            const double upper = b + (1.0 - b) * d;
                            const double u = half == 0 ? upper : lower;
                            const double s = half == 0 ? lower : upper;
                            const Point3 r = on_triangle(
                                edge_start, edge_end, test_apex, u, v);
                            const Point3 rp = on_triangle(
                                edge_start, edge_end, source_apex, s, w);
                            result += weight * (1.0 - v) * (1.0 - w) *
                                      integrand(r, rp);
                        }
                    }
                }
            }
        }
    }
    return result;
}

double shared_edge_integral(Point3 edge_start, Point3 edge_end,
                            Point3 test_apex, Point3 source_apex,
                            int points, Kernel kernel) {
    const double area_product = 0.25 *
        twice_area(edge_start, edge_end, test_apex) *
        twice_area(edge_start, edge_end, source_apex);
    return area_product * shared_edge_average(
        edge_start, edge_end, test_apex, source_apex, points,
        [kernel](Point3 r, Point3 rp) {
            return kernel == Kernel::Constant ? 1.0 : 1.0 / norm(r - rp);
        });
}

using Complex = std::complex<double>;
using Matrix3 = std::array<std::array<Complex, 3>, 3>;

// The local edges are (0,1), (1,2), (2,0); the free vertex is (j+2)%3.
double signed_edge_length(const Point3 triangle[3], const int ids[3], int j) {
    const int next = (j + 1) % 3;
    const double sign = ids[j] > ids[next] ? 1.0 : -1.0;
    return sign * norm(triangle[next] - triangle[j]);
}

double vector_basis_dot(Point3 r, Point3 rp, Point3 test_free,
                        Point3 source_free, Point3 test_normal) {
    return dot(cross(r - test_free, test_normal), rp - source_free);
}

template <typename Green>
Matrix3 duffy_vector_area(Point3 a, Point3 b,
                          Point3 test_apex, Point3 source_apex,
                          int points, const Green& green) {
    const Point3 test_tri[3] = {a, b, test_apex};
    const Point3 source_tri[3] = {b, a, source_apex};
    const int test_ids[3] = {0, 1, 2};
    const int source_ids[3] = {1, 0, 3};
    const Point3 test_normal =
        (1.0 / twice_area(a, b, test_apex)) * cross(b - a, test_apex - a);
    Matrix3 result{};

    for (int j = 0; j < 3; ++j) {
        const Point3 test_free = test_tri[(j + 2) % 3];
        const double test_edge = signed_edge_length(test_tri, test_ids, j);
        for (int i = 0; i < 3; ++i) {
            const Point3 source_free = source_tri[(i + 2) % 3];
            const double source_edge = signed_edge_length(source_tri, source_ids, i);
            const auto integrand = [&](Point3 r, Point3 rp) {
                return vector_basis_dot(r, rp, test_free, source_free,
                                        test_normal) * green(norm(r - rp));
            };
            // shared_edge_average already contains normalized triangle weights.
            result[j][i] = (test_edge * source_edge / 4.0) *
                shared_edge_average(a, b, test_apex, source_apex,
                                    points, integrand);
        }
    }
    return result;
}


// Normalized line x source-triangle integration for the three test edges.
// Edge 0 shares a full edge with the source; edges 1 and 2 meet it at a vertex.
template <typename Green>
Matrix3 duffy_line_area(Point3 a, Point3 b,
                        Point3 test_apex, Point3 source_apex,
                        double k, int points, const Green& green) {
    std::vector<double> nodes(points), weights(points);
    gauss_legendre_rule(points, nodes.data(), weights.data());
    const Point3 test_tri[3] = {a, b, test_apex};
    const Point3 source_tri[3] = {b, a, source_apex};
    const int test_ids[3] = {0, 1, 2};
    const int source_ids[3] = {1, 0, 3};
    const double twice_test_area = twice_area(a, b, test_apex);
    const Point3 test_normal =
        (1.0 / twice_test_area) * cross(b - a, test_apex - a);
    Matrix3 result{};

    for (int edge = 0; edge < 3; ++edge) {
        const Point3 edge_start = test_tri[edge];
        const Point3 edge_end = test_tri[(edge + 1) % 3];
        const Point3 edge_vector = edge_end - edge_start;
        const double length = norm(edge_vector);
        const Point3 outward = (1.0 / length) * cross(edge_vector, test_normal);

        for (int j = 0; j < 3; ++j) {
            const Point3 test_free = test_tri[(j + 2) % 3];
            Complex integral{};
            for (int ix = 0; ix < points; ++ix) {
                const double x = nodes[ix];
                for (int ir = 0; ir < points; ++ir) {
                    const double radial = nodes[ir];
                    for (int ia = 0; ia < points; ++ia) {
                        const double alpha = nodes[ia];
                        const double gauss_weight =
                            weights[ix] * weights[ir] * weights[ia];
                        for (int sector = 0; sector < 2; ++sector) {
                            if (edge == 0) {
                                // Shared-edge singularity: t=s and source depth v=0.
                                const double d = sector == 0 ? radial : radial * alpha;
                                const double v = sector == 0 ? radial * alpha : radial;
                                const double low = x;
                                const double high = x + (1.0 - x) * d;
                                for (int half = 0; half < 2; ++half) {
                                    const double t = half == 0 ? high : low;
                                    const double s = half == 0 ? low : high;
                                    const Point3 r = a + t * (b - a);
                                    const Point3 rp = on_triangle(
                                        a, b, source_apex, s, v);
                                    const double jacobian =
                                        2.0 * (1.0 - v) * (1.0 - x) * radial;
                                    const double basis = dot(
                                        cross(r - test_free, test_normal), outward);
                                    integral += gauss_weight * jacobian * basis *
                                                green(norm(r - rp));
                                }
                            } else {
                                // Vertex singularity: line distance t=0 and
                                // source radial coordinate v=0.
                                const double t = sector == 0 ? radial : radial * alpha;
                                const double v = sector == 0 ? radial * alpha : radial;
                                const Point3 shared = edge == 1 ? b : a;
                                const Point3 source_other = edge == 1 ? a : b;
                                const Point3 r = shared + t * (test_apex - shared);
                                const Point3 rp = shared + v * (
                                    (1.0 - x) * (source_other - shared) +
                                    x * (source_apex - shared));
                                const double jacobian = 2.0 * v * radial;
                                const double basis = dot(
                                    cross(r - test_free, test_normal), outward);
                                integral += gauss_weight * jacobian * basis *
                                            green(norm(r - rp));
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < 3; ++i) {
                const double edge_factor =
                    signed_edge_length(test_tri, test_ids, j) *
                    signed_edge_length(source_tri, source_ids, i) *
                    length / (twice_test_area * k * k);
                result[j][i] += edge_factor * integral;
            }
        }
    }
    return result;
}

template <typename Green>
Matrix3 duffy_form_a(Point3 a, Point3 b,
                     Point3 test_apex, Point3 source_apex,
                     double k, int points, const Green& green) {
    Matrix3 area = duffy_vector_area(
        a, b, test_apex, source_apex, points, green);
    const Matrix3 line = duffy_line_area(
        a, b, test_apex, source_apex, k, points, green);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            area[j][i] += line[j][i];
    return area;
}

vec3_t to_vec3(Point3 p) {
    return make_vec3(p.x, p.y, p.z);
}

Matrix3 extracted_form_a(Point3 a, Point3 b,
                         Point3 test_apex, Point3 source_apex,
                         double k, int triangle_rule, int line_points) {
    const vec3_t test_tri[3] = {
        to_vec3(a), to_vec3(b), to_vec3(test_apex)
    };
    const vec3_t source_tri[3] = {
        to_vec3(b), to_vec3(a), to_vec3(source_apex)
    };
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {1, 0, 3};
    const int triangle_points = dunavant_order_num(triangle_rule);
    std::vector<double> nodes_double(2 * triangle_points);
    std::vector<double> weights_double(triangle_points);
    dunavant_rule(triangle_rule, triangle_points,
                  nodes_double.data(), weights_double.data());
    std::vector<float> nodes(nodes_double.begin(), nodes_double.end());
    std::vector<float> weights(weights_double.begin(), weights_double.end());
    std::vector<double> line_nodes_double(line_points);
    std::vector<double> line_weights_double(line_points);
    if (line_points > 0) {
        gauss_legendre_rule(line_points, line_nodes_double.data(),
                            line_weights_double.data());
    }
    std::vector<float> line_nodes(line_nodes_double.begin(), line_nodes_double.end());
    std::vector<float> line_weights(line_weights_double.begin(), line_weights_double.end());
    const float* line_node_ptr = line_points > 0 ? line_nodes.data() : nullptr;
    const float* line_weight_ptr = line_points > 0 ? line_weights.data() : nullptr;

    complex__float reg_ntl_k1[3][3], reg_ntl_k2[3][3];
    complex__float reg_ntk_k1[3][3], reg_ntk_k2[3][3];
    // G_reg: area plus line when line_points > 0.
    calculateFormAQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        nodes.data(), weights.data(), triangle_points,
        line_node_ptr, line_weight_ptr, line_points, real_t(k), real_t(k), true,
        reg_ntl_k1, reg_ntl_k2, reg_ntk_k1, reg_ntk_k2);

    complex__float sing_ntl_k1[3][3], sing_ntl_k2[3][3];
    complex__float sing_ntk_k1[3][3], sing_ntk_k2[3][3];
    // Extracted 1/(4*pi*R): area plus line when line_points > 0.
    calculateFormASingularPartQuadrature_nTL_nTK(
        test_tri, test_indices, source_tri, source_indices,
        nodes.data(), weights.data(), triangle_points,
        line_node_ptr, line_weight_ptr, line_points, real_t(k), real_t(k),
        sing_ntl_k1, sing_ntl_k2, sing_ntk_k1, sing_ntk_k2);

    Matrix3 result{};
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            result[j][i] = {
                double(reg_ntl_k1[j][i].x) + sing_ntl_k1[j][i].x,
                double(reg_ntl_k1[j][i].y) + sing_ntl_k1[j][i].y};
        }
    }
    return result;
}

} // namespace

TEST_CASE("Shared-edge Duffy mapping integrates a constant kernel",
          "[Quadrature][DuffyReference]") {
    const Point3 a{0, 0, 0}, b{0.04, 0, 0};
    const Point3 test_apex{0, 0.04, 0};
    const double angle = 20.0 * std::acos(-1.0) / 180.0;
    const Point3 source_apex{0, -0.04 * std::cos(angle),
                             0.04 * std::sin(angle)};
    const double product_of_areas =
        0.25 * twice_area(a, b, test_apex) *
        twice_area(a, b, source_apex);

    for (int points : {4, 8, 12}) {
        const double actual = shared_edge_integral(
            a, b, test_apex, source_apex, points, Kernel::Constant);
        INFO("Gauss points per dimension: " << points);
        CHECK(actual == Approx(product_of_areas).epsilon(1e-12));
    }
}

TEST_CASE("Shared-edge Duffy 1/R integral converges and obeys reciprocity",
          "[Quadrature][DuffyReference]") {
    const Point3 a{0, 0, 0}, b{0.04, 0, 0};
    const Point3 test_apex{0, 0.04, 0};
    const double angle = 20.0 * std::acos(-1.0) / 180.0;
    const Point3 source_apex{0, -0.04 * std::cos(angle),
                             0.04 * std::sin(angle)};
    const double coarse = shared_edge_integral(
        a, b, test_apex, source_apex, 12, Kernel::InverseDistance);
    const double fine = shared_edge_integral(
        a, b, test_apex, source_apex, 20, Kernel::InverseDistance);
    const double swapped = shared_edge_integral(
        a, b, source_apex, test_apex, 20, Kernel::InverseDistance);

    const double scaled = shared_edge_integral(
        2.0 * a, 2.0 * b, 2.0 * test_apex, 2.0 * source_apex,
        20, Kernel::InverseDistance);

    INFO("12-point result=" << coarse << ", 20-point result=" << fine);
    CHECK(std::abs(fine - coarse) / std::abs(fine) < 1e-5);
    CHECK(swapped == Approx(fine).epsilon(1e-12));
    CHECK(scaled == Approx(8.0 * fine).epsilon(1e-12));
}

TEST_CASE("Shared-edge Duffy vector nTL area matrix versus extracted Form A area matrix",
          "[Quadrature][DuffyReference][nTLVector]") {
    // Same geometry as [NearSharedEdge] with h=0: the triangles share an edge.
    const double side = 0.04;
    const double angle = 20.0 * std::acos(-1.0) / 180.0;
    const Point3 a{0, 0, 0}, b{side, 0, 0};
    const Point3 test_apex{0, side, 0};
    const Point3 source_apex{0, -side * std::cos(angle),
                             side * std::sin(angle)};
    const double frequency_hz = 1e8;
    const double epsilon0 = 8.8541878128e-12;
    const double mu0 = 1.25663706212e-6;
    const double k = 2.0 * std::acos(-1.0) * frequency_hz *
                     std::sqrt(epsilon0 * mu0);
    const auto green = [k](double distance) {
        return std::exp(Complex(0.0, -k * distance)) /
               (4.0 * std::acos(-1.0) * distance);
    };

    const Matrix3 duffy_12 = duffy_vector_area(
        a, b, test_apex, source_apex, 12, green);
    const Matrix3 duffy_20 = duffy_vector_area(
        a, b, test_apex, source_apex, 20, green);
    const Matrix3 extracted = extracted_form_a(
        a, b, test_apex, source_apex, k, 20, 0);

    double scale = 0.0;
    for (const auto& row : duffy_20)
        for (const auto& entry : row)
            scale = std::max(scale, std::abs(entry));
    REQUIRE(scale > 0.0);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("row=" << j << ", col=" << i
                 << ", Duffy=" << duffy_20[j][i]
                 << ", extracted=" << extracted[j][i]);
            CHECK(std::abs(duffy_20[j][i] - duffy_12[j][i]) / scale < 1e-4);
            CHECK(std::abs(extracted[j][i] - duffy_20[j][i]) / scale < 1.3e-4);
        }
    }
}

TEST_CASE("Shared-edge Duffy line integral vanishes for constant Green kernel",
          "[Quadrature][DuffyReference][nTLLine]") {
    const double side = 0.04;
    const double angle = 20.0 * std::acos(-1.0) / 180.0;
    const Point3 a{0, 0, 0}, b{side, 0, 0};
    const Point3 test_apex{0, side, 0};
    const Point3 source_apex{0, -side * std::cos(angle),
                             side * std::sin(angle)};
    const Matrix3 line = duffy_line_area(
        a, b, test_apex, source_apex, 1.0, 8,
        [](double) { return Complex(1.0, 0.0); });
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            INFO("row=" << j << ", col=" << i);
            CHECK(std::abs(line[j][i]) < 1e-12);
        }
}

TEST_CASE("Shared-edge Duffy full Form A nTL matrix versus extracted Form A matrix",
          "[Quadrature][DuffyReference][nTLFull]") {
    // Same triangles as [NearSharedEdge] with h=0.
    const double side = 0.04;
    const double angle = 0.0 * std::acos(-1.0) / 180.0;
    const Point3 a{0, 0, 0}, b{side, 0, 0};
    const Point3 test_apex{0, side, 0};
    const Point3 source_apex{0, -side * std::cos(angle),
                             side * std::sin(angle)};
    const double frequency_hz = 1e8;
    const double epsilon0 = 8.8541878128e-12;
    const double mu0 = 1.25663706212e-6;
    const double k = 2.0 * std::acos(-1.0) * frequency_hz *
                     std::sqrt(epsilon0 * mu0);
    const auto green = [k](double distance) {
        return std::exp(Complex(0.0, -k * distance)) /
               (4.0 * std::acos(-1.0) * distance);
    };

    const Matrix3 duffy_12 = duffy_form_a(
        a, b, test_apex, source_apex, k, 12, green);
    const Matrix3 duffy_20 = duffy_form_a(
        a, b, test_apex, source_apex, k, 20, green);
    const Matrix3 extracted = extracted_form_a(
        a, b, test_apex, source_apex, k, 20, 20);

    double scale = 0.0;
    for (const auto& row : duffy_20)
        for (const auto& entry : row)
            scale = std::max(scale, std::abs(entry));
    REQUIRE(scale > 0.0);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            INFO("row=" << j << ", col=" << i
                 << ", Duffy=" << duffy_20[j][i]
                 << ", extracted=" << extracted[j][i]);
            CHECK(std::abs(duffy_20[j][i] - duffy_12[j][i]) / scale < 1e-4);
            CHECK(std::abs(extracted[j][i] - duffy_20[j][i]) / scale < 1e-4);
        }
    }
}

TEST_CASE("Form A self-face line contribution cancels after singularity extraction",
          "[Quadrature][nTL][FormA][SelfLine]") {
    // The test and source are the same geometrical triangle. Its scalar nTL
    // contribution is zero after integrating over the complete face boundary.
    const vec3_t triangle[3] = {
        make_vec3(0.0f, 0.0f, 0.0f),
        make_vec3(0.04f, 0.0f, 0.0f),
        make_vec3(0.007f, 0.035f, 0.004f)
    };
    // Distinct labels keep the production self shortcut from bypassing the
    // edge loops; the test and source positions remain exactly coincident.
    const int test_indices[3] = {0, 1, 2};
    const int source_indices[3] = {3, 4, 5};
    const double frequency_hz = 1e9;
    const double epsilon0 = 8.8541878128e-12;
    const double mu0 = 1.25663706212e-6;
    const real_t k = real_t(2.0 * std::acos(-1.0) * frequency_hz *
                            std::sqrt(epsilon0 * mu0));

    const int triangle_rule = 20;
    const int triangle_points = dunavant_order_num(triangle_rule);
    std::vector<double> triangle_nodes_double(2 * triangle_points);
    std::vector<double> triangle_weights_double(triangle_points);
    dunavant_rule(triangle_rule, triangle_points,
                  triangle_nodes_double.data(), triangle_weights_double.data());
    const std::vector<float> triangle_nodes(
        triangle_nodes_double.begin(), triangle_nodes_double.end());
    const std::vector<float> triangle_weights(
        triangle_weights_double.begin(), triangle_weights_double.end());

    const int line_points = 20;
    std::vector<double> line_nodes_double(line_points);
    std::vector<double> line_weights_double(line_points);
    gauss_legendre_rule(line_points, line_nodes_double.data(),
                        line_weights_double.data());
    const std::vector<float> line_nodes(
        line_nodes_double.begin(), line_nodes_double.end());
    const std::vector<float> line_weights(
        line_weights_double.begin(), line_weights_double.end());

    // Zero edge points removes only the line integral; the area integral
    // still uses the same triangle rule in both evaluations.
    const auto evaluate = [&](bool singular, int edge_points) {
        complex__float ntl1[3][3], ntl2[3][3], ntk1[3][3], ntk2[3][3];
        if (singular) {
            calculateFormASingularPartQuadrature_nTL_nTK(
                triangle, test_indices, triangle, source_indices,
                triangle_nodes.data(), triangle_weights.data(), triangle_points,
                line_nodes.data(), line_weights.data(), edge_points,
                k, k, ntl1, ntl2, ntk1, ntk2);
        } else {
            calculateFormAQuadrature_nTL_nTK(
                triangle, test_indices, triangle, source_indices,
                triangle_nodes.data(), triangle_weights.data(), triangle_points,
                line_nodes.data(), line_weights.data(), edge_points,
                k, k, true, ntl1, ntl2, ntk1, ntk2);
        }
        Matrix3 result{};
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                result[j][i] = {double(ntl1[j][i].x), double(ntl1[j][i].y)};
        return result;
    };

    const Matrix3 regular_full = evaluate(false, line_points);
    const Matrix3 regular_area = evaluate(false, 0);
    const Matrix3 singular_full = evaluate(true, line_points);
    const Matrix3 singular_area = evaluate(true, 0);

    double largest_line_part = 0.0;
    double largest_area_entry = 0.0;
    double largest_residual = 0.0;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const Complex regular_line = regular_full[j][i] - regular_area[j][i];
            const Complex singular_line = singular_full[j][i] - singular_area[j][i];
            INFO("row=" << j << ", col=" << i
                 << ", regular line=" << regular_line
                 << ", singular line=" << singular_line);
            CHECK(std::isfinite(std::abs(regular_line)));
            CHECK(std::isfinite(std::abs(singular_line)));
            largest_line_part = std::max({largest_line_part, std::abs(regular_line),
                                          std::abs(singular_line)});
            largest_area_entry = std::max(
                largest_area_entry, std::abs(regular_area[j][i] + singular_area[j][i]));
            largest_residual = std::max(largest_residual,
                                        std::abs(regular_line + singular_line));
        }
    }
    REQUIRE(largest_line_part > 0.0); // The edge loops were exercised.
    REQUIRE(largest_area_entry > 0.0);
    INFO("largest self-line residual=" << largest_residual
         << ", largest individual line part=" << largest_line_part
         << ", largest self-face area entry=" << largest_area_entry);
    CHECK(largest_residual / largest_area_entry < 1e-4);
}
