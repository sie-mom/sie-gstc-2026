#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

namespace {
constexpr double kWeightTolerance = 1e-12;
constexpr double kCoordinateTolerance = 1e-12;
}

TEST_CASE("Dunavant metadata is consistent", "[dunavant]")
{
    const int rule_num = dunavant_rule_num();
    REQUIRE(rule_num > 0);

    for (int rule = 1; rule <= rule_num; ++rule)
    {
        INFO("rule=" << rule);
        const int order_num = dunavant_order_num(rule);
        const int degree = dunavant_degree(rule);

        CHECK(order_num > 0);
        CHECK(degree >= 1);
    }
}

TEST_CASE("Dunavant weights sum to unity", "[dunavant]")
{
    const int rule_num = dunavant_rule_num();
    REQUIRE(rule_num > 0);

    for (int rule = 1; rule <= rule_num; ++rule)
    {
        const int order_num = dunavant_order_num(rule);
        std::vector<double> xy(2 * order_num);
        std::vector<double> w(order_num);

        dunavant_rule(rule, order_num, xy.data(), w.data());

        const double weight_sum = std::accumulate(w.begin(), w.end(), 0.0);
        INFO("rule=" << rule << ", order=" << order_num);
        CHECK(weight_sum == Approx(1.0).margin(kWeightTolerance));
    }
}

TEST_CASE("Dunavant suborders give valid barycentric coordinates", "[dunavant]")
{
    const int rule_num = dunavant_rule_num();
    REQUIRE(rule_num > 0);

    for (int rule = 1; rule <= rule_num; ++rule)
    {
        const int suborder_num = dunavant_suborder_num(rule);
        std::vector<double> suborder_xyz(3 * suborder_num);
        std::vector<double> suborder_w(suborder_num);

        dunavant_subrule(rule, suborder_num, suborder_xyz.data(), suborder_w.data());

        for (int idx = 0; idx < suborder_num; ++idx)
        {
            const double sum = suborder_xyz[idx * 3 + 0]
                             + suborder_xyz[idx * 3 + 1]
                             + suborder_xyz[idx * 3 + 2];

            INFO("rule=" << rule << ", suborder=" << idx);
            CHECK(sum == Approx(1.0).margin(kCoordinateTolerance));
        }
    }
}

TEST_CASE("Dunavant rules integrate monomials exactly up to their degree", "[dunavant]")
{
    const int rule_num = dunavant_rule_num();
    REQUIRE(rule_num > 0);

    const int rule_max = std::min(rule_num, 10);
    const double reference_area = 0.5; // Unit reference triangle area.

    for (int a = 0; a <= 10; ++a)
    {
        for (int b = 0; b <= 10 - a; ++b)
        {
            // Scale factor so that integral of the monomial over the reference triangle is 1.
            double coef = static_cast<double>(a + b + 2) * static_cast<double>(a + b + 1);
            for (int i = 1; i <= b; ++i)
            {
                coef *= static_cast<double>(a + i) / static_cast<double>(i);
            }

            const int monomial_degree = a + b;

            for (int rule = 1; rule <= rule_max; ++rule)
            {
                const int rule_degree = dunavant_degree(rule);
                if (monomial_degree > rule_degree)
                {
                    continue;
                }

                const int order_num = dunavant_order_num(rule);
                std::vector<double> xy(2 * order_num);
                std::vector<double> w(order_num);

                dunavant_rule(rule, order_num, xy.data(), w.data());

                double quad = 0.0;
                for (int order = 0; order < order_num; ++order)
                {
                    const double x = xy[order * 2 + 0];
                    const double y = xy[order * 2 + 1];

                    double value = coef;
                    if (a != 0)
                    {
                        value *= std::pow(x, a);
                    }
                    if (b != 0)
                    {
                        value *= std::pow(y, b);
                    }

                    quad += w[order] * value;
                }

                quad *= reference_area;
                const double err = std::abs(1.0 - quad);

                INFO("rule=" << rule
                              << ", a=" << a
                              << ", b=" << b
                              << ", monomial_degree=" << monomial_degree
                              << ", rule_degree=" << rule_degree);
                CHECK(err == Approx(0.0).margin(1e-10));
            }
        }
    }
}

TEST_CASE("Reference-to-physical mapping is consistent", "[dunavant]")
{
    std::array<double, 6> ref_nodes{0.0, 0.0,
                                    1.0, 0.0,
                                    0.0, 1.0};
    std::array<double, 6> phys_nodes{1.0, 2.0,
                                     1.0, 1.0,
                                     3.0, 2.0};

    const int rule = 2;
    const int order_num = dunavant_order_num(rule);

    std::vector<double> xy(2 * order_num);
    std::vector<double> xy_phys(2 * order_num);
    std::vector<double> w(order_num);

    dunavant_rule(rule, order_num, xy.data(), w.data());

    reference_to_physical_t3(phys_nodes.data(),
                             order_num,
                             xy.data(),
                             xy_phys.data());

    const double area_ref = triangle_area(ref_nodes.data());
    const double area_phys = triangle_area(phys_nodes.data());

    CHECK(area_ref == Approx(0.5).margin(kCoordinateTolerance));
    CHECK(area_phys == Approx(1.0).margin(kCoordinateTolerance));

    for (int order = 0; order < order_num; ++order)
    {
        const double r = xy[order * 2 + 0];
        const double s = xy[order * 2 + 1];
        const double t = 1.0 - r - s;

        const double expected_x = t * phys_nodes[0] + r * phys_nodes[2] + s * phys_nodes[4];
        const double expected_y = t * phys_nodes[1] + r * phys_nodes[3] + s * phys_nodes[5];

        INFO("order=" << order);
        CHECK(xy_phys[order * 2 + 0] == Approx(expected_x).margin(kCoordinateTolerance));
        CHECK(xy_phys[order * 2 + 1] == Approx(expected_y).margin(kCoordinateTolerance));
    }

    const double weight_sum = std::accumulate(w.begin(), w.end(), 0.0);
    CHECK(weight_sum == Approx(1.0).margin(kWeightTolerance));

    const double constant_integral = area_phys * weight_sum;
    CHECK(constant_integral == Approx(area_phys).margin(1e-12));
}