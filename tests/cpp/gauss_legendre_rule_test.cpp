#include "catch.hpp"

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "sie_mom/quadrature/gauss_legendre_rule.hpp"

TEST_CASE("Gauss-Legendre rules use the normalized unit interval", "[gauss_legendre]")
{
    const int max_points = gauss_legendre_max_points();
    REQUIRE(max_points >= 20);

    for (int point_count = 1; point_count <= max_points; ++point_count)
    {
        std::vector<double> nodes(point_count);
        std::vector<double> weights(point_count);
        gauss_legendre_rule(point_count, nodes.data(), weights.data());

        INFO("point_count=" << point_count);
        CHECK(std::abs(std::accumulate(weights.begin(), weights.end(), 0.0) - 1.0) < 1e-12);
        for (int index = 0; index < point_count; ++index)
        {
            CHECK(nodes[index] > 0.0);
            CHECK(nodes[index] < 1.0);
            CHECK(weights[index] > 0.0);
            CHECK(std::abs(nodes[index] + nodes[point_count - 1 - index] - 1.0) < 1e-12);
            CHECK(std::abs(weights[index] - weights[point_count - 1 - index]) < 1e-12);
            if (index > 0)
            {
                CHECK(nodes[index - 1] < nodes[index]);
            }
        }
    }
}

TEST_CASE("Gauss-Legendre rules integrate polynomials through degree 2N-1", "[gauss_legendre]")
{
    for (int point_count = 1; point_count <= gauss_legendre_max_points(); ++point_count)
    {
        std::vector<double> nodes(point_count);
        std::vector<double> weights(point_count);
        gauss_legendre_rule(point_count, nodes.data(), weights.data());

        for (int degree = 0; degree <= 2 * point_count - 1; ++degree)
        {
            double approximation = 0.0;
            for (int index = 0; index < point_count; ++index)
            {
                approximation += weights[index] * std::pow(nodes[index], degree);
            }

            // Integral from 0 to 1 of t^degree dt = 1 / (degree + 1).
            const double exact = 1.0 / (degree + 1);
            INFO("point_count=" << point_count << ", degree=" << degree);
            CHECK(std::abs(approximation - exact) < 1e-12);
        }
    }
}

TEST_CASE("Gauss-Legendre error decreases for a smooth nonpolynomial", "[gauss_legendre]")
{
    // Integral from 0 to 1 of 1/(1+t) dt = log(2).
    constexpr std::array<int, 6> point_counts{{1, 2, 3, 4, 6, 8}};
    const double exact = std::log(2.0);
    double previous_error = 1.0;

    for (int point_count : point_counts)
    {
        std::vector<double> nodes(point_count);
        std::vector<double> weights(point_count);
        gauss_legendre_rule(point_count, nodes.data(), weights.data());

        double approximation = 0.0;
        for (int index = 0; index < point_count; ++index)
        {
            approximation += weights[index] / (1.0 + nodes[index]);
        }

        const double error = std::abs(approximation - exact);
        INFO("point_count=" << point_count << ", error=" << error);
        CHECK(error < previous_error);
        previous_error = error;
    }
    CHECK(previous_error < 1e-11);
}

TEST_CASE("Gauss-Legendre rejects invalid point counts and null arrays", "[gauss_legendre]")
{
    double node = 0.0;
    double weight = 0.0;
    CHECK_THROWS_AS(gauss_legendre_rule(0, &node, &weight), std::out_of_range);
    CHECK_THROWS_AS(gauss_legendre_rule(gauss_legendre_max_points() + 1, &node, &weight),
                    std::out_of_range);
    CHECK_THROWS_AS(gauss_legendre_rule(1, nullptr, &weight), std::invalid_argument);
    CHECK_THROWS_AS(gauss_legendre_rule(1, &node, nullptr), std::invalid_argument);
}
