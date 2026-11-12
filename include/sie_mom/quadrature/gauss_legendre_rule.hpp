#pragma once

// Gauss-Legendre quadrature rules for the reference interval [0, 1].
// An N-point rule integrates polynomials through degree 2*N - 1 in exact
// arithmetic. Nodes are ordered increasingly, and the weights sum to 1.
// For a physical edge, map t to a + t*(b - a) and apply the edge length
// separately; neither operation is performed by this interface.

// Largest supported number of quadrature points in the compiled tables.
int gauss_legendre_max_points() noexcept;

// Copy the selected rule into arrays of at least point_count doubles each.
// Valid point counts are 1 through gauss_legendre_max_points().
// Throws std::out_of_range for an unsupported count and std::invalid_argument
// if either output array is null.
void gauss_legendre_rule(int point_count, double nodes[], double weights[]);
