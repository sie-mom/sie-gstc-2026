#pragma once

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/excitation/excitation.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace sie_mom::cli {

/**
 * @brief Ensures an option has a following value and advances the argument index.
 */
inline const char* require_value(int argc, char* argv[], int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for option ") + argv[index]);
    }
    return argv[++index];
}

/**
 * @brief Parses a real floating-point value with finite-precision validation.
 */
inline real_t parse_real(const char* text, const std::string& option) {
    std::size_t consumed = 0;
    const std::string value(text);
    const double parsed = std::stod(value, &consumed);
    const real_t converted = static_cast<real_t>(parsed);
    if (consumed != value.size() || !std::isfinite(parsed) || !std::isfinite(converted)) {
        throw std::invalid_argument("invalid numeric value for " + option + ": " + value);
    }
    return converted;
}

/**
 * @brief Parses an integer value with range bounds checking.
 */
inline int parse_integer(const char* text, const std::string& option) {
    std::size_t consumed = 0;
    const std::string value(text);
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size()
        || parsed < static_cast<long>(std::numeric_limits<int>::min())
        || parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid integer value for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

/**
 * @brief Parses "interior" or "exterior" into a SourceRegion enum.
 */
inline excitation::SourceRegion parse_region(const std::string& region) {
    if (region == "interior") return excitation::SourceRegion::interior;
    if (region == "exterior") return excitation::SourceRegion::exterior;
    throw std::invalid_argument("source region must be interior or exterior");
}

/**
 * @brief Parses three consecutive real values as a 3D vector (x, y, z).
 */
inline vec3_t parse_vector(int argc, char* argv[], int& index, const std::string& option) {
    const real_t x = parse_real(require_value(argc, argv, index), option);
    const real_t y = parse_real(require_value(argc, argv, index), option);
    const real_t z = parse_real(require_value(argc, argv, index), option);
    return make_vec3(x, y, z);
}

} // namespace sie_mom::cli

