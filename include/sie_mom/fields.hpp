#pragma once

#include "sie_mom/cuda/platform.cuh"

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace sie_mom {

/** Material parameters of one homogeneous isotropic region. */
struct IsotropicMedium {
    real_t relative_permittivity = real_t(1);
    real_t relative_permeability = real_t(1);
};

using ComplexCartesianVector = std::array<std::complex<real_t>, 3>;

/** Physical electric and magnetic fields at one observation point. */
struct ElectromagneticFieldSample {
    ComplexCartesianVector electric;
    ComplexCartesianVector magnetic;
};

/** Computes difference (left - right) between two electromagnetic field samples. */
inline ElectromagneticFieldSample subtract_fields(
    const ElectromagneticFieldSample& left,
    const ElectromagneticFieldSample& right
) {
    ElectromagneticFieldSample diff;
    for (std::size_t component = 0; component < 3; ++component) {
        diff.electric[component] = left.electric[component] - right.electric[component];
        diff.magnetic[component] = left.magnetic[component] - right.magnetic[component];
    }
    return diff;
}

inline ElectromagneticFieldSample operator-(
    const ElectromagneticFieldSample& left,
    const ElectromagneticFieldSample& right
) {
    return subtract_fields(left, right);
}

/**
 * @brief Extracts a sub-vector corresponding to one RWG block (e.g. J1, M1, J2, M2)
 * from a contiguous multi-block current vector.
 */
inline std::vector<std::complex<real_t>> current_block(
    const std::vector<std::complex<real_t>>& currents,
    std::size_t block,
    std::size_t edge_count
) {
    const auto first = currents.begin() + static_cast<std::ptrdiff_t>(block * edge_count);
    return {first, first + static_cast<std::ptrdiff_t>(edge_count)};
}

} // namespace sie_mom
