#pragma once

#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/fields.hpp"

#include <complex>
#include <vector>

namespace sie_mom {
namespace solvers {

/**
 * Evaluate the positive surface-integral terms for one physical current pair.
 *
 * With the exp(+j omega t) convention, the returned quantities are
 *
 *   Q_E = +j omega mu L[J] + K[M],
 *   Q_H = -K[J] + j omega epsilon L[M].
 *
 * They are not the physical scattered or total fields by themselves.  The
 * regional representation is F_total = F_incident - Q.
 *
 * electric_currents and magnetic_currents contain one physical RWG
 * coefficient per mesh edge. In particular, magnetic_currents must contain M,
 * not the internally scaled M / eta0 used during the dense SIE solve.
 *
 * The kernel independently selects ordinary quadrature or regular quadrature
 * plus singularity extraction for each observation/triangle pair, using a
 * centroid-distance threshold of 0.1 times the wavelength in this medium.
 * The caller supplies regional points/current pairs; integration does not
 * classify points. Exact on-surface trace conventions are separate from this
 * near-surface integration treatment.
 */
std::vector<ElectromagneticFieldSample> evaluate_surface_integral_fields(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& electric_currents,
    const std::vector<std::complex<real_t>>& magnetic_currents,
    const std::vector<vec3_t>& observation_points,
    real_t frequency_hz,
    const IsotropicMedium& medium,
    int quadrature_rule = 17,
    real_t vacuum_permittivity = real_t(8.8541878188e-12),
    real_t vacuum_permeability = real_t(1.25663706127e-6)
);

} // namespace solvers
} // namespace sie_mom
