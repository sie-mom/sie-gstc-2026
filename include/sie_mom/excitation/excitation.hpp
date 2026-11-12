#pragma once

#include "sie_mom/excitation/source_functors.cuh"
#include "sie_mom/fields.hpp"

#include <utility>
#include <variant>
#include <vector>

namespace sie_mom {
namespace excitation {

enum class SourceRegion : int {
    interior = 1,
    exterior = 2,
};

/** Effective moment convention is the one used by cuda::DipoleSource. */
struct DipoleExcitation {
    vec3_t effective_moment = make_vec3(real_t(0), real_t(0), real_t(1));
    vec3_t position = make_vec3(real_t(0), real_t(0), real_t(0));
    bool is_magnetic = false;
    SourceRegion region = SourceRegion::interior;
};

struct PlaneWaveExcitation {
    vec3_t electric_amplitude = make_vec3(real_t(1), real_t(0), real_t(0));
    vec3_t propagation_direction = make_vec3(real_t(0), real_t(0), real_t(1));
    SourceRegion region = SourceRegion::exterior;
};

using SourceSpec = std::variant<DipoleExcitation, PlaneWaveExcitation>;

/**
 * A validated source with its frequency and source-medium constants bound once.
 * Pass the same object to RHS assembly and incident-field evaluation. The
 * variant is visited on the host; only the concrete functor goes to the GPU.
 */
class PreparedExcitation {
public:
    SourceRegion region() const { return region_; }

    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) const {
        return std::visit(std::forward<Visitor>(visitor), source_);
    }

    /** Reject use with a solver whose source-medium physics differ. */
    void validate_environment(
        real_t frequency_hz,
        const IsotropicMedium& interior,
        const IsotropicMedium& exterior,
        real_t vacuum_permittivity,
        real_t vacuum_permeability
    ) const;

private:
    using Source = std::variant<cuda::DipoleSource, cuda::PlaneWave>;

    PreparedExcitation(
        Source source, SourceRegion region, real_t frequency_hz,
        IsotropicMedium medium, real_t vacuum_permittivity,
        real_t vacuum_permeability
    ) : source_(std::move(source)), region_(region),
        frequency_hz_(frequency_hz), medium_(medium),
        vacuum_permittivity_(vacuum_permittivity),
        vacuum_permeability_(vacuum_permeability) {}

    Source source_;
    SourceRegion region_;
    real_t frequency_hz_;
    IsotropicMedium medium_;
    real_t vacuum_permittivity_;
    real_t vacuum_permeability_;

    friend PreparedExcitation prepare_source(
        const SourceSpec&, real_t, const IsotropicMedium&,
        const IsotropicMedium&, real_t, real_t
    );
};

/** Select the source region's medium, validate inputs, and bind k and eta. */
PreparedExcitation prepare_source(
    const SourceSpec& specification,
    real_t frequency_hz,
    const IsotropicMedium& interior,
    const IsotropicMedium& exterior,
    real_t vacuum_permittivity = real_t(8.8541878188e-12),
    real_t vacuum_permeability = real_t(1.25663706127e-6)
);

/**
 * Evaluate the source's homogeneous incident field at arbitrary points.
 * No medium or frequency is supplied again. Regional reconstruction must add
 * this field only in source.region(); this function does not classify points.
 */
std::vector<ElectromagneticFieldSample> evaluate_incident_fields(
    const std::vector<vec3_t>& observation_points,
    const PreparedExcitation& source
);

} // namespace excitation
} // namespace sie_mom
