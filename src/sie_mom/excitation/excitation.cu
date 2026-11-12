#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/sie_kernels_utils.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace sie_mom {
namespace excitation {
namespace {

void require_positive(real_t value, const char* name) {
    if (!(value > real_t(0)) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be positive and finite");
    }
}

void require_finite(const vec3_t& value, const char* name) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

const IsotropicMedium& source_medium(
    SourceRegion region, const IsotropicMedium& interior, const IsotropicMedium& exterior
) {
    if (region == SourceRegion::interior) return interior;
    if (region == SourceRegion::exterior) return exterior;
    throw std::invalid_argument("source region must be interior or exterior");
}

cuda::DipoleSource make_functor(const DipoleExcitation& spec, real_t k, real_t eta) {
    require_finite(spec.position, "dipole position");
    require_finite(spec.effective_moment, "dipole effective moment");
    require_positive(std::hypot(spec.effective_moment.x, spec.effective_moment.y,
                                spec.effective_moment.z), "dipole moment magnitude");
    return {spec.effective_moment, spec.position, k, eta, spec.is_magnetic};
}

cuda::PlaneWave make_functor(const PlaneWaveExcitation& spec, real_t k, real_t eta) {
    require_finite(spec.electric_amplitude, "plane-wave electric amplitude");
    require_finite(spec.propagation_direction, "plane-wave propagation direction");
    const auto& direction = spec.propagation_direction;
    const real_t direction_norm = std::hypot(direction.x, direction.y, direction.z);
    const auto& amplitude = spec.electric_amplitude;
    const real_t amplitude_norm = std::hypot(amplitude.x, amplitude.y, amplitude.z);
    require_positive(direction_norm, "plane-wave direction magnitude");
    require_positive(amplitude_norm, "plane-wave amplitude magnitude");
    const vec3_t unit_direction = direction / direction_norm;
    const real_t longitudinal_fraction = dot(unit_direction, amplitude / amplitude_norm);
    if (std::abs(longitudinal_fraction) > real_t(64) * std::numeric_limits<real_t>::epsilon()) {
        throw std::invalid_argument("plane-wave electric amplitude must be transverse to propagation");
    }
    return {amplitude, unit_direction, k, eta};
}

using namespace sie_mom::cuda;

void require_cuda_device() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices for field evaluation");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable device is available for field evaluation");
    }
}

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source, const char* label) {
    if (!source.empty()) {
        check_cuda(
            cudaMemcpy(
                destination.data(), source.data(), source.size() * sizeof(T),
                cudaMemcpyHostToDevice
            ),
            label
        );
    }
}

ComplexCartesianVector to_host_vector(const cvec3_t& value) {
    return {
        std::complex<real_t>(value.x.x, value.x.y),
        std::complex<real_t>(value.y.x, value.y.y),
        std::complex<real_t>(value.z.x, value.z.y),
    };
}

std::vector<ElectromagneticFieldSample> to_host_samples(
    const std::vector<cvec3_t>& electric,
    const std::vector<cvec3_t>& magnetic
) {
    std::vector<ElectromagneticFieldSample> samples;
    samples.reserve(electric.size());
    for (std::size_t index = 0; index < electric.size(); ++index) {
        samples.push_back({
            to_host_vector(electric[index]),
            to_host_vector(magnetic[index]),
        });
    }
    return samples;
}

} // namespace

PreparedExcitation prepare_source(
    const SourceSpec& specification,
    real_t frequency_hz,
    const IsotropicMedium& interior,
    const IsotropicMedium& exterior,
    real_t vacuum_permittivity,
    real_t vacuum_permeability
) {
    require_positive(frequency_hz, "source frequency");
    require_positive(vacuum_permittivity, "vacuum permittivity");
    require_positive(vacuum_permeability, "vacuum permeability");
    return std::visit([&](const auto& spec) {
        const auto& medium = source_medium(spec.region, interior, exterior);
        require_positive(medium.relative_permittivity, "source relative permittivity");
        require_positive(medium.relative_permeability, "source relative permeability");
        const real_t epsilon = medium.relative_permittivity * vacuum_permittivity;
        const real_t mu = medium.relative_permeability * vacuum_permeability;
        const real_t omega = real_t(2) * std::acos(real_t(-1)) * frequency_hz;
        require_positive(epsilon, "source permittivity");
        require_positive(mu, "source permeability");
        const real_t k = omega * std::sqrt(mu * epsilon);
        const real_t eta = std::sqrt(mu / epsilon);
        require_positive(k, "source wavenumber");
        require_positive(eta, "source impedance");
        return PreparedExcitation(make_functor(spec, k, eta), spec.region, frequency_hz,
                                  medium, vacuum_permittivity, vacuum_permeability);
    }, specification);
}

void PreparedExcitation::validate_environment(
    real_t frequency_hz,
    const IsotropicMedium& interior,
    const IsotropicMedium& exterior,
    real_t vacuum_permittivity,
    real_t vacuum_permeability
) const {
    const auto& medium = source_medium(region_, interior, exterior);
    if (frequency_hz != frequency_hz_
        || medium.relative_permittivity != medium_.relative_permittivity
        || medium.relative_permeability != medium_.relative_permeability
        || vacuum_permittivity != vacuum_permittivity_
        || vacuum_permeability != vacuum_permeability_) {
        throw std::invalid_argument(
            "prepared excitation does not match the solver frequency, source medium, or vacuum constants"
        );
    }
}

std::vector<ElectromagneticFieldSample> evaluate_incident_fields(
    const std::vector<vec3_t>& observation_points,
    const PreparedExcitation& prepared
) {
    for (const auto& point : observation_points) {
        require_finite(point, "incident-field observation point");
    }
    if (observation_points.empty()) {
        return {};
    }
    require_cuda_device();
    DeviceBuffer<vec3_t> device_observation_points(observation_points.size());
    DeviceBuffer<cvec3_t> device_electric_fields(observation_points.size());
    DeviceBuffer<cvec3_t> device_magnetic_fields(observation_points.size());
    copy_to_device(
        device_observation_points, observation_points,
        "copy incident observation points"
    );

    constexpr unsigned int threads = 256;
    const dim3 blocks(static_cast<unsigned int>(
        (observation_points.size() + threads - 1) / threads
    ));
    prepared.visit([&](const auto& source) {
        compute_incident_field_kernel<<<blocks, threads>>>(
            device_observation_points.data(),
            observation_points.size(),
            source,
            device_electric_fields.data(),
            device_magnetic_fields.data()
        );
    });
    check_last_kernel("evaluate incident fields");

    std::vector<cvec3_t> host_electric(observation_points.size());
    std::vector<cvec3_t> host_magnetic(observation_points.size());
    check_cuda(
        cudaMemcpy(
            host_electric.data(), device_electric_fields.data(),
            host_electric.size() * sizeof(cvec3_t), cudaMemcpyDeviceToHost
        ),
        "copy incident electric fields to host"
    );
    check_cuda(
        cudaMemcpy(
            host_magnetic.data(), device_magnetic_fields.data(),
            host_magnetic.size() * sizeof(cvec3_t), cudaMemcpyDeviceToHost
        ),
        "copy incident magnetic fields to host"
    );
    return to_host_samples(host_electric, host_magnetic);
}

} // namespace excitation
} // namespace sie_mom
