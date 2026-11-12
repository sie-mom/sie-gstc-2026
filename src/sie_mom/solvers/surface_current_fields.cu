/**
 * @file surface_current_fields.cu
 * @brief High-performance GPU evaluation of electromagnetic fields radiated by surface currents.
 *
 * This file provides routines (like `evaluate_surface_integral_fields`) to compute the exact
 * near and far-field electromagnetic vectors resulting from a solved set of RWG surface currents.
 *
 * Crucially, these functions expect the raw, unscaled physical electric (J) and magnetic (M)
 * current coefficients. They internally apply the exact free-space or dielectric L and K
 * continuous integral operators to compute E and H without any assumptions about the conditioning
 * scaling that may have occurred during the Method of Moments (MoM) matrix solve.
 */

#include "sie_mom/solvers/surface_current_fields.hpp"

#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/sie_kernels_utils.cuh"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sie_mom {
namespace solvers {
namespace {

using namespace sie_mom::cuda;
using sie_mom::quadrature::QuadratureData;
using sie_mom::quadrature::make_quadrature;

struct MediumConstants {
    real_t omega;
    real_t epsilon;
    real_t mu;
    real_t wavenumber;
};

bool finite(real_t value) {
    return std::isfinite(static_cast<double>(value));
}

void validate_common_inputs(
    const std::vector<vec3_t>& observation_points,
    real_t frequency_hz,
    const IsotropicMedium& medium,
    int quadrature_rule,
    real_t vacuum_permittivity,
    real_t vacuum_permeability
) {
    if (!(frequency_hz > real_t(0)) || !finite(frequency_hz)) {
        throw std::invalid_argument("field-evaluation frequency must be positive and finite");
    }
    if (!(medium.relative_permittivity > real_t(0))
        || !(medium.relative_permeability > real_t(0))
        || !finite(medium.relative_permittivity)
        || !finite(medium.relative_permeability)) {
        throw std::invalid_argument(
            "field-evaluation relative permittivity and permeability must be positive and finite"
        );
    }
    if (!(vacuum_permittivity > real_t(0))
        || !(vacuum_permeability > real_t(0))
        || !finite(vacuum_permittivity)
        || !finite(vacuum_permeability)) {
        throw std::invalid_argument(
            "field-evaluation vacuum constants must be positive and finite"
        );
    }
    if (quadrature_rule < 1 || quadrature_rule > 20) {
        throw std::invalid_argument("Dunavant quadrature rule must be in [1, 20]");
    }
    for (std::size_t index = 0; index < observation_points.size(); ++index) {
        const auto& point = observation_points[index];
        if (!finite(point.x) || !finite(point.y) || !finite(point.z)) {
            throw std::invalid_argument(
                "non-finite observation point at index " + std::to_string(index)
            );
        }
    }
}

void require_cuda_device() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices for field evaluation");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable device is available for field evaluation");
    }
}

MediumConstants make_medium_constants(
    real_t frequency_hz,
    const IsotropicMedium& medium,
    real_t vacuum_permittivity,
    real_t vacuum_permeability
) {
    const real_t pi = std::acos(real_t(-1));
    const real_t omega = real_t(2) * pi * frequency_hz;
    const real_t epsilon = medium.relative_permittivity * vacuum_permittivity;
    const real_t mu = medium.relative_permeability * vacuum_permeability;
    return {
        omega,
        epsilon,
        mu,
        omega * std::sqrt(mu * epsilon),
    };
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

std::vector<complex__float> to_cuda_complex(
    const std::vector<std::complex<real_t>>& values
) {
    std::vector<complex__float> converted;
    converted.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto& value = values[index];
        if (!finite(value.real()) || !finite(value.imag())) {
            throw std::invalid_argument(
                "non-finite surface-current coefficient at index "
                + std::to_string(index)
            );
        }
        converted.push_back(make_complex(value.real(), value.imag()));
    }
    return converted;
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

std::vector<ElectromagneticFieldSample> evaluate_surface_integral_fields(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& electric_currents,
    const std::vector<std::complex<real_t>>& magnetic_currents,
    const std::vector<vec3_t>& observation_points,
    real_t frequency_hz,
    const IsotropicMedium& medium,
    int quadrature_rule,
    real_t vacuum_permittivity,
    real_t vacuum_permeability
) {
    validate_common_inputs(
        observation_points, frequency_hz, medium, quadrature_rule,
        vacuum_permittivity, vacuum_permeability
    );
    if (observation_points.empty()) {
        return {};
    }
    if (mesh.vertices.empty() || mesh.faces.empty() || mesh.edges.empty()) {
        throw std::invalid_argument(
            "field-evaluation mesh must contain vertices, faces, and edges"
        );
    }
    if (mesh.faces_edges.size() != mesh.faces.size()) {
        throw std::invalid_argument(
            "field-evaluation face-to-edge mapping size does not match face count"
        );
    }
    if (mesh.faces.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("field-evaluation face count exceeds the CUDA kernel limit");
    }
    if (electric_currents.size() != mesh.edges.size()
        || magnetic_currents.size() != mesh.edges.size()) {
        throw std::invalid_argument(
            "field evaluation requires one electric and magnetic coefficient per mesh edge"
        );
    }

    require_cuda_device();
    const auto quadrature = make_quadrature(quadrature_rule);
    const auto physical = make_medium_constants(
        frequency_hz, medium, vacuum_permittivity, vacuum_permeability
    );
    const auto cuda_electric_currents = to_cuda_complex(electric_currents);
    const auto cuda_magnetic_currents = to_cuda_complex(magnetic_currents);

    DeviceBuffer<vec3_t> device_vertices(mesh.vertices.size());
    DeviceBuffer<int3> device_faces(mesh.faces.size());
    DeviceBuffer<int3> device_faces_edges(mesh.faces_edges.size());
    DeviceBuffer<float> device_quadrature_points(quadrature.points.size());
    DeviceBuffer<float> device_quadrature_weights(quadrature.weights.size());
    DeviceBuffer<vec3_t> device_observation_points(observation_points.size());
    DeviceBuffer<complex__float> device_electric_currents(electric_currents.size());
    DeviceBuffer<complex__float> device_magnetic_currents(magnetic_currents.size());
    DeviceBuffer<cvec3_t> device_electric_fields(observation_points.size());
    DeviceBuffer<cvec3_t> device_magnetic_fields(observation_points.size());

    copy_to_device(device_vertices, mesh.vertices, "copy field-evaluation vertices");
    copy_to_device(device_faces, mesh.faces, "copy field-evaluation faces");
    copy_to_device(
        device_faces_edges, mesh.faces_edges, "copy field-evaluation face-edge mapping"
    );
    copy_to_device(
        device_quadrature_points, quadrature.points,
        "copy field-evaluation quadrature points"
    );
    copy_to_device(
        device_quadrature_weights, quadrature.weights,
        "copy field-evaluation quadrature weights"
    );
    copy_to_device(
        device_observation_points, observation_points,
        "copy field-evaluation observation points"
    );
    copy_to_device(
        device_electric_currents, cuda_electric_currents,
        "copy physical electric-current coefficients"
    );
    copy_to_device(
        device_magnetic_currents, cuda_magnetic_currents,
        "copy physical magnetic-current coefficients"
    );

    constexpr unsigned int threads = 128;
    const dim3 blocks(static_cast<unsigned int>(observation_points.size()));
    const std::size_t shared_memory_bytes = 2 * threads * sizeof(cvec3_t);
    compute_scattered_fields_kernel<<<blocks, threads, shared_memory_bytes>>>(
        device_vertices.data(),
        device_faces.data(),
        device_faces_edges.data(),
        static_cast<int>(mesh.faces.size()),
        device_quadrature_points.data(),
        device_quadrature_weights.data(),
        quadrature.order,
        device_observation_points.data(),
        observation_points.size(),
        device_electric_currents.data(),
        device_magnetic_currents.data(),
        make_complex(real_t(0), physical.omega * physical.mu),
        make_complex(real_t(1), real_t(0)),
        make_complex(real_t(-1), real_t(0)),
        make_complex(real_t(0), physical.omega * physical.epsilon),
        physical.wavenumber,
        device_electric_fields.data(),
        device_magnetic_fields.data()
    );
    check_last_kernel("evaluate surface-current integral fields");

    std::vector<cvec3_t> host_electric(observation_points.size());
    std::vector<cvec3_t> host_magnetic(observation_points.size());
    check_cuda(
        cudaMemcpy(
            host_electric.data(), device_electric_fields.data(),
            host_electric.size() * sizeof(cvec3_t), cudaMemcpyDeviceToHost
        ),
        "copy evaluated electric fields to host"
    );
    check_cuda(
        cudaMemcpy(
            host_magnetic.data(), device_magnetic_fields.data(),
            host_magnetic.size() * sizeof(cvec3_t), cudaMemcpyDeviceToHost
        ),
        "copy evaluated magnetic fields to host"
    );
    return to_host_samples(host_electric, host_magnetic);
}


} // namespace solvers
} // namespace sie_mom
