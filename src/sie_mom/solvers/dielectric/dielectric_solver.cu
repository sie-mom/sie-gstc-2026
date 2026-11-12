/**
 * @file dielectric_solver.cu
 * @brief Unified GPU-accelerated dense solver for dielectric bodies using T-PMCHWT and N-PMCHWT formulations.
 *
 * This file implements the `solve_pmchwt_dense()` function, which acts as a unified coordinator
 * for both the standard RWG-tested PMCHWT and the rotated n x RWG tested N-PMCHWT methods.
 * It manages the dense matrix assembly using massively parallel CUDA kernels and solves the
 * resulting 2x2 block linear system using cuSolverDn.
 *
 * Both the T-PMCHWT and N-PMCHWT kernels implemented in this codebase (`pmchwt_ii_kernel` and `n_pmchwt_kernel`)
 * construct their unscaled algebraic systems, naturally returning raw, un-conditioned physical J and M currents.
 * Thus, this solver avoids any artificial eta0 pre-conditioning steps on the unknowns.
 */

#include "sie_mom/solvers/dielectric/dielectric_solver.hpp"

#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/dense_lu_solver.cuh"
#include "sie_mom/cuda/sie_kernels_assembly.cuh"
#include "sie_mom/cuda/sie_kernels_utils.cuh"
#include "sie_mom/solvers/dielectric/dielectric_kernels.cuh"
#include "sie_mom/quadrature/gauss_legendre_rule.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sie_mom {
namespace solvers {
namespace {

using Clock = std::chrono::steady_clock;

using namespace sie_mom::cuda;

struct QuadratureData {
    // Triangle rule, used by the matrix and right-hand side.
    std::vector<float> points;
    std::vector<float> weights;
    int order = 0;
    // Testing-edge rule, populated only for N-PMCHWT Form A.
    std::vector<float> line_points;
    std::vector<float> line_weights;
    int line_order = 0;
};

struct PhysicalParameters {
    real_t omega;
    real_t k_interior;
    real_t k_exterior;
    real_t eta0;
    real_t singularity_threshold;
};

struct DeviceProblemData {
    DeviceBuffer<vec3_t> vertices;
    DeviceBuffer<int3> faces;
    DeviceBuffer<int3> faces_edges;
    DeviceBuffer<float> quadrature_points;
    DeviceBuffer<float> quadrature_weights;
    DeviceBuffer<float> line_quadrature_points;
    DeviceBuffer<float> line_quadrature_weights;
};

double elapsed_seconds(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

double vector_norm(const std::vector<complex__float>& values) {
    long double sum = 0.0L;
    for (const auto& value : values) {
        const long double real = static_cast<long double>(value.x);
        const long double imag = static_cast<long double>(value.y);
        sum += real * real + imag * imag;
    }
    return std::sqrt(static_cast<double>(sum));
}

void validate_problem(
    const mesh::MeshData& mesh,
    const DielectricOptions& options
) {
    if (mesh.vertices.empty() || mesh.faces.empty() || mesh.edges.empty()) {
        throw std::invalid_argument("SIE mesh must contain vertices, faces, and edges");
    }
    if (mesh.faces_edges.size() != mesh.faces.size()) {
        throw std::invalid_argument("mesh face-to-edge mapping size does not match face count");
    }
    if (!(options.frequency_hz > real_t(0))) {
        throw std::invalid_argument("SIE frequency must be positive");
    }
    if (!(options.interior.relative_permittivity > real_t(0))
        || !(options.interior.relative_permeability > real_t(0))
        || !(options.exterior.relative_permittivity > real_t(0))
        || !(options.exterior.relative_permeability > real_t(0))) {
        throw std::invalid_argument("relative permittivities and permeabilities must be positive");
    }
    if (!(options.vacuum_permittivity > real_t(0))
        || !(options.vacuum_permeability > real_t(0))) {
        throw std::invalid_argument("vacuum permittivity and permeability must be positive");
    }
    if (options.quadrature_rule < 1 || options.quadrature_rule > 20) {
        throw std::invalid_argument("Dunavant quadrature rule must be in [1, 20]");
    }
    if ((options.formulation == DielectricFormulation::N_PMCHWT ||
         options.formulation == DielectricFormulation::N_Muller)
        && options.ntl_form == NtlIntegralForm::Line
        && (options.line_quadrature_points < 1
            || options.line_quadrature_points > gauss_legendre_max_points())) {
        throw std::invalid_argument("Gauss-Legendre point count must be in [1, 20]");
    }
    if (options.formulation == DielectricFormulation::T_PMCHWT
        && options.ntl_form == NtlIntegralForm::Line) {
        throw std::invalid_argument("line nTL form requires N-PMCHWT or N-Muller formulation");
    }
    if (!(options.singularity_threshold_lambda0 > real_t(0))) {
        throw std::invalid_argument("singularity threshold in lambda0 must be positive");
    }
}

PhysicalParameters make_physical_parameters(const DielectricOptions& options) {
    const real_t pi = std::acos(real_t(-1));
    const real_t omega = real_t(2) * pi * options.frequency_hz;

    const real_t epsilon1 =
        options.interior.relative_permittivity * options.vacuum_permittivity;
    const real_t mu1 =
        options.interior.relative_permeability * options.vacuum_permeability;
    const real_t k1 = omega * std::sqrt(mu1 * epsilon1);

    const real_t epsilon2 =
        options.exterior.relative_permittivity * options.vacuum_permittivity;
    const real_t mu2 =
        options.exterior.relative_permeability * options.vacuum_permeability;
    const real_t k2 = omega * std::sqrt(mu2 * epsilon2);

    const real_t eta0 = std::sqrt(
        options.vacuum_permeability / options.vacuum_permittivity
    );
    const real_t lambda0 = real_t(2) * pi / (omega * std::sqrt(
        options.vacuum_permeability * options.vacuum_permittivity
    ));
    const real_t singularity_threshold =
        options.singularity_threshold_lambda0 * lambda0;

    return {omega, k1, k2, eta0, singularity_threshold};
}

QuadratureData make_quadrature(const DielectricOptions& options) {
    QuadratureData quadrature;
    quadrature.order = dunavant_order_num(options.quadrature_rule);
    std::vector<double> points_double(2 * quadrature.order);
    std::vector<double> weights_double(quadrature.order);
    dunavant_rule(
        options.quadrature_rule,
        quadrature.order,
        points_double.data(),
        weights_double.data()
    );
    quadrature.points.resize(2 * quadrature.order);
    quadrature.weights.resize(quadrature.order);
    for (int index = 0; index < quadrature.order; ++index) {
        quadrature.points[2 * index] = static_cast<float>(points_double[2 * index]);
        quadrature.points[2 * index + 1] =
            static_cast<float>(points_double[2 * index + 1]);
        quadrature.weights[index] = static_cast<float>(weights_double[index]);
    }
    if ((options.formulation == DielectricFormulation::N_PMCHWT ||
         options.formulation == DielectricFormulation::N_Muller)
        && options.ntl_form == NtlIntegralForm::Line) {
        quadrature.line_order = options.line_quadrature_points;
        std::vector<double> line_points_double(quadrature.line_order);
        std::vector<double> line_weights_double(quadrature.line_order);
        gauss_legendre_rule(
            quadrature.line_order, line_points_double.data(), line_weights_double.data()
        );
        quadrature.line_points.resize(quadrature.line_order);
        quadrature.line_weights.resize(quadrature.line_order);
        for (int index = 0; index < quadrature.line_order; ++index) {
            quadrature.line_points[index] = static_cast<float>(line_points_double[index]);
            quadrature.line_weights[index] = static_cast<float>(line_weights_double[index]);
        }
    }
    return quadrature;
}

DeviceProblemData copy_problem_to_device(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature
) {
    DeviceProblemData device;
    device.vertices.allocate(mesh.vertices.size());
    check_cuda(
        cudaMemcpy(
            device.vertices.data(), mesh.vertices.data(),
            mesh.vertices.size() * sizeof(vec3_t), cudaMemcpyHostToDevice
        ),
        "copy mesh vertices"
    );

    device.faces.allocate(mesh.faces.size());
    check_cuda(
        cudaMemcpy(
            device.faces.data(), mesh.faces.data(),
            mesh.faces.size() * sizeof(int3), cudaMemcpyHostToDevice
        ),
        "copy mesh faces"
    );

    device.faces_edges.allocate(mesh.faces_edges.size());
    check_cuda(
        cudaMemcpy(
            device.faces_edges.data(), mesh.faces_edges.data(),
            mesh.faces_edges.size() * sizeof(int3), cudaMemcpyHostToDevice
        ),
        "copy mesh face-to-edge mapping"
    );

    device.quadrature_points.allocate(quadrature.points.size());
    check_cuda(
        cudaMemcpy(
            device.quadrature_points.data(), quadrature.points.data(),
            quadrature.points.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "copy quadrature points"
    );

    device.quadrature_weights.allocate(quadrature.weights.size());
    check_cuda(
        cudaMemcpy(
            device.quadrature_weights.data(), quadrature.weights.data(),
            quadrature.weights.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "copy quadrature weights"
    );

    if (quadrature.line_order > 0) {
        device.line_quadrature_points.allocate(quadrature.line_points.size());
        check_cuda(
            cudaMemcpy(
                device.line_quadrature_points.data(), quadrature.line_points.data(),
                quadrature.line_points.size() * sizeof(float), cudaMemcpyHostToDevice
            ),
            "copy line quadrature points"
        );
        device.line_quadrature_weights.allocate(quadrature.line_weights.size());
        check_cuda(
            cudaMemcpy(
                device.line_quadrature_weights.data(), quadrature.line_weights.data(),
                quadrature.line_weights.size() * sizeof(float), cudaMemcpyHostToDevice
            ),
            "copy line quadrature weights"
        );
    }

    return device;
}

void assemble_system_matrix(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const DielectricOptions& options,
    const PhysicalParameters& physical,
    DeviceProblemData& device,
    complex__float* matrix
) {
    check_cuda(
        cudaMemset(
            matrix, 0,
            4 * mesh.edges.size() * mesh.edges.size() * sizeof(complex__float)
        ),
        "zero system matrix memory"
    );

    const int num_faces = static_cast<int>(mesh.faces.size());
    const int num_edges = static_cast<int>(mesh.edges.size());
    const int matrix_dimension = 2 * num_edges; // 2x2 PMCHWT blocks
    constexpr unsigned int block_size = 16;
    const dim3 threads(block_size, block_size);
    const dim3 blocks(
        (num_faces + block_size - 1) / block_size,
        (num_faces + block_size - 1) / block_size
    );

    if (options.formulation == DielectricFormulation::N_PMCHWT) {
        dielectric::launch_n_pmchwt_kernel(
            blocks, threads,
            device.vertices.data(),
            device.faces.data(),
            device.faces_edges.data(),
            device.quadrature_points.data(),
            device.quadrature_weights.data(),
            quadrature.order,
            device.line_quadrature_points.data(),
            device.line_quadrature_weights.data(),
            quadrature.line_order,
            options.ntl_form == NtlIntegralForm::Line,
            num_faces,
            num_edges,
            matrix_dimension,
            static_cast<float>(physical.k_interior),
            static_cast<float>(physical.k_exterior),
            static_cast<float>(options.interior.relative_permeability),
            static_cast<float>(options.exterior.relative_permeability),
            static_cast<float>(options.interior.relative_permittivity),
            static_cast<float>(options.exterior.relative_permittivity),
            static_cast<float>(options.vacuum_permeability),
            static_cast<float>(options.vacuum_permittivity),
            static_cast<float>(physical.omega),
            static_cast<float>(physical.singularity_threshold),
            matrix
        );
        check_last_kernel("assemble dense N-PMCHWT matrix");
    } else if (options.formulation == DielectricFormulation::T_PMCHWT) {
        dielectric::launch_t_pmchwt_kernel(
            blocks, threads,
            device.vertices.data(),
            device.faces.data(),
            device.faces_edges.data(),
            device.quadrature_points.data(),
            device.quadrature_weights.data(),
            quadrature.order,
            num_faces,
            num_edges,
            matrix_dimension,
            static_cast<float>(physical.k_interior),
            static_cast<float>(physical.k_exterior),
            static_cast<float>(options.interior.relative_permeability),
            static_cast<float>(options.exterior.relative_permeability),
            static_cast<float>(options.interior.relative_permittivity),
            static_cast<float>(options.exterior.relative_permittivity),
            static_cast<float>(options.vacuum_permeability),
            static_cast<float>(options.vacuum_permittivity),
            static_cast<float>(physical.omega),
            static_cast<float>(physical.singularity_threshold),
            matrix
        );
        check_last_kernel("assemble dense PMCHWT-II matrix");
    } else if (options.formulation == DielectricFormulation::N_Muller) {
        dielectric::launch_n_muller_kernel(
            blocks, threads,
            device.vertices.data(),
            device.faces.data(),
            device.faces_edges.data(),
            device.quadrature_points.data(),
            device.quadrature_weights.data(),
            quadrature.order,
            device.line_quadrature_points.data(),
            device.line_quadrature_weights.data(),
            quadrature.line_order,
            options.ntl_form == NtlIntegralForm::Line,
            num_faces,
            num_edges,
            matrix_dimension,
            static_cast<float>(physical.k_interior),
            static_cast<float>(physical.k_exterior),
            static_cast<float>(options.interior.relative_permeability),
            static_cast<float>(options.exterior.relative_permeability),
            static_cast<float>(options.interior.relative_permittivity),
            static_cast<float>(options.exterior.relative_permittivity),
            static_cast<float>(options.vacuum_permeability),
            static_cast<float>(options.vacuum_permittivity),
            static_cast<float>(physical.omega),
            static_cast<float>(physical.singularity_threshold),
            matrix
        );
        check_last_kernel("assemble dense N-Muller matrix");
    }
}

/**
 * Dielectric solver right-hand side excitation assembly.
 *
 * Region numbering convention (matches docs/source/matrix_assembly_v2.rst
 * and excitation::SourceRegion):
 *   - Region 1 = Interior scatterer  (excitation::SourceRegion::interior = 1)
 *   - Region 2 = Exterior background (excitation::SourceRegion::exterior = 2)
 *
 * Evaluates tested incident fields and scales them appropriately for the formulation:
 * - PMCHWT / N-PMCHWT:
 *     Tested incident field vector scaled by 'sign':
 *     sign = -1 for interior excitation (Region 1)
 *     sign = +1 for exterior excitation (Region 2)
 * - N-Müller:
 *     Exterior (Region 2): RHS_E =  eps2 * V_E, RHS_H = -mu2 * V_H
 *     Interior (Region 1): RHS_E =  eps1 * V_E, RHS_H = -mu1 * V_H
 */
template <TestingFunction TestFunc, typename ExcitationSource>
__global__ void assemble_dielectric_rhs_kernel(
    const vec3_t* vertices,
    const int3* faces,
    const int3* faces_edges,
    const float* quad_points,
    const float* quad_weights,
    int order_num,
    size_t num_faces,
    size_t num_edges,
    ExcitationSource source,
    real_t scale_e,
    real_t scale_h,
    complex__float* rhs
) {
    const size_t face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= num_faces) return;
    const int3 face = faces[face_index];
    const int indices[3] = {face.x, face.y, face.z};
    const vec3_t triangle[3] = {vertices[face.x], vertices[face.y], vertices[face.z]};
    const int3 edge_map = faces_edges[face_index];
    const int edges[3] = {edge_map.x, edge_map.y, edge_map.z};
    complex__float electric[3], magnetic[3];
    calculateExcitationQuadrature<TestFunc>(
        triangle, indices, quad_points, quad_weights, order_num, source, electric, magnetic
    );
    for (int local_edge = 0; local_edge < 3; ++local_edge) {
        atomicAddComplex(&rhs[edges[local_edge]], complexScale(electric[local_edge], scale_e));
        atomicAddComplex(&rhs[edges[local_edge] + num_edges], complexScale(magnetic[local_edge], scale_h));
    }
}

void assemble_rhs(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const DielectricOptions& options,
    const excitation::PreparedExcitation& prepared,
    DeviceProblemData& device,
    complex__float* rhs
) {
    check_cuda(
        cudaMemset(rhs, 0, 2 * mesh.edges.size() * sizeof(complex__float)),
        "zero right-hand side memory"
    );

    const int num_faces = static_cast<int>(mesh.faces.size());
    const int num_edges = static_cast<int>(mesh.edges.size());
    constexpr unsigned int threads = 256;
    const unsigned int blocks = (num_faces + threads - 1) / threads;

    prepared.visit([&](const auto& source) {
        // Region convention:
        //   Region 1 = Interior (SourceRegion::interior = 1)
        //   Region 2 = Exterior (SourceRegion::exterior = 2)
        const bool is_exterior = (prepared.region() == excitation::SourceRegion::exterior);
        const int source_region = static_cast<int>(prepared.region());

        const real_t eta0 = std::sqrt(
            static_cast<real_t>(options.vacuum_permeability) / 
            static_cast<real_t>(options.vacuum_permittivity)
        );

        if (options.formulation == DielectricFormulation::N_Muller) {
            real_t scale_e = real_t(0);
            real_t scale_h = real_t(0);
            if (is_exterior) {
                // Exterior plane wave (Region 2, background):
                // RHS_E = eps2 * V_E, RHS_H = -mu2 * V_H
                scale_e = static_cast<real_t>(options.exterior.relative_permittivity * options.vacuum_permittivity);
                scale_h = -static_cast<real_t>(options.exterior.relative_permeability * options.vacuum_permeability) * eta0;
            } else {
                // Interior source (Region 1, scatterer):
                // RHS_E = eps1 * V_E, RHS_H = -mu1 * V_H
                scale_e = static_cast<real_t>(options.interior.relative_permittivity * options.vacuum_permittivity);
                scale_h = -static_cast<real_t>(options.interior.relative_permeability * options.vacuum_permeability) * eta0;
            }
            assemble_dielectric_rhs_kernel<TestingFunction::NxRWG><<<blocks, threads>>>(
                device.vertices.data(),
                device.faces.data(),
                device.faces_edges.data(),
                device.quadrature_points.data(),
                device.quadrature_weights.data(),
                quadrature.order,
                num_faces,
                num_edges,
                source,
                scale_e,
                scale_h,
                rhs
            );
            check_last_kernel("assemble N-Muller excitation right-hand side");
        } else if (options.formulation == DielectricFormulation::N_PMCHWT) {
            // N-PMCHWT excitation scaling:
            // Region 1 (interior) -> +1, Region 2 (exterior) -> -1
            const real_t sign = is_exterior ? real_t(1) : real_t(-1);
            assemble_dielectric_rhs_kernel<TestingFunction::NxRWG><<<blocks, threads>>>(
                device.vertices.data(),
                device.faces.data(),
                device.faces_edges.data(),
                device.quadrature_points.data(),
                device.quadrature_weights.data(),
                quadrature.order,
                num_faces,
                num_edges,
                source,
                sign,
                sign * eta0,
                rhs
            );
            check_last_kernel("assemble N-PMCHWT excitation right-hand side");
        } else {
            // T-PMCHWT excitation scaling:
            // Region 1 (interior) -> +1, Region 2 (exterior) -> -1
            const real_t sign = is_exterior ? real_t(1) : real_t(-1);
            assemble_dielectric_rhs_kernel<TestingFunction::RWG><<<blocks, threads>>>(
                device.vertices.data(),
                device.faces.data(),
                device.faces_edges.data(),
                device.quadrature_points.data(),
                device.quadrature_weights.data(),
                quadrature.order,
                num_faces,
                num_edges,
                source,
                sign,
                sign * eta0,
                rhs
            );
            check_last_kernel("assemble PMCHWT-II excitation right-hand side");
        }
    });
}

void dense_matrix_vector_residual(
    int matrix_dimension,
    const complex__float* matrix,
    const complex__float* scaled_solution,
    const std::vector<complex__float>& original_rhs,
    double& absolute_residual,
    double& relative_residual
) {
    DeviceBuffer<complex__float> residual(static_cast<std::size_t>(matrix_dimension));
    check_cuda(
        cudaMemcpy(
            residual.data(), original_rhs.data(),
            original_rhs.size() * sizeof(complex__float), cudaMemcpyHostToDevice
        ),
        "copy original right-hand side for residual"
    );

    CublasHandle blas;
#ifdef USE_DOUBLE_PRECISION
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(-1.0, 0.0);
    check_cublas(
        cublasZgemv(
            blas.get(), CUBLAS_OP_N, matrix_dimension, matrix_dimension,
            &alpha, reinterpret_cast<const cuDoubleComplex*>(matrix),
            matrix_dimension,
            reinterpret_cast<const cuDoubleComplex*>(scaled_solution), 1,
            &beta, reinterpret_cast<cuDoubleComplex*>(residual.data()), 1
        ),
        "cublasZgemv for dense residual"
    );
#else
    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(-1.0f, 0.0f);
    check_cublas(
        cublasCgemv(
            blas.get(), CUBLAS_OP_N, matrix_dimension, matrix_dimension,
            &alpha, reinterpret_cast<const cuComplex*>(matrix),
            matrix_dimension,
            reinterpret_cast<const cuComplex*>(scaled_solution), 1,
            &beta, reinterpret_cast<cuComplex*>(residual.data()), 1
        ),
        "cublasCgemv for dense residual"
    );
#endif

    std::vector<complex__float> host_residual(original_rhs.size());
    check_cuda(
        cudaMemcpy(
            host_residual.data(), residual.data(),
            host_residual.size() * sizeof(complex__float), cudaMemcpyDeviceToHost
        ),
        "copy dense residual to host"
    );
    absolute_residual = vector_norm(host_residual);
    const double rhs_norm = vector_norm(original_rhs);
    relative_residual = rhs_norm > 0.0
        ? absolute_residual / rhs_norm
        : std::numeric_limits<double>::infinity();
}

void validate_and_summarize_physical_currents(
    const std::vector<complex__float>& currents,
    std::size_t num_edges,
    DielectricMetrics& metrics
) {
    double* maxima[2] = {
        &metrics.maximum_j_coefficient,
        &metrics.maximum_m_coefficient,
    };

    for (std::size_t block = 0; block < 2; ++block) {
        for (std::size_t edge = 0; edge < num_edges; ++edge) {
            const std::size_t index = block * num_edges + edge;
            const auto& value = currents[index];
            if (!std::isfinite(static_cast<double>(value.x))
                || !std::isfinite(static_cast<double>(value.y))) {
                throw std::runtime_error(
                    "non-finite physical current coefficient at index "
                    + std::to_string(index)
                );
            }
            const double magnitude = std::hypot(
                static_cast<double>(value.x), static_cast<double>(value.y)
            );
            *maxima[block] = std::max(*maxima[block], magnitude);
        }
    }
}

} // namespace

DielectricResult solve_dielectric_dense(
    const mesh::MeshData& mesh,
    const DielectricOptions& options,
    const excitation::PreparedExcitation& source
) {
    validate_problem(mesh, options);
    source.validate_environment(
        options.frequency_hz, options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable device is available for the SIE solve");
    }

    DielectricResult result;
    result.metrics.vertex_count = mesh.vertices.size();
    result.metrics.face_count = mesh.faces.size();
    result.metrics.edge_count = mesh.edges.size();
    result.metrics.matrix_dimension = 2 * mesh.edges.size();
    result.metrics.dense_matrix_bytes =
        result.metrics.matrix_dimension
        * result.metrics.matrix_dimension
        * sizeof(complex__float);

    const auto physical = make_physical_parameters(options);
    const auto quadrature = make_quadrature(options);
    const std::size_t matrix_dimension = result.metrics.matrix_dimension;
    const int matrix_dimension_int = static_cast<int>(matrix_dimension);

    const auto preparation_start = Clock::now();
    auto device = copy_problem_to_device(mesh, quadrature);
    DeviceBuffer<complex__float> matrix(matrix_dimension * matrix_dimension);
    DeviceBuffer<complex__float> solution(matrix_dimension);
    const auto preparation_end = Clock::now();
    result.metrics.device_preparation_seconds = elapsed_seconds(
        preparation_start, preparation_end
    );

    const auto assembly_start = Clock::now();
    assemble_system_matrix(
        mesh, quadrature, options, physical, device, matrix.data()
    );
    assemble_rhs(
        mesh, quadrature, options, source, device,
        solution.data()
    );
    const auto assembly_end = Clock::now();
    result.metrics.assembly_seconds = elapsed_seconds(assembly_start, assembly_end);

    std::vector<complex__float> original_rhs(matrix_dimension);
    check_cuda(
        cudaMemcpy(
            original_rhs.data(), solution.data(),
            original_rhs.size() * sizeof(complex__float), cudaMemcpyDeviceToHost
        ),
        "copy original right-hand side to host"
    );

    const auto solve_start = Clock::now();
    solve_dense_lu(matrix_dimension_int, matrix.data(), solution.data());
    const auto solve_end = Clock::now();
    result.metrics.solve_seconds = elapsed_seconds(solve_start, solve_end);

    if (options.compute_residual) {
        const auto residual_start = Clock::now();
        // LU factorization overwrites the matrix; restore A before computing A*x - b.
        assemble_system_matrix(
            mesh, quadrature, options, physical, device, matrix.data()
        );
        dense_matrix_vector_residual(
            matrix_dimension_int, matrix.data(), solution.data(), original_rhs,
            result.metrics.absolute_residual, result.metrics.relative_residual
        );
        result.metrics.residual_computed = true;
        const auto residual_end = Clock::now();
        result.metrics.residual_seconds = elapsed_seconds(
            residual_start, residual_end
        );
    }

    std::vector<complex__float> physical_currents(matrix_dimension);
    check_cuda(
        cudaMemcpy(
            physical_currents.data(), solution.data(),
            physical_currents.size() * sizeof(complex__float),
            cudaMemcpyDeviceToHost
        ),
        "copy PMCHWT solution to host"
    );

    // Unscale the magnetic currents
    // We solved the system for [J, M/eta0] to improve matrix conditioning.
    // Thus, physical M = (M/eta0) * eta0.
    const real_t eta0 = std::sqrt(
        static_cast<real_t>(options.vacuum_permeability) / 
        static_cast<real_t>(options.vacuum_permittivity)
    );
    for (size_t i = 0; i < mesh.edges.size(); ++i) {
        physical_currents[i + mesh.edges.size()] = complexScale(
            physical_currents[i + mesh.edges.size()], eta0
        );
    }

    validate_and_summarize_physical_currents(
        physical_currents, mesh.edges.size(), result.metrics
    );

    result.physical_currents.reserve(matrix_dimension);
    for (const auto& value : physical_currents) {
        result.physical_currents.emplace_back(value.x, value.y);
    }
    return result;
}

} // namespace solvers
} // namespace sie_mom
