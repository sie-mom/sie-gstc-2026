/**
 * @file gstc_sie.cu
 * @brief Modern GPU-accelerated SIE-GSTC solver (4x4 multitrace method) for closed triangular metasurfaces.
 *
 * Implements single-case and batch dense GSTC solves on arbitrary closed, outward-oriented
 * triangular meshes, managing matrix assembly, linear system solve, and physical scaling
 * for the 4x4 block SIE-GSTC formulation. Supports both intrinsic face-edge tangent frames
 * (`GstcBasisType::face`) and origin-centered spherical tangent frames (`GstcBasisType::spherical`).
 *
 * Internally, the assembly scales the equations (e.g., by 1/eta0) to improve matrix conditioning.
 * However, `recover_physical_currents` within `solve_gstc_dense()` and `solve_gstc_dense_batch()`
 * un-scales the resulting coefficients before returning them to the user, ensuring the final
 * output consists of physical electric (J) and magnetic (M) current coefficients.
 */

#include "sie_mom/solvers/gstc_sie.hpp"

#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/sie_kernels_assembly.cuh"
#include "sie_mom/cuda/dense_lu_solver.cuh"
#include "sie_mom/cuda/sie_kernels_utils.cuh"
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
using sie_mom::quadrature::QuadratureData;
using sie_mom::quadrature::make_quadrature;

struct DeviceProblemData {
    DeviceBuffer<vec3_t> vertices;
    DeviceBuffer<int3> faces;
    DeviceBuffer<int3> faces_edges;
    DeviceBuffer<float> quadrature_points;
    DeviceBuffer<float> quadrature_weights;
    DeviceBuffer<complex__float> gstc_tensors;
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
    const std::vector<std::complex<real_t>>& gstc_tensors,
    const GstcSieOptions& options
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
    if (!(options.singularity_threshold_lambda0 > real_t(0))) {
        throw std::invalid_argument("singularity threshold in lambda0 must be positive");
    }
    if (options.basis != GstcBasisType::face
        && options.basis != GstcBasisType::spherical) {
        throw std::invalid_argument("unsupported GSTC basis type");
    }

    const std::size_t records = options.tensors_are_uniform ? 1 : mesh.faces.size();
    if (records > std::numeric_limits<std::size_t>::max() / 32) {
        throw std::overflow_error("GSTC tensor count overflows size_t");
    }
    const std::size_t expected_tensor_count = records * 32;
    if (gstc_tensors.size() != expected_tensor_count) {
        throw std::invalid_argument(
            "GSTC tensor count mismatch: expected "
            + std::to_string(expected_tensor_count) + ", got "
            + std::to_string(gstc_tensors.size())
        );
    }

    if (mesh.edges.size() > std::numeric_limits<std::size_t>::max() / 4) {
        throw std::overflow_error("SIE matrix dimension overflows size_t");
    }
    const std::size_t matrix_dimension = 4 * mesh.edges.size();
    if (matrix_dimension > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("SIE matrix dimension exceeds the cuSolver int limit");
    }
    if (matrix_dimension > std::numeric_limits<std::size_t>::max() / matrix_dimension) {
        throw std::overflow_error("dense SIE matrix element count overflows size_t");
    }
    const std::size_t matrix_elements = matrix_dimension * matrix_dimension;
    if (matrix_elements > std::numeric_limits<std::size_t>::max() / sizeof(complex__float)) {
        throw std::overflow_error("dense SIE matrix byte count overflows size_t");
    }
}

void validate_batch_cases(
    const std::vector<GstcExcitationCase>& cases,
    const GstcSieOptions& options,
    std::size_t matrix_dimension
) {
    if (cases.empty()) {
        throw std::invalid_argument("SIE batch must contain at least one excitation case");
    }
    if (cases.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("SIE batch case count exceeds the cuSolver int limit");
    }
    if (cases.size() > std::numeric_limits<std::size_t>::max() / matrix_dimension
        || matrix_dimension * cases.size()
            > std::numeric_limits<std::size_t>::max() / sizeof(complex__float)) {
        throw std::overflow_error("SIE batch RHS byte count overflows size_t");
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const auto& incident = cases[index];
        const std::string prefix = "SIE batch case " + std::to_string(index) + ": ";
        if (!incident.interior && !incident.exterior) {
            throw std::invalid_argument(prefix + "at least one incident region is required");
        }
        const auto validate_source = [&](
            const std::optional<excitation::PreparedExcitation>& source,
            excitation::SourceRegion region
        ) {
            if (!source) {
                return;
            }
            if (source->region() != region) {
                throw std::invalid_argument(prefix + "excitation does not match its regional slot");
            }
            try {
                source->validate_environment(
                    options.frequency_hz, options.interior, options.exterior,
                    options.vacuum_permittivity, options.vacuum_permeability
                );
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument(prefix + error.what());
            }
        };
        validate_source(incident.interior, excitation::SourceRegion::interior);
        validate_source(incident.exterior, excitation::SourceRegion::exterior);
    }
}

DeviceProblemData copy_problem_to_device(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const std::vector<std::complex<real_t>>& gstc_tensors
) {
    DeviceProblemData device;
    device.vertices.allocate(mesh.vertices.size());
    device.faces.allocate(mesh.faces.size());
    device.faces_edges.allocate(mesh.faces_edges.size());
    device.quadrature_points.allocate(quadrature.points.size());
    device.quadrature_weights.allocate(quadrature.weights.size());
    device.gstc_tensors.allocate(gstc_tensors.size());

    check_cuda(
        cudaMemcpy(
            device.vertices.data(), mesh.vertices.data(),
            mesh.vertices.size() * sizeof(vec3_t), cudaMemcpyHostToDevice
        ),
        "copy mesh vertices to device"
    );
    check_cuda(
        cudaMemcpy(
            device.faces.data(), mesh.faces.data(),
            mesh.faces.size() * sizeof(int3), cudaMemcpyHostToDevice
        ),
        "copy mesh faces to device"
    );
    check_cuda(
        cudaMemcpy(
            device.faces_edges.data(), mesh.faces_edges.data(),
            mesh.faces_edges.size() * sizeof(int3), cudaMemcpyHostToDevice
        ),
        "copy face-to-edge mapping to device"
    );
    check_cuda(
        cudaMemcpy(
            device.quadrature_points.data(), quadrature.points.data(),
            quadrature.points.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "copy quadrature points to device"
    );
    check_cuda(
        cudaMemcpy(
            device.quadrature_weights.data(), quadrature.weights.data(),
            quadrature.weights.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "copy quadrature weights to device"
    );
    std::vector<complex__float> cuda_tensors;
    cuda_tensors.reserve(gstc_tensors.size());
    for (const auto& value : gstc_tensors) {
        cuda_tensors.push_back(make_complex(value.real(), value.imag()));
    }
    check_cuda(
        cudaMemcpy(
            device.gstc_tensors.data(), cuda_tensors.data(),
            cuda_tensors.size() * sizeof(complex__float), cudaMemcpyHostToDevice
        ),
        "copy GSTC tensors to device"
    );
    return device;
}

void assemble_system_matrix(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const GstcSieOptions& options,
    const PhysicalParameters& physical,
    const DeviceProblemData& device,
    complex__float* matrix
) {
    const std::size_t num_faces = mesh.faces.size();
    const std::size_t num_edges = mesh.edges.size();
    const std::size_t matrix_dimension = 4 * num_edges;
    const std::size_t matrix_elements = matrix_dimension * matrix_dimension;

    check_cuda(
        cudaMemset(matrix, 0, matrix_elements * sizeof(complex__float)),
        "clear dense SIE matrix"
    );

    const dim3 calderon_threads(16, 16);
    const dim3 calderon_blocks(
        static_cast<unsigned int>(
            (num_faces + calderon_threads.x - 1) / calderon_threads.x
        ),
        static_cast<unsigned int>(
            (num_faces + calderon_threads.y - 1) / calderon_threads.y
        )
    );
    assemble_Z_calderon_eta0_scaled_kernel<<<calderon_blocks, calderon_threads>>>(
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
    check_last_kernel("assemble scaled Calderon matrix");

    const dim3 mass_threads(256);
    const dim3 mass_blocks(
        static_cast<unsigned int>((num_faces + mass_threads.x - 1) / mass_threads.x)
    );
    assemble_Z_mass_eta0_scaled_kernel<<<mass_blocks, mass_threads>>>(
        device.vertices.data(),
        device.faces.data(),
        device.faces_edges.data(),
        device.quadrature_points.data(),
        device.quadrature_weights.data(),
        quadrature.order,
        num_faces,
        num_edges,
        matrix_dimension,
        device.gstc_tensors.data(),
        options.tensors_are_uniform,
        static_cast<float>(options.vacuum_permeability),
        static_cast<float>(options.vacuum_permittivity),
        matrix,
        options.basis
    );
    check_last_kernel("assemble scaled GSTC mass matrix");
}

void assemble_rhs(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const GstcSieOptions& options,
    const excitation::PreparedExcitation& prepared,
    const DeviceProblemData& device,
    complex__float* rhs
) {
    const std::size_t num_faces = mesh.faces.size();
    const std::size_t num_edges = mesh.edges.size();
    const std::size_t matrix_dimension = 4 * num_edges;
    check_cuda(
        cudaMemset(rhs, 0, matrix_dimension * sizeof(complex__float)),
        "clear scaled SIE right-hand side"
    );

    const dim3 threads(256);
    const dim3 blocks(
        static_cast<unsigned int>((num_faces + threads.x - 1) / threads.x)
    );
    prepared.visit([&](const auto& source) {
        assemble_V_vector_eta0_scaled_kernel<<<blocks, threads>>>(
            device.vertices.data(),
            device.faces.data(),
            device.faces_edges.data(),
            device.quadrature_points.data(),
            device.quadrature_weights.data(),
            quadrature.order,
            num_faces,
            num_edges,
            source,
            static_cast<int>(prepared.region()),
            static_cast<float>(options.vacuum_permeability),
            static_cast<float>(options.vacuum_permittivity),
            rhs
        );
    });
    check_last_kernel("assemble scaled incident right-hand side");
}

void assemble_case_rhs(
    const mesh::MeshData& mesh,
    const QuadratureData& quadrature,
    const GstcSieOptions& options,
    const GstcExcitationCase& incident,
    const DeviceProblemData& device,
    complex__float* rhs
) {
    const std::size_t num_faces = mesh.faces.size();
    const std::size_t num_edges = mesh.edges.size();
    check_cuda(
        cudaMemset(rhs, 0, 4 * num_edges * sizeof(complex__float)),
        "clear scaled batch right-hand side"
    );
    const dim3 threads(256);
    const dim3 blocks(
        static_cast<unsigned int>((num_faces + threads.x - 1) / threads.x)
    );
    const auto accumulate = [&](const excitation::PreparedExcitation& prepared) {
        prepared.visit([&](const auto& source) {
            assemble_V_vector_eta0_scaled_kernel<<<blocks, threads>>>(
                device.vertices.data(), device.faces.data(), device.faces_edges.data(),
                device.quadrature_points.data(), device.quadrature_weights.data(),
                quadrature.order, num_faces, num_edges, source,
                static_cast<int>(prepared.region()),
                static_cast<float>(options.vacuum_permeability),
                static_cast<float>(options.vacuum_permittivity), rhs
            );
        });
        check_last_kernel("accumulate scaled batch incident right-hand side");
    };
    if (incident.interior) {
        accumulate(*incident.interior);
    }
    if (incident.exterior) {
        accumulate(*incident.exterior);
    }
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

void recover_physical_currents(
    std::vector<complex__float>& scaled_currents,
    std::size_t num_edges,
    real_t eta0
) {
    for (std::size_t edge = 0; edge < num_edges; ++edge) {
        scaled_currents[num_edges + edge] = complexScale(
            scaled_currents[num_edges + edge], eta0
        );
        scaled_currents[3 * num_edges + edge] = complexScale(
            scaled_currents[3 * num_edges + edge], eta0
        );
    }
}

void validate_and_summarize_physical_currents(
    const std::vector<complex__float>& currents,
    std::size_t num_edges,
    GstcSieMetrics& metrics
) {
    double* maxima[4] = {
        &metrics.maximum_j1_coefficient,
        &metrics.maximum_m1_coefficient,
        &metrics.maximum_j2_coefficient,
        &metrics.maximum_m2_coefficient,
    };

    for (std::size_t block = 0; block < 4; ++block) {
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

GstcSieResult solve_gstc_dense(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& gstc_tensors,
    const GstcSieOptions& options,
    const excitation::PreparedExcitation& source
) {
    validate_problem(mesh, gstc_tensors, options);
    source.validate_environment(
        options.frequency_hz, options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable device is available for the SIE solve");
    }

    GstcSieResult result;
    result.metrics.vertex_count = mesh.vertices.size();
    result.metrics.face_count = mesh.faces.size();
    result.metrics.edge_count = mesh.edges.size();
    result.metrics.matrix_dimension = 4 * mesh.edges.size();
    result.metrics.dense_matrix_bytes =
        result.metrics.matrix_dimension
        * result.metrics.matrix_dimension
        * sizeof(complex__float);

    const auto physical = make_physical_parameters(options);
    const auto quadrature = make_quadrature(options.quadrature_rule);
    const std::size_t matrix_dimension = result.metrics.matrix_dimension;
    const int matrix_dimension_int = static_cast<int>(matrix_dimension);

    const auto preparation_start = Clock::now();
    auto device = copy_problem_to_device(mesh, quadrature, gstc_tensors);
    DeviceBuffer<complex__float> matrix(matrix_dimension * matrix_dimension);
    DeviceBuffer<complex__float> scaled_rhs_and_solution(matrix_dimension);
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
        scaled_rhs_and_solution.data()
    );
    const auto assembly_end = Clock::now();
    result.metrics.assembly_seconds = elapsed_seconds(assembly_start, assembly_end);

    std::vector<complex__float> original_scaled_rhs(matrix_dimension);
    check_cuda(
        cudaMemcpy(
            original_scaled_rhs.data(), scaled_rhs_and_solution.data(),
            matrix_dimension * sizeof(complex__float), cudaMemcpyDeviceToHost
        ),
        "copy original scaled right-hand side to host"
    );
    result.metrics.rhs_norm = vector_norm(original_scaled_rhs);
    if (!(result.metrics.rhs_norm > 0.0)) {
        throw std::runtime_error("assembled incident right-hand side has zero norm");
    }

    const auto solve_start = Clock::now();
    solve_dense_lu(
        matrix_dimension_int, matrix.data(), scaled_rhs_and_solution.data()
    );
    const auto solve_end = Clock::now();
    result.metrics.solve_seconds = elapsed_seconds(solve_start, solve_end);

    std::vector<complex__float> scaled_currents(matrix_dimension);
    check_cuda(
        cudaMemcpy(
            scaled_currents.data(), scaled_rhs_and_solution.data(),
            matrix_dimension * sizeof(complex__float), cudaMemcpyDeviceToHost
        ),
        "copy scaled current solution to host"
    );

    if (options.compute_residual) {
        const auto residual_start = Clock::now();
        assemble_system_matrix(
            mesh, quadrature, options, physical, device, matrix.data()
        );
        dense_matrix_vector_residual(
            matrix_dimension_int,
            matrix.data(),
            scaled_rhs_and_solution.data(),
            original_scaled_rhs,
            result.metrics.absolute_residual,
            result.metrics.relative_residual
        );
        result.metrics.residual_computed = true;
        const auto residual_end = Clock::now();
        result.metrics.residual_seconds = elapsed_seconds(
            residual_start, residual_end
        );
    }

    recover_physical_currents(
        scaled_currents, mesh.edges.size(), physical.eta0
    );
    validate_and_summarize_physical_currents(
        scaled_currents, mesh.edges.size(), result.metrics
    );
    result.physical_currents.reserve(matrix_dimension);
    for (const auto& value : scaled_currents) {
        result.physical_currents.emplace_back(value.x, value.y);
    }
    return result;
}

GstcSieBatchResult solve_gstc_dense_batch(
    const mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& gstc_tensors,
    const GstcSieOptions& options,
    const std::vector<GstcExcitationCase>& cases
) {
    validate_problem(mesh, gstc_tensors, options);
    const std::size_t matrix_dimension = 4 * mesh.edges.size();
    validate_batch_cases(cases, options, matrix_dimension);
    const int dimension = static_cast<int>(matrix_dimension);
    const int case_count = static_cast<int>(cases.size());

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0) {
        throw std::runtime_error("no CUDA-capable device is available for the SIE solve");
    }

    GstcSieBatchResult result;
    result.cases.resize(cases.size());
    result.metrics.vertex_count = mesh.vertices.size();
    result.metrics.face_count = mesh.faces.size();
    result.metrics.edge_count = mesh.edges.size();
    result.metrics.matrix_dimension = matrix_dimension;
    result.metrics.dense_matrix_bytes =
        matrix_dimension * matrix_dimension * sizeof(complex__float);
    result.metrics.case_count = cases.size();
    const auto physical = make_physical_parameters(options);
    const auto quadrature = make_quadrature(options.quadrature_rule);

    const auto preparation_start = Clock::now();
    auto device = copy_problem_to_device(mesh, quadrature, gstc_tensors);
    DeviceBuffer<complex__float> matrix(matrix_dimension * matrix_dimension);
    // cuSolver stores RHS and solution columns contiguously in column-major order.
    DeviceBuffer<complex__float> rhs_and_solutions(matrix_dimension * cases.size());
    DenseLuSolver solver(dimension);
    result.metrics.device_preparation_seconds = elapsed_seconds(preparation_start, Clock::now());

    const auto assembly_start = Clock::now();
    assemble_system_matrix(mesh, quadrature, options, physical, device, matrix.data());
    result.metrics.matrix_assembly_seconds = elapsed_seconds(assembly_start, Clock::now());

    std::vector<std::vector<complex__float>> original_rhs(
        options.compute_residual ? cases.size() : 0
    );
    std::vector<complex__float> host_column(matrix_dimension);
    for (std::size_t index = 0; index < cases.size(); ++index) {
        auto& metrics = result.cases[index].metrics;
        auto* column = rhs_and_solutions.data() + index * matrix_dimension;
        const auto rhs_start = Clock::now();
        assemble_case_rhs(mesh, quadrature, options, cases[index], device, column);
        metrics.rhs_assembly_seconds = elapsed_seconds(rhs_start, Clock::now());
        check_cuda(
            cudaMemcpy(
                host_column.data(), column, matrix_dimension * sizeof(complex__float),
                cudaMemcpyDeviceToHost
            ),
            "copy batch scaled right-hand side to host"
        );
        metrics.rhs_norm = vector_norm(host_column);
        if (!(metrics.rhs_norm > 0.0) || !std::isfinite(metrics.rhs_norm)) {
            throw std::runtime_error(
                "SIE batch case " + std::to_string(index)
                + ": incident RHS norm must be finite and nonzero"
            );
        }
        if (options.compute_residual) {
            original_rhs[index] = host_column;
        }
    }

    const auto factorization_start = Clock::now();
    solver.factorize(matrix.data());
    result.metrics.factorization_seconds = elapsed_seconds(factorization_start, Clock::now());
    const auto solve_start = Clock::now();
    solver.solve(CUBLAS_OP_N, case_count, matrix.data(), rhs_and_solutions.data());
    result.metrics.solve_seconds = elapsed_seconds(solve_start, Clock::now());

    if (options.compute_residual) {
        // All solves are complete: the LU factors can now be replaced by A.
        const auto reassembly_start = Clock::now();
        assemble_system_matrix(mesh, quadrature, options, physical, device, matrix.data());
        result.metrics.residual_matrix_assembly_seconds = elapsed_seconds(
            reassembly_start, Clock::now()
        );
    }
    for (std::size_t index = 0; index < cases.size(); ++index) {
        auto& case_result = result.cases[index];
        const auto* column = rhs_and_solutions.data() + index * matrix_dimension;
        check_cuda(
            cudaMemcpy(
                host_column.data(), column, matrix_dimension * sizeof(complex__float),
                cudaMemcpyDeviceToHost
            ),
            "copy batch scaled current solution to host"
        );
        if (options.compute_residual) {
            const auto residual_start = Clock::now();
            dense_matrix_vector_residual(
                dimension, matrix.data(), column, original_rhs[index],
                case_result.metrics.absolute_residual, case_result.metrics.relative_residual
            );
            case_result.metrics.residual_computed = true;
            case_result.metrics.residual_seconds = elapsed_seconds(residual_start, Clock::now());
        }
        recover_physical_currents(host_column, mesh.edges.size(), physical.eta0);
        GstcSieMetrics diagnostics;
        validate_and_summarize_physical_currents(host_column, mesh.edges.size(), diagnostics);
        case_result.metrics.maximum_j1_coefficient = diagnostics.maximum_j1_coefficient;
        case_result.metrics.maximum_m1_coefficient = diagnostics.maximum_m1_coefficient;
        case_result.metrics.maximum_j2_coefficient = diagnostics.maximum_j2_coefficient;
        case_result.metrics.maximum_m2_coefficient = diagnostics.maximum_m2_coefficient;
        case_result.physical_currents.reserve(matrix_dimension);
        for (const auto& value : host_column) {
            case_result.physical_currents.emplace_back(value.x, value.y);
        }
    }
    return result;
}

} // namespace solvers
} // namespace sie_mom