#include "sie_mom/solvers/pec_solver.hpp"

#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/dense_lu_solver.cuh"
#include "sie_mom/cuda/sie_kernels_assembly.cuh"
#include "sie_mom/quadrature/gauss_legendre_rule.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sie_mom::solvers {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

void check_kernel(const char* action) {
    check_cuda(cudaGetLastError(), action);
    check_cuda(cudaDeviceSynchronize(), action);
}

void check_blas(cublasStatus_t status, const char* action) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(action) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

using sie_mom::cuda::DeviceBuffer;

template <typename T>
void upload(DeviceBuffer<T>& destination, const std::vector<T>& source, const char* action) {
    (void)action;
    destination.copy_from_host(source.data(), source.size());
}

struct Quadrature {
    std::vector<float> points;
    std::vector<float> weights;
    int count = 0;
    std::vector<float> line_points;
    std::vector<float> line_weights;
    int line_count = 0;
};

Quadrature make_quadrature(const PecOptions& options) {
    Quadrature result;
    result.count = dunavant_order_num(options.quadrature_rule);
    std::vector<double> points(2 * result.count), weights(result.count);
    dunavant_rule(options.quadrature_rule, result.count, points.data(), weights.data());
    result.points.assign(points.begin(), points.end());
    result.weights.assign(weights.begin(), weights.end());
    if (options.formulation == PecFormulation::NEfie &&
        options.ntl_form == PecNtlForm::Line) {
        result.line_count = options.line_quadrature_points;
        std::vector<double> line_points(result.line_count), line_weights(result.line_count);
        gauss_legendre_rule(result.line_count, line_points.data(), line_weights.data());
        result.line_points.assign(line_points.begin(), line_points.end());
        result.line_weights.assign(line_weights.begin(), line_weights.end());
    }
    return result;
}

void validate(const mesh::MeshData& mesh, const PecOptions& options,
              const excitation::PreparedExcitation& source) {
    if (mesh.vertices.empty() || mesh.faces.empty() || mesh.edges.empty() ||
        mesh.faces_edges.size() != mesh.faces.size())
        throw std::invalid_argument("PEC mesh needs vertices, faces, edges, and a face-edge map");
    if (!(options.frequency_hz > real_t(0)) ||
        !(options.exterior.relative_permittivity > real_t(0)) ||
        !(options.exterior.relative_permeability > real_t(0)) ||
        !(options.vacuum_permittivity > real_t(0)) ||
        !(options.vacuum_permeability > real_t(0)))
        throw std::invalid_argument("PEC frequency and medium constants must be positive");
    if (options.quadrature_rule < 1 || options.quadrature_rule > 20)
        throw std::invalid_argument("Dunavant rule must be in [1, 20]");
    if (options.formulation == PecFormulation::NEfie &&
        options.ntl_form == PecNtlForm::Line &&
        (options.line_quadrature_points < 1 ||
         options.line_quadrature_points > gauss_legendre_max_points()))
        throw std::invalid_argument("Gauss-Legendre point count must be in [1, 20]");
    if (options.formulation != PecFormulation::NEfie &&
        options.ntl_form == PecNtlForm::Line)
        throw std::invalid_argument("line nTL form requires N-EFIE");
    if (!(options.singularity_threshold_lambda0 > real_t(0)))
        throw std::invalid_argument("singularity threshold in lambda0 must be positive");
    if (source.region() != excitation::SourceRegion::exterior)
        throw std::invalid_argument("PEC solver requires an exterior (region 2) source");
    // The source interface validates only its own region's medium. PEC has no
    // interior propagation medium, so pass the exterior medium for both slots.
    source.validate_environment(options.frequency_hz, options.exterior, options.exterior,
                                options.vacuum_permittivity, options.vacuum_permeability);
    if (mesh.edges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("PEC edge count exceeds the dense solver integer limit");
}

// One electric-current block. Existing quadrature functions return nTK too,
// but only exterior nTL enters this PEC EFIE matrix.
__global__ void assemble_pec_n_efie_operator_kernel(
    const vec3_t* vertices, const int3* faces, const int3* faces_edges,
    const float* quad_points, const float* quad_weights, int quad_count,
    const float* line_points, const float* line_weights, int line_count,
    bool use_form_a, std::size_t face_count, std::size_t edge_count,
    float k2, float mu2, float omega, float threshold,
    complex__float* matrix)
{
    const std::size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= face_count || n >= face_count) return;
    const int3 fm = faces[m], fn = faces[n];
    const int tm[3] = {fm.x, fm.y, fm.z};
    const int sn[3] = {fn.x, fn.y, fn.z};
    const vec3_t tri_m[3] = {vertices[fm.x], vertices[fm.y], vertices[fm.z]};
    const vec3_t tri_n[3] = {vertices[fn.x], vertices[fn.y], vertices[fn.z]};
    const int3 me = faces_edges[m], ne = faces_edges[n];
    const int test_edges[3] = {me.x, me.y, me.z};
    const int source_edges[3] = {ne.x, ne.y, ne.z};
    const vec3_t cm = (tri_m[0] + tri_m[1] + tri_m[2]) / real_t(3);
    const vec3_t cn = (tri_n[0] + tri_n[1] + tri_n[2]) / real_t(3);
    const bool extract = length(cm - cn) < threshold;

    complex__float ntl1[3][3], ntl2[3][3], ntk1[3][3], ntk2[3][3];
    if (use_form_a) {
        calculateFormAQuadrature_nTL_nTK(
            tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
            line_points, line_weights, line_count, k2, k2, extract,
            ntl1, ntl2, ntk1, ntk2);
    } else {
        calculateDoubleQuadrature_nTL_nTK(
            tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
            k2, k2, extract, ntl1, ntl2, ntk1, ntk2);
    }
    if (extract) {
        complex__float singular_ntl1[3][3], singular_ntl2[3][3];
        complex__float singular_ntk1[3][3], singular_ntk2[3][3];
        if (use_form_a) {
            calculateFormASingularPartQuadrature_nTL_nTK(
                tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
                line_points, line_weights, line_count, k2, k2,
                singular_ntl1, singular_ntl2, singular_ntk1, singular_ntk2);
        } else {
            calculateSingularPartQuadrature_nTL_nTK(
                tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
                k2, k2, singular_ntl1, singular_ntl2,
                singular_ntk1, singular_ntk2);
        }
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                complexAddInPlace(ntl2[j][i], singular_ntl2[j][i]);
    }
    const complex__float factor = make_complex(real_t(0), omega * mu2);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            atomicAddComplex(&matrix[columnMajorIndex(
                test_edges[j], source_edges[i], edge_count)], factor * ntl2[j][i]);
}


// Galerkin n-MFIE: (1/2 M - nTK_2) J2 = <f x n, H_inc,2>.
// M[mj,ni] = <f_mj, f_ni>, while nTK_2[mj,ni] is the PV
// <f_mj x n_m, K_2[f_ni]>. The 1/2 M term is assembled only for m == n.
__global__ void assemble_pec_n_mfie_operator_kernel(
    const vec3_t* vertices, const int3* faces, const int3* faces_edges,
    const float* quad_points, const float* quad_weights, int quad_count,
    std::size_t face_count, std::size_t edge_count, float k2, float threshold,
    complex__float* matrix)
{
    const std::size_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= face_count || n >= face_count) return;

    const int3 fm = faces[m], fn = faces[n];
    const int tm[3] = {fm.x, fm.y, fm.z};
    const int sn[3] = {fn.x, fn.y, fn.z};
    const vec3_t tri_m[3] = {vertices[fm.x], vertices[fm.y], vertices[fm.z]};
    const vec3_t tri_n[3] = {vertices[fn.x], vertices[fn.y], vertices[fn.z]};
    const int3 me = faces_edges[m], ne = faces_edges[n];
    const int test_edges[3] = {me.x, me.y, me.z};
    const int source_edges[3] = {ne.x, ne.y, ne.z};

    if (m == n) {
        // The flat self-face PV nTK contribution is zero. Its jump is the
        // 1/2 M term, including local off-diagonal RWG edge pairs j != i.
        complex__float mass[3][3];
        calculateMassMatrixQuadrature(
            tri_m, tm, quad_points, quad_weights, quad_count, mass);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                atomicAddComplex(&matrix[columnMajorIndex(
                    test_edges[j], source_edges[i], edge_count)],
                    complexScale(mass[j][i], real_t(0.5)));
        return;
    }

    const vec3_t cm = (tri_m[0] + tri_m[1] + tri_m[2]) / real_t(3);
    const vec3_t cn = (tri_n[0] + tri_n[1] + tri_n[2]) / real_t(3);
    const bool extract = length(cm - cn) < threshold;

    // Use Form A's combined routine, but disable its nTL-only line loop.
    // Only the area-area PV nTK_2 output enters MFIE.
    complex__float ntl1[3][3], ntl2[3][3], ntk1[3][3], ntk2[3][3];
    calculateFormAQuadrature_nTL_nTK(
        tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
        nullptr, nullptr, 0, k2, k2, extract,
        ntl1, ntl2, ntk1, ntk2);

    if (extract) {
        // Add the analytically integrated Coulomb part to the regular nTK_2.
        complex__float singular_ntl1[3][3], singular_ntl2[3][3];
        complex__float singular_ntk1[3][3], singular_ntk2[3][3];
        calculateFormASingularPartQuadrature_nTL_nTK(
            tri_m, tm, tri_n, sn, quad_points, quad_weights, quad_count,
            nullptr, nullptr, 0, k2, k2,
            singular_ntl1, singular_ntl2, singular_ntk1, singular_ntk2);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                complexAddInPlace(ntk2[j][i], singular_ntk2[j][i]);
    }

    // Complete the operator 1/2 M - nTK_2 for non-self face pairs.
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            atomicAddComplex(&matrix[columnMajorIndex(
                test_edges[j], source_edges[i], edge_count)],
                complexScale(ntk2[j][i], real_t(-1)));
}

template <TestingFunction TestFunc, typename ExcitationSource>
__global__ void assemble_pec_rhs_kernel(
    const vec3_t* vertices, const int3* faces, const int3* faces_edges,
    const float* quad_points, const float* quad_weights, int count,
    std::size_t face_count, ExcitationSource source,
    bool magnetic_rhs, complex__float* rhs)
{
    const std::size_t face_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_index >= face_count) return;
    const int3 face = faces[face_index];
    const int indices[3] = {face.x, face.y, face.z};
    const vec3_t triangle[3] = {vertices[face.x], vertices[face.y], vertices[face.z]};
    const int3 edge_map = faces_edges[face_index];
    const int edges[3] = {edge_map.x, edge_map.y, edge_map.z};
    complex__float electric[3], magnetic[3];
    calculateExcitationQuadrature<TestFunc>(
        triangle, indices, quad_points, quad_weights, count, source, electric, magnetic);
    // EFIE uses +<f or f x n, E_inc,2>; n-MFIE uses +<f x n, H_inc,2>.
    // Both RHS signs follow the exterior PEC trace equations.
    for (int local = 0; local < 3; ++local)
        atomicAddComplex(&rhs[edges[local]],
                         magnetic_rhs ? magnetic[local] : electric[local]);
}

double norm(const std::vector<complex__float>& values) {
    long double sum = 0;
    for (const auto& value : values)
        sum += static_cast<long double>(value.x) * value.x +
               static_cast<long double>(value.y) * value.y;
    return std::sqrt(static_cast<double>(sum));
}

} // namespace

PecResult solve_pec_dense(
    const mesh::MeshData& mesh, const PecOptions& options,
    const excitation::PreparedExcitation& source)
{
    validate(mesh, options, source);
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
    if (device_count == 0)
        throw std::runtime_error("no CUDA-capable device is available");

    PecResult result;
    auto& metrics = result.metrics;
    metrics.vertex_count = mesh.vertices.size();
    metrics.face_count = mesh.faces.size();
    metrics.edge_count = mesh.edges.size();
    metrics.matrix_dimension = mesh.edges.size();
    metrics.dense_matrix_bytes = metrics.matrix_dimension * metrics.matrix_dimension *
                                 sizeof(complex__float);

    const real_t pi = std::acos(real_t(-1));
    const real_t omega = real_t(2) * pi * options.frequency_hz;
    const real_t mu2 = options.exterior.relative_permeability * options.vacuum_permeability;
    const real_t epsilon2 = options.exterior.relative_permittivity * options.vacuum_permittivity;
    const real_t k2 = omega * std::sqrt(mu2 * epsilon2);
    const real_t lambda0 = real_t(2) * pi /
        (omega * std::sqrt(options.vacuum_permeability * options.vacuum_permittivity));
    const real_t threshold = options.singularity_threshold_lambda0 * lambda0;
    const auto quadrature = make_quadrature(options);

    DeviceBuffer<vec3_t> vertices(mesh.vertices.size());
    DeviceBuffer<int3> faces(mesh.faces.size()), face_edges(mesh.faces_edges.size());
    DeviceBuffer<float> quad_points(quadrature.points.size());
    DeviceBuffer<float> quad_weights(quadrature.weights.size());
    DeviceBuffer<float> line_points(quadrature.line_points.size());
    DeviceBuffer<float> line_weights(quadrature.line_weights.size());
    DeviceBuffer<complex__float> matrix(metrics.matrix_dimension * metrics.matrix_dimension);
    DeviceBuffer<complex__float> rhs(metrics.matrix_dimension);
    upload(vertices, mesh.vertices, "copy PEC vertices");
    upload(faces, mesh.faces, "copy PEC faces");
    upload(face_edges, mesh.faces_edges, "copy PEC face-edge mapping");
    upload(quad_points, quadrature.points, "copy PEC quadrature points");
    upload(quad_weights, quadrature.weights, "copy PEC quadrature weights");
    if (quadrature.line_count > 0) {
        upload(line_points, quadrature.line_points, "copy PEC line points");
        upload(line_weights, quadrature.line_weights, "copy PEC line weights");
    }

    const auto assemble_matrix = [&]() {
        check_cuda(cudaMemset(matrix.data(), 0, metrics.dense_matrix_bytes),
                   "zero PEC matrix");
        constexpr unsigned int block_size = 16;
        const dim3 threads(block_size, block_size);
        const dim3 blocks(
            (mesh.faces.size() + block_size - 1) / block_size,
            (mesh.faces.size() + block_size - 1) / block_size);
        if (options.formulation == PecFormulation::NMfie) {
            // n-MFIE: (1/2 M - nTK_2) J2 = <f x n, H_inc,2>.
            // The kernel places 1/2 M on same-face pairs and uses Form A's
            // nTK_2 output elsewhere, with its nTL line loop disabled.
            assemble_pec_n_mfie_operator_kernel<<<blocks, threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                mesh.faces.size(), mesh.edges.size(), static_cast<float>(k2),
                static_cast<float>(threshold), matrix.data());
            check_kernel("assemble N-MFIE matrix");
        } else if (options.formulation == PecFormulation::NEfie) {
            assemble_pec_n_efie_operator_kernel<<<blocks, threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                line_points.data(), line_weights.data(), quadrature.line_count,
                options.ntl_form == PecNtlForm::Line,
                mesh.faces.size(), mesh.edges.size(), static_cast<float>(k2),
                static_cast<float>(mu2), static_cast<float>(omega),
                static_cast<float>(threshold), matrix.data());
            check_kernel("assemble N-EFIE matrix");
        } else {
            assemble_pec_efie_operator_kernel<<<blocks, threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                mesh.faces.size(), mesh.edges.size(), metrics.matrix_dimension,
                static_cast<float>(k2), static_cast<float>(k2),
                static_cast<float>(options.exterior.relative_permeability),
                static_cast<float>(options.vacuum_permeability),
                static_cast<float>(omega), static_cast<float>(threshold), matrix.data());
            check_kernel("assemble T-EFIE matrix");
        }
    };

    const auto assembly_start = Clock::now();
    assemble_matrix();
    check_cuda(cudaMemset(rhs.data(), 0, metrics.matrix_dimension * sizeof(complex__float)),
               "zero PEC RHS");
    constexpr unsigned int rhs_threads = 256;
    const unsigned int rhs_blocks =
        (mesh.faces.size() + rhs_threads - 1) / rhs_threads;
    source.visit([&](const auto& concrete_source) {
        if (options.formulation == PecFormulation::NMfie) {
            // Rotated testing of the exterior magnetic trace gives +<f x n, H_inc,2>.
            assemble_pec_rhs_kernel<TestingFunction::NxRWG><<<rhs_blocks, rhs_threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                mesh.faces.size(), concrete_source, true, rhs.data());
            check_kernel("assemble N-MFIE RHS");
        } else if (options.formulation == PecFormulation::NEfie) {
            assemble_pec_rhs_kernel<TestingFunction::NxRWG><<<rhs_blocks, rhs_threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                mesh.faces.size(), concrete_source, false, rhs.data());
            check_kernel("assemble N-EFIE RHS");
        } else {
            assemble_pec_rhs_kernel<TestingFunction::RWG><<<rhs_blocks, rhs_threads>>>(
                vertices.data(), faces.data(), face_edges.data(),
                quad_points.data(), quad_weights.data(), quadrature.count,
                mesh.faces.size(), concrete_source, false, rhs.data());
            check_kernel("assemble T-EFIE RHS");
        }
    });
    metrics.assembly_seconds =
        std::chrono::duration<double>(Clock::now() - assembly_start).count();

    std::vector<complex__float> original_rhs(metrics.matrix_dimension);
    check_cuda(cudaMemcpy(original_rhs.data(), rhs.data(),
                          original_rhs.size() * sizeof(complex__float),
                          cudaMemcpyDeviceToHost), "copy PEC RHS");

    if (options.capture_assembled_matrix) {
        // LU overwrites matrix, so capture the fully assembled LHS first.
        std::vector<complex__float> host_matrix(
            metrics.matrix_dimension * metrics.matrix_dimension);
        check_cuda(cudaMemcpy(host_matrix.data(), matrix.data(),
                              metrics.dense_matrix_bytes, cudaMemcpyDeviceToHost),
                   "copy assembled PEC matrix");
        result.assembled_matrix.reserve(host_matrix.size());
        for (const auto& value : host_matrix)
            result.assembled_matrix.emplace_back(value.x, value.y);
    }

    const auto solve_start = Clock::now();
    sie_mom::cuda::solve_dense_lu(static_cast<int>(metrics.matrix_dimension), matrix.data(), rhs.data());
    metrics.solve_seconds =
        std::chrono::duration<double>(Clock::now() - solve_start).count();

    if (options.compute_residual) {
        const auto residual_start = Clock::now();
        // LU overwrote the matrix. Reassemble it before evaluating A*J2 - b.
        assemble_matrix();
        DeviceBuffer<complex__float> product(metrics.matrix_dimension);
        cublasHandle_t blas = nullptr;
        check_blas(cublasCreate(&blas), "create cuBLAS handle");
        try {
#ifdef USE_DOUBLE_PRECISION
            const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            check_blas(cublasZgemv(blas, CUBLAS_OP_N,
                static_cast<int>(metrics.matrix_dimension),
                static_cast<int>(metrics.matrix_dimension), &alpha,
                reinterpret_cast<const cuDoubleComplex*>(matrix.data()),
                static_cast<int>(metrics.matrix_dimension),
                reinterpret_cast<const cuDoubleComplex*>(rhs.data()), 1,
                &beta, reinterpret_cast<cuDoubleComplex*>(product.data()), 1),
                "compute PEC matrix-vector product");
#else
            const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
            const cuComplex beta = make_cuComplex(0.0f, 0.0f);
            check_blas(cublasCgemv(blas, CUBLAS_OP_N,
                static_cast<int>(metrics.matrix_dimension),
                static_cast<int>(metrics.matrix_dimension), &alpha,
                reinterpret_cast<const cuComplex*>(matrix.data()),
                static_cast<int>(metrics.matrix_dimension),
                reinterpret_cast<const cuComplex*>(rhs.data()), 1,
                &beta, reinterpret_cast<cuComplex*>(product.data()), 1),
                "compute PEC matrix-vector product");
#endif
        } catch (...) {
            cublasDestroy(blas);
            throw;
        }
        cublasDestroy(blas);
        std::vector<complex__float> host_product(metrics.matrix_dimension);
        check_cuda(cudaMemcpy(host_product.data(), product.data(),
                              host_product.size() * sizeof(complex__float),
                              cudaMemcpyDeviceToHost), "copy PEC matrix-vector product");
        for (std::size_t i = 0; i < host_product.size(); ++i) {
            host_product[i].x -= original_rhs[i].x;
            host_product[i].y -= original_rhs[i].y;
        }
        metrics.absolute_residual = norm(host_product);
        const double rhs_norm = norm(original_rhs);
        metrics.relative_residual = rhs_norm > 0.0
            ? metrics.absolute_residual / rhs_norm
            : std::numeric_limits<double>::infinity();
        metrics.residual_computed = true;
        metrics.residual_seconds =
            std::chrono::duration<double>(Clock::now() - residual_start).count();
    }

    std::vector<complex__float> host_current(metrics.matrix_dimension);
    check_cuda(cudaMemcpy(host_current.data(), rhs.data(),
                          host_current.size() * sizeof(complex__float),
                          cudaMemcpyDeviceToHost), "copy PEC solution");
    result.exterior_current.reserve(host_current.size());
    for (const auto& value : host_current) {
        if (!std::isfinite(static_cast<double>(value.x)) ||
            !std::isfinite(static_cast<double>(value.y)))
            throw std::runtime_error("PEC solution contains a non-finite coefficient");
        metrics.maximum_j_coefficient = std::max(
            metrics.maximum_j_coefficient,
            std::hypot(static_cast<double>(value.x), static_cast<double>(value.y)));
        result.exterior_current.emplace_back(value.x, value.y);
    }
    return result;
}

} // namespace sie_mom::solvers
