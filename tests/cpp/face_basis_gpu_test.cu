// Eigen is used only for host-side tensor construction in this test.
// Avoid Eigen/MSVC complex-operator conflicts under NVCC.
#if defined(_MSC_VER) && defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_NO_CUDA
#endif

#include "catch.hpp"

#include "sie_mom/cuda/device_buffer.cuh"
#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/cuda/sie_kernels_assembly.cuh"
#include "sie_mom/cuda/sie_kernels_matvec.cuh"
#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/quadrature/triangle_dunavant_rule.hpp"
#include "sie_mom/solvers/gstc_sie.hpp"

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

#ifdef USE_DOUBLE_PRECISION
constexpr double kExactKernelTolerance = 1e-12;
constexpr double kPermutedQuadratureTolerance = 1e-7;
#else
constexpr double kExactKernelTolerance = 2e-3;
constexpr double kPermutedQuadratureTolerance = 2e-3;
#endif

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

sie_mom::mesh::MeshData make_triaxial_ellipsoid_mesh(int refinement_level = 0) {
    auto mesh = sie_mom::mesh::SphericalSurface(0.03f, refinement_level).getMeshData();
    // Deform sphere into a non-spherical triaxial ellipsoid (preserves positive CCW volume).
    for (auto& v : mesh.vertices) {
        v.x *= real_t(1.40);
        v.y *= real_t(0.85);
        v.z *= real_t(1.15);
    }
    return mesh;
}

__global__ void evaluate_face_bases_device_kernel(
    const vec3_t* vertices,
    const int3* faces,
    std::size_t num_faces,
    vec3_t* out_t1,
    vec3_t* out_t2
) {
    const std::size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= num_faces) {
        return;
    }
    const int3 f = faces[m];
    get_local_basis_vectors(
        vertices[f.x], vertices[f.y], vertices[f.z],
        sie_mom::GstcBasisType::face,
        out_t1[m], out_t2[m]
    );
}

std::vector<std::complex<real_t>> make_anisotropic_bianisotropic_face_tensors(
    std::size_t num_faces
) {
    using Matrix4c = Eigen::Matrix<std::complex<double>, 4, 4>;
    const double eta0 = 376.730313668;
    std::vector<std::complex<real_t>> flat(32 * num_faces);

    for (std::size_t m = 0; m < num_faces; ++m) {
        const double phase = 0.23 * static_cast<double>(m + 1);
        Matrix4c Y = Matrix4c::Zero();
        // Strongly anisotropic Y_ee (rows 0..1, cols 0..1)
        Y(0, 0) = std::complex<double>(0.012, 0.005 + 0.001 * std::cos(phase));
        Y(0, 1) = std::complex<double>(0.003, -0.002 * std::sin(phase));
        Y(1, 0) = std::complex<double>(-0.0025, 0.0015);
        Y(1, 1) = std::complex<double>(0.021, -0.006);

        // Magnetoelectric coupling Y_em and Y_me
        Y(0, 2) = std::complex<double>(0.0, 0.04 * std::cos(phase));
        Y(1, 3) = std::complex<double>(0.0, -0.03 * std::sin(phase));
        Y(2, 0) = std::complex<double>(0.0, -0.035 * std::sin(phase));
        Y(3, 1) = std::complex<double>(0.0, 0.025 * std::cos(phase));

        // Strongly anisotropic Y_mm (rows 2..3, cols 2..3)
        Y(2, 2) = std::complex<double>(0.85 * eta0, 0.25 * eta0);
        Y(2, 3) = std::complex<double>(0.15 * eta0, -0.10 * eta0 * std::cos(phase));
        Y(3, 2) = std::complex<double>(-0.12 * eta0, 0.08 * eta0);
        Y(3, 3) = std::complex<double>(1.35 * eta0, -0.30 * eta0);

        const Matrix4c Z = Y.inverse();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                flat[32 * m + r * 4 + c] = {
                    static_cast<real_t>(Y(r, c).real()),
                    static_cast<real_t>(Y(r, c).imag())
                };
                flat[32 * m + 16 + r * 4 + c] = {
                    static_cast<real_t>(Z(r, c).real()),
                    static_cast<real_t>(Z(r, c).imag())
                };
            }
        }
    }
    return flat;
}

void rotate_face_record_in_plane(
    std::complex<real_t>* record32,
    const real_t Q[2][2]
) {
    // Rotate each of the eight 2x2 blocks (4 in Y, 4 in Z): T' = Q * T * Q^T
    for (int mat = 0; mat < 2; ++mat) {
        std::complex<real_t>* mat_ptr = record32 + mat * 16;
        for (int rb = 0; rb < 2; ++rb) {
            for (int cb = 0; cb < 2; ++cb) {
                std::complex<real_t> block[2][2] = {
                    {mat_ptr[(2 * rb) * 4 + (2 * cb)],     mat_ptr[(2 * rb) * 4 + (2 * cb + 1)]},
                    {mat_ptr[(2 * rb + 1) * 4 + (2 * cb)], mat_ptr[(2 * rb + 1) * 4 + (2 * cb + 1)]},
                };
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 2; ++c) {
                        std::complex<real_t> sum{real_t(0), real_t(0)};
                        for (int p = 0; p < 2; ++p) {
                            for (int q = 0; q < 2; ++q) {
                                sum += Q[r][p] * block[p][q] * Q[c][q];
                            }
                        }
                        mat_ptr[(2 * rb + r) * 4 + (2 * cb + c)] = sum;
                    }
                }
            }
        }
    }
}

double relative_l2_difference(
    const std::vector<std::complex<real_t>>& a,
    const std::vector<std::complex<real_t>>& b
) {
    REQUIRE(a.size() == b.size());
    long double diff_sq = 0.0L;
    long double ref_sq = 0.0L;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const long double dr = static_cast<long double>(a[i].real()) - static_cast<long double>(b[i].real());
        const long double di = static_cast<long double>(a[i].imag()) - static_cast<long double>(b[i].imag());
        const long double rr = static_cast<long double>(b[i].real());
        const long double ri = static_cast<long double>(b[i].imag());
        diff_sq += dr * dr + di * di;
        ref_sq += rr * rr + ri * ri;
    }
    REQUIRE(ref_sq > 0.0L);
    return std::sqrt(static_cast<double>(diff_sq / ref_sq));
}

} // namespace

TEST_CASE(
    "CUDA device get_local_basis_vectors matches host get_face_basis_vectors on triaxial ellipsoid",
    "[sie][basis][gpu]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }

    const auto mesh = make_triaxial_ellipsoid_mesh(1);
    const std::size_t num_faces = mesh.faces.size();

    sie_mom::cuda::DeviceBuffer<vec3_t> d_vertices(mesh.vertices.size());
    sie_mom::cuda::DeviceBuffer<int3> d_faces(num_faces);
    sie_mom::cuda::DeviceBuffer<vec3_t> d_t1(num_faces);
    sie_mom::cuda::DeviceBuffer<vec3_t> d_t2(num_faces);

    REQUIRE(cudaMemcpy(
        d_vertices.data(), mesh.vertices.data(),
        mesh.vertices.size() * sizeof(vec3_t), cudaMemcpyHostToDevice
    ) == cudaSuccess);
    REQUIRE(cudaMemcpy(
        d_faces.data(), mesh.faces.data(),
        num_faces * sizeof(int3), cudaMemcpyHostToDevice
    ) == cudaSuccess);

    evaluate_face_bases_device_kernel<<<1, 128>>>(
        d_vertices.data(), d_faces.data(), num_faces, d_t1.data(), d_t2.data()
    );
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<vec3_t> host_t1(num_faces), host_t2(num_faces);
    REQUIRE(cudaMemcpy(
        host_t1.data(), d_t1.data(), num_faces * sizeof(vec3_t), cudaMemcpyDeviceToHost
    ) == cudaSuccess);
    REQUIRE(cudaMemcpy(
        host_t2.data(), d_t2.data(), num_faces * sizeof(vec3_t), cudaMemcpyDeviceToHost
    ) == cudaSuccess);

    const double eps = static_cast<double>(std::numeric_limits<real_t>::epsilon()) * 64.0;
    for (std::size_t m = 0; m < num_faces; ++m) {
        const int3 f = mesh.faces[m];
        vec3_t expected_t1{}, expected_t2{};
        get_face_basis_vectors(
            mesh.vertices[f.x], mesh.vertices[f.y], mesh.vertices[f.z],
            expected_t1, expected_t2
        );
        CHECK(host_t1[m].x == Approx(expected_t1.x).margin(eps));
        CHECK(host_t1[m].y == Approx(expected_t1.y).margin(eps));
        CHECK(host_t1[m].z == Approx(expected_t1.z).margin(eps));
        CHECK(host_t2[m].x == Approx(expected_t2.x).margin(eps));
        CHECK(host_t2[m].y == Approx(expected_t2.y).margin(eps));
        CHECK(host_t2[m].z == Approx(expected_t2.z).margin(eps));
    }
}

TEST_CASE(
    "dense mass assembly and matrix-free matvec agree under GstcBasisType::face on triaxial ellipsoid",
    "[sie][basis][matvec][gpu]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }

    const auto mesh = make_triaxial_ellipsoid_mesh(0);
    const std::size_t num_faces = mesh.faces.size();
    const std::size_t num_edges = mesh.edges.size();
    const std::size_t dim = 4 * num_edges;
    const auto quad = sie_mom::quadrature::make_quadrature(5);
    const auto host_tensors = make_anisotropic_bianisotropic_face_tensors(num_faces);

    std::vector<complex__float> cuda_tensors(host_tensors.size());
    for (std::size_t i = 0; i < host_tensors.size(); ++i) {
        cuda_tensors[i] = make_complex(host_tensors[i].real(), host_tensors[i].imag());
    }
    std::vector<complex__float> host_x(dim);
    for (std::size_t i = 0; i < dim; ++i) {
        host_x[i] = make_complex(
            real_t(0.13) * static_cast<real_t>((i % 7) + 1),
            real_t(-0.09) * static_cast<real_t>((i % 5) - 2)
        );
    }

    sie_mom::cuda::DeviceBuffer<vec3_t> d_vertices(mesh.vertices.size());
    sie_mom::cuda::DeviceBuffer<int3> d_faces(num_faces);
    sie_mom::cuda::DeviceBuffer<int3> d_faces_edges(num_faces);
    sie_mom::cuda::DeviceBuffer<float> d_qpts(quad.points.size());
    sie_mom::cuda::DeviceBuffer<float> d_qwts(quad.weights.size());
    sie_mom::cuda::DeviceBuffer<complex__float> d_tensors(cuda_tensors.size());
    sie_mom::cuda::DeviceBuffer<complex__float> d_Z(dim * dim);
    sie_mom::cuda::DeviceBuffer<complex__float> d_x(dim);
    sie_mom::cuda::DeviceBuffer<complex__float> d_y(dim);

    REQUIRE(cudaMemcpy(d_vertices.data(), mesh.vertices.data(), mesh.vertices.size() * sizeof(vec3_t), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_faces.data(), mesh.faces.data(), num_faces * sizeof(int3), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_faces_edges.data(), mesh.faces_edges.data(), num_faces * sizeof(int3), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_qpts.data(), quad.points.data(), quad.points.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_qwts.data(), quad.weights.data(), quad.weights.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tensors.data(), cuda_tensors.data(), cuda_tensors.size() * sizeof(complex__float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x.data(), host_x.data(), dim * sizeof(complex__float), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemset(d_Z.data(), 0, dim * dim * sizeof(complex__float)) == cudaSuccess);
    REQUIRE(cudaMemset(d_y.data(), 0, dim * sizeof(complex__float)) == cudaSuccess);

    assemble_Z_mass_kernel<<<1, 64>>>(
        d_vertices.data(), d_faces.data(), d_faces_edges.data(),
        d_qpts.data(), d_qwts.data(), quad.order,
        num_faces, num_edges, dim,
        d_tensors.data(), false, d_Z.data(),
        sie_mom::GstcBasisType::face
    );
    REQUIRE(cudaGetLastError() == cudaSuccess);

    matvec_Z_mass_kernel<<<1, 64>>>(
        d_vertices.data(), d_faces.data(), d_faces_edges.data(),
        d_qpts.data(), d_qwts.data(), quad.order,
        num_faces, num_edges,
        d_tensors.data(), false, d_x.data(), d_y.data(),
        sie_mom::GstcBasisType::face
    );
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<complex__float> host_Z(dim * dim);
    std::vector<complex__float> host_y_matvec(dim);
    REQUIRE(cudaMemcpy(host_Z.data(), d_Z.data(), dim * dim * sizeof(complex__float), cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(host_y_matvec.data(), d_y.data(), dim * sizeof(complex__float), cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<std::complex<real_t>> y_dense(dim, {real_t(0), real_t(0)});
    std::vector<std::complex<real_t>> y_matvec(dim);
    for (std::size_t col = 0; col < dim; ++col) {
        const std::complex<real_t> x_col(host_x[col].x, host_x[col].y);
        for (std::size_t row = 0; row < dim; ++row) {
            const auto z_rc = host_Z[col * dim + row]; // column-major
            y_dense[row] += std::complex<real_t>(z_rc.x, z_rc.y) * x_col;
        }
    }
    for (std::size_t row = 0; row < dim; ++row) {
        y_matvec[row] = {host_y_matvec[row].x, host_y_matvec[row].y};
    }

    CHECK(relative_l2_difference(y_matvec, y_dense) < kExactKernelTolerance);
}

TEST_CASE(
    "solve_gstc_dense matches to double precision between GstcBasisType::spherical and GstcBasisType::face on regular icosahedron",
    "[sie][basis][solver][gpu]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }

    // Every face of a regular icosahedron (refinement 0) has equal edge lengths and
    // equal vertex radii, so its planar face normal equals its centroid radial unit
    // vector to machine precision. Rotating 2x2 GSTC blocks from (theta_hat, phi_hat)
    // to (t1, t2) must therefore produce identical solved currents without quadrature shift.
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    const std::size_t num_faces = mesh.faces.size();
    const auto spherical_tensors = make_anisotropic_bianisotropic_face_tensors(num_faces);
    auto face_tensors = spherical_tensors;

    for (std::size_t m = 0; m < num_faces; ++m) {
        const int3 f = mesh.faces[m];
        vec3_t th_hat{}, ph_hat{};
        vec3_t t1{}, t2{};
        get_spherical_basis_vectors(mesh.vertices[f.x], mesh.vertices[f.y], mesh.vertices[f.z], th_hat, ph_hat);
        get_face_basis_vectors(mesh.vertices[f.x], mesh.vertices[f.y], mesh.vertices[f.z], t1, t2);

        const real_t Q[2][2] = {
            {dot(t1, th_hat), dot(t1, ph_hat)},
            {dot(t2, th_hat), dot(t2, ph_hat)},
        };
        rotate_face_record_in_plane(&face_tensors[32 * m], Q);
    }

    sie_mom::solvers::GstcSieOptions options;
    options.frequency_hz = real_t(1.0e9);
    options.interior = {real_t(2.4), real_t(1.1)};
    options.exterior = {real_t(1.0), real_t(1.0)};
    options.quadrature_rule = 7;
    options.tensors_are_uniform = false;

    sie_mom::excitation::DipoleExcitation dipole;
    dipole.region = sie_mom::excitation::SourceRegion::interior;
    dipole.position = make_vec3(real_t(0.004), real_t(-0.003), real_t(0.002));
    dipole.effective_moment = make_vec3(real_t(0.3), real_t(-0.5), real_t(0.8));
    const auto source = sie_mom::excitation::prepare_source(
        dipole, options.frequency_hz, options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    options.basis = sie_mom::GstcBasisType::spherical;
    const auto spherical_result = sie_mom::solvers::solve_gstc_dense(mesh, spherical_tensors, options, source);

    options.basis = sie_mom::GstcBasisType::face;
    const auto face_result = sie_mom::solvers::solve_gstc_dense(mesh, face_tensors, options, source);

    CHECK(spherical_result.metrics.relative_residual < kExactKernelTolerance);
    CHECK(face_result.metrics.relative_residual < kExactKernelTolerance);
    CHECK(relative_l2_difference(face_result.physical_currents, spherical_result.physical_currents) < kExactKernelTolerance);
}

TEST_CASE(
    "solve_gstc_dense with GstcBasisType::face is gauge-invariant under cyclic vertex permutation + 2D frame rotation on triaxial ellipsoid",
    "[sie][basis][solver][gpu]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }

    const auto mesh = make_triaxial_ellipsoid_mesh(0);
    const std::size_t num_faces = mesh.faces.size();
    const auto tensors = make_anisotropic_bianisotropic_face_tensors(num_faces);

    sie_mom::solvers::GstcSieOptions options;
    options.frequency_hz = real_t(1.0e9);
    options.interior = {real_t(2.4), real_t(1.1)};
    options.exterior = {real_t(1.0), real_t(1.0)};
    options.quadrature_rule = 7;
    options.tensors_are_uniform = false;
    options.compute_residual = true;
    options.basis = sie_mom::GstcBasisType::face;

    sie_mom::excitation::DipoleExcitation dipole;
    dipole.region = sie_mom::excitation::SourceRegion::interior;
    dipole.position = make_vec3(real_t(0.004), real_t(-0.003), real_t(0.002));
    dipole.effective_moment = make_vec3(real_t(0.3), real_t(-0.5), real_t(0.8));
    const auto source = sie_mom::excitation::prepare_source(
        dipole, options.frequency_hz, options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    const auto baseline = sie_mom::solvers::solve_gstc_dense(mesh, tensors, options, source);
    CHECK(baseline.metrics.relative_residual < kExactKernelTolerance);

    // Construct a mesh with cyclically permuted vertices (v0, v1, v2) -> (v1, v2, v0)
    // on odd-indexed faces (and two cyclic steps on multiples of 3), preserving CCW outward
    // normal and global edge numbering while rotating each face's local (t1, t2) frame.
    auto permuted_mesh = mesh;
    auto rotated_tensors = tensors;
    for (std::size_t m = 0; m < num_faces; ++m) {
        const int3 f = mesh.faces[m];
        const int3 fe = mesh.faces_edges[m];
        vec3_t u_orig{}, v_orig{};
        get_face_basis_vectors(
            mesh.vertices[f.x], mesh.vertices[f.y], mesh.vertices[f.z],
            u_orig, v_orig
        );

        if (m % 2 == 1) {
            permuted_mesh.faces[m] = make_int3(f.y, f.z, f.x);
            permuted_mesh.faces_edges[m] = make_int3(fe.y, fe.z, fe.x);
        } else if (m % 3 == 0) {
            permuted_mesh.faces[m] = make_int3(f.z, f.x, f.y);
            permuted_mesh.faces_edges[m] = make_int3(fe.z, fe.x, fe.y);
        } else {
            continue;
        }

        const int3 fp = permuted_mesh.faces[m];
        vec3_t u_perm{}, v_perm{};
        get_face_basis_vectors(
            permuted_mesh.vertices[fp.x],
            permuted_mesh.vertices[fp.y],
            permuted_mesh.vertices[fp.z],
            u_perm, v_perm
        );

        const real_t Q[2][2] = {
            {dot(u_perm, u_orig), dot(u_perm, v_orig)},
            {dot(v_perm, u_orig), dot(v_perm, v_orig)},
        };
        rotate_face_record_in_plane(&rotated_tensors[32 * m], Q);
    }

    // 1. Without rotating the local 2x2 tensors, permuting (v0, v1, v2) changes the physical operator
    const auto mismatched = sie_mom::solvers::solve_gstc_dense(
        permuted_mesh, tensors, options, source
    );
    CHECK(relative_l2_difference(mismatched.physical_currents, baseline.physical_currents) > 1e-2);

    // 2. With the exact 2D frame rotation Q, the physical currents match to 32-bit quadrature node precision
    const auto gauge_transformed = sie_mom::solvers::solve_gstc_dense(
        permuted_mesh, rotated_tensors, options, source
    );
    CHECK(gauge_transformed.metrics.relative_residual < kExactKernelTolerance);
    CHECK(relative_l2_difference(gauge_transformed.physical_currents, baseline.physical_currents) < kPermutedQuadratureTolerance);
}