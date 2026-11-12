#include "catch.hpp"

#include "sie_mom/cuda/gstc_tensor_utils.cuh"
#include "sie_mom/solvers/gstc_sie.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using SingleSolve = sie_mom::solvers::GstcSieResult (*)(
    const sie_mom::mesh::MeshData&,
    const std::vector<std::complex<real_t>>&,
    const sie_mom::solvers::GstcSieOptions&,
    const sie_mom::excitation::PreparedExcitation&
);
using BatchSolve = sie_mom::solvers::GstcSieBatchResult (*)(
    const sie_mom::mesh::MeshData&,
    const std::vector<std::complex<real_t>>&,
    const sie_mom::solvers::GstcSieOptions&,
    const std::vector<sie_mom::solvers::GstcExcitationCase>&
);

static_assert(std::is_same_v<
    decltype(&sie_mom::solvers::solve_gstc_dense), SingleSolve
>);
static_assert(std::is_same_v<
    decltype(&sie_mom::solvers::solve_gstc_dense_batch), BatchSolve
>);

sie_mom::mesh::MeshData one_face_mesh() {
    sie_mom::mesh::MeshData mesh;
    mesh.vertices = {
        make_vec3(real_t(0), real_t(0), real_t(0)),
        make_vec3(real_t(1), real_t(0), real_t(0)),
        make_vec3(real_t(0), real_t(1), real_t(0)),
    };
    mesh.faces = {make_int3(0, 1, 2)};
    mesh.edges = {
        make_int2(0, 1),
        make_int2(1, 2),
        make_int2(2, 0),
    };
    mesh.faces_edges = {make_int3(0, 1, 2)};
    return mesh;
}

} // namespace

TEST_CASE(
    "GstcSieOptions defaults to the geometry-agnostic face basis",
    "[sie][basis][solver_contract]"
) {
    sie_mom::solvers::GstcSieOptions options;
    CHECK(options.basis == sie_mom::GstcBasisType::face);
}

TEST_CASE(
    "get_face_basis_vectors builds a right-handed orthonormal triad aligned with v0->v1",
    "[sie][basis][geometry]"
) {
    const double eps = static_cast<double>(std::numeric_limits<real_t>::epsilon()) * 64.0;

    // Arbitrary skewed non-spherical 3D triangle
    const vec3_t v0 = make_vec3(real_t(0.35), real_t(-0.70), real_t(1.15));
    const vec3_t v1 = make_vec3(real_t(1.60), real_t(0.25), real_t(0.80));
    const vec3_t v2 = make_vec3(real_t(0.10), real_t(0.95), real_t(2.05));

    vec3_t t1{}, t2{};
    get_face_basis_vectors(v0, v1, v2, t1, t2);

    const vec3_t e1 = v1 - v0;
    const vec3_t e2 = v2 - v0;
    const vec3_t n_expected = normalize(cross(e1, e2));
    const vec3_t e1_unit = normalize(e1);
    const vec3_t n_actual = cross(t1, t2);

    // Unit length and mutual orthogonality
    CHECK(length(t1) == Approx(1.0).margin(eps));
    CHECK(length(t2) == Approx(1.0).margin(eps));
    CHECK(dot(t1, t2) == Approx(0.0).margin(eps));
    CHECK(dot(t1, n_expected) == Approx(0.0).margin(eps));
    CHECK(dot(t2, n_expected) == Approx(0.0).margin(eps));

    // Alignment of t1 with edge v0 -> v1 and right-handedness t1 x t2 = n_hat
    CHECK(dot(t1, e1_unit) == Approx(1.0).margin(eps));
    CHECK(n_actual.x == Approx(n_expected.x).margin(eps));
    CHECK(n_actual.y == Approx(n_expected.y).margin(eps));
    CHECK(n_actual.z == Approx(n_expected.z).margin(eps));

    // Cyclic vertex permutation (v1, v2, v0) preserves outward normal n_hat
    // while rotating (t1, t2) within the same tangent plane.
    vec3_t t1_perm{}, t2_perm{};
    get_face_basis_vectors(v1, v2, v0, t1_perm, t2_perm);
    const vec3_t n_perm = cross(t1_perm, t2_perm);
    CHECK(n_perm.x == Approx(n_expected.x).margin(eps));
    CHECK(n_perm.y == Approx(n_expected.y).margin(eps));
    CHECK(n_perm.z == Approx(n_expected.z).margin(eps));
    CHECK(std::abs(dot(t1, t1_perm)) < real_t(0.99));

    // Reversed winding (v0, v2, v1) flips normal to -n_hat while keeping t1 x t2 = -n_hat
    vec3_t t1_flip{}, t2_flip{};
    get_face_basis_vectors(v0, v2, v1, t1_flip, t2_flip);
    const vec3_t n_flip = cross(t1_flip, t2_flip);
    CHECK(n_flip.x == Approx(-n_expected.x).margin(eps));
    CHECK(n_flip.y == Approx(-n_expected.y).margin(eps));
    CHECK(n_flip.z == Approx(-n_expected.z).margin(eps));
}

TEST_CASE(
    "rotate_tensor_2x2_to_3x3 is invariant under 2D tangent-frame rotation",
    "[sie][basis][tensors]"
) {
    const double tol = static_cast<double>(std::numeric_limits<real_t>::epsilon()) * 128.0;

    const vec3_t v0 = make_vec3(real_t(-0.4), real_t(0.9), real_t(0.2));
    const vec3_t v1 = make_vec3(real_t(0.8), real_t(0.3), real_t(-0.5));
    const vec3_t v2 = make_vec3(real_t(0.1), real_t(1.7), real_t(0.6));

    vec3_t u{}, v{};
    vec3_t u_alt{}, v_alt{};
    get_face_basis_vectors(v0, v1, v2, u, v);
    get_face_basis_vectors(v1, v2, v0, u_alt, v_alt);

    // General complex asymmetric 2x2 block in (u, v)
    const std::complex<real_t> T[2][2] = {
        {{real_t(1.7), real_t(-0.4)}, {real_t(-0.3), real_t(0.85)}},
        {{real_t(0.6), real_t(0.25)}, {real_t(-1.1), real_t(0.95)}},
    };

    // 2x2 orthogonal change-of-basis matrix Q_ij = basis_alt_i . basis_j
    const real_t Q[2][2] = {
        {dot(u_alt, u), dot(u_alt, v)},
        {dot(v_alt, u), dot(v_alt, v)},
    };
    const real_t det_Q = Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0];
    CHECK(det_Q == Approx(1.0).margin(tol));

    // T_alt = Q * T * Q^T
    std::complex<real_t> T_alt[2][2]{};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            std::complex<real_t> sum{real_t(0), real_t(0)};
            for (int p = 0; p < 2; ++p) {
                for (int q = 0; q < 2; ++q) {
                    sum += Q[r][p] * T[p][q] * Q[c][q];
                }
            }
            T_alt[r][c] = sum;
        }
    }

    complex__float glob[3][3]{};
    complex__float glob_alt[3][3]{};
    rotate_tensor_2x2_to_3x3(
        make_complex(T[0][0].real(), T[0][0].imag()),
        make_complex(T[0][1].real(), T[0][1].imag()),
        make_complex(T[1][0].real(), T[1][0].imag()),
        make_complex(T[1][1].real(), T[1][1].imag()),
        u, v, glob
    );
    rotate_tensor_2x2_to_3x3(
        make_complex(T_alt[0][0].real(), T_alt[0][0].imag()),
        make_complex(T_alt[0][1].real(), T_alt[0][1].imag()),
        make_complex(T_alt[1][0].real(), T_alt[1][0].imag()),
        make_complex(T_alt[1][1].real(), T_alt[1][1].imag()),
        u_alt, v_alt, glob_alt
    );

    const vec3_t n_hat = cross(u, v);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CHECK(glob[i][j].x == Approx(glob_alt[i][j].x).margin(tol));
            CHECK(glob[i][j].y == Approx(glob_alt[i][j].y).margin(tol));
        }
        // Normal nullspace check: T_glob * n_hat == 0
        const real_t re = glob[i][0].x * n_hat.x + glob[i][1].x * n_hat.y + glob[i][2].x * n_hat.z;
        const real_t im = glob[i][0].y * n_hat.x + glob[i][1].y * n_hat.y + glob[i][2].y * n_hat.z;
        CHECK(re == Approx(0.0).margin(tol));
        CHECK(im == Approx(0.0).margin(tol));
    }
}

TEST_CASE(
    "GSTC batch cases retain independent regional prepared sources",
    "[sie][batch_interface]"
) {
    using namespace sie_mom;
    const real_t frequency = real_t(1e9);
    const IsotropicMedium interior{real_t(7), real_t(2)};
    const IsotropicMedium exterior{real_t(3), real_t(1.5)};
    const auto interior_source = excitation::prepare_source(
        excitation::DipoleExcitation{}, frequency, interior, exterior
    );
    const auto exterior_source = excitation::prepare_source(
        excitation::PlaneWaveExcitation{}, frequency, interior, exterior
    );

    const std::vector<solvers::GstcExcitationCase> cases{
        {interior_source, std::nullopt},
        {std::nullopt, exterior_source},
        {interior_source, exterior_source},
    };
    CHECK(cases[0].interior.has_value());
    CHECK_FALSE(cases[0].exterior.has_value());
    CHECK_FALSE(cases[1].interior.has_value());
    CHECK(cases[1].exterior.has_value());
    REQUIRE(cases[2].interior.has_value());
    REQUIRE(cases[2].exterior.has_value());
    CHECK(cases[2].interior->region() == excitation::SourceRegion::interior);
    CHECK(cases[2].exterior->region() == excitation::SourceRegion::exterior);
    CHECK_NOTHROW(cases[2].interior->validate_environment(
        frequency, interior, exterior, real_t(8.8541878188e-12), real_t(1.25663706127e-6)
    ));
    CHECK_NOTHROW(cases[2].exterior->validate_environment(
        frequency, interior, exterior, real_t(8.8541878188e-12), real_t(1.25663706127e-6)
    ));
}

TEST_CASE(
    "GSTC batch results separate shared timings and ordered case diagnostics",
    "[sie][batch_interface]"
) {
    sie_mom::solvers::GstcSieBatchResult result;
    result.metrics.case_count = 2;
    result.metrics.factorization_seconds = 3.0;
    result.metrics.solve_seconds = 0.5;
    result.cases.resize(2);
    result.cases[0].physical_currents = {{real_t(1), real_t(2)}};
    result.cases[0].metrics.rhs_norm = 4.0;
    result.cases[0].metrics.relative_residual = 1e-10;
    result.cases[0].metrics.residual_computed = true;
    result.cases[1].physical_currents = {{real_t(-2), real_t(3)}};
    result.cases[1].metrics.rhs_norm = 8.0;

    CHECK(result.cases.size() == result.metrics.case_count);
    CHECK(result.metrics.factorization_seconds == 3.0);
    CHECK(result.metrics.solve_seconds == 0.5);
    CHECK(result.cases[0].physical_currents[0] == std::complex<real_t>(1, 2));
    CHECK(result.cases[1].physical_currents[0] == std::complex<real_t>(-2, 3));
    CHECK(result.cases[0].metrics.rhs_norm == 4.0);
    CHECK(result.cases[1].metrics.rhs_norm == 8.0);
    CHECK(result.cases[0].metrics.residual_computed);
    CHECK(result.cases[0].metrics.relative_residual == Approx(1e-10));
    CHECK_FALSE(result.cases[1].metrics.residual_computed);
}

TEST_CASE(
    "GSTC SIE rejects an empty mesh before CUDA work",
    "[sie][solver_contract]"
) {
    sie_mom::mesh::MeshData mesh;
    std::vector<std::complex<real_t>> tensors;
    sie_mom::solvers::GstcSieOptions options;
    options.frequency_hz = real_t(1.0e9);
    const auto source = sie_mom::excitation::prepare_source(
        sie_mom::excitation::DipoleExcitation{}, options.frequency_hz,
        options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    CHECK_THROWS_AS(
        sie_mom::solvers::solve_gstc_dense(mesh, tensors, options, source),
        std::invalid_argument
    );
    CHECK_THROWS_AS(
        sie_mom::solvers::solve_gstc_dense_batch(
            mesh, tensors, options, {{source, std::nullopt}}
        ),
        std::invalid_argument
    );
}

TEST_CASE(
    "GSTC SIE validates per-face tensor record count",
    "[sie][solver_contract]"
) {
    const auto mesh = one_face_mesh();
    std::vector<std::complex<real_t>> tensors(31, {real_t(0), real_t(0)});
    sie_mom::solvers::GstcSieOptions options;
    options.frequency_hz = real_t(1.0e9);
    const auto source = sie_mom::excitation::prepare_source(
        sie_mom::excitation::DipoleExcitation{}, options.frequency_hz,
        options.interior, options.exterior,
        options.vacuum_permittivity, options.vacuum_permeability
    );

    CHECK_THROWS_AS(
        sie_mom::solvers::solve_gstc_dense(mesh, tensors, options, source),
        std::invalid_argument
    );
    CHECK_THROWS_AS(
        sie_mom::solvers::solve_gstc_dense_batch(
            mesh, tensors, options, {{source, std::nullopt}}
        ),
        std::invalid_argument
    );
}

TEST_CASE(
    "GSTC batch validates every excitation case before CUDA work",
    "[sie][batch][solver_contract]"
) {
    using namespace sie_mom;
    const auto mesh = one_face_mesh();
    const std::vector<std::complex<real_t>> tensors(32);
    solvers::GstcSieOptions options;
    options.frequency_hz = real_t(1e9);
    options.tensors_are_uniform = true;
    const auto interior = excitation::prepare_source(
        excitation::DipoleExcitation{}, options.frequency_hz, options.interior, options.exterior
    );
    const auto exterior = excitation::prepare_source(
        excitation::PlaneWaveExcitation{}, options.frequency_hz, options.interior, options.exterior
    );
    std::vector<solvers::GstcExcitationCase> cases{{interior, std::nullopt}};

    SECTION("empty batch") {
        cases.clear();
    }
    SECTION("empty later case") {
        cases.push_back({std::nullopt, std::nullopt});
    }
    SECTION("interior source in the exterior slot") {
        cases.push_back({std::nullopt, interior});
    }
    SECTION("exterior source in the interior slot") {
        cases.push_back({exterior, std::nullopt});
    }
    SECTION("interior source at a different frequency") {
        options.frequency_hz *= real_t(2);
    }
    SECTION("later exterior source at a different frequency") {
        cases.push_back({std::nullopt, excitation::prepare_source(
            excitation::PlaneWaveExcitation{}, options.frequency_hz * real_t(2),
            options.interior, options.exterior
        )});
    }
    SECTION("interior source in a different medium") {
        options.interior.relative_permittivity = real_t(2);
    }
    SECTION("exterior source in a different medium") {
        cases.push_back({std::nullopt, exterior});
        options.exterior.relative_permeability = real_t(2);
    }
    CHECK_THROWS_AS(
        solvers::solve_gstc_dense_batch(mesh, tensors, options, cases),
        std::invalid_argument
    );
}

