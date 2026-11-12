#include "catch.hpp"

#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/solvers/dielectric/dielectric_solver.hpp"
#include "sie_mom/solvers/gstc_sie.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace sie_mom;
using namespace sie_mom::excitation;
using namespace sie_mom::solvers;

constexpr real_t frequency = real_t(1e9);
constexpr real_t eps0 = real_t(8.8541878188e-12);
constexpr real_t mu0 = real_t(1.25663706127e-6);
#ifdef USE_DOUBLE_PRECISION
constexpr double tolerance = 1e-8;
#else
constexpr double tolerance = 2e-3;
#endif

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename Options>
PreparedExcitation prepare(const SourceSpec& spec, const Options& options) {
    return prepare_source(spec, options.frequency_hz, options.interior, options.exterior,
                          options.vacuum_permittivity, options.vacuum_permeability);
}

void check_scaled_currents(
    const std::vector<std::complex<real_t>>& actual,
    const std::vector<std::complex<real_t>>& reference,
    real_t scale
) {
    REQUIRE(actual.size() == reference.size());
    double error = 0;
    double norm = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        error += std::norm(actual[index] - scale * reference[index]);
        norm += std::norm(scale * reference[index]);
    }
    REQUIRE(norm > 0);
    CHECK(std::sqrt(error / norm) < tolerance);
}

} // namespace

TEST_CASE("prepared sources bind the source region and reject mismatched physics", "[excitation]") {
    const IsotropicMedium interior{real_t(7), real_t(2)};
    const IsotropicMedium exterior{real_t(3), real_t(1.5)};
    for (auto region : {SourceRegion::interior, SourceRegion::exterior}) {
        DipoleExcitation spec;
        spec.region = region;
        const auto source = prepare_source(spec, frequency, interior, exterior, eps0, mu0);
        CHECK(source.region() == region);
        const auto& medium = region == SourceRegion::interior ? interior : exterior;
        source.visit([&](const auto& functor) {
            const real_t epsilon = medium.relative_permittivity * eps0;
            const real_t mu = medium.relative_permeability * mu0;
            CHECK(functor.k == Approx(real_t(2) * std::acos(real_t(-1)) * frequency * std::sqrt(mu * epsilon)));
            CHECK(functor.eta == Approx(std::sqrt(mu / epsilon)));
        });
        CHECK_NOTHROW(source.validate_environment(frequency, interior, exterior, eps0, mu0));
        CHECK_THROWS_AS(source.validate_environment(frequency * 2, interior, exterior, eps0, mu0), std::invalid_argument);
        CHECK_THROWS_AS(source.validate_environment(frequency, exterior, interior, eps0, mu0), std::invalid_argument);
        CHECK_THROWS_AS(source.validate_environment(frequency, interior, exterior, eps0 * 2, mu0), std::invalid_argument);
    }
}

TEST_CASE("invalid excitation settings are rejected before GPU work", "[excitation]") {
    DipoleExcitation dipole;
    dipole.region = static_cast<SourceRegion>(0);
    CHECK_THROWS_AS(prepare_source(dipole, frequency, {}, {}), std::invalid_argument);
    dipole.region = SourceRegion::interior;
    dipole.effective_moment = make_vec3(0, 0, 0);
    CHECK_THROWS_AS(prepare_source(dipole, frequency, {}, {}), std::invalid_argument);
    dipole = DipoleExcitation{};
    dipole.position.x = std::numeric_limits<real_t>::quiet_NaN();
    CHECK_THROWS_AS(prepare_source(dipole, frequency, {}, {}), std::invalid_argument);
    CHECK_THROWS_AS(prepare_source(PlaneWaveExcitation{}, 0, {}, {}), std::invalid_argument);
    PlaneWaveExcitation wave;
    wave.electric_amplitude = wave.propagation_direction;
    CHECK_THROWS_AS(prepare_source(wave, frequency, {}, {}), std::invalid_argument);
    wave = PlaneWaveExcitation{};
    wave.propagation_direction = make_vec3(0, 0, 0);
    CHECK_THROWS_AS(prepare_source(wave, frequency, {}, {}), std::invalid_argument);
    const auto source = prepare_source(DipoleExcitation{}, frequency, {}, {});
    CHECK(evaluate_incident_fields({}, source).empty());
    CHECK_THROWS_AS(evaluate_incident_fields({make_vec3(std::numeric_limits<real_t>::infinity(), 0, 0)}, source), std::invalid_argument);
}

TEST_CASE("GPU plane-wave fields use the prepared amplitude phase and impedance", "[excitation][gpu]") {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    PlaneWaveExcitation spec;
    spec.electric_amplitude = make_vec3(2, 0, 0);
    spec.propagation_direction = make_vec3(0, 0, 3);
    const IsotropicMedium exterior{real_t(4), real_t(2)};
    const auto source = prepare_source(spec, frequency, {}, exterior, eps0, mu0);
    spec.electric_amplitude = make_vec3(9, 0, 0); // The prepared value is independent of later edits.
    const std::vector<vec3_t> points{make_vec3(0, 0, 0), make_vec3(0.1, -0.03, 0.017)};
    const auto fields = evaluate_incident_fields(points, source);
    const real_t eta = std::sqrt(real_t(2) * mu0 / (real_t(4) * eps0));
    const real_t k = real_t(2) * std::acos(real_t(-1)) * frequency * std::sqrt(real_t(8) * mu0 * eps0);
    REQUIRE(fields.size() == points.size());
    for (std::size_t index = 0; index < points.size(); ++index) {
        const auto expected_e = real_t(2) * std::exp(std::complex<real_t>(0, -k * points[index].z));
        CHECK(std::abs(fields[index].electric[0] - expected_e) < tolerance);
        CHECK(std::abs(fields[index].magnetic[1] - expected_e / eta) < tolerance);
        CHECK(std::abs(fields[index].electric[1]) < tolerance);
        CHECK(std::abs(fields[index].electric[2]) < tolerance);
        CHECK(std::abs(fields[index].magnetic[0]) < tolerance);
        CHECK(std::abs(fields[index].magnetic[2]) < tolerance);
    }
}

TEST_CASE("electric and magnetic dipoles retain duality on the GPU", "[excitation][gpu]") {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    DipoleExcitation spec;
    spec.position = make_vec3(0.003, -0.002, 0.001);
    spec.effective_moment = make_vec3(0.2, -0.4, 0.7);
    spec.region = SourceRegion::exterior;
    const IsotropicMedium medium{real_t(3), real_t(2)};
    const std::vector<vec3_t> points{make_vec3(0.1, 0.04, -0.02)};
    const auto electric = evaluate_incident_fields(points, prepare_source(spec, frequency, {}, medium, eps0, mu0));
    spec.is_magnetic = true;
    const auto magnetic = evaluate_incident_fields(points, prepare_source(spec, frequency, {}, medium, eps0, mu0));
    const real_t eta = std::sqrt(real_t(2) * mu0 / (real_t(3) * eps0));
    for (int component = 0; component < 3; ++component) {
        CHECK(std::abs(magnetic[0].electric[component] + eta * electric[0].magnetic[component])
              < tolerance * std::abs(magnetic[0].electric[component]));
        CHECK(std::abs(magnetic[0].magnetic[component] - electric[0].electric[component] / eta)
              < tolerance * std::abs(magnetic[0].magnetic[component]));
    }
}

TEST_CASE("both solvers accept both prepared source types and regions", "[excitation][gpu][solver]") {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    DielectricOptions pmchwt;
    GstcSieOptions gstc;
    pmchwt.frequency_hz = gstc.frequency_hz = frequency;
    pmchwt.interior = gstc.interior = {real_t(2.5), real_t(1.2)};
    pmchwt.exterior = gstc.exterior = {real_t(1.1), real_t(0.9)};
    pmchwt.quadrature_rule = gstc.quadrature_rule = 5;
    gstc.tensors_are_uniform = true;
    std::vector<std::complex<real_t>> tensors(32);
    // A diagonal sheet map and its inverse, stored as two row-major 4x4 matrices.
    for (int index = 0; index < 4; ++index) {
        const real_t admittance = index < 2 ? real_t(0.01) : real_t(100);
        tensors[5 * index] = admittance;
        tensors[16 + 5 * index] = real_t(1) / admittance;
    }
    for (auto region : {SourceRegion::interior, SourceRegion::exterior}) {
        DipoleExcitation dipole;
        dipole.region = region;
        dipole.position = region == SourceRegion::interior
            ? make_vec3(0.003, -0.002, 0.001) : make_vec3(0.1, -0.02, 0.01);
        PlaneWaveExcitation wave;
        wave.region = region;
        for (const SourceSpec& spec : {SourceSpec(dipole), SourceSpec(wave)}) {
            SourceSpec doubled = spec;
            std::visit([](auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, DipoleExcitation>) value.effective_moment = value.effective_moment * real_t(2);
                else value.electric_amplitude = value.electric_amplitude * real_t(2);
            }, doubled);
            for (auto formulation : {DielectricFormulation::T_PMCHWT, DielectricFormulation::N_PMCHWT}) {
                pmchwt.formulation = formulation;
                const auto result = solve_dielectric_dense(mesh, pmchwt, prepare(spec, pmchwt));
                const auto scaled = solve_dielectric_dense(mesh, pmchwt, prepare(doubled, pmchwt));
                CHECK(result.metrics.relative_residual < tolerance);
                check_scaled_currents(scaled.physical_currents, result.physical_currents, real_t(2));
            }
            const auto result = solve_gstc_dense(mesh, tensors, gstc, prepare(spec, gstc));
            const auto scaled = solve_gstc_dense(mesh, tensors, gstc, prepare(doubled, gstc));
            CHECK(result.metrics.relative_residual < tolerance);
            check_scaled_currents(scaled.physical_currents, result.physical_currents, real_t(2));
        }
    }
}

TEST_CASE("solvers reject a source prepared at another frequency", "[excitation][solver_contract]") {
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    const auto source = prepare_source(DipoleExcitation{}, frequency, {}, {});
    DielectricOptions pmchwt;
    GstcSieOptions gstc;
    pmchwt.frequency_hz = gstc.frequency_hz = frequency * 2;
    gstc.tensors_are_uniform = true;
    const std::vector<std::complex<real_t>> tensors(32);
    CHECK_THROWS_AS(solve_dielectric_dense(mesh, pmchwt, source), std::invalid_argument);
    CHECK_THROWS_AS(solve_gstc_dense(mesh, tensors, gstc, source), std::invalid_argument);
}

TEST_CASE(
    "GSTC batch matches independent solves for uniform and per-face sheets",
    "[sie][batch][excitation][gpu][solver]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    GstcSieOptions options;
    options.frequency_hz = frequency;
    options.interior = {real_t(2.5), real_t(1.2)};
    options.exterior = {real_t(1.1), real_t(0.9)};
    options.quadrature_rule = 5;

    DipoleExcitation electric;
    electric.position = make_vec3(0.003, -0.002, 0.001);
    electric.effective_moment = make_vec3(0.2, -0.4, 0.7);
    DipoleExcitation magnetic = electric;
    magnetic.position = make_vec3(0, 0, 0);
    magnetic.effective_moment = make_vec3(0, 0, 0.5);
    magnetic.is_magnetic = true;
    PlaneWaveExcitation wave;
    wave.electric_amplitude = make_vec3(0.7, 0.3, 0);
    wave.propagation_direction = make_vec3(0, 0, 3);
    DipoleExcitation outside = magnetic;
    outside.position = make_vec3(0.1, -0.02, 0.01);
    outside.effective_moment = make_vec3(0.5, -0.1, 0.3);
    outside.region = SourceRegion::exterior;
    const std::vector<PreparedExcitation> sources{
        prepare(electric, options), prepare(magnetic, options),
        prepare(wave, options), prepare(outside, options),
    };
    std::vector<GstcExcitationCase> cases;
    for (const auto& source : sources) {
        cases.push_back(source.region() == SourceRegion::interior
            ? GstcExcitationCase{source, std::nullopt}
            : GstcExcitationCase{std::nullopt, source});
    }

    for (bool uniform : {true, false}) {
        options.tensors_are_uniform = uniform;
        const std::size_t records = uniform ? 1 : mesh.faces.size();
        std::vector<std::complex<real_t>> tensors(32 * records);
        for (std::size_t face = 0; face < records; ++face) {
            for (int block = 0; block < 4; ++block) {
                const real_t admittance = (block < 2 ? real_t(0.01) : real_t(100))
                    * (real_t(1) + real_t(0.01) * static_cast<real_t>(face));
                tensors[32 * face + 5 * block] = admittance;
                tensors[32 * face + 16 + 5 * block] = real_t(1) / admittance;
            }
        }
        const auto batch = solve_gstc_dense_batch(mesh, tensors, options, cases);
        REQUIRE(batch.cases.size() == sources.size());
        CHECK(batch.metrics.case_count == sources.size());
        CHECK(batch.metrics.vertex_count == mesh.vertices.size());
        CHECK(batch.metrics.face_count == mesh.faces.size());
        CHECK(batch.metrics.edge_count == mesh.edges.size());
        CHECK(batch.metrics.matrix_dimension == 4 * mesh.edges.size());
        CHECK(batch.metrics.dense_matrix_bytes
            == batch.metrics.matrix_dimension * batch.metrics.matrix_dimension * sizeof(complex__float));
        CHECK(batch.metrics.device_preparation_seconds > 0);
        CHECK(batch.metrics.matrix_assembly_seconds > 0);
        CHECK(batch.metrics.factorization_seconds > 0);
        CHECK(batch.metrics.solve_seconds > 0);
        CHECK(batch.metrics.residual_matrix_assembly_seconds > 0);
        for (std::size_t index = 0; index < sources.size(); ++index) {
            INFO("uniform=" << uniform << ", case=" << index);
            const auto single = solve_gstc_dense(mesh, tensors, options, sources[index]);
            const auto& actual = batch.cases[index];
            check_scaled_currents(actual.physical_currents, single.physical_currents, real_t(1));
            CHECK(actual.metrics.rhs_norm == Approx(single.metrics.rhs_norm).epsilon(tolerance));
            CHECK(actual.metrics.residual_computed);
            CHECK(actual.metrics.relative_residual < tolerance);
            CHECK(actual.metrics.rhs_assembly_seconds > 0);
            CHECK(actual.metrics.residual_seconds > 0);
            CHECK(actual.metrics.maximum_j1_coefficient
                == Approx(single.metrics.maximum_j1_coefficient).epsilon(tolerance));
            CHECK(actual.metrics.maximum_m1_coefficient
                == Approx(single.metrics.maximum_m1_coefficient).epsilon(tolerance));
            CHECK(actual.metrics.maximum_j2_coefficient
                == Approx(single.metrics.maximum_j2_coefficient).epsilon(tolerance));
            CHECK(actual.metrics.maximum_m2_coefficient
                == Approx(single.metrics.maximum_m2_coefficient).epsilon(tolerance));
        }
    }
}

TEST_CASE(
    "GSTC batch combines both incident regions and can skip residual reassembly",
    "[sie][batch][excitation][gpu][solver]"
) {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    GstcSieOptions options;
    options.frequency_hz = frequency;
    options.interior = {real_t(2.5), real_t(1.2)};
    options.exterior = {real_t(1.1), real_t(0.9)};
    options.quadrature_rule = 5;
    options.tensors_are_uniform = true;
    std::vector<std::complex<real_t>> tensors(32);
    for (int block = 0; block < 4; ++block) {
        const real_t admittance = block < 2 ? real_t(0.01) : real_t(100);
        tensors[5 * block] = admittance;
        tensors[16 + 5 * block] = real_t(1) / admittance;
    }
    DipoleExcitation dipole;
    dipole.position = make_vec3(0.003, -0.002, 0.001);
    // Comparable RHS magnitudes expose loss of either regional contribution.
    dipole.effective_moment = make_vec3(real_t(2e-6), real_t(-4e-6), real_t(7e-6));
    const auto interior = prepare(dipole, options);
    const auto exterior = prepare(PlaneWaveExcitation{}, options);
    const std::vector<GstcExcitationCase> cases{
        {interior, std::nullopt}, {std::nullopt, exterior}, {interior, exterior},
    };
    const auto batch = solve_gstc_dense_batch(mesh, tensors, options, cases);
    REQUIRE(batch.cases.size() == 3);
    const double rhs_ratio = batch.cases[0].metrics.rhs_norm / batch.cases[1].metrics.rhs_norm;
    REQUIRE(rhs_ratio > 0.01);
    REQUIRE(rhs_ratio < 100.0);
    auto sum = batch.cases[0].physical_currents;
    REQUIRE(sum.size() == batch.cases[1].physical_currents.size());
    for (std::size_t index = 0; index < sum.size(); ++index) {
        sum[index] += batch.cases[1].physical_currents[index];
    }
    check_scaled_currents(batch.cases[2].physical_currents, sum, real_t(1));
    CHECK(batch.cases[2].metrics.rhs_norm == Approx(std::hypot(
        batch.cases[0].metrics.rhs_norm, batch.cases[1].metrics.rhs_norm
    )).epsilon(tolerance));
    for (const auto& result : batch.cases) {
        CHECK(result.metrics.residual_computed);
        CHECK(result.metrics.relative_residual < tolerance);
    }

    options.compute_residual = false;
    const auto no_residual = solve_gstc_dense_batch(mesh, tensors, options, {cases[2]});
    REQUIRE(no_residual.cases.size() == 1);
    CHECK(no_residual.metrics.case_count == 1);
    CHECK(no_residual.metrics.residual_matrix_assembly_seconds == 0.0);
    const auto& actual = no_residual.cases[0];
    CHECK_FALSE(actual.metrics.residual_computed);
    CHECK(actual.metrics.residual_seconds == 0.0);
    CHECK(actual.metrics.absolute_residual == 0.0);
    CHECK(actual.metrics.relative_residual == 0.0);
    CHECK(actual.metrics.rhs_norm == Approx(batch.cases[2].metrics.rhs_norm).epsilon(tolerance));
    check_scaled_currents(actual.physical_currents, batch.cases[2].physical_currents, real_t(1));
}

TEST_CASE("PMCHWT exterior illumination uses the opposite RHS sign in two blocks", "[excitation][gpu][pmchwt]") {
    if (!has_cuda_device()) { WARN("No CUDA device; skipping GPU test"); return; }
    const auto mesh = sie_mom::mesh::SphericalSurface(0.03f, 0).getMeshData();
    DielectricOptions options;
    options.frequency_hz = frequency;
    options.quadrature_rule = 5;
    // Equal media make the same plane wave identical in either region.
    options.interior = options.exterior = {real_t(2), real_t(1)};
    PlaneWaveExcitation wave;
    for (auto formulation : {DielectricFormulation::T_PMCHWT, DielectricFormulation::N_PMCHWT}) {
        options.formulation = formulation;
        wave.region = SourceRegion::interior;
        const auto interior = solve_dielectric_dense(mesh, options, prepare(wave, options));
        wave.region = SourceRegion::exterior;
        const auto exterior = solve_dielectric_dense(mesh, options, prepare(wave, options));
        REQUIRE(exterior.physical_currents.size() == 2 * mesh.edges.size());
        check_scaled_currents(exterior.physical_currents, interior.physical_currents, real_t(-1));
    }
}
