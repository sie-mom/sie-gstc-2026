#include "catch.hpp"

#include "sie_mom/cuda/sie_scaling.cuh"

#include <cmath>

namespace {

void check_complex_equal(
    const complex__float& actual,
    const complex__float& expected
) {
    CHECK(actual.x == Approx(expected.x));
    CHECK(actual.y == Approx(expected.y));
}

} // namespace

TEST_CASE(
    "eta0 scaling equals D A D for unequal material parameters",
    "[sie][scaling]"
) {
    const real_t mu0 = static_cast<real_t>(4.0e-7 * M_PI);
    const real_t c0 = static_cast<real_t>(299792458.0);
    const real_t eps0 = real_t(1) / (mu0 * c0 * c0);
    const real_t omega = static_cast<real_t>(2.0 * M_PI * 10.0e9);
    const real_t mu_r = real_t(1);
    const real_t eps_r = real_t(7);
    const real_t eta0 = sqrt(mu0 / eps0);

    const complex__float tl = make_complex(real_t(0.25), real_t(-0.5));
    const complex__float tk = make_complex(real_t(-0.75), real_t(0.125));
    const complex__float j_omega = make_complex(real_t(0), omega);

    const complex__float unscaled_jj = complexMul(
        j_omega,
        complexScale(tl, mu_r * mu0)
    );
    const complex__float unscaled_jm = tk;
    const complex__float unscaled_mj = complexNeg(tk);
    const complex__float unscaled_mm = complexMul(
        j_omega,
        complexScale(tl, eps_r * eps0)
    );

    const auto factors = sie_mom::cuda::make_eta0_scaled_calderon_factors(
        omega,
        mu_r * mu0,
        eps_r * eps0,
        eta0
    );
    const auto scaled = sie_mom::cuda::make_eta0_scaled_calderon_block(
        tl,
        tk,
        factors
    );

    check_complex_equal(scaled.jj, unscaled_jj);
    check_complex_equal(scaled.jm, complexScale(unscaled_jm, eta0));
    check_complex_equal(scaled.mj, complexScale(unscaled_mj, eta0));
    check_complex_equal(
        scaled.mm,
        complexScale(unscaled_mm, eta0 * eta0)
    );

    const real_t correct_mm_material_factor = eta0 * eta0 * eps_r * eps0;
    const real_t legacy_mm_material_factor = mu_r * mu0;
    CHECK(correct_mm_material_factor == Approx(real_t(7) * mu0));
    CHECK(correct_mm_material_factor != Approx(legacy_mm_material_factor));
}
