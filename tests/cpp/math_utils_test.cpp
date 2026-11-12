#include "catch.hpp"

#include "sie_mom/cuda/math_utils.cuh"

TEST_CASE("complex conjugation preserves the build precision", "[math][complex]") {
    const auto value = make_complex(real_t(1.23456789012345), real_t(-2.34567890123456));
    const auto conjugate = complexConj(value);
    CHECK(conjugate.x == value.x);
    CHECK(conjugate.y == -value.y);

    const auto restored = complexConj(conjugate);
    CHECK(restored.x == value.x);
    CHECK(restored.y == value.y);

    const auto product = complexMul(value, conjugate);
    CHECK(product.x == Approx(value.x * value.x + value.y * value.y));
    CHECK(product.y == real_t(0));
}
