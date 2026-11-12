#pragma once

#include "sie_mom/cuda/math_utils.cuh"

namespace sie_mom {
namespace cuda {

/**
 * @brief One medium's eta0-scaled Calderon block.
 *
 * The entries are ordered by electric and magnetic surface-current unknowns:
 *
 *     [ jj  jm ]
 *     [ mj  mm ]
 */
struct Eta0ScaledCalderonBlock {
    complex__float jj;
    complex__float jm;
    complex__float mj;
    complex__float mm;
};

/**
 * @brief Invariant factors for one medium's eta0-scaled Calderon block.
 *
 * These factors depend only on frequency, material properties, and eta0. They
 * should be computed before applying the block scaling to individual TL and TK
 * entries.
 */
struct Eta0ScaledCalderonFactors {
    complex__float jj_tl;
    real_t jm_tk;
    complex__float mm_tl;
};

/**
 * @brief Prepare the invariant factors for one medium block.
 *
 * Starting from
 *
 *     A = [ j omega mu TL    TK              ]
 *         [ -TK              j omega eps TL  ],
 *
 * the factors produce D A D for D = diag(1, eta0). The lower-right
 * factor deliberately retains eta0^2 * eps: replacing it by mu is valid only
 * when eps_r == mu_r.
 */
__host__ __device__ inline Eta0ScaledCalderonFactors make_eta0_scaled_calderon_factors(
    real_t omega,
    real_t mu,
    real_t eps,
    real_t eta0
) {
    const complex__float j_omega = make_complex(real_t(0), omega);

    Eta0ScaledCalderonFactors factors;
    factors.jj_tl = complexScale(j_omega, mu);
    factors.jm_tk = eta0;
    factors.mm_tl = complexScale(j_omega, eta0 * eta0 * eps);
    return factors;
}

/**
 * @brief Apply precomputed eta0 scaling factors to one TL/TK entry pair.
 */
__host__ __device__ inline Eta0ScaledCalderonBlock make_eta0_scaled_calderon_block(
    const complex__float& tl,
    const complex__float& tk,
    const Eta0ScaledCalderonFactors& factors
) {
    Eta0ScaledCalderonBlock block;
    block.jj = complexMul(factors.jj_tl, tl);
    block.jm = complexScale(tk, factors.jm_tk);
    block.mj = complexNeg(block.jm);
    block.mm = complexMul(factors.mm_tl, tl);
    return block;
}

} // namespace cuda
} // namespace sie_mom
