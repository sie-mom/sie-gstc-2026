#ifndef GREEN_FUNCTIONS_UTILS_CUH
#define GREEN_FUNCTIONS_UTILS_CUH

#include "platform.cuh"
#include "math_utils.cuh"

/**
 * @brief Computes the scalar 3D Green's function for the Helmholtz equation.
 *
 * This function calculates the free-space Green's function, which is the solution to
 * the Helmholtz equation for a point source. The formula is:
 * \f[
 * G(\mathbf{r}, \mathbf{r}') = \frac{e^{-jk|\mathbf{r}-\mathbf{r}'|}}{4\pi|\mathbf{r}-\mathbf{r}'|}
 * \f]
 * where \f$ k \f$ is the wavenumber, \f$ \mathbf{r} \f$ is the observation point, and
 * \f$ \mathbf{r}' \f$ is the source point.
 *
 * For efficiency, this implementation uses the `sincosf` intrinsic to compute the
 * real and imaginary parts of the complex exponential \f$ e^{-jkR} = \cos(kR) - j\sin(kR) \f$
 * in a single operation.
 *
 * @param r1 The observation point vector \f$ \mathbf{r} \f$.
 * @param r2 The source point vector \f$ \mathbf{r}' \f$.
 * @param k The wavenumber \f$ k \f$.
 * @return A `complex__float` representing the value of the Green's function.
 */
__host__ __device__ inline complex__float greenFunction3D(const vec3_t &r1, const vec3_t &r2, const real_t k)
{
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    // Avoid division by zero if the points are coincident.
    // The calling code (e.g., singularity extraction) should typically prevent this.
    if (R_mag < static_cast<real_t>(1e-12)) {
        return make_complex(real_t(0), real_t(0));
    }

    const real_t inv_4_pi_R = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI) * R_mag);
    const real_t kR = k * R_mag;

    real_t s, c;
    SIE_MOM_SINCOS(kR, &s, &c); // Precision-aware trig helper

    // G = (cos(kR) - j*sin(kR)) / (4*pi*R)
    return make_complex(c * inv_4_pi_R, -s * inv_4_pi_R);
}

/**
 * @brief Computes the regular part of the 3D Green's function for singularity subtraction.
 *
 * This function implements the "regular" part of the Green's function, defined as
 * \f$ G_{\mathrm{reg}} = G - \frac{1}{4\pi R} \f$, which simplifies to:
 * \f[
 * G_{\mathrm{reg}}(R) = \frac{e^{-jkR} - 1}{4\pi R}
 * \f]
 * This form is used in singularity extraction schemes to isolate the Coulomb-type
 * singularity, which can then be handled analytically.
 *
 * To avoid numerical instability for small \f$ R \f$, this function uses the
 * limiting value as \f$ R \to 0 \f$, which is \f$ -jk/(4\pi) \f$. 
 * \verbatim embed:rst
 * See :eq:`eq:Green_reg_definition` in the theory documentation for more details. 
 * \endverbatim
 * @param r1 The observation point vector \f$ \mathbf{r} \f$.
 * @param r2 The source point vector \f$ \mathbf{r}' \f$.
 * @param k The wavenumber \f$ k \f$.
 * @return A `complex__float` representing the value of the regularized Green's function.
 */
__host__ __device__ inline complex__float greenFunction3DRegular(const vec3_t &r1, const vec3_t &r2, const real_t k)
{
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);
    if (R_mag < static_cast<real_t>(1e-12))
    {
        // return -\dfrac{jk}{4\pi}
        return make_complex(real_t(0), -k / (real_t(4) * static_cast<real_t>(M_PI)));
    }
   
    // \dfrac{e^{-jkR} - 1}{4\pi R}
    const real_t inv_4_pi_R = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI) * R_mag);
    const real_t kR = k * R_mag;
    real_t s, c;
    SIE_MOM_SINCOS(kR, &s, &c); // Precision-aware trig helper
    return make_complex((c - real_t(1)) * inv_4_pi_R, -s * inv_4_pi_R);
    
}

/**
 * @brief Computes the gradient of the 3D Green's function with respect to the source coordinates.
 *
 * This function calculates the quantity \f$ \nabla' G(\mathbf{r}, \mathbf{r}') \f$.
 * This term appears in the kernel of the \f$\mathcal{K}\f$ operator.
 *
 * The formula is derived from:
 * \f[
 * \nabla' G = \frac{e^{-jkR}}{4\pi R} (jk + \frac{1}{R}) \hat{\mathbf{R}} = G(R) (jk + 1/R) \hat{\mathbf{R}}
 * \f]
 * Note that \f$ \nabla' G = -\nabla G \f$.
 *
 * @param r1 The observation point vector \f$ \mathbf{r} \f$.
 * @param r2 The source point vector \f$ \mathbf{r}' \f$.
 * @param k The wavenumber \f$ k \f$.
 * @return A `cvec3_t` representing the vector value.
 */
__host__ __device__ inline cvec3_t gradSourceGreenFunction(const vec3_t &r1, const vec3_t &r2, const real_t k)
{
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    if (R_mag < static_cast<real_t>(1e-12))
    {
        const complex__float zero = make_complex(real_t(0), real_t(0));
        return make_cvec3(zero, zero, zero);
    }

    const real_t inv_R = real_t(1) / R_mag;
    const vec3_t R_hat = make_vec3(R_vec.x * inv_R, R_vec.y * inv_R, R_vec.z * inv_R);

    // Re-implement greenFunction3D locally to reuse R_mag
    const real_t inv_4_pi_R = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI) * R_mag);
    const real_t kR = k * R_mag;
    real_t s, c;
    SIE_MOM_SINCOS(kR, &s, &c);
    const complex__float green = make_complex(c * inv_4_pi_R, -s * inv_4_pi_R);

    const complex__float scalar = make_complex(inv_R, k);
    const complex__float weighted_green = complexMul(scalar, green);

    const complex__float x = complexScale(weighted_green, R_hat.x);
    const complex__float y = complexScale(weighted_green, R_hat.y);
    const complex__float z = complexScale(weighted_green, R_hat.z);

    return make_cvec3(x, y, z);
}

struct greenAndGradGreen
{
    complex__float green;
    cvec3_t grad_green;
};

__host__ __device__ inline greenAndGradGreen greenAndGradSourceGreenFunction(const vec3_t &r1, const vec3_t &r2, const real_t k)
{
    greenAndGradGreen result;
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    if (R_mag < static_cast<real_t>(1e-12))
    {
        result.green = make_complex(real_t(0), real_t(0));
        const complex__float zero = make_complex(real_t(0), real_t(0));
        result.grad_green = make_cvec3(zero, zero, zero);
        return result;
    }

    const real_t inv_R = real_t(1) / R_mag;
    const vec3_t R_hat = make_vec3(R_vec.x * inv_R, R_vec.y * inv_R, R_vec.z * inv_R);

    // Re-implement greenFunction3D locally to reuse R_mag
    const real_t inv_4_pi_R = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI) * R_mag);
    const real_t kR = k * R_mag;
    real_t s, c;
    SIE_MOM_SINCOS(kR, &s, &c);
    result.green = make_complex(c * inv_4_pi_R, -s * inv_4_pi_R);

    const complex__float scalar = make_complex(inv_R, k);
    const complex__float weighted_green = complexMul(scalar, result.green);

    const complex__float x = complexScale(weighted_green, R_hat.x);
    const complex__float y = complexScale(weighted_green, R_hat.y);
    const complex__float z = complexScale(weighted_green, R_hat.z);

    result.grad_green = make_cvec3(x, y, z);
    return result;
}

__host__ __device__ inline greenAndGradGreen greenAndGradSourceGreenFunctionRegular(const vec3_t &r1, const vec3_t &r2, const real_t k)
{
    greenAndGradGreen result;
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    if (R_mag < static_cast<real_t>(1e-6))
    {
        // G_reg -> -jk / (4*pi)
        result.green = make_complex(real_t(0), -k / (real_t(4) * static_cast<real_t>(M_PI)));
        
        // K_reg(R) -> -j * k^3 / 3, grad_G_reg = 1/(4*pi) * K_reg(R) * R_vec -> 0 vector
        const complex__float zero = make_complex(real_t(0), real_t(0));
        result.grad_green = make_cvec3(zero, zero, zero);
        return result;
    }

    const real_t inv_4_pi = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI));
    const real_t inv_R = real_t(1) / R_mag;
    const real_t inv_R2 = inv_R * inv_R;
    const real_t inv_R3 = inv_R2 * inv_R;
    const real_t inv_4_pi_R = inv_4_pi * inv_R;

    const real_t kR = k * R_mag;
    real_t s, c;
    SIE_MOM_SINCOS(kR, &s, &c); 
    // Note: exp(-jkR) = c - j*s

    // G_reg = (exp(-jkR) - 1) / (4*pi*R)
    result.green = make_complex((c - real_t(1)) * inv_4_pi_R, -s * inv_4_pi_R);

    // K_reg = (1/R^3 + j*k/R^2) * exp(-jkR) - (1/R^3 + k^2 / 2R)
    const complex__float term1 = make_complex(inv_R3, k * inv_R2);
    const complex__float exp_term = make_complex(c, -s);
    const real_t term3_real = -(inv_R3 + k * k * real_t(0.5) * inv_R);
    
    complex__float K_reg_val = complexMul(term1, exp_term);
    K_reg_val.x += term3_real;

    // grad_G_reg = 1/(4*pi) * K_reg_val * R_vec
    const complex__float scalar = complexScale(K_reg_val, inv_4_pi);
    result.grad_green = make_cvec3(complexScale(scalar, R_vec.x),
                                   complexScale(scalar, R_vec.y),
                                   complexScale(scalar, R_vec.z));
                                   
    return result;
}

struct IntegrandKernels
{
    complex__float tl_k1;
    complex__float tl_k2;
    cvec3_t tk_k1;
    cvec3_t tk_k2;
};

// This function will calculate at once the 4 integrand kernels: TL(k1), TL(k2), TK(k1), TK(k2)
// i.e., the kernels for the TL and TK operators with two different wavenumbers.
// we will resuse computations where possible, like R_vec and R_mag, inv_R, R_hat, sincosf, etc.
__host__ __device__ inline IntegrandKernels integrandKernels(const vec3_t &r1, const vec3_t &r2, const real_t k1, const real_t k2)
{
    IntegrandKernels kernels;

    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    if (R_mag < static_cast<real_t>(1e-12))
    {
        const complex__float zero = make_complex(real_t(0), real_t(0));
        kernels.tl_k1 = zero;
        kernels.tl_k2 = zero;
        kernels.tk_k1 = make_cvec3(zero, zero, zero);
        kernels.tk_k2 = make_cvec3(zero, zero, zero);
        return kernels;
    }

    const real_t inv_R = real_t(1) / R_mag;
    const vec3_t R_hat = make_vec3(R_vec.x * inv_R, R_vec.y * inv_R, R_vec.z * inv_R);
    const real_t inv_4_pi_R = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI) * R_mag);

    // --- Kernels for k1 ---
    const real_t k1R = k1 * R_mag;
    real_t s1, c1;
    SIE_MOM_SINCOS(k1R, &s1, &c1);
    
    // TL kernel for k1
    kernels.tl_k1 = make_complex(c1 * inv_4_pi_R, -s1 * inv_4_pi_R);

    // TK kernel for k1
    const complex__float scalar1 = make_complex(inv_R, k1);
    const complex__float weighted_green1 = complexMul(scalar1, kernels.tl_k1);
    kernels.tk_k1 = make_cvec3(complexScale(weighted_green1, R_hat.x),
                                        complexScale(weighted_green1, R_hat.y),
                                        complexScale(weighted_green1, R_hat.z));

    // --- Kernels for k2 ---
    const real_t k2R = k2 * R_mag;
    real_t s2, c2;
    SIE_MOM_SINCOS(k2R, &s2, &c2);

    // TL kernel for k2
    kernels.tl_k2 = make_complex(c2 * inv_4_pi_R, -s2 * inv_4_pi_R);

    // TK kernel for k2
    const complex__float scalar2 = make_complex(inv_R, k2);
    const complex__float weighted_green2 = complexMul(scalar2, kernels.tl_k2);
    kernels.tk_k2 = make_cvec3(complexScale(weighted_green2, R_hat.x),
                                        complexScale(weighted_green2, R_hat.y),
                                        complexScale(weighted_green2, R_hat.z));

    return kernels;
}

// This function will calculate at once the 4 integrand Regularkernels: TLRegular(k1), TLRegular(k2), TKRegular(k1), TKRegular(k2)
// i.e., the kernels for the TL and TK operators with two different wavenumbers.
// we will resuse computations where possible, like R_vec and R_mag, inv_R, R_hat, sincosf, etc.
// TLRegular and TKRegular use the regularized Green's function and they are defined in 
// greenFunction3DRegular and TKDpqRegular
__host__ __device__ inline IntegrandKernels integrandKernelsRegular(const vec3_t &r1, const vec3_t &r2, const real_t k1, const real_t k2)
{
    IntegrandKernels kernels;
    const vec3_t R_vec = make_vec3(r1.x - r2.x, r1.y - r2.y, r1.z - r2.z);
    const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

    if (R_mag < static_cast<real_t>(1e-9)) // Use a slightly larger epsilon for regular kernels
    {
        // --- Handle R -> 0 case ---
        // TL regular kernel limit: -jk / (4*pi)
        kernels.tl_k1 = make_complex(real_t(0), -k1 / (real_t(4) * static_cast<real_t>(M_PI)));
        kernels.tl_k2 = make_complex(real_t(0), -k2 / (real_t(4) * static_cast<real_t>(M_PI)));

        // TK regular kernel limit: K_reg -> -j*k^3/3, D_pq -> (1/(4pi)) * K_reg * R_vec -> 0
        const complex__float zero = make_complex(real_t(0), real_t(0));
        kernels.tk_k1 = make_cvec3(zero, zero, zero);
        kernels.tk_k2 = make_cvec3(zero, zero, zero);
        return kernels;
    }

    // --- General Case: R > 0 ---
    const real_t inv_4_pi = real_t(1) / (real_t(4) * static_cast<real_t>(M_PI));
    const real_t inv_R = real_t(1) / R_mag;
    const real_t inv_R2 = inv_R * inv_R;
    const real_t inv_R3 = inv_R2 * inv_R;
    const real_t inv_4_pi_R = inv_4_pi * inv_R;

    // --- Kernels for k1 ---
    const real_t k1R = k1 * R_mag;
    real_t s1, c1;
    SIE_MOM_SINCOS(k1R, &s1, &c1);
    
    // TL regular kernel for k1: (exp(-jk1R) - 1) / (4*pi*R)
    kernels.tl_k1 = make_complex((c1 - real_t(1)) * inv_4_pi_R, -s1 * inv_4_pi_R);

    // TK regular kernel for k1
    const complex__float term1_k1 = make_complex(inv_R3, k1 * inv_R2);
    const complex__float exp_term1 = make_complex(c1, -s1); // exp(-jk1R)
    const real_t term3_real_k1 = -(inv_R3 + k1 * k1 * real_t(0.5) * inv_R);
    complex__float K_reg_val1 = complexMul(term1_k1, exp_term1);
    K_reg_val1.x += term3_real_k1;
    const complex__float scalar_tk1 = complexScale(K_reg_val1, inv_4_pi);
    kernels.tk_k1 = make_cvec3(complexScale(scalar_tk1, R_vec.x),
                                        complexScale(scalar_tk1, R_vec.y),
                                        complexScale(scalar_tk1, R_vec.z));

    // --- Kernels for k2 ---
    const real_t k2R = k2 * R_mag;
    real_t s2, c2;
    SIE_MOM_SINCOS(k2R, &s2, &c2);

    // TL regular kernel for k2: (exp(-jk2R) - 1) / (4*pi*R)
    kernels.tl_k2 = make_complex((c2 - real_t(1)) * inv_4_pi_R, -s2 * inv_4_pi_R);

    // TK regular kernel for k2
    const complex__float term1_k2 = make_complex(inv_R3, k2 * inv_R2);
    const complex__float exp_term2 = make_complex(c2, -s2); // exp(-jk2R)
    const real_t term3_real_k2 = -(inv_R3 + k2 * k2 * real_t(0.5) * inv_R);
    complex__float K_reg_val2 = complexMul(term1_k2, exp_term2);
    K_reg_val2.x += term3_real_k2;
    const complex__float scalar_tk2 = complexScale(K_reg_val2, inv_4_pi);
    kernels.tk_k2 = make_cvec3(complexScale(scalar_tk2, R_vec.x),
                                        complexScale(scalar_tk2, R_vec.y),
                                        complexScale(scalar_tk2, R_vec.z));

    return kernels;
}

struct TLDpq // TL double quadrature kernel D_pq.
{
    __host__ __device__ complex__float operator()(const vec3_t& rmp,
                                                  const vec3_t& rnq,
                                                  real_t k) const
    {
        return greenFunction3D(rmp, rnq, k);
    }
};

struct TLDpqRegular // TL double quadrature kernel D_pq with regularized Green's function.
{
    __host__ __device__ complex__float operator()(const vec3_t& rmp,
                                                  const vec3_t& rnq,
                                                  real_t k) const
    {
        return greenFunction3DRegular(rmp, rnq, k);
    }
};


struct TKDpq // TK double quadrature kernel D_pq.
{
    __host__ __device__ cvec3_t operator()(const vec3_t& rmp,
                                                  const vec3_t& rnq,
                                                  real_t k) const
    {
        return gradSourceGreenFunction(rmp, rnq, k);
    }
};

/**
 * @brief Functor for the regular part of the TK operator kernel.
 *
 * This functor implements the regularized kernel for the double quadrature
 * of the TK operator, as defined in :eq:`eq:TK_D_pq_regular`.
 *
 * The formula is:
 * \f[
 * D_{pq}^{(\mathrm{reg})} = \frac{1}{4\pi} K_{\mathrm{reg}}(R_{pq}) \mathbf{R}_{pq}
 * \f]
 *
 * where \f$ K_{\mathrm{reg}} \f$ is the regularized kernel function from
 * :eq:`eq:TK_K_reg_func_definition`. This implementation calculates all terms
 * internally to avoid recomputing the distance \f$ R \f$.
 */
struct TKDpqRegular
{
    __host__ __device__ cvec3_t operator()(const vec3_t& rmp,
                                                  const vec3_t& rnq,
                                                  real_t k) const
    {
        const vec3_t R_vec = make_vec3(rmp.x - rnq.x, rmp.y - rnq.y, rmp.z - rnq.z);
        const real_t R_mag = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

        complex__float K_reg_val;
        if (R_mag < static_cast<real_t>(1e-6)) {
            // Use limiting value for small R: K_reg(R) -> -j * k^3 / 3
            K_reg_val = make_complex(real_t(0), -k * k * k / real_t(3));
        } else {
            // K_reg(R) = (jk/R^2 + 1/R^3) * exp(-jkR) - (1/R^3 + k^2/(2R))
            const real_t R2 = R_mag * R_mag;
            const real_t R3 = R2 * R_mag;

            const complex__float term1 = make_complex(real_t(1) / R3, k / R2);

            real_t s, c;
            SIE_MOM_SINCOS(-k * R_mag, &s, &c);
            const complex__float exp_term = make_complex(c, s);

            const real_t term3_real = -(real_t(1) / R3 + k * k / (real_t(2) * R_mag));

            K_reg_val = complexMul(term1, exp_term);
            K_reg_val.x += term3_real;
        }

        // D_pq = (1 / (4*pi)) * K_reg(R) * R_vec
        const complex__float scalar = complexScale(K_reg_val, real_t(1) / (real_t(4) * static_cast<real_t>(M_PI)));
        return make_cvec3(complexScale(scalar, R_vec.x),
                                   complexScale(scalar, R_vec.y),
                                   complexScale(scalar, R_vec.z));
    }
};


#endif