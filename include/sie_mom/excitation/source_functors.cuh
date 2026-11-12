#pragma once

#include "sie_mom/cuda/math_utils.cuh"

namespace sie_mom {
namespace cuda {

/**
 * @brief Functor to compute the fields of a uniform plane wave.
 *
 * This struct calculates the electric and magnetic fields of a plane wave
 * with a given polarization, propagation direction, and wavenumber.
 */
struct PlaneWave {
    vec3_t E0;           ///< Electric field amplitude and polarization vector (real vector).
    vec3_t propagation_k;  ///< Propagation direction vector (should be a unit vector).
    real_t k;             ///< Wavenumber of the medium.
    real_t eta;           ///< Intrinsic impedance of the medium.

    /**
     * @brief Calculates the electric and magnetic fields at a given position.
     *
     * @param r The position vector at which to evaluate the fields.
     * @param E_inc (Output) The calculated incident electric field (complex vector).
     * @param H_inc (Output) The calculated incident magnetic field (complex vector).
     */
    __host__ __device__ void operator()(const vec3_t& r, cvec3_t& E_inc, cvec3_t& H_inc) const {
        // Phase of the plane wave at position r: exp(-j * k_vec . r)
        // Note: k_vec = k * propagation_k
        const real_t k_dot_r = dot(propagation_k, r) * k;
        real_t sin_term, cos_term;
        SIE_MOM_SINCOS(-k_dot_r, &sin_term, &cos_term);
        const complex__float phase = make_complex(cos_term, sin_term);

        // Incident Electric Field: E_inc = E0 * phase
        E_inc = E0 * phase;

        // Incident Magnetic Field: H_inc = (1/eta) * (k_hat x E_inc)
        const cvec3_t k_cross_E = cross(propagation_k, E_inc);
        H_inc = k_cross_E * (real_t(1) / eta);
    }
};


/**
 * @brief Functor to compute the fields of an internal electric or magnetic dipole.
 *
 * This struct calculates the Cartesian electric and magnetic fields of a dipole
 * located at r_0, with a specified moment p_vec.
 */
struct DipoleSource {
    vec3_t p_vec;        ///< Effective dipole moment vector (amplitude and polarization).
    vec3_t r_0;          ///< Position of the dipole in Cartesian coords.
    real_t k;             ///< Wavenumber of the medium.
    real_t eta;           ///< Intrinsic impedance of the medium.
    bool is_magnetic;    ///< True for magnetic dipole, False for electric dipole.

    /**
     * @brief Calculates the electric and magnetic fields at a given position.
     *
     * @param r The observation position vector.
     * @param E_inc (Output) The calculated incident electric field (complex vector).
     * @param H_inc (Output) The calculated incident magnetic field (complex vector).
     */
    __host__ __device__ void operator()(const vec3_t& r, cvec3_t& E_inc, cvec3_t& H_inc) const {
        // Displacement vector
        vec3_t R_vec = make_vec3(r.x - r_0.x, r.y - r_0.y, r.z - r_0.z);
        real_t R = sqrt(R_vec.x * R_vec.x + R_vec.y * R_vec.y + R_vec.z * R_vec.z);

        // Guard against origin singularity
        if (R < static_cast<real_t>(1e-6)) {
            E_inc = make_cvec3(make_complex(real_t(0), real_t(0)), make_complex(real_t(0), real_t(0)), make_complex(real_t(0), real_t(0)));
            H_inc = make_cvec3(make_complex(real_t(0), real_t(0)), make_complex(real_t(0), real_t(0)), make_complex(real_t(0), real_t(0)));
            return;
        }

        vec3_t R_hat = make_vec3(R_vec.x / R, R_vec.y / R, R_vec.z / R);
        real_t R_dot_p = R_hat.x * p_vec.x + R_hat.y * p_vec.y + R_hat.z * p_vec.z;
        
        vec3_t R_cross_p = make_vec3(
            R_hat.y * p_vec.z - R_hat.z * p_vec.y,
            R_hat.z * p_vec.x - R_hat.x * p_vec.z,
            R_hat.x * p_vec.y - R_hat.y * p_vec.x
        );

        vec3_t triple = make_vec3(
            R_cross_p.y * R_hat.z - R_cross_p.z * R_hat.y,
            R_cross_p.z * R_hat.x - R_cross_p.x * R_hat.z,
            R_cross_p.x * R_hat.y - R_cross_p.y * R_hat.x
        );

        // Phase term = exp(-j * k * R)
        real_t sin_kR, cos_kR;
        SIE_MOM_SINCOS(-k * R, &sin_kR, &cos_kR);
        complex__float phase_term = make_complex(cos_kR, sin_kR);

        // Electric Field (Cartesian)
        real_t term1_real = (k * k) / R;
        real_t term2_real = real_t(1) / (R * R * R);
        real_t term2_imag = k / (R * R);

        vec3_t term2_vec = make_vec3(
            real_t(3) * R_hat.x * R_dot_p - p_vec.x,
            real_t(3) * R_hat.y * R_dot_p - p_vec.y,
            real_t(3) * R_hat.z * R_dot_p - p_vec.z
        );

        cvec3_t E_cart;
        E_cart.x = complexMul(make_complex(term1_real * triple.x + term2_vec.x * term2_real, term2_vec.x * term2_imag), phase_term);
        E_cart.y = complexMul(make_complex(term1_real * triple.y + term2_vec.y * term2_real, term2_vec.y * term2_imag), phase_term);
        E_cart.z = complexMul(make_complex(term1_real * triple.z + term2_vec.z * term2_real, term2_vec.z * term2_imag), phase_term);

        // Magnetic Field (Cartesian)
        complex__float H_total_coeff = complexMul(make_complex(-(k * k) / (eta * R), k / (eta * R * R)), phase_term);
        
        cvec3_t H_cart;
        H_cart.x = make_complex(H_total_coeff.x * -R_cross_p.x, H_total_coeff.y * -R_cross_p.x);
        H_cart.y = make_complex(H_total_coeff.x * -R_cross_p.y, H_total_coeff.y * -R_cross_p.y);
        H_cart.z = make_complex(H_total_coeff.x * -R_cross_p.z, H_total_coeff.y * -R_cross_p.z);

        if (is_magnetic) {
            // Apply electromagnetic duality
            E_inc = make_cvec3(make_complex(H_cart.x.x * -eta, H_cart.x.y * -eta), make_complex(H_cart.y.x * -eta, H_cart.y.y * -eta), make_complex(H_cart.z.x * -eta, H_cart.z.y * -eta));
            H_inc = make_cvec3(make_complex(E_cart.x.x / eta, E_cart.x.y / eta), make_complex(E_cart.y.x / eta, E_cart.y.y / eta), make_complex(E_cart.z.x / eta, E_cart.z.y / eta));
        } else {
            E_inc = E_cart;
            H_inc = H_cart;
        }
    }
};

}  // namespace cuda
}  // namespace sie_mom
