#ifndef RWG_BASIS_UTILS_CUH
#define RWG_BASIS_UTILS_CUH

#include "math_utils.cuh"


/**
 * @brief Computes the position vector component (rho) of the RWG basis function.
 *
 * This function calculates the classic RWG \f$ \rho \f$ vector, which is defined as
 * \f$ \rho(\mathbf{r}) = \mathbf{r} - \mathbf{r}_v \f$, where \f$ \mathbf{r}_v \f$ is the free vertex
 * of the triangle.
 *
 * The full RWG basis function is typically expressed as \f$ \mathbf{f}(\mathbf{r}) = \frac{l}{2A} \rho(\mathbf{r}) \f$.
 * This function only returns the \f$ \rho(\mathbf{r}) \f$ part. The scaling by edge length (\f$l\f$) and
 * twice the triangle area (\f$2A\f$) is handled separately.
 *
 * The choice of which free vertex to pass as `r_free` determines whether this corresponds
 * to the basis function on the positive or negative triangle relative to the edge's orientation.
 *
 * @param r The observation point \f$ \mathbf{r} \f$.
 * @param r_free The position of the free vertex \f$ \mathbf{r}_v \f$.
 * @return The position vector \f$ \rho(\mathbf{r}) \f$.
 */
__host__ __device__ inline vec3_t rhoRwg(const vec3_t &r, const vec3_t &r_free)
{
    return make_vec3(r.x - r_free.x, r.y - r_free.y, r.z - r_free.z);
}

__host__ __device__ inline real_t rwgDoubleQuadTL(const vec3_t &rmp,
                                                 const vec3_t &rm_free,
                                                 const vec3_t &rnq,
                                                 const vec3_t &rn_free,
                                                 const real_t k)
{
    const vec3_t rho_mj = rhoRwg(rmp, rm_free);
    const vec3_t rho_ni = rhoRwg(rnq, rn_free);
    const real_t inv_k2 = real_t(1) / (k * k);

    return real_t(0.25) * dot(rho_mj, rho_ni) - inv_k2;
}

struct TLModifier
{
    __host__ __device__ real_t operator()(const vec3_t& rmp,
                                         const vec3_t& rm_free,
                                         const vec3_t& rnq,
                                         const vec3_t& rn_free,
                                         real_t k) const
    {
        return rwgDoubleQuadTL(rmp, rm_free, rnq, rn_free, k);
    }
};

struct TKModifier
{
    __host__ __device__ vec3_t operator()(const vec3_t& rmp,
                                          const vec3_t& rm_free,
                                          const vec3_t& rnq,
                                          const vec3_t& rn_free,
                                          real_t /*k*/) const
    {
        const vec3_t rho_mj = rhoRwg(rmp, rm_free);
        const vec3_t rho_ni = rhoRwg(rnq, rn_free);
        const vec3_t cross_rho = cross(rho_mj, rho_ni);
        return real_t(0.25) * cross_rho;
    }
};

#endif // RWG_BASIS_UTILS_CUH