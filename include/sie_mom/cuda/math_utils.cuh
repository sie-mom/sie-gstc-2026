#ifndef MATH_UTILS_CUH
#define MATH_UTILS_CUH

#include "platform.cuh" // For __host__, __device__ portability and vector types
#include <cmath>

__host__ __device__ inline complex__float complexAdd(const complex__float& a, const complex__float& b)
{
    return make_complex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline complex__float complexScale(const complex__float& z, real_t scalar)
{
    return make_complex(z.x * scalar, z.y * scalar);
}

__host__ __device__ inline complex__float complexScale(real_t scalar, const complex__float& z)
{
    return complexScale(z, scalar);
}

__host__ __device__ inline void complexAddInPlace(complex__float& a, const complex__float& b)
{
    a = make_complex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline complex__float complexSub(const complex__float& a, const complex__float& b)
{
    return make_complex(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline complex__float complexMul(const complex__float& a, const complex__float& b)
{
    return make_complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__host__ __device__ inline complex__float complexDiv(const complex__float& a, const complex__float& b)
{
    const real_t denom = b.x * b.x + b.y * b.y;
    // Avoid division by zero
    if (denom < static_cast<real_t>(1e-20)) return make_complex(real_t(0), real_t(0));
    const real_t inv_denom = real_t(1) / denom;
    return make_complex((a.x * b.x + a.y * b.y) * inv_denom, (a.y * b.x - a.x * b.y) * inv_denom);
}

__host__ __device__ inline complex__float complexInverse(const complex__float& z)
{
    const real_t denom = z.x * z.x + z.y * z.y;
    if (denom < static_cast<real_t>(1e-20)) return make_complex(real_t(0), real_t(0));
    const real_t inv_denom = real_t(1) / denom;
    return make_complex(z.x * inv_denom, -z.y * inv_denom);
}

__host__ __device__ inline complex__float complexNeg(const complex__float& z)
{
    return make_complex(-z.x, -z.y);
}

__host__ __device__ inline complex__float complexConj(const complex__float& z)
{
    return make_complex(z.x, -z.y);
}

__host__ __device__ inline real_t complexAbs(const complex__float& z)
{
    return sqrt(z.x * z.x + z.y * z.y);
}


// --- Operator Overloads for complex__float ---

__host__ __device__ inline complex__float operator+(const complex__float& a, const complex__float& b)
{
    return complexAdd(a, b);
}

__host__ __device__ inline complex__float operator-(const complex__float& a, const complex__float& b)
{
    return complexSub(a, b);
}

__host__ __device__ inline complex__float operator*(const complex__float& a, const complex__float& b)
{
    return complexMul(a, b);
}

__host__ __device__ inline complex__float operator*(real_t scalar, const complex__float& z)
{
    return complexScale(scalar, z);
}

__host__ __device__ inline complex__float operator*(const complex__float& z, real_t scalar)
{
    return complexScale(z, scalar);
}

// dot and cross products

__host__ __device__ inline real_t dot(const vec3_t& a, const vec3_t& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline complex__float dot(const vec3_t& a, const cvec3_t& b) {
    return make_complex(a.x * b.x.x + a.y * b.y.x + a.z * b.z.x,
                        a.x * b.x.y + a.y * b.y.y + a.z * b.z.y);
}

__host__ __device__ inline complex__float dot(const cvec3_t& a, const vec3_t& b) {
    return dot(b, a);
}

/**
 * @brief Complex Euclidean Hermitian inner product a^H * b = conj(a.x)*b.x + conj(a.y)*b.y + conj(a.z)*b.z
 */
__host__ __device__ inline complex__float complexDotH(const cvec3_t& a, const cvec3_t& b) {
    return complexConj(a.x) * b.x + complexConj(a.y) * b.y + complexConj(a.z) * b.z;
}

__host__ __device__ inline vec3_t cross(const vec3_t& a, const vec3_t& b) {
    return make_vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline cvec3_t cross(const vec3_t& a, const cvec3_t& b)
{
    const complex__float x = complexSub(complexMul(make_complex(a.y, real_t(0)), b.z),
                                        complexMul(make_complex(a.z, real_t(0)), b.y));
    const complex__float y = complexSub(complexMul(make_complex(a.z, real_t(0)), b.x),
                                        complexMul(make_complex(a.x, real_t(0)), b.z));
    const complex__float z = complexSub(complexMul(make_complex(a.x, real_t(0)), b.y),
                                        complexMul(make_complex(a.y, real_t(0)), b.x));
    return make_cvec3(x, y, z);
}

// length and normalize operations
__host__ __device__ inline real_t length(const vec3_t& v) {
    return sqrt(dot(v, v));
}

__host__ __device__ inline vec3_t normalize(const vec3_t& v) {
    real_t inv_len = real_t(1) / length(v);
    return make_vec3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}



// + and - operators
__host__ __device__ inline vec3_t operator-(const vec3_t& a, const vec3_t& b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline vec3_t operator-(const vec3_t& a) {
    return make_vec3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline vec3_t operator+(const vec3_t& a, const vec3_t& b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vec3_t& operator+=(vec3_t& a, const vec3_t& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__host__ __device__ inline vec3_t& operator-=(vec3_t& a, const vec3_t& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__host__ __device__ inline cvec3_t operator+(const cvec3_t& a, const cvec3_t& b)
{
    return make_cvec3(complexAdd(a.x, b.x),
                      complexAdd(a.y, b.y),
                      complexAdd(a.z, b.z));
}

__host__ __device__ inline cvec3_t operator-(const cvec3_t& a, const cvec3_t& b)
{
    return make_cvec3(complexSub(a.x, b.x),
                      complexSub(a.y, b.y),
                      complexSub(a.z, b.z));
}

__host__ __device__ inline cvec3_t& operator+=(cvec3_t& a, const cvec3_t& b)
{
    a.x = complexAdd(a.x, b.x);
    a.y = complexAdd(a.y, b.y);
    a.z = complexAdd(a.z, b.z);
    return a;
}

// * and / operators
__host__ __device__ inline vec3_t operator*(real_t s, const vec3_t& a) {
    return make_vec3(s * a.x, s * a.y, s * a.z);
}

__host__ __device__ inline vec3_t operator*(const vec3_t& a, real_t s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline cvec3_t operator*(const vec3_t& v, const complex__float& c)
{
    return make_cvec3(complexScale(c, v.x),
                      complexScale(c, v.y),
                      complexScale(c, v.z));
}

__host__ __device__ inline vec3_t& operator*=(vec3_t& a, real_t s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}

__host__ __device__ inline cvec3_t operator*(const complex__float& c, const vec3_t& v)
{
    return v * c;
}

__host__ __device__ inline cvec3_t operator*(const cvec3_t& v, real_t s)
{
    return make_cvec3(complexScale(v.x, s),
                      complexScale(v.y, s),
                      complexScale(v.z, s));
}

__host__ __device__ inline cvec3_t operator*(real_t s, const cvec3_t& v)
{
    return v * s;
}

__host__ __device__ inline cvec3_t operator*(const complex__float& c, const cvec3_t& v)
{
    return make_cvec3(complexMul(c, v.x),
                      complexMul(c, v.y),
                      complexMul(c, v.z));
}

__host__ __device__ inline cvec3_t operator*(const cvec3_t& v, const complex__float& c)
{
    return c * v;
}

__host__ __device__ inline vec3_t operator/(const vec3_t& a, real_t s) {
    return make_vec3(a.x / s, a.y / s, a.z / s);
}

__device__ inline real_t hypot3f(real_t x, real_t y, real_t z) {
    return hypot(hypot(fabs(x), fabs(y)), fabs(z));
}

#ifndef __CUDACC__
    // Host-side implementations for functions provided by CUDA libraries

    // Provide a host-side implementation for complex multiplication
    static inline float2 complexMul(float2 a, float2 b) {
        return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    }

    // Provide a host-side sincos helper. The system one can cause conflicts.
    // Use a different name to avoid re-declaration errors with system headers.
    static inline void host_sincos(real_t x, real_t* s, real_t* c) {
        *s = sin(x);
        *c = cos(x);
    }

    // When not in CUDA, make the precision-aware helper use the host implementation.
    #define SIE_MOM_SINCOS host_sincos
#else
    #if defined(USE_DOUBLE_PRECISION)
        #define SIE_MOM_SINCOS sincos
    #else
        #define SIE_MOM_SINCOS sincosf
    #endif
#endif

#ifdef __CUDACC__
    // Helper for atomic add (only available in device code compilation)
    __device__ inline void atomicAddComplex(complex__float* addr, const complex__float& val) {
        atomicAdd(&(addr->x), val.x);
        atomicAdd(&(addr->y), val.y);
}
#endif


#endif // MATH_UTILS_CUH
