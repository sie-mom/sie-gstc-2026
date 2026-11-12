#ifndef PLATFORM_CUH
#define PLATFORM_CUH

// This header centralizes platform- and compiler-specific definitions
// to ensure CUDA source files can be processed by both NVCC and a
// standard C++ compiler (e.g., for host-side testing).
#include <cstdio>

#ifdef _MSC_VER
    // Add any Microsoft Visual C++ specific includes or definitions here.
#endif

#if defined(USE_DOUBLE_PRECISION)
    using real_t = double;
#else
    using real_t = float;
#endif

#ifdef __CUDACC__
    // When compiling with NVCC, use the CUDA-provided headers and types.
    #include <cuda_runtime.h>
    #include <cuComplex.h>

    #if defined(USE_DOUBLE_PRECISION)
        using complex_t = cuDoubleComplex;
        using vec3_t = double3;
    #else
        using complex_t = cuComplex;
        using vec3_t = float3;
    #endif

    using complex__float = complex_t;

    __host__ __device__ inline complex__float make_complex(real_t r, real_t i)
    {
        #if defined(USE_DOUBLE_PRECISION)
            return make_cuDoubleComplex(r, i);
        #else
            return make_cuComplex(r, i);
        #endif
    }

    __host__ __device__ inline vec3_t make_vec3(real_t x, real_t y, real_t z)
    {
        #if defined(USE_DOUBLE_PRECISION)
            return make_double3(x, y, z);
        #else
            return make_float3(x, y, z);
        #endif
    }

#else
    // When compiling with a standard host compiler, provide CUDA stubs.
    #ifndef _USE_MATH_DEFINES
        #define _USE_MATH_DEFINES
    #endif
    #include <cmath>
    using std::isfinite;

    // CUDA function specifiers become no-ops on the host.
    #define __host__
    #define __device__
    #define __global__
    #define __shared__

    struct float2 { real_t x, y; };
    struct float3 { float x, y, z; };
    struct double3 { double x, y, z; };
    #if defined(USE_DOUBLE_PRECISION)
        using vec3_t = double3;
    #else
        using vec3_t = float3;
    #endif

    inline float2 make_float2(real_t x, real_t y) { return {x, y}; }
    inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
    inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }
    inline vec3_t make_vec3(real_t x, real_t y, real_t z) {
        #if defined(USE_DOUBLE_PRECISION)
            return make_double3(x, y, z);
        #else
            return make_float3(x, y, z);
        #endif
    }

    struct complex_t { real_t x, y; };
    using complex__float = complex_t;

    inline complex__float make_complex(real_t r, real_t i)
    {
        return {r, i};
    }

    // Some host-side code still calls make_cuComplex/ make_float3, etc.
    inline complex__float make_cuComplex(real_t r, real_t i)
    {
        return {r, i};
    }

    struct int2 { int x, y; };
    struct int3 { int x, y, z; };

    inline int2 make_int2(int x, int y) { return {x, y}; }
    inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }

#endif

struct cvec3_t {
    complex__float x;
    complex__float y;
    complex__float z;
};

__host__ __device__ inline cvec3_t make_cvec3(const complex__float& x,
                                              const complex__float& y,
                                              const complex__float& z)
{
    return {x, y, z};
}

#endif // PLATFORM_CUH
