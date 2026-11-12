#ifndef TESTING_UTILS_CUH
#define TESTING_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

// Macro to wrap CUDA calls for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#endif // TESTING_UTILS_CUH
