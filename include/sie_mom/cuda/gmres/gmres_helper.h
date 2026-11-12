#ifndef GMRES_HELPER_H
#define GMRES_HELPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Helper wrapper for memory allocation compatible with what code expects
// but implementation uses cudaMalloc
inline cublasStatus_t cublas_alloc(int n, int elemSize, void** devPtr) {
    cudaError_t err = cudaMalloc(devPtr, n * elemSize);
    return (err == cudaSuccess) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_ALLOC_FAILED;
}

inline cublasStatus_t cublas_free(void* devPtr) {
    cudaError_t err = cudaFree(devPtr);
    return (err == cudaSuccess) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}

#endif