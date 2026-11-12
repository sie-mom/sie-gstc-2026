#pragma once

#include "sie_mom/cuda/platform.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace sie_mom::cuda {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

inline void check_last_kernel(const char* operation) {
    check_cuda(cudaGetLastError(), operation);
    check_cuda(cudaDeviceSynchronize(), operation);
}

inline void check_cusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuSolver status "
            + std::to_string(static_cast<int>(status))
        );
    }
}

inline void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status "
            + std::to_string(static_cast<int>(status))
        );
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (data_ != nullptr) {
            throw std::logic_error("device buffer is already allocated");
        }
        if (count == 0) {
            return;
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc"
        );
        count_ = count;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t count() const { return count_; }

    void copy_from_host(const T* host_data, std::size_t element_count) {
        if (element_count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_from_host exceeds buffer capacity");
        }
        check_cuda(
            cudaMemcpy(data_, host_data, element_count * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpyHostToDevice"
        );
    }

    void copy_to_host(T* host_data, std::size_t element_count) const {
        if (element_count > count_) {
            throw std::out_of_range("DeviceBuffer::copy_to_host exceeds buffer capacity");
        }
        check_cuda(
            cudaMemcpy(host_data, data_, element_count * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpyDeviceToHost"
        );
    }

    void fill_zero() {
        if (data_ != nullptr && count_ > 0) {
            check_cuda(
                cudaMemset(data_, 0, count_ * sizeof(T)),
                "cudaMemset"
            );
        }
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class CusolverHandle {
public:
    CusolverHandle() {
        check_cusolver(cusolverDnCreate(&handle_), "cusolverDnCreate");
    }

    ~CusolverHandle() {
        if (handle_ != nullptr) {
            cusolverDnDestroy(handle_);
        }
    }

    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;

    cusolverDnHandle_t get() const { return handle_; }

private:
    cusolverDnHandle_t handle_ = nullptr;
};

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

} // namespace sie_mom::cuda
