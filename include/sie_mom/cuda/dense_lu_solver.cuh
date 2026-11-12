#pragma once

#include "sie_mom/cuda/device_buffer.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <string>

namespace sie_mom {
namespace cuda {

/**
 * @brief High-level RAII solver for dense linear systems using cuSolver LU factorization.
 * Supports single and double precision complex types.
 * Solves A * X = B (CUBLAS_OP_N), A^H * X = B (CUBLAS_OP_C), or A^T * X = B (CUBLAS_OP_T).
 */
class DenseLuSolver {
public:
    DenseLuSolver() = default;

    explicit DenseLuSolver(int dimension) {
        allocate(dimension);
    }

    void allocate(int dimension) {
        if (dimension <= 0) {
            throw std::invalid_argument("DenseLuSolver dimension must be positive");
        }
        dimension_ = dimension;
        pivots_.allocate(static_cast<std::size_t>(dimension));
        information_.allocate(1);
    }

    int dimension() const { return dimension_; }
    const int* pivots() const { return pivots_.data(); }
    int* pivots() { return pivots_.data(); }
    const CusolverHandle& handle() const { return solver_; }

    /**
     * @brief Computes in-place LU factorization of square matrix A of size dimension x dimension.
     */
    void factorize(complex__float* matrix, int dimension = 0) {
        if (dimension > 0 && dimension != dimension_) {
            allocate(dimension);
        }
        if (dimension_ <= 0) {
            throw std::invalid_argument("DenseLuSolver dimension must be initialized before factorize");
        }

        int workspace_size = 0;
#ifdef USE_DOUBLE_PRECISION
        check_cusolver(
            cusolverDnZgetrf_bufferSize(
                solver_.get(), dimension_, dimension_,
                reinterpret_cast<cuDoubleComplex*>(matrix), dimension_,
                &workspace_size
            ),
            "cusolverDnZgetrf_bufferSize"
        );
#else
        check_cusolver(
            cusolverDnCgetrf_bufferSize(
                solver_.get(), dimension_, dimension_,
                reinterpret_cast<cuComplex*>(matrix), dimension_,
                &workspace_size
            ),
            "cusolverDnCgetrf_bufferSize"
        );
#endif

        if (workspace_.size() < static_cast<std::size_t>(workspace_size)) {
            workspace_.allocate(static_cast<std::size_t>(workspace_size));
        }

#ifdef USE_DOUBLE_PRECISION
        check_cusolver(
            cusolverDnZgetrf(
                solver_.get(), dimension_, dimension_,
                reinterpret_cast<cuDoubleComplex*>(matrix), dimension_,
                reinterpret_cast<cuDoubleComplex*>(workspace_.data()),
                pivots_.data(), information_.data()
            ),
            "cusolverDnZgetrf"
        );
#else
        check_cusolver(
            cusolverDnCgetrf(
                solver_.get(), dimension_, dimension_,
                reinterpret_cast<cuComplex*>(matrix), dimension_,
                reinterpret_cast<cuComplex*>(workspace_.data()),
                pivots_.data(), information_.data()
            ),
            "cusolverDnCgetrf"
        );
#endif

        int status = 0;
        check_cuda(
            cudaMemcpy(&status, information_.data(), sizeof(int), cudaMemcpyDeviceToHost),
            "copy LU factorization status"
        );
        if (status < 0) {
            throw std::runtime_error(
                "LU factorization received an invalid argument at position " + std::to_string(-status)
            );
        }
        if (status > 0) {
            throw std::runtime_error(
                "dense matrix is singular at U diagonal " + std::to_string(status)
            );
        }
    }

    /**
     * @brief Solves op(A) * X = B for multiple right-hand sides in place.
     * @param op CUBLAS_OP_N for A*X = B, CUBLAS_OP_C for A^H*X = B, CUBLAS_OP_T for A^T*X = B
     * @param nrhs Number of right-hand side columns in B
     * @param factored_matrix The LU-factored matrix produced by factorize()
     * @param rhs Pointer to device memory with dimensions ldb x nrhs, overwritten with solution X
     * @param ldb Leading dimension of B (defaults to dimension_)
     */
    void solve(
        cublasOperation_t op,
        int nrhs,
        const complex__float* factored_matrix,
        complex__float* rhs,
        int ldb = 0
    ) {
        if (ldb == 0) {
            ldb = dimension_;
        }
#ifdef USE_DOUBLE_PRECISION
        check_cusolver(
            cusolverDnZgetrs(
                solver_.get(), op, dimension_, nrhs,
                reinterpret_cast<const cuDoubleComplex*>(factored_matrix), dimension_,
                pivots_.data(),
                reinterpret_cast<cuDoubleComplex*>(rhs), ldb,
                information_.data()
            ),
            "cusolverDnZgetrs"
        );
#else
        check_cusolver(
            cusolverDnCgetrs(
                solver_.get(), op, dimension_, nrhs,
                reinterpret_cast<const cuComplex*>(factored_matrix), dimension_,
                pivots_.data(),
                reinterpret_cast<cuComplex*>(rhs), ldb,
                information_.data()
            ),
            "cusolverDnCgetrs"
        );
#endif

        int status = 0;
        check_cuda(
            cudaMemcpy(&status, information_.data(), sizeof(int), cudaMemcpyDeviceToHost),
            "copy dense LU solve status"
        );
        if (status != 0) {
            throw std::runtime_error(
                "dense LU solve failed with status " + std::to_string(status)
            );
        }
    }

    /**
     * @brief Convenience overload for single right-hand side forward solve A * x = b.
     */
    void solve(
        const complex__float* factored_matrix,
        complex__float* rhs
    ) {
        solve(CUBLAS_OP_N, 1, factored_matrix, rhs, dimension_);
    }

private:
    CusolverHandle solver_;
    int dimension_ = 0;
    DeviceBuffer<complex__float> workspace_;
    DeviceBuffer<int> pivots_;
    DeviceBuffer<int> information_;
};

/**
 * @brief Convenience function to factorize and solve a dense linear system A * x = b in place.
 */
inline void solve_dense_lu(
    int dimension,
    complex__float* matrix,
    complex__float* rhs
) {
    DenseLuSolver solver(dimension);
    solver.factorize(matrix);
    solver.solve(matrix, rhs);
}

} // namespace cuda
} // namespace sie_mom

