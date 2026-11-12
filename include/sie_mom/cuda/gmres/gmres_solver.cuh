#ifndef GMRES_SOLVER_CUH
#define GMRES_SOLVER_CUH

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/cuda/gmres/gmres_helper.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

// --- Helper Functions (Wrapper around cuBLAS) ---

inline void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " failed with error " + std::to_string(status));
    }
}

// Complex Dot Product (conjugate): result = x^H * y
inline complex__float cublas_dotc(cublasHandle_t handle, int n, const complex__float* x, const complex__float* y) {
    #ifdef USE_DOUBLE_PRECISION
    cuDoubleComplex result;
    check_cublas(cublasZdotc(handle, n,
                             reinterpret_cast<const cuDoubleComplex*>(x), 1,
                             reinterpret_cast<const cuDoubleComplex*>(y), 1,
                             &result),
                 "cublasZdotc");
    return make_complex(result.x, result.y);
    #else
    cuComplex result;
    check_cublas(cublasCdotc(handle, n,
                             reinterpret_cast<const cuComplex*>(x), 1,
                             reinterpret_cast<const cuComplex*>(y), 1,
                             &result),
                 "cublasCdotc");
    return make_complex(result.x, result.y);
    #endif
}

// Euclidean Norm: result = ||x||_2
inline real_t cublas_nrm2(cublasHandle_t handle, int n, const complex__float* x) {
    real_t result;
    #ifdef USE_DOUBLE_PRECISION
    check_cublas(cublasDznrm2(handle, n, reinterpret_cast<const cuDoubleComplex*>(x), 1, &result), "cublasDznrm2");
    #else
    check_cublas(cublasScnrm2(handle, n, reinterpret_cast<const cuComplex*>(x), 1, &result), "cublasScnrm2");
    #endif
    return result;
}

// Scale: x = alpha * x
inline void cublas_scal(cublasHandle_t handle, int n, complex__float alpha, complex__float* x) {
    #ifdef USE_DOUBLE_PRECISION
    cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.x, alpha.y);
    check_cublas(cublasZscal(handle, n, &alpha_cu, reinterpret_cast<cuDoubleComplex*>(x), 1), "cublasZscal");
    #else
    cuComplex alpha_cu = make_cuComplex(alpha.x, alpha.y);
    check_cublas(cublasCscal(handle, n, &alpha_cu, reinterpret_cast<cuComplex*>(x), 1), "cublasCscal");
    #endif
}

// Scale (Real scalar): x = alpha * x
inline void cublas_csscal(cublasHandle_t handle, int n, real_t alpha, complex__float* x) {
    #ifdef USE_DOUBLE_PRECISION
    check_cublas(cublasZdscal(handle, n, &alpha, reinterpret_cast<cuDoubleComplex*>(x), 1), "cublasZdscal");
    #else
    check_cublas(cublasCsscal(handle, n, &alpha, reinterpret_cast<cuComplex*>(x), 1), "cublasCsscal");
    #endif
}

// AXPY: y = alpha * x + y
inline void cublas_axpy(cublasHandle_t handle, int n, complex__float alpha, const complex__float* x, complex__float* y) {
    #ifdef USE_DOUBLE_PRECISION
    cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.x, alpha.y);
    check_cublas(cublasZaxpy(handle, n, &alpha_cu,
                             reinterpret_cast<const cuDoubleComplex*>(x), 1,
                             reinterpret_cast<cuDoubleComplex*>(y), 1),
                 "cublasZaxpy");
    #else
    cuComplex alpha_cu = make_cuComplex(alpha.x, alpha.y);
    check_cublas(cublasCaxpy(handle, n, &alpha_cu,
                             reinterpret_cast<const cuComplex*>(x), 1,
                             reinterpret_cast<cuComplex*>(y), 1),
                 "cublasCaxpy");
    #endif
}

// Copy: y = x
inline void cublas_copy(cublasHandle_t handle, int n, const complex__float* x, complex__float* y) {
    #ifdef USE_DOUBLE_PRECISION
    check_cublas(cublasZcopy(handle, n,
                             reinterpret_cast<const cuDoubleComplex*>(x), 1,
                             reinterpret_cast<cuDoubleComplex*>(y), 1),
                 "cublasZcopy");
    #else
    check_cublas(cublasCcopy(handle, n,
                             reinterpret_cast<const cuComplex*>(x), 1,
                             reinterpret_cast<cuComplex*>(y), 1),
                 "cublasCcopy");
    #endif
}

// Zero: x = 0
inline void device_zero(int n, complex__float* x) {
    cudaMemset(x, 0, n * sizeof(complex__float));
}


// --- Givens Rotation Helper ---
// Applies Previous Givens Rotations to column k
inline void apply_previous_givens(
    std::vector<complex__float>& hes_col, // Column k of Hessenberg matrix (size k+1)
    const std::vector<float>& cs,         // Cosine terms of previous rotations
    const std::vector<complex__float>& sn,// Sine terms of previous rotations
    int k
) {
    for (int i = 0; i < k; ++i) {
        complex__float temp_i = hes_col[i];
        complex__float temp_ip1 = hes_col[i + 1];
        
        // [ cs[i]       sn[i] ] [ h[i]   ]
        // [ -conj(sn[i]) cs[i] ] [ h[i+1] ]
        
        // h[i]   = cs[i]*h[i] + sn[i]*h[i+1]
        hes_col[i]     = complexAdd(complexScale(temp_i, cs[i]), complexMul(sn[i], temp_ip1));
        // h[i+1] = -conj(sn[i])*h[i] + cs[i]*h[i+1]
        complex__float sn_conj = make_complex(sn[i].x, -sn[i].y);
        hes_col[i + 1] = complexAdd(complexMul(complexNeg(sn_conj), temp_i), complexScale(temp_ip1, cs[i]));
    }
}

// Generates new Givens Rotation to eliminate h[k+1]
inline void generate_givens(
    complex__float h1,     // h[k]
    complex__float h2,     // h[k+1]
    real_t& cs_out,        // Output Cosine (Real)
    complex__float& sn_out // Output Sine (Complex)
) {
    real_t nrm = std::sqrt(h1.x*h1.x + h1.y*h1.y + h2.x*h2.x + h2.y*h2.y);
    
    if (nrm == 0.0f) {
        cs_out = 1.0f;
        sn_out = make_complex(0.f, 0.f);
    } else {
        // To zero out h2:
        // cs = |h1| / nrm
        // sn = cs * h2 / h1 (if h1 != 0)
        // Standard stable implementation for complex numbers:
        
        // We want updated h1_new = nrm.
        // Rotated vector: [cs  sn] * [h1] = [r]
        //                 [-sn' cs]   [h2]   [0]
        
        real_t h1_mag = std::sqrt(h1.x*h1.x + h1.y*h1.y);
        if (h1_mag == 0.0f) {
            cs_out = 0.0f;
            sn_out = make_complex(1.0f, 0.0f); // Or complex conjugate depending on definition
        } else {
            cs_out = h1_mag / nrm;
            // sn = h2 / h1 * cs (conceptually)
            // sn = (h2 * conj(h1)) / (|h1| * nrm)? No.
            
            // Using LAPACK formula for stable complex givens:
            // clartg( f, g, cs, sn, r )
            
            // Let's implement manually:
            // r = sign(sqrt(|f|^2+|g|^2)) * (f/|f|) * sqrt(...) ?
            // Simplified:
            // cs = |f| / sqrt(|f|^2+|g|^2)
            // sn = (f / |f|) * (conj(g) / sqrt(...))
            
            // Keep it simple matching classic GMRES logic locally:
            // Just satisfy: -conj(sn)*h1 + cs*h2 = 0
            
            // cs * h2 = conj(sn) * h1
            
            // If we choose cs to be real:
            // sn = cs * conj(h2) / conj(h1) ? No.
            
            // Let's follow a specific reference (e.g. Saad):
            // if g == 0, cs=1, sn=0
            // if f == 0, cs=0, sn=conj(g)/|g| ?
            
            // Working Implementation:
            // alpha = h1, beta = h2
            // norm = sqrt(|a|^2 + |b|^2)
            // cs = |a| / norm
            // sn = (a / |a|) * (conj(b) / norm)
            
            complex__float alpha = h1;
            complex__float beta = h2;
            
            cs_out = h1_mag / nrm;
            
            // sn = (alpha / |alpha|) * (conj(beta) / norm)
            complex__float sign_alpha = complexScale(alpha, 1.0f/h1_mag);
            complex__float conj_beta = make_complex(beta.x, -beta.y);
            sn_out = complexMul(sign_alpha, complexScale(conj_beta, 1.0f/nrm));
        }
    }
}

// Upper Triangular Solve: H * y = rhs
inline void upper_triangular_solve(
    int n,
    std::vector<complex__float>& sol,
    const std::vector<complex__float>& hes_mat, // Row-major or custom index?
    int m1, // Leading dimension (rows) of hes_mat
    const std::vector<complex__float>& hes_rhs
) {
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        complex__float sum = make_complex(0.f, 0.f);
        for (int j = i + 1; j < n; ++j) {
            // hes_mat[j*m1 + i] ? No, the template stored it as:
            // *(hes_mat+k*m1+j) where k is column, j is row.
            // Wait, template: *(hes_mat + k*m1 + j)
            // In iteration k (column k), we file rows j=0..k+1.
            // So hes_mat is Column-Major effectively if accessed as [col][row]
            // Or Row-Major m1 x m.
            
            // Let's assume passed hes_mat is stored linearly such that H(i, j)
            // can be retrieved.
            // Template: *(hes_mat+k*m1+j) -> Col k, Row j.
            // Index = k * m1 + j.
            
            complex__float H_ji = hes_mat[j * m1 + i]; // Col j, Row i (Upper triangular)
            sum = complexAdd(sum, complexMul(H_ji, sol[j]));
        }
        
        // sol[i] = (rhs[i] - sum) / H_ii
        complex__float H_ii = hes_mat[i * m1 + i];
        sol[i] = complexDiv(complexSub(hes_rhs[i], sum), H_ii);
    }
}


/**
 * @brief Generalized Minimal Residual Method (GMRES) - Restarted
 *        Solves Ax = b for complex matrices A and vectors x, b.
 * 
 * @param neqns     Number of unknowns (vector size)
 * @param m         Restart frequency (Krylov subspace dimension)
 * @param maxnit    Maximum number of restarts (outer iterations)
 * @param tol       Relative convergence tolerance
 * @param d_solution  Device ptr: Initial guess x0 on entry, solution x on exit (size neqns)
 * @param d_rhs       Device ptr: Right-hand side vector b (size neqns)
 * @param matvec    Function object calling the custom MatVec kernel: y = A * x
 * @param handle    cuBLAS handle
 * @param out_res   Validation output: Final residual norm
 * @param out_iters Validation output: Total number of inner iterations
 * 
 * @return int 0 if converged, 1 if not converged
 */

template <typename MatVecFunc>
inline int gmres_restarted(
    int neqns,
    int m,
    int maxnit,
    real_t tol,
    complex__float* d_solution,
    const complex__float* d_rhs,
    MatVecFunc matvec,
    cublasHandle_t handle,
    real_t* out_res = nullptr,
    int* out_iters = nullptr, // Total iterations
    std::vector<real_t>* out_res_history = nullptr // Residual history tracking
) {
    // 0. Setup
    int info = 0;
    real_t rhs_nrm = cublas_nrm2(handle, neqns, d_rhs);
    
    // Safety check for zero RHS
    if (rhs_nrm < real_t(1e-20)) {
        // Solution is zero.
        device_zero(neqns, d_solution);
        if(out_res) *out_res = real_t(0);
        if(out_iters) *out_iters = 0;
        return 0;
    }
    
    real_t res_tol = tol * rhs_nrm;
    
    // Allocate Device Memory
    complex__float *d_res, *d_tmp, *d_V;
    // d_V stores the Krylov basis vectors v_0 ... v_m. (Size: neqns * (m+1))
    // Stored column-major: V = [v0, v1, ..., vm]
    
    check_cublas(cublas_alloc(neqns, sizeof(complex__float), (void**)&d_res), "alloc d_res");
    check_cublas(cublas_alloc(neqns, sizeof(complex__float), (void**)&d_tmp), "alloc d_tmp");
    check_cublas(cublas_alloc(neqns * (m + 1), sizeof(complex__float), (void**)&d_V), "alloc d_V");
    
    // Allocate Host Memory for Hessenberg Matrix and Rotations
    // H is (m+1) x m.
    // Stored in a flattened vector. Access: Col k, Row j -> [k * (m+1) + j]
    int ldH = m + 1;
    std::vector<complex__float> h_H(ldH * m); 
    std::vector<complex__float> h_sn(m);
    std::vector<real_t> h_cs(m);
    std::vector<complex__float> h_g(ldH); // Right hand side of the least squares problem (g)
    
    std::vector<complex__float> h_y(m);   // Solution of the least squares problem

    int total_iters = 0;
    real_t current_res = rhs_nrm;

    // --- Outer Loop (Restarts) ---
    for (int iter = 0; iter < maxnit; ++iter) {
        
        // 1. Compute Initial Residual: r0 = b - A * x0
        // Calculate A * x0
        matvec(d_solution, d_tmp); // d_tmp = A * x0
        
        // d_res = b
        cublas_copy(handle, neqns, d_rhs, d_res);
        // d_res = d_res - d_tmp (i.e., b - Ax0)
        complex__float alpha = make_complex(-1.f, 0.f);
        cublas_axpy(handle, neqns, alpha, d_tmp, d_res);
        
        // 2. Normalize Residual
        real_t beta = cublas_nrm2(handle, neqns, d_res);
        current_res = beta;
        
        // Check convergence before inner loop
        if (iter == 0) std::cout << "Initial Absolute Residual: " << current_res << std::endl;
        else std::cout << "Restart " << iter << " Relative Residual: " << (current_res / rhs_nrm) << std::endl;

        if (current_res <= res_tol) {
            info = 0;
            break;
        }

        // v_0 = r0 / beta
        // Store v_0 in first column of d_V
        cublas_copy(handle, neqns, d_res, d_V); // Copy r0 to v0 location
        cublas_csscal(handle, neqns, real_t(1) / beta, d_V);
        
        // Initialize g = [beta, 0, ..., 0]^T
        std::fill(h_g.begin(), h_g.end(), make_complex(0.f, 0.f));
        h_g[0] = make_complex(beta, 0.f);
        
        // --- Inner Loop (Arnoldi) ---
        int k = 0;
        for (; k < m; ++k) {
            total_iters++;
            
            // 3. Arnoldi Step: w = A * v_k
            complex__float* v_k = d_V + k * neqns;
            complex__float* v_kp1 = d_V + (k + 1) * neqns; // Location for v_{k+1}, currently used as w
            
            matvec(v_k, v_kp1); // Compute w directly into v_{k+1} slot
            
            // Modified Gram-Schmidt Orthogonalization
            for (int j = 0; j <= k; ++j) {
                complex__float* v_j = d_V + j * neqns;
                
                // h_{j,k} = v_j^H * w
                complex__float h_jk = cublas_dotc(handle, neqns, v_j, v_kp1);
                
                // Store in H matrix (Col k, Row j)
                h_H[k * ldH + j] = h_jk;
                
                // w = w - h_{j,k} * v_j
                // v_{k+1} is holding w
                complex__float neg_h_jk = complexNeg(h_jk);
                cublas_axpy(handle, neqns, neg_h_jk, v_j, v_kp1);
            }
            
            // h_{k+1, k} = ||w||
            real_t h_kp1_k = cublas_nrm2(handle, neqns, v_kp1);
            h_H[k * ldH + (k + 1)] = make_complex(h_kp1_k, 0.f);
            
            // Scale v_{k+1} = w / h_{k+1, k}
            if (h_kp1_k > real_t(1e-12)) { // Check breakdown
                cublas_csscal(handle, neqns, real_t(1) / h_kp1_k, v_kp1);
            } else {
                // Happy breakdown? or lucky convergence.
                // Assuming we continue or break.
                // If it's truly 0, we found invariant subspace.
            }
            
            // 4. Update QR Factorization of H via Givens Rotations
            
            // Isolate column k of H: h_H[k*ldH ... ]
            // We need to form a local vector to manipulate
            // H is stored column-major in the vector, so we can access directly.
            
            // a) Apply previous updates 0..k-1
            // Manually loop because apply_previous needs std::vector
            // Let's reuse logic.
            // H_col_k is effectively accessible via pointers or indices.
            
            for (int i = 0; i < k; ++i) {
                // Indices in H for column k, rows i and i+1
                int idx_i = k * ldH + i;
                int idx_ip1 = k * ldH + i + 1;
                
                complex__float temp_i = h_H[idx_i];
                complex__float temp_ip1 = h_H[idx_ip1];
                
                // Apply Given's Rotation (i)
                // h[i]   = cs[i]*h[i] + sn[i]*h[i+1]
                h_H[idx_i] = complexAdd(complexScale(temp_i, h_cs[i]), complexMul(h_sn[i], temp_ip1));
                
                // h[i+1] = -conj(sn[i])*h[i] + cs[i]*h[i+1]
                complex__float sn_conj = make_complex(h_sn[i].x, -h_sn[i].y);
                h_H[idx_ip1] = complexAdd(complexMul(complexNeg(sn_conj), temp_i), complexScale(temp_ip1, h_cs[i]));
            }
            
            // b) Compute NEW rotation k to eliminate h_{k+1, k}
            complex__float h_diag = h_H[k * ldH + k];
            complex__float h_sub = h_H[k * ldH + k + 1];
            
            real_t cs_k;
            complex__float sn_k;
            generate_givens(h_diag, h_sub, cs_k, sn_k);
            
            // Store rotation
            h_cs[k] = cs_k;
            h_sn[k] = sn_k;
            
            // Apply new rotation to H (rows k and k+1)
            // h[k] = cs*h[k] + sn*h[k+1] (which becomes 'r')
            // h[k+1] = ... (which becomes 0)
            
            h_H[k * ldH + k] = complexAdd(complexScale(h_diag, cs_k), complexMul(sn_k, h_sub));
            h_H[k * ldH + k + 1] = make_complex(0.f, 0.f); 
            
            // c) Apply rotation to RHS vector g
            // g[k]   = cs*g[k] + sn*g[k+1]
            // g[k+1] = -conj(sn)*g[k] + cs*g[k+1]
            // Note: g[k+1] starts as 0.
            
            complex__float g_k = h_g[k];
            // g[k+1] is 0
            
            h_g[k] = complexScale(g_k, cs_k);
            complex__float sn_conj = make_complex(sn_k.x, -sn_k.y);
            h_g[k + 1] = complexMul(complexNeg(sn_conj), g_k);
            
            // Residual Check ( |g_{k+1}| )
            real_t abs_res = std::sqrt(h_g[k+1].x*h_g[k+1].x + h_g[k+1].y*h_g[k+1].y);
            current_res = abs_res;
            
            // Track relative residual (scaled by initial RHS norm)
            if (out_res_history) {
                out_res_history->push_back(current_res / rhs_nrm);
            }

            // std::cout << "   Inner " << k << ", Res: " << current_res << std::endl;
            
            if (current_res <= res_tol) {
                // Converged inside inner loop
                k++; // Increment to signal we did this step
                break;
            }
        }
        
        // 5. Form Approximate Solution
        // Solve H_k * y_k = g_k (Upper triangular)
        // Use k steps (0 to k-1)
        int meff = k;   
        
        // Extract needed part of g
        std::vector<complex__float> y_sol(meff);
        std::vector<complex__float> g_rhs(meff);
        for(int i=0; i<meff; ++i) g_rhs[i] = h_g[i];
        
        // Back substitution
        // Need to pass the full 'h_H' but carefully index it.
        // H is stored column-major. H(row, col) = h_H[col * ldH + row]
        // But the solve function expects... let's inline it to be safe.
        
        for (int i = meff - 1; i >= 0; --i) {
             complex__float sum = make_complex(0.f, 0.f);
             for (int j = i + 1; j < meff; ++j) {
                 // H(i, j) -> Col j, Row i.
                 complex__float H_ij = h_H[j * ldH + i]; 
                 sum = complexAdd(sum, complexMul(H_ij, y_sol[j]));
             }
             // y[i] = (g[i] - sum) / H(i,i)
             complex__float H_ii = h_H[i * ldH + i];
             y_sol[i] = complexDiv(complexSub(g_rhs[i], sum), H_ii);
        }
        
        // 6. Update x: x = x0 + V_k * y_k
        // Calculate linear combination V_k * y_k
        // V is (neqns x (m+1)). We use first k columns.
        
        // We can do this with cublasCaxpy k times.
        // d_solution already has x0 (start of restart).
        // Wait, d_solution is x0 at start of outer loop ONLY.
        // If we update d_solution now, it becomes x_new which serves as x0 for next restart. Correct.
        
        for (int j = 0; j < meff; ++j) {
            complex__float* v_j = d_V + j * neqns;
            cublas_axpy(handle, neqns, y_sol[j], v_j, d_solution);
        }
        
        if (current_res <= res_tol) {
            info = 0;
            break;
        } else {
            info = 1; // Need another restart
        }
    }
    
    // Cleanup
    cudaFree(d_res);
    cudaFree(d_tmp);
    cudaFree(d_V);
    
    if(out_res) *out_res = current_res;
    if(out_iters) *out_iters = total_iters;
    
    return info;
}


// Wrapper Class for Convenience
template <typename MatVecOp>
class GMRESSolver {
    cublasHandle_t _handle;
    int _neqns;
    MatVecOp _matvec;

public:
    GMRESSolver(cublasHandle_t handle, int neqns, MatVecOp matvec)
        : _handle(handle), _neqns(neqns), _matvec(matvec) {}

    int solve(const complex__float* d_rhs, complex__float* d_solution, int m, int maxnit, real_t tol, real_t* out_res = nullptr, int* out_iters = nullptr, std::vector<real_t>* out_res_history = nullptr) {
        return gmres_restarted(_neqns, m, maxnit, tol, d_solution, d_rhs, _matvec, _handle, out_res, out_iters, out_res_history);
    }
};

#endif // GMRES_SOLVER_CUH
