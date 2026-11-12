#ifndef SIE_MOM_GSTC_UTILS_HPP
#define SIE_MOM_GSTC_UTILS_HPP

// Eigen is used only for host-side GSTC tensor construction in this header.
// Avoid Eigen/MSVC complex-operator conflicts in CUDA translation units.
// This must precede the first Eigen include; CUDA solvers remain enabled.
#if defined(_MSC_VER) && defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_NO_CUDA
#endif

#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <fstream>     // For std::ifstream
#include <string>      // For std::string
#include <stdexcept>   // For std::runtime_error
#include "sie_mom/cuda/math_utils.cuh" // For complex__float and float3
#include "sie_mom/mesh/spherical_surface.hpp" // For MeshData, faces, etc.
#include "sie_mom/utils/binary_io.hpp"

namespace sie_mom {
namespace gstc {

// Define a 2x2 complex matrix type for input parameters
using Matrix2cd = Eigen::Matrix<std::complex<float>, 2, 2>;
using Matrix3cd = Eigen::Matrix<std::complex<float>, 3, 3>;

/**
 * @brief Parameters defining the 4 GSTC tensors (Y, Chi, Gamma, Z) in the local (theta, phi) basis.
 * These are assumed to be uniform over the sphere (constant coefficients in the spherical basis).
 */
struct UniformGSTCParams {
    Matrix2cd Y_ee; // Electric Admittance
    Matrix2cd Y_em; // Chi (Magneto-Electric)
    Matrix2cd Y_me; // Gamma (Electro-Magnetic)
    Matrix2cd Y_mm; // Magnetic Impedance
};

/**
 * @brief Helper to convert an Eigen complex<float> to our CUDA-compatible complex__float
 */
inline complex__float to_cu(std::complex<float> c) {
    return make_complex(c.real(), c.imag());
}

/**
 * @brief Generates a SINGLE set of 4x4 GSTC tensors (Y and Z) for uniform materials.
 * 
 * This is a simplified version of generate_face_tensors for validation cases where
 * the material properties are constant in the LOCAL coordinate system.
 * across the entire mesh.
 * 
 * Output Format:
 * - A flattened vector of complex floats (Size 32).
 *   - First 16: Y_tot (4x4, row-major)
 *   - Next 16:  Z_tot (4x4, row-major)
 * 
 * @param params The uniform GSTC parameters.
 * @return std::vector<complex__float> Flattened array of size 32.
 */
inline std::vector<complex__float> generate_uniform_tensors(
    const UniformGSTCParams& params
) {
    std::vector<complex__float> flat_tensors;
    flat_tensors.reserve(32); 

    // Pre-calculate the 4x4 Inverse
    Eigen::Matrix<std::complex<float>, 4, 4> Y_tot;
    Y_tot.block<2,2>(0,0) = params.Y_ee;
    Y_tot.block<2,2>(0,2) = params.Y_em;
    Y_tot.block<2,2>(2,0) = params.Y_me;
    Y_tot.block<2,2>(2,2) = params.Y_mm;
    
    Eigen::Matrix<std::complex<float>, 4, 4> Z_tot = Y_tot.inverse();
    
    // 1. Store Y_tot (4x4)
    for(int r=0; r<4; ++r) {
        for(int c=0; c<4; ++c) {
            flat_tensors.push_back(to_cu(Y_tot(r,c)));
        }
    }
    
    // 2. Store Z_tot (4x4)
    for(int r=0; r<4; ++r) {
        for(int c=0; c<4; ++c) {
            flat_tensors.push_back(to_cu(Z_tot(r,c)));
        }
    }
    
    return flat_tensors;
}

/**
 * @brief Generates the per-face GSTC tensors.
 * 
 * For each face, this function provides the full 4x4 Admittance (Y) and Impedance (Z) matrices.
 * 
 * Logic:
 * 1. Takes the uniform 2x2 input blocks (Yee, Yem, Yme, Ymm).
 * 2. Assembles them into a 4x4 Admittance matrix Y_tot.
 * 3. Inverts Y_tot to get the 4x4 Impedance matrix Z_tot.
 * 4. Stores both 4x4 matrices for every face.
 * 
 * note:: No rotation is performed here. The matrices are in the local surface basis (u, v).
 * The basis rotation to global Cartesian coordinates is handled inside the GPU kernel.
 * 
 * Output Format:
 * - A flattened vector of complex floats.
 * - Per face: 32 complex numbers.
 *   - First 16: Y_tot (4x4, row-major)
 *   - Next 16:  Z_tot (4x4, row-major)
 * 
 * @param mesh The spherical mesh data (used for face count).
 * @param params The uniform GSTC parameters.
 * @return std::vector<complex__float> Flattened array of size num_faces * 32.
 */
inline std::vector<complex__float> generate_face_tensors(
    const sie_mom::mesh::MeshData& mesh,
    const UniformGSTCParams& params
) {
    size_t num_faces = mesh.faces.size();
    std::vector<complex__float> flat_tensors;
    // Each face needs 2 matrices (Y_tot, Z_tot) of 4x4 elements = 32 complex floats
    flat_tensors.reserve(num_faces * 32); 

    // Pre-calculate the 4x4 Inverse once (to get Z parameters from Y parameters)
    Eigen::Matrix<std::complex<float>, 4, 4> Y_tot;
    Y_tot.block<2,2>(0,0) = params.Y_ee;
    Y_tot.block<2,2>(0,2) = params.Y_em;
    Y_tot.block<2,2>(2,0) = params.Y_me;
    Y_tot.block<2,2>(2,2) = params.Y_mm;
    
    // Invert to get Z blocks
    // Vector = Y_tot * Field -> Field = Z_tot * Vector
    Eigen::Matrix<std::complex<float>, 4, 4> Z_tot = Y_tot.inverse();
    
    // Fill the vector
    // We replicate the same 4x4 matrices for every face.
    // In a non-uniform implementation, we would compute different Y/Z per face here.
    for (size_t i = 0; i < num_faces; ++i) {
        // 1. Store Y_tot (4x4)
        for(int r=0; r<4; ++r) {
            for(int c=0; c<4; ++c) {
                flat_tensors.push_back(to_cu(Y_tot(r,c)));
            }
        }
        
        // 2. Store Z_tot (4x4)
        for(int r=0; r<4; ++r) {
            for(int c=0; c<4; ++c) {
                flat_tensors.push_back(to_cu(Z_tot(r,c)));
            }
        }
    }
    
    return flat_tensors;
}

/**
 * @brief Loads pre-computed tensor data from a binary file.
 * Expects 32 complex floats per face (16 for Y, 16 for Z).
 * 
 * Used for loading spatially varying cloaking tensors from Python.
 */
inline std::vector<complex__float> load_face_tensors_from_file(
    const std::string& filename,
    size_t num_faces
) {
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()) throw std::runtime_error("Could not open tensor file: " + filename);
    
    // Check file size
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    const size_t expected_values = num_faces * 32 * 2; // 32 complex = 64 scalars
    const size_t expected_bytes_f32 = expected_values * sizeof(float);
    const size_t expected_bytes_f64 = expected_values * sizeof(double);

    if(file_size != expected_bytes_f32 && file_size != expected_bytes_f64) {
        throw std::runtime_error("Tensor file size mismatch. Expected " +
            std::to_string(expected_bytes_f32) + " or " +
            std::to_string(expected_bytes_f64) + " bytes, got " +
            std::to_string(file_size));
    }

    std::vector<complex__float> tensors(num_faces * 32);

    if (file_size == expected_bytes_f32) {
        std::vector<float> raw(expected_values);
        in.read(reinterpret_cast<char*>(raw.data()), expected_bytes_f32);
        if(!in) throw std::runtime_error("Error reading tensor data");

        for (size_t i = 0; i < tensors.size(); ++i) {
            const real_t re = static_cast<real_t>(raw[i * 2]);
            const real_t im = static_cast<real_t>(raw[i * 2 + 1]);
            tensors[i] = make_complex(re, im);
        }
    } else {
        std::vector<double> raw(expected_values);
        in.read(reinterpret_cast<char*>(raw.data()), expected_bytes_f64);
        if(!in) throw std::runtime_error("Error reading tensor data");

        for (size_t i = 0; i < tensors.size(); ++i) {
            const real_t re = static_cast<real_t>(raw[i * 2]);
            const real_t im = static_cast<real_t>(raw[i * 2 + 1]);
            tensors[i] = make_complex(re, im);
        }
    }

    return tensors;
}

/**
 * @brief Converts 10 dimensionless optimization parameters per face into the full
 *        physical 4x4 forward map Lambda_f and inverse map Lambda_f^-1 (32 complex
 *        numbers per face) for reciprocal-lossless metasurfaces.
 *
 * ====================================================================================
 * RECIPROCAL-LOSSLESS PARAMETERIZATION & ROSETTA STONE
 * ====================================================================================
 * An arbitrary bi-anisotropic surface map Lambda_f maps average tangential fields to
 * equivalent surface current densities:
 *
 *     [ J_t ]   [ Y_ee   Y_em ] [ E_avg ]
 *     [ M_t ] = [ Y_me   Y_mm ] [ H_avg ]
 *
 * When restricted to the reciprocal and lossless sub-manifold:
 * 1. Reciprocity requires:
 *      Y_ee = Y_ee^T  (symmetric electric admittance)
 *      Y_mm = Y_mm^T  (symmetric magnetic impedance; named Z_mm in literature)
 *      Y_me = -Y_em^T (skew-transposed magneto-electric coupling; gamma = -chi^T)
 *
 * 2. Losslessness requires the map to be skew-Hermitian (Lambda_f + Lambda_f^H = 0):
 *      Re(Y_ee) = 0   (purely imaginary / reactive electric admittance)
 *      Re(Y_mm) = 0   (purely imaginary / reactive magnetic impedance)
 *      Im(Y_em) = 0   (purely real magneto-electric coupling)
 *
 * This reduces the degrees of freedom from 32 real numbers to exactly 10 real numbers
 * per face: theta_m = (y1, y2, y3, c1, c2, c3, c4, z1, z2, z3)
 *
 * Dimensional Normalization with reference impedance eta0 = sqrt(mu0 / eps0):
 *   - Electric admittance (units: Siemens [S]):
 *       Y_ee = (j / eta0) * [ y1,  y3 ]
 *                           [ y3,  y2 ]
 *   - Magneto-electric coupling (dimensionless [1]):
 *       chi   = [ c1,  c2 ]
 *               [ c3,  c4 ]
 *       gamma = -chi^T = [ -c1, -c3 ]
 *                        [ -c2, -c4 ]
 *   - Magnetic impedance (units: Ohms [Ohm]):
 *       Z_mm = (j * eta0) * [ z1,  z3 ]
 *                           [ z3,  z2 ]
 *
 * Output Layout per Face (32 complex numbers):
 *   - Elements 0..15  (Y_ptr) : The 4x4 forward map Lambda_f (row-major)
 *   - Elements 16..31 (Z_ptr) : The 4x4 inverse map Lambda_f^-1 = Z_solver,f (row-major)
 *
 * Performance & Parallelization Note:
 *   Eigen computes the inverse of 4x4 complex matrices using an exact closed-form
 *   analytical cofactor formula rather than general Gaussian elimination. On modern
 *   x86 CPUs, this requires only ~150-200 clock cycles per face (benchmarked at ~1.8 ms
 *   for 320 faces and ~12.3 ms for 1,372 faces; projected ~45 ms for 5,000 faces).
 *
 *   Because this CPU setup is strictly O(N_faces) while the subsequent dense GPU LU
 *   factorization is O(N^3) (~620 ms for 1,372 faces; ~30 s for 5,000 faces), this loop
 *   accounts for <0.2% - 2% of the total iteration runtime and is not a bottleneck.
 *
 *   Future Parallelization:
 *   Faces are strictly independent with zero inter-face coupling. If scaling to
 *   massive meshes (>10,000 faces) where CPU reconstruction becomes non-negligible,
 *   pre-allocate `tensors.resize(num_faces * 32)` and apply `#pragma omp parallel for`
 *   with direct indexed assignment (`tensors[m * 32 + idx] = ...`).
 *
 * @param thetas     Flat array of size 10 * num_faces with normalized parameters
 * @param num_faces  Number of triangular faces in the mesh
 * @param eta0       Physical background reference impedance sqrt(mu0 / eps0)
 * @return Flattened array of size num_faces * 32 containing Lambda_f and Lambda_f^-1
 */
inline std::vector<complex__float> build_face_tensors_from_thetas(
    const std::vector<real_t>& thetas,
    size_t num_faces,
    real_t eta0
) {
    if (thetas.size() != num_faces * 10) {
        throw std::runtime_error(
            "Theta vector size mismatch: expected " + std::to_string(num_faces * 10) +
            " parameters, got " + std::to_string(thetas.size())
        );
    }

    std::vector<complex__float> tensors;
    tensors.reserve(num_faces * 32);

    const std::complex<double> j(0.0, 1.0);
    const double eta0_d = static_cast<double>(eta0);

    for (size_t m = 0; m < num_faces; ++m) {
        const size_t base = m * 10;
        const double y1 = static_cast<double>(thetas[base + 0]);
        const double y2 = static_cast<double>(thetas[base + 1]);
        const double y3 = static_cast<double>(thetas[base + 2]);
        const double c1 = static_cast<double>(thetas[base + 3]);
        const double c2 = static_cast<double>(thetas[base + 4]);
        const double c3 = static_cast<double>(thetas[base + 5]);
        const double c4 = static_cast<double>(thetas[base + 6]);
        const double z1 = static_cast<double>(thetas[base + 7]);
        const double z2 = static_cast<double>(thetas[base + 8]);
        const double z3 = static_cast<double>(thetas[base + 9]);

        Eigen::Matrix<std::complex<double>, 4, 4> Lambda = Eigen::Matrix<std::complex<double>, 4, 4>::Zero();

        // 1. Electric admittance: Y_ee = (j / eta0) * [y1, y3; y3, y2]
        Lambda(0, 0) = j * (y1 / eta0_d);
        Lambda(1, 1) = j * (y2 / eta0_d);
        Lambda(0, 1) = j * (y3 / eta0_d);
        Lambda(1, 0) = j * (y3 / eta0_d);

        // 2. Magneto-electric coupling: chi = [c1, c2; c3, c4]
        Lambda(0, 2) = c1;
        Lambda(0, 3) = c2;
        Lambda(1, 2) = c3;
        Lambda(1, 3) = c4;

        // 3. Electro-magnetic coupling: gamma = -chi^T = [-c1, -c3; -c2, -c4]
        Lambda(2, 0) = -c1;
        Lambda(3, 0) = -c2;
        Lambda(2, 1) = -c3;
        Lambda(3, 1) = -c4;

        // 4. Magnetic impedance: Z_mm = (j * eta0) * [z1, z3; z3, z2]
        Lambda(2, 2) = j * (z1 * eta0_d);
        Lambda(3, 3) = j * (z2 * eta0_d);
        Lambda(2, 3) = j * (z3 * eta0_d);
        Lambda(3, 2) = j * (z3 * eta0_d);

        // Fast 4x4 matrix inversion in Eigen
        Eigen::Matrix<std::complex<double>, 4, 4> Lambda_inv = Lambda.inverse();

        // Store Lambda_f (16 complex numbers, row-major)
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                tensors.push_back(make_complex(
                    static_cast<real_t>(Lambda(r, c).real()),
                    static_cast<real_t>(Lambda(r, c).imag())
                ));
            }
        }

        // Store Lambda_f^-1 (16 complex numbers, row-major)
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                tensors.push_back(make_complex(
                    static_cast<real_t>(Lambda_inv(r, c).real()),
                    static_cast<real_t>(Lambda_inv(r, c).imag())
                ));
            }
        }
    }

    return tensors;
}

/**
 * @brief Loads the 10 normalized optimization parameters per face from a binary file
 *        and converts them into the physical 4x4 forward and inverse maps (32 complex
 *        values per face) ready for GPU consumption.
 *
 * @param filename   Path to binary file containing 10 * num_faces real values
 * @param num_faces  Number of triangular faces in the mesh
 * @param eta0       Physical background reference impedance sqrt(mu0 / eps0)
 * @return Flattened array of size num_faces * 32
 */
inline std::vector<complex__float> load_face_thetas_from_file(
    const std::string& filename,
    size_t num_faces,
    real_t eta0
) {
    const auto thetas = sie_mom::utils::read_binary_vector<real_t>(filename);
    return build_face_tensors_from_thetas(thetas, num_faces, eta0);
}

} // namespace gstc
} // namespace sie_mom

#endif // SIE_MOM_GSTC_UTILS_HPP
