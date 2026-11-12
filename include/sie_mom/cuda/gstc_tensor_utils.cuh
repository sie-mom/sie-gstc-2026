#pragma once

/**
 * @file gstc_tensor_utils.cuh
 * @brief Local tangential coordinate frames and 2x2-to-3x3 GSTC tensor lifting utilities.
 *
 * Provides portable (`__host__ __device__`, `real_t` single/double precision) helpers
 * used during SIE-GSTC mass-matrix assembly and matrix-free matvec evaluation
 * to lift 2x2 tangential sheet blocks into 3x3 global Cartesian operators.
 */

#include "sie_mom/cuda/math_utils.cuh" // For complex arithmetic, vector ops, and real_t

namespace sie_mom {

/**
 * @brief Local tangential coordinate basis convention for 2x2 GSTC tensor blocks.
 *
 * Selects the orthonormal tangent pair \f$(\hat{\mathbf{u}}, \hat{\mathbf{v}})\f$ forming the
 * columns of the \f$3 \times 2\f$ embedding matrix \f$\mathbf{R} = [\hat{\mathbf{u}}, \hat{\mathbf{v}}]\f$
 * used in \f$\mathbf{T}_{\mathrm{glob}} = \mathbf{R}\,\mathbf{T}_{\mathrm{loc}}\,\mathbf{R}^T\f$:
 * - `face`: Intrinsic triangle-edge frame \f$(\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2)\f$ derived
 *   algebraically from ordered face vertices \f$(\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)\f$
 *   (see `get_face_basis_vectors`). Geometry-agnostic; valid on any closed, outward-oriented
 *   triangular surface mesh.
 * - `spherical`: Origin-centered spherical frame \f$(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})\f$
 *   evaluated at the face centroid (see `get_spherical_basis_vectors`). Intended for canonical
 *   origin-centered spherical validation cases.
 */
enum class GstcBasisType : int {
    face = 0,      ///< Intrinsic triangle-edge frame (t1, t2) from ordered (v0, v1, v2)
    spherical = 1  ///< Origin-centered spherical frame (theta_hat, phi_hat) at face centroid
};

// --- Local Tangent Frame & Anisotropic Tensor Helpers ---

/**
 * @brief Computes the intrinsic orthonormal tangent basis \f$(\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2)\f$
 *        from a triangle's ordered vertices \f$(\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)\f$.
 *
 * Given a triangular mesh face `faces[m] = (i0, i1, i2)` with vertex coordinates
 * \f$\mathbf{v}_0 = \text{vertices}[i_0]\f$, \f$\mathbf{v}_1 = \text{vertices}[i_1]\f$, and
 * \f$\mathbf{v}_2 = \text{vertices}[i_2]\f$ ordered counter-clockwise when viewed from the
 * exterior region (outward unit normal \f$\hat{\mathbf{n}}\f$), the local right-handed triad
 * is defined algebraically on the fly without trigonometric evaluations:
 * \f[
 *   \mathbf{e}_1 = \mathbf{v}_1 - \mathbf{v}_0, \qquad
 *   \mathbf{e}_2 = \mathbf{v}_2 - \mathbf{v}_0,
 * \f]
 * \f[
 *   \hat{\mathbf{n}} = \frac{\mathbf{e}_1 \times \mathbf{e}_2}{\|\mathbf{e}_1 \times \mathbf{e}_2\|}, \qquad
 *   \hat{\mathbf{t}}_1 = \frac{\mathbf{e}_1}{\|\mathbf{e}_1\|}, \qquad
 *   \hat{\mathbf{t}}_2 = \hat{\mathbf{n}} \times \hat{\mathbf{t}}_1.
 * \f]
 *
 * By construction, \f$(\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2, \hat{\mathbf{n}})\f$ is strictly
 * orthonormal and right-handed (\f$\hat{\mathbf{t}}_1 \times \hat{\mathbf{t}}_2 = \hat{\mathbf{n}}\f$),
 * matching the planar face normal used in the RWG perpendicular operator \f$\hat{\mathbf{n}} \times (\cdot)\f$.
 *
 * @warning Vertex-Order Contract: Because \f$\hat{\mathbf{t}}_1\f$ aligns with the directed edge
 *          \f$\mathbf{v}_0 \to \mathbf{v}_1\f$, cyclic permutations of \f$(\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)\f$
 *          rotate \f$(\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2)\f$ in the tangent plane. Both Python
 *          GSTC parameter synthesis (`sie_mom.mesh.face_frames.compute_face_frames`) and C++ mesh
 *          loading (`ObjMeshLoader`) must preserve the exact `(i0, i1, i2)` vertex order of each face.
 *
 * @param v0 First ordered vertex \f$\mathbf{v}_0\f$ of the triangular face.
 * @param v1 Second ordered vertex \f$\mathbf{v}_1\f$ of the triangular face.
 * @param v2 Third ordered vertex \f$\mathbf{v}_2\f$ of the triangular face.
 * @param[out] t1 First unit tangent vector \f$\hat{\mathbf{t}}_1\f$ (column 0 of \f$\mathbf{R}\f$).
 * @param[out] t2 Second unit tangent vector \f$\hat{\mathbf{t}}_2\f$ (column 1 of \f$\mathbf{R}\f$).
 */
__host__ __device__ inline void get_face_basis_vectors(
    vec3_t v0, vec3_t v1, vec3_t v2,
    vec3_t& t1, vec3_t& t2
) {
    const vec3_t e1 = v1 - v0;
    const vec3_t e2 = v2 - v0;
    const vec3_t normal_unnormalized = cross(e1, e2);
    const real_t e1_len = length(e1);
    const real_t normal_len = length(normal_unnormalized);

    if (e1_len < static_cast<real_t>(1e-20) || normal_len < static_cast<real_t>(1e-20)) {
        t1 = make_vec3(real_t(1), real_t(0), real_t(0));
        t2 = make_vec3(real_t(0), real_t(1), real_t(0));
        return;
    }

    const vec3_t n_hat = (real_t(1) / normal_len) * normal_unnormalized;
    t1 = (real_t(1) / e1_len) * e1;
    t2 = cross(n_hat, t1);
}

/**
 * @brief Computes the origin-centered spherical tangent basis \f$(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\phi}})\f$
 *        evaluated at the face centroid.
 *
 * Returns the columns of the \f$3 \times 2\f$ embedding matrix \f$\mathbf{R} = [\hat{\mathbf{u}}, \hat{\mathbf{v}}]\f$
 * with \f$\hat{\mathbf{u}} = \hat{\boldsymbol{\theta}}\f$ and \f$\hat{\mathbf{v}} = \hat{\boldsymbol{\phi}}\f$.
 *
 * @param v0 Vertex 0 of the face.
 * @param v1 Vertex 1 of the face.
 * @param v2 Vertex 2 of the face.
 * @param[out] R_col0 Output: First column of R (\f$\hat{\boldsymbol{\theta}}\f$).
 * @param[out] R_col1 Output: Second column of R (\f$\hat{\boldsymbol{\phi}}\f$).
 */
__host__ __device__ inline void get_spherical_basis_vectors(
    vec3_t v0, vec3_t v1, vec3_t v2,
    vec3_t& R_col0, vec3_t& R_col1
) {
    // 1. Compute Centroid
    vec3_t centroid;
    centroid.x = (v0.x + v1.x + v2.x) / real_t(3);
    centroid.y = (v0.y + v1.y + v2.y) / real_t(3);
    centroid.z = (v0.z + v1.z + v2.z) / real_t(3);

    // 2. Local Spherical Coordinates
    const real_t r = length(centroid);

    // Avoid division by zero at origin (though mesh shouldn't be there)
    if (r < static_cast<real_t>(1e-12)) {
        R_col0 = make_vec3(real_t(1), real_t(0), real_t(0));
        R_col1 = make_vec3(real_t(0), real_t(1), real_t(0));
        return;
    }

    real_t cos_theta = centroid.z / r;
    if (cos_theta > real_t(1)) cos_theta = real_t(1);
    if (cos_theta < real_t(-1)) cos_theta = real_t(-1);

    const real_t theta = acos(cos_theta);
    const real_t phi = atan2(centroid.y, centroid.x);

    const real_t ct = cos(theta);
    const real_t st = sin(theta);
    const real_t cp = cos(phi);
    const real_t sp = sin(phi);

    // theta_hat = cos(th)cos(ph) x + cos(th)sin(ph) y - sin(th) z
    R_col0.x = ct * cp;
    R_col0.y = ct * sp;
    R_col0.z = -st;

    // phi_hat = -sin(ph) x + cos(ph) y
    R_col1.x = -sp;
    R_col1.y = cp;
    R_col1.z = real_t(0);
}

/**
 * @brief Dispatches local tangent basis evaluation according to `basis_type`.
 *
 * @param v0 First ordered vertex of the triangular face.
 * @param v1 Second ordered vertex of the triangular face.
 * @param v2 Third ordered vertex of the triangular face.
 * @param basis_type Coordinate frame convention (`GstcBasisType::face` or `GstcBasisType::spherical`).
 * @param[out] u_hat First unit tangent basis vector (column 0 of \f$\mathbf{R}\f$).
 * @param[out] v_hat Second unit tangent basis vector (column 1 of \f$\mathbf{R}\f$).
 */
__host__ __device__ inline void get_local_basis_vectors(
    vec3_t v0, vec3_t v1, vec3_t v2,
    sie_mom::GstcBasisType basis_type,
    vec3_t& u_hat, vec3_t& v_hat
) {
    if (basis_type == sie_mom::GstcBasisType::face) {
        get_face_basis_vectors(v0, v1, v2, u_hat, v_hat);
    } else {
        get_spherical_basis_vectors(v0, v1, v2, u_hat, v_hat);
    }
}

/**
 * @brief Lifts a 2x2 local tangential tensor \f$\mathbf{T}_{\mathrm{local}}\f$ into a 3x3 global
 *        Cartesian tensor \f$\mathbf{T}_{\mathrm{global}} = \mathbf{R}\,\mathbf{T}_{\mathrm{local}}\,\mathbf{R}^T\f$,
 *        where \f$\mathbf{R} = [\mathbf{u}, \mathbf{v}]\f$.
 *
 * @param T_loc_00 Element (0,0) of local 2x2 tensor
 * @param T_loc_01 Element (0,1) of local 2x2 tensor
 * @param T_loc_10 Element (1,0) of local 2x2 tensor
 * @param T_loc_11 Element (1,1) of local 2x2 tensor
 * @param u First tangent basis vector (column 0 of R)
 * @param v Second tangent basis vector (column 1 of R)
 * @param[out] T_glob Output 3x3 global Cartesian tensor
 */
__host__ __device__ inline void rotate_tensor_2x2_to_3x3(
    complex__float T_loc_00, complex__float T_loc_01,
    complex__float T_loc_10, complex__float T_loc_11,
    vec3_t u, vec3_t v,
    complex__float T_glob[3][3]
) {
    // We want T_glob = R * T_loc * R^T
    // Let R = [u  v] (3x2 matrix)
    // T_loc = [[T00, T01], [T10, T11]]
    //
    // Intermediate term M = T_loc * R^T (2x3 matrix)
    // M = [[T00, T01], [T10, T11]] * [[u.x, u.y, u.z], [v.x, v.y, v.z]]
    // M_row0 = T00 * u^T + T01 * v^T
    // M_row1 = T10 * u^T + T11 * v^T

    // We can compute components directly:
    // T_glob(i, j) = R(i, 0)*M(0, j) + R(i, 1)*M(1, j)

    auto get_comp = [](vec3_t vec, int idx) -> real_t {
        return (idx == 0) ? vec.x : (idx == 1) ? vec.y : vec.z;
    };

    for (int i = 0; i < 3; ++i) { // Row of T_glob
        const real_t Ri0 = get_comp(u, i);
        const real_t Ri1 = get_comp(v, i);

        for (int j = 0; j < 3; ++j) { // Col of T_glob
            const real_t Rj0 = get_comp(u, j);
            const real_t Rj1 = get_comp(v, j);

            // Expansion of (R * T * R^T)ij
            // = Ri0*T00*Rj0 + Ri0*T01*Rj1 + Ri1*T10*Rj0 + Ri1*T11*Rj1

            complex__float val = make_complex(real_t(0), real_t(0));

            // Term 1: u_i * T00 * u_j
            complex__float term1 = complexScale(T_loc_00, Ri0 * Rj0);
            val = complexAdd(val, term1);

            // Term 2: u_i * T01 * v_j
            complex__float term2 = complexScale(T_loc_01, Ri0 * Rj1);
            val = complexAdd(val, term2);

            // Term 3: v_i * T10 * u_j
            complex__float term3 = complexScale(T_loc_10, Ri1 * Rj0);
            val = complexAdd(val, term3);

            // Term 4: v_i * T11 * v_j
            complex__float term4 = complexScale(T_loc_11, Ri1 * Rj1);
            val = complexAdd(val, term4);

            T_glob[i][j] = val;
        }
    }
}

/**
 * @brief Computes the "perpendicular" tensor transformation:
 *        \f$\mathbf{Y}^{\perp} = - [\hat{\mathbf{n}}]_\times \, \mathbf{T}_{\mathrm{in}} \, [\hat{\mathbf{n}}]_\times\f$.
 *
 * In the weak-form boundary integral equations, the admittance terms involve cross-normal
 * fields appearing in both the testing function and the trial current:
 * \f[
 *    \text{Term} \propto (\hat{\mathbf{n}} \times \mathbf{f}_i) \cdot \mathbf{Y}_{3D} \cdot (\hat{\mathbf{n}} \times \mathbf{f}_j).
 * \f]
 * Using the vector identity \f$(\hat{\mathbf{n}} \times \mathbf{A}) \cdot \mathbf{B} = -\mathbf{A} \cdot (\hat{\mathbf{n}} \times \mathbf{B})\f$,
 * this term is converted to standard form \f$\mathbf{f}_i \cdot (\mathbf{Y}^{\perp} \cdot \mathbf{f}_j)\f$,
 * where \f$[\hat{\mathbf{n}}]_\times\f$ is the skew-symmetric cross-product matrix:
 * \f[
 *    [\hat{\mathbf{n}}]_\times = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}.
 * \f]
 * Geometrically, \f$[\hat{\mathbf{n}}]_\times\f$ represents a 90-degree in-plane rotation about \f$\hat{\mathbf{n}}\f$
 * (\f$\hat{\mathbf{n}} \times \mathbf{v}\f$). The minus sign comes from moving the cross product from the testing
 * function onto the operator.
 *
 * The implementation explicitly computes:
 * 1. \f$\mathbf{T}_{\mathrm{temp}} = \mathbf{T}_{\mathrm{in}} \, [\hat{\mathbf{n}}]_\times\f$
 * 2. \f$\mathbf{T}_{\mathrm{out}} = - [\hat{\mathbf{n}}]_\times \, \mathbf{T}_{\mathrm{temp}}\f$
 *
 * @param T_in  Input 3x3 Cartesian tensor (e.g. \f$\mathbf{Y}_{3D}\f$)
 * @param n_hat Unit normal vector \f$\hat{\mathbf{n}}\f$
 * @param[out] T_out Output 3x3 perpendicular tensor \f$\mathbf{Y}^{\perp}\f$
 *
 * @see docs/source/matrix_assembly_v2.rst (Section "The Perp Transform")
 */
__host__ __device__ inline void apply_perp_transform(
    const complex__float T_in[3][3],
    const vec3_t& n_hat,
    complex__float T_out[3][3]
) {
    const real_t nx = n_hat.x;
    const real_t ny = n_hat.y;
    const real_t nz = n_hat.z;
    const real_t N[3][3] = {
        {real_t(0), -nz, ny},
        {nz, real_t(0), -nx},
        {-ny, nx, real_t(0)}
    };

    complex__float T_temp[3][3];
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            complex__float sum = make_complex(0.0f, 0.0f);
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                sum = complexAdd(sum, complexScale(T_in[r][k], static_cast<float>(N[k][c])));
            }
            T_temp[r][c] = sum;
        }
    }

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            complex__float sum = make_complex(0.0f, 0.0f);
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                sum = complexAdd(sum, complexScale(T_temp[k][c], static_cast<float>(N[r][k])));
            }
            T_out[r][c] = complexNeg(sum);
        }
    }
}

/**
 * @brief Copies elements of a 3x3 complex tensor from `src` to `dest`.
 *
 * @param src  Source 3x3 tensor
 * @param dest Destination 3x3 tensor
 */
__host__ __device__ inline void copy_tensor_3x3(
    const complex__float src[3][3],
    complex__float dest[3][3]
) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            dest[r][c] = src[r][c];
        }
    }
}

} // namespace sie_mom

using sie_mom::GstcBasisType;
using sie_mom::get_face_basis_vectors;
using sie_mom::get_spherical_basis_vectors;
using sie_mom::get_local_basis_vectors;
using sie_mom::rotate_tensor_2x2_to_3x3;
using sie_mom::apply_perp_transform;
using sie_mom::copy_tensor_3x3;
