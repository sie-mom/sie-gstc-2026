#ifndef SEMI_ANALYTIC_INTEGRALS_CUH
#define SEMI_ANALYTIC_INTEGRALS_CUH

#include "platform.cuh"
#include "math_utils.cuh"
#include "mesh_geometry.cuh"

#ifdef __CUDACC__
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#endif


/**
 * @brief Computes the geometric quantities required for Wilton's analytic potentials.
 *
 * This helper function calculates all the necessary geometric terms for a given
 * observation point `r` and a source triangle, as described in Wilton et al.,
 * "Potential Integrals for Uniform and Linear Source Distributions on
 * Polygonal and Polyhedral Domains," IEEE Trans. Antennas Propagat., 1984.
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @param d Output: Signed distance from `r` to the triangle's plane.
 * @param r_pi Output: Projection of `r` onto the triangle's plane.
 * @param u_i Output: Array of the three outward-pointing co-normal vectors for each edge.
 * @param p0_i Output: Array of the three perpendicular distances from `r_pi` to each edge line.
 * @param l_i Output: 2D array for the signed distances from `r_pi` to the start/end vertices of each edge. l_i[edge][0] is l_minus, l_i[edge][1] is l_plus.
 * @param P_i Output: 2D array for the in-plane distances from `r_pi` to the start/end vertices of each edge. P_i[edge][0] is P_minus, P_i[edge][1] is P_plus.
 * @param R0_i Output: Array of the three distances from `r` to each edge line.
 * @param R_i Output: 2D array for the 3D distances from `r` to the start/end vertices of each edge. R_i[edge][0] is R_minus, R_i[edge][1] is R_plus.
 * @param orientation_i Output: Array of the three orientation factors (+1 or -1).
 */
__host__ __device__ inline void calculateWiltonGeometricTerms(
    const vec3_t& r,
    const vec3_t tri[3],
    double& d,
    vec3_t& r_pi,
    vec3_t u_i[3],
    double p0_i[3],
    double l_i[3][2],
    double P_i[3][2],
    double R0_i[3],
    double R_i[3][2],
    double orientation_i[3])
{
    // Triangle normal
    vec3_t e1 = tri[1] - tri[0];
    vec3_t e2 = tri[2] - tri[0];
    vec3_t n = cross(e1, e2);
    n = normalize(n);

    // Signed distance `d` and projection `r_pi`
    d = dot(r - tri[0], n);
    r_pi = r - d * n;

    for (int edge_idx = 0; edge_idx < 3; ++edge_idx)
    {
        const vec3_t& v_start = tri[edge_idx];
        const vec3_t& v_end = tri[(edge_idx + 1) % 3];

        // Edge tangent and co-normal
        vec3_t t_hat = normalize(v_end - v_start);
        u_i[edge_idx] = cross(t_hat, n);

        // In-plane projections of vertices
        // Project vertices onto the plane defined by the normal `n` and the origin.
        vec3_t rho_start = v_start - n * dot(n, v_start);
        vec3_t rho_end   = v_end - n * dot(n, v_end);

        // Perpendicular distance to edge line (p0)
        vec3_t vec_to_start = rho_start - r_pi;
        p0_i[edge_idx] = fabs(dot(vec_to_start, u_i[edge_idx]));

        // Orientation factor
        orientation_i[edge_idx] = (dot(vec_to_start, u_i[edge_idx])) >= 0.f ? 1.f : -1.f;

        // Along-edge distances (l_plus, l_minus)
        l_i[edge_idx][1] = dot(rho_end - r_pi, t_hat);   // l_plus
        l_i[edge_idx][0] = dot(rho_start - r_pi, t_hat); // l_minus

        // In-plane distances to vertices (P_plus, P_minus)
        P_i[edge_idx][1] = hypot(p0_i[edge_idx], l_i[edge_idx][1]);
        P_i[edge_idx][0] = hypot(p0_i[edge_idx], l_i[edge_idx][0]);

        // 3D distances (R0, R_plus, R_minus)
        R0_i[edge_idx]      = hypot(p0_i[edge_idx], fabs(d));      // √(p0^2 + d^2)
        R_i[edge_idx][1]    = hypot(P_i[edge_idx][1], fabs(d));      // √(P+^2 + d^2) = √(p0^2 + l+^2 + d^2)
        R_i[edge_idx][0]    = hypot(P_i[edge_idx][0], fabs(d));      // √(P-^2 + d^2) = √(p0^2 + l-^2 + d^2)
    }
}


/**
 * @brief Computes the scalar potential phi_0 for a triangular source.
 *
 * This function implements the analytic formula for the scalar potential from
 * a uniform charge distribution over a planar triangle, as defined in
 * `eq:phi0_wilton` in the documentation.
 *
 * \f[ \phi_0(\mathbf r) = \int_{f_n} \frac{1}{\|\mathbf r - \mathbf r'\|} dS' \f]
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @return The scalar potential value.
 */
__host__ __device__ inline double calculate_phi_0(const vec3_t& r, const vec3_t tri[3])
{
    double d, p0_i[3], l_i[3][2], P_i[3][2], R0_i[3], R_i[3][2], orientation_i[3];
    vec3_t r_pi, u_i[3];

    calculateWiltonGeometricTerms(r, tri, d, r_pi, u_i, p0_i, l_i, P_i, R0_i, R_i, orientation_i);

    double phi0 = 0.0;
    double abs_d = fabs(d);
    for (int i = 0; i < 3; ++i)
    {
        double p0 = p0_i[i];
        double l_plus = l_i[i][1];
        double l_minus = l_i[i][0];
        double R_plus = R_i[i][1];
        double R_minus = R_i[i][0];
        double R0 = R0_i[i];

        // If the observation point is on one of the edge's vertices,
        // either R_plus or R_minus will be zero. The contribution from this
        // edge is zero, so we can safely skip it to avoid log(0).
        if (R_plus < 1e-9 || R_minus < 1e-9) continue;

        // If the observation point is on the line containing the edge (p0=0),
        // the contribution from this edge is also zero.
        // This is a singularity in the Wilton formula for the coplanar case.
        //   numerical issues when r lies in S along the extension of an edge (Ri0=0). 
        // In such cases, the observation should be moved a very small distance away from the edge in the plane, i.e.,
        // r = r + epsilon * u^; epsilon is a very small number and u^ is the unit vector along the edge.
        
        // TODO: For robustness across different mesh scales (e.g., microscopic vs macro), 
        // the threshold and epsilon below should ideally be relative to the local geometry.
        // For example: `const double epsilon = 1e-8 * length_of_edge_i;` 
        // A similar relative approach should be considered for the global singularity threshold.
        if (R0_i[i] < 1e-8) {
             const double epsilon = 1e-8;
             R_plus = sqrt(pow(R_plus, 2) + pow(epsilon, 2));
             R_minus = sqrt(pow(R_minus, 2) + pow(epsilon, 2));

        }

        // Logarithm term
        // Robust evaluation using log1p to handle cases where l is small.
        // log(R+l) = log(R) + log(1 + l/R) = log(R) + log1p(l/R)
        double log_term;
        double term_plus = log(R_plus) + log1p(l_plus / R_plus);
        double term_minus = log(R_minus) + log1p(l_minus / R_minus);
        log_term = term_plus - term_minus;


        // Arctangent term
        double atan_term = 0.0;
        if (abs_d > 1e-9) // Avoid division by zero for coplanar case
        {
            double num_plus = p0 * l_plus;
            double den_plus = R0 * R0 + abs_d * R_plus;
            double num_minus = p0 * l_minus;
            double den_minus = R0 * R0 + abs_d * R_minus;
            atan_term = atan2(num_plus, den_plus) - atan2(num_minus, den_minus);
        }

        phi0 += orientation_i[i] * (p0 * log_term - abs_d * atan_term);
    }

    return phi0;
}

/**
 * @brief Computes the vector potential phi_1 for a triangular source.
 *
 * This function implements the analytic formula for the vector potential from
 * a linear source distribution over a planar triangle, as defined in
 * `eq:phi1_wilton` in the documentation.
 *
 * \f[ \boldsymbol{\phi}_1(\mathbf r) = \int_{f_n} \frac{\mathbf{r}' - \mathbf{r}}{\|\mathbf r - \mathbf r'\|} dS' \f]
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @return The vector potential value.
 */
__host__ __device__ inline vec3_t calculate_phi_1(const vec3_t& r, const vec3_t tri[3])
{
    double d, p0_i[3], l_i[3][2], P_i[3][2], R0_i[3], R_i[3][2], orientation_i[3];
    vec3_t r_pi, u_i[3];

    calculateWiltonGeometricTerms(r, tri, d, r_pi, u_i, p0_i, l_i, P_i, R0_i, R_i, orientation_i);

    vec3_t phi1 = make_vec3(0.f, 0.f, 0.f);
    double abs_d = fabs(d);

    for (int i = 0; i < 3; ++i)
    {        
        double l_plus = l_i[i][1];
        double l_minus = l_i[i][0];
        double R_plus = R_i[i][1];
        double R_minus = R_i[i][0];
        double R0 = R0_i[i];

        //   numerical issues when r lies in S along the extension of an edge (Ri0=0). 
        // In such cases, the observation should be moved a very small distance away from the edge in the plane, i.e.,
        // r = r + epsilon * u^; epsilon is a very small number and u^ is the unit vector along the edge.
        if (R0_i[i] < 1e-8) {
             const double epsilon = 1e-8;
             R_plus = sqrt(pow(R_plus, 2) + pow(epsilon, 2));
             R_minus = sqrt(pow(R_minus, 2) + pow(epsilon, 2));
        }


        // Logarithm term
        double log_term;
        // Use log1p for robust evaluation when l/R is small
        double term_plus = log(R_plus) + log1p(l_plus / R_plus);
        double term_minus = log(R_minus) + log1p(l_minus / R_minus);
        log_term = term_plus - term_minus;

        // The scalar term inside the summation in eq:phi1_wilton
        double scalar_term;
        scalar_term = (R0 * R0) * log_term + (l_plus * R_plus) - (l_minus * R_minus);
        
        // Accumulate the vector component for this edge
        phi1 += u_i[i] * scalar_term;
    }

    // The formula from Wilton gives the integral of (r' - r_pi)/R.
    // We need to scale by 0.5 and then add the correction term for (r' - r)/R.
    phi1 *= 0.5;

    // The formula from Wilton gives the integral of (r' - r_pi)/R.
    // We need the integral of (r' - r)/R.
    // (r' - r) = (r' - r_pi) - (r - r_pi) = (r' - r_pi) - d*n
    // So we need to subtract the integral of (d*n)/R, which is d*n*phi_0.
    vec3_t n = cross(tri[1] - tri[0], tri[2] - tri[0]);
    n = normalize(n);
    double phi0_val = calculate_phi_0(r, tri);
    phi1 -= d * n * phi0_val;

    return phi1;
}

/**
 * @brief Computes the scalar potential phi_2 for a triangular source.
 *
 * This function implements the analytic formula for the scalar potential integral
 * over 1/R^3, as defined in `eq:phi_2_hodges` in the documentation.
 *
 * \f[ \phi_2(\mathbf r) = \int_{f_n} \frac{1}{\|\mathbf r - \mathbf r'\|^3} dS' \f]
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @return The scalar potential value.
 */
__host__ __device__ inline double calculate_phi_2(const vec3_t& r, const vec3_t tri[3])
{
    double d, p0_i[3], l_i[3][2], P_i[3][2], R0_i[3], R_i[3][2], orientation_i[3];
    vec3_t r_pi, u_i[3];

    calculateWiltonGeometricTerms(r, tri, d, r_pi, u_i, p0_i, l_i, P_i, R0_i, R_i, orientation_i);

    double phi2 = 0.0;
    double abs_d = fabs(d);

    for (int i = 0; i < 3; ++i)
    {
        double p0 = p0_i[i];
        double l_plus = l_i[i][1];
        double l_minus = l_i[i][0];
        double R_plus = R_i[i][1];
        double R_minus = R_i[i][0];
        double R0 = R0_i[i];

        double term = 0.0;
        if (abs_d > 1e-6) // General case (d is not small)
        {
            double num_plus  = p0 * l_plus;
            double den_plus  = R0 * R0 + abs_d * R_plus;
            double num_minus = p0 * l_minus;
            double den_minus = R0 * R0 + abs_d * R_minus;
            term = (1.0 / abs_d) * (atan2(num_plus, den_plus) - atan2(num_minus, den_minus));
        }
        else // Coplanar or nearly-coplanar case (d is small or zero)
        {
            // For small d, the general formula suffers from catastrophic cancellation.
            // We use the coplanar formula, which is the limit as d->0.
            double P_plus = P_i[i][1];
            double P_minus = P_i[i][0];

            if (p0 > 1e-9) // Avoid division by zero if r is on the edge line
            {
                term = -((l_plus / (p0 * P_plus)) - (l_minus / (p0 * P_minus)));
                
            }
            else
            {
                // print a warning or handle the case as needed
                // printf("Warning: P0 < 1e-9 for triangle %d. very close to edge line.\n", i);
            }
        }
        phi2 += static_cast<double>(orientation_i[i]) * term;
    }

    return phi2;
}

/**
 * @brief Computes the vector potential phi_3 for a triangular source.
 *
 * This function implements the analytic formula for the vector potential integral
 * of rho_ni/R^3, as defined in `eq:phi3_hodges` in the documentation.
 *
 * \f[ \boldsymbol{\phi}_3(\mathbf r) = \int_{f_n} \frac{\boldsymbol{\rho}_{ni}(\mathbf r')}{\|\mathbf r - \mathbf r'\|^3} dS' \f]
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @param free_vertex_idx The index (0, 1, or 2) of the free vertex associated with the basis function.
 * @return The vector potential value.
 */
__host__ __device__ inline vec3_t calculate_phi_3(const vec3_t& r, const vec3_t tri[3], int free_vertex_idx)
{
    double d, p0_i[3], l_i[3][2], P_i[3][2], R0_i[3], R_i[3][2], orientation_i[3];
    vec3_t r_pi, u_i[3];

    calculateWiltonGeometricTerms(r, tri, d, r_pi, u_i, p0_i, l_i, P_i, R0_i, R_i, orientation_i);

    // First term: -sum(u_i * log_term)
    vec3_t phi3_sum_term = make_vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < 3; ++i)
    {
        double l_plus = static_cast<double>(l_i[i][1]);
        double l_minus = static_cast<double>(l_i[i][0]);
        double R_plus = static_cast<double>(R_i[i][1]);
        double R_minus = static_cast<double>(R_i[i][0]);

        // Robust evaluation using log1p
        double log_term;
        if (R_plus < 1e-9 || R_minus < 1e-9) { log_term = 0.0; }
        else {
            log_term = (log(R_plus) + log1p(l_plus / R_plus)) - (log(R_minus) + log1p(l_minus / R_minus));
        }
        phi3_sum_term -= make_vec3(u_i[i].x, u_i[i].y, u_i[i].z) * log_term;
    }

    // Second term: (r_pi - r_free_vertex) * phi_2
    double phi2_val = calculate_phi_2(r, tri);
    const vec3_t& r_free_vertex = tri[free_vertex_idx];
    vec3_t phi3_phi2_term = make_vec3(r_pi.x - r_free_vertex.x, r_pi.y - r_free_vertex.y, r_pi.z - r_free_vertex.z) * phi2_val;

    // Combine terms
    vec3_t result = phi3_sum_term + phi3_phi2_term;
    return result;
}

/**
 * @brief Computes the composite potential Psi_1 for a half-basis function.
 *
 * This function implements the formula for the composite potential Psi_1,
 * which is used in the singularity extraction for the TL operator. It combines
 * the scalar (phi_0) and vector (phi_1) potentials, as defined in
 * `eq:Psi1_via_phi` in the documentation.
 *
 * \f[ \boldsymbol{\Psi}_1(\mathbf r) = \boldsymbol{\phi}_1(\mathbf r) + (\mathbf r - \mathbf r_{\overline{\partial e_{ni}}})\,\phi_0(\mathbf r) \f]
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @param free_vertex_idx The index (0, 1, or 2) of the free vertex associated with the basis function.
 * @return The composite potential vector.
 */
__host__ __device__ inline vec3_t calculate_Psi_1(const vec3_t& r, const vec3_t tri[3], int free_vertex_idx)
{
    // Calculate the fundamental scalar and vector potentials
    double phi0_val = calculate_phi_0(r, tri);
    vec3_t phi1_val = calculate_phi_1(r, tri);

    // Get the position of the free vertex
    const vec3_t& r_free_vertex = tri[free_vertex_idx];

    // Combine according to the formula
    vec3_t r_minus_r_free = r - r_free_vertex;
    vec3_t Psi_1 = phi1_val + r_minus_r_free * phi0_val;

    return Psi_1;
}

/**
 * @brief A helper struct to hold the results of the free-vertex-independent
 * semi-analytic potential calculations.
 */
struct WiltonPotentials {
    double  phi_0;
    vec3_t phi_1_base; // The part of phi_1 before the phi_0 correction
    double  phi_2;
    vec3_t phi_3_base; // The part of phi_3 that is independent of the free vertex
    vec3_t r_pi;       // Projection of r onto the triangle plane
    double  d;          // Signed distance from r to the triangle plane
};

/**
 * @brief A simple struct to hold the scalar and vector parts of a semi-analytic integral.
 */
struct IntegrandComponents {
    double  scalar_part;
    vec3_t vector_part;
};

/**
 * @brief Calculates Psi_1 potential using pre-computed Wilton potentials.
 *
 * Since phi_1 = phi_1_base - d*n*phi_0 and r = r_pi + d*n, the normal
 * corrections cancel in Psi_1 = phi_1 + (r - free_vertex)*phi_0. Thus
 * Psi_1 = phi_1_base + (r_pi - free_vertex)*phi_0.
 *
 * @param pots The pre-computed Wilton potentials and geometric terms.
 * @param free_vertex The free vertex of the source basis function.
 * @return The Psi_1 vector potential.
 */
__host__ __device__ inline vec3_t calculate_Psi_1(const WiltonPotentials& pots, const vec3_t& free_vertex)
{
    // Reconstruct the integral of r'/R from phi_1_base and phi_0
    const vec3_t int_r_prime_over_R = pots.phi_1_base + pots.phi_0 * pots.r_pi;
    return int_r_prime_over_R - free_vertex * pots.phi_0;
}

/**
 * @brief Computes the true phi_1 vector potential from the pre-computed Wilton potentials.
 *
 * \f[ \boldsymbol{\phi}_1(\mathbf r) = \boldsymbol{\phi}_{1,\mathrm{base}} - d \hat{\mathbf{n}} \phi_0 \f]
 *
 * @param pots The pre-computed Wilton potentials.
 * @param n The normal vector of the source triangle.
 * @return The phi_1 vector potential.
 */
__host__ __device__ inline vec3_t calculate_phi_1(const WiltonPotentials& pots, const vec3_t& n)
{
    return pots.phi_1_base - (pots.d * pots.phi_0) * n;
}

/**
 * @brief Computes the Coulomb part of the inner integral for the TK operator.
 *
 * This function implements the formula for the singular part of the inner
 * vector integral, as defined in `eq:Upsilon_coul_final_assembly`.
 *
 * \f[ \boldsymbol{\Upsilon}^{(\mathrm{coul})}(\mathbf r) = -\frac{1}{4\pi} (\mathbf{r} - \mathbf{r}_{\overline{\partial e_{ni}}}) \times \left( \boldsymbol{\phi}_3(\mathbf r) + \frac{k^2}{2} \boldsymbol{\Psi}_1(\mathbf r) \right) \f]
 *
 * @param r The observation point (field point).
 * @param tri_j The three vertices of the source triangle.
 * @param free_vertex_n_idx_local The local index (0, 1, or 2) of the source basis function's free vertex.
 * @param k The wavenumber.
 * @return The vector value of the integral.
 */
__host__ __device__ inline vec3_t calculate_Upsilon_coul(const vec3_t& r, const vec3_t tri_j[3], int free_vertex_n_idx_local, float k)
{
    // Calculate phi_3 and Psi_1 potentials
    vec3_t phi_3_val = calculate_phi_3(r, tri_j, free_vertex_n_idx_local);
    vec3_t Psi_1_val = calculate_Psi_1(r, tri_j, free_vertex_n_idx_local);

    // Combine them according to eq:TK_coul_split
    float k2_over_2 = k * k * 0.5f;
    vec3_t combined_potentials = phi_3_val + k2_over_2 * Psi_1_val;

    // Get R_i = r - r'_free_vertex
    const vec3_t& r_free_vertex_n = tri_j[free_vertex_n_idx_local];
    vec3_t R_i = r - r_free_vertex_n;

    // Calculate the cross product and scale
    vec3_t Upsilon = -1.0f*cross(R_i, combined_potentials);
    // Upsilon *= -1.f / (4.f * (float)M_PI);

    return Upsilon;
}



// add overload to calculate_Upsilon_coul that takes WiltonPotentials here
__host__ __device__ inline vec3_t calculate_Upsilon_coul(const WiltonPotentials& pots, const vec3_t& free_vertex_n, const vec3_t& r, float k)
{
    // 1. Reconstruct phi_3 from its base and phi_2
    const vec3_t phi_3 = pots.phi_3_base + (pots.r_pi - free_vertex_n) * pots.phi_2;

    // 2. Reconstruct Psi_1 using its dedicated efficient function
    const vec3_t psi_1 = calculate_Psi_1(pots, free_vertex_n);

    // 3. Combine them according to eq:TK_coul_split
    const float k2_over_2 = k * k * 0.5f;
    const vec3_t combined_potentials = phi_3 + k2_over_2 * psi_1;

    // 4. Get R_i = r - r'_free_vertex
    const vec3_t R_i = r - free_vertex_n;

    // 5. Calculate the final cross product
    return -1.0f * cross(R_i, combined_potentials);
}

/**
 * @brief Functor for the singular part of the TK operator kernel.
 *
 * This functor implements the integrand for `eq:TK_coul_outer_sum`. It calculates
 * the `Upsilon_coul` potential at the observation point and returns it as the
 * vector part of the integrand components.
 */
struct TKSingularKernel
{
    const vec3_t* source_tri;
    int source_free_vertex_idx;
    float k;

    __device__ IntegrandComponents operator()(const vec3_t& r_obs) const
    {
        IntegrandComponents components;
        vec3_t upsilon_coul = calculate_Upsilon_coul(r_obs, source_tri, source_free_vertex_idx, k);
        components.vector_part = upsilon_coul * 0.25f; // The 1/4 is from eq:TK_coul_outer_sum
        components.scalar_part = 0.f;
        return components;
    }
};

/**
 * @brief Functor for the singular part of the TL operator kernel.
 *
 * This functor implements the integrand for `eq:TL_coulomb_outer_sum`. It
 * efficiently calculates both the vector and scalar parts by computing phi_0
 * and phi_1 only once, avoiding the redundant calculations inherent in calling
 * `calculate_Psi_1` and `calculate_phi_0` separately.
 */
struct TLSingularKernel
{
    const vec3_t* source_tri;
    int source_free_vertex_idx;
    float k_inv_sq;

    __device__ IntegrandComponents operator()(const vec3_t& r_obs) const
    {
        IntegrandComponents components;
        // const vec3_t& r_free_vertex = source_tri[source_free_vertex_idx];

        vec3_t psi_1_val = calculate_Psi_1(r_obs, source_tri, source_free_vertex_idx);
        float phi_0_val = calculate_phi_0(r_obs, source_tri);

        components.vector_part = psi_1_val * 0.25f;
        components.scalar_part = -k_inv_sq * phi_0_val;
        return components;
    }
};


/**
 * @brief Computes all free-vertex-independent Wilton potentials in a single, efficient call.
 *
 * This function consolidates the calculation of phi_0, phi_2, and the base
 * parts of phi_1 and phi_3. It calls calculateWiltonGeometricTerms only once and reuses all
 * intermediate geometric and potential calculations to maximize performance.
 *
 * @param r The observation point (field point).
 * @param tri The three vertices of the source triangle.
 * @return A WiltonPotentials struct containing the calculated potentials and key geometric terms.
 */
__host__ __device__ inline WiltonPotentials calculateAllWiltonPotentials(const vec3_t& r, const vec3_t tri[3])
{
    WiltonPotentials pots;

    // 1. Calculate all geometric terms once.
    double p0_i[3], l_i[3][2], P_i[3][2], R0_i[3], R_i[3][2], orientation_i[3];
    vec3_t u_i[3];
    calculateWiltonGeometricTerms(r, tri, pots.d, pots.r_pi, u_i, p0_i, l_i, P_i, R0_i, R_i, orientation_i);

    // 2. Initialize potentials
    double  phi0_val = 0.0;
    vec3_t phi1_base_val = make_vec3(0.f, 0.f, 0.f);
    double phi2_val = 0.0;
    vec3_t phi3_base_val = make_vec3(real_t(0), real_t(0), real_t(0));
    
    double abs_d = fabs(pots.d);

    // 3. Loop over edges to compute all potentials simultaneously
    for (int i = 0; i < 3; ++i)
    {
        // --- Common terms for all potentials ---
        double l_plus = l_i[i][1];
        double l_minus = l_i[i][0];
        double R_plus = R_i[i][1];
        double R_minus = R_i[i][0];
         
        //   numerical issues when r lies in S along the extension of an edge (Ri0=0). 
        // In such cases, the observation should be moved a very small distance away from the edge in the plane, i.e.,
        // r = r + epsilon * u^; epsilon is a very small number and u^ is the unit vector along the edge.
        // if (R0_i[i] < 1e-5) {
        //     const double epsilon = 1e-5;
        //     R_plus = sqrt(pow(R_plus, 2) + pow(epsilon, 2));
        //     R_minus = sqrt(pow(R_minus, 2) + pow(epsilon, 2));
        // }

        // Robust log term calculation
        double log_term;
        double term_plus = log(R_plus) + log1p(l_plus / R_plus);
        double term_minus = log(R_minus) + log1p(l_minus / R_minus);
        log_term = term_plus - term_minus;

        // --- phi_0 Calculation ---
        double p0 = p0_i[i];
        if (p0 > 1e-9f)
        {
            double atan_term = 0.0;
            if (abs_d > 1e-9f)
            {
                double R0 = R0_i[i];
                double num_plus = p0 * l_plus;
                double den_plus = R0 * R0 + abs_d * R_plus;
                double num_minus = p0 * l_minus;
                double den_minus = R0 * R0 + abs_d * R_minus;
                atan_term = atan2(num_plus, den_plus) - atan2(num_minus, den_minus);
            }
            phi0_val += orientation_i[i] * (p0 * log_term - abs_d * atan_term);
        }

        // --- phi_1_base Calculation ---
        double R0 = R0_i[i];
        double scalar_term = (R0 * R0) * log_term + (l_plus * R_plus) - (l_minus * R_minus);
        phi1_base_val += u_i[i] * scalar_term;

        // --- phi_2 Calculation (using double for precision) ---
        double p0_d = p0_i[i];
        double term_d = 0.0;
        if (abs_d > 1e-6)
        {
            double l_plus_d = l_i[i][1];
            double l_minus_d = l_i[i][0];
            double R_plus_d = R_i[i][1];
            double R_minus_d = R_i[i][0];
            double R0_d = R0_i[i];
            double num_plus  = p0_d * l_plus_d;
            double den_plus  = R0_d * R0_d + abs_d * R_plus_d;
            double num_minus = p0_d * l_minus_d;
            double den_minus = R0_d * R0_d + abs_d * R_minus_d;
            term_d = (1.0 / abs_d) * (atan2(num_plus, den_plus) - atan2(num_minus, den_minus));
        }
        else
        {
            if (p0_d > 1e-9)
            {
                double P_plus_d = P_i[i][1];
                double P_minus_d = P_i[i][0];
                double l_plus_d = l_i[i][1];
                double l_minus_d = l_i[i][0];
                term_d = -((l_plus_d / (p0_d * P_plus_d)) - (l_minus_d / (p0_d * P_minus_d)));
            }
        }
        phi2_val += static_cast<double>(orientation_i[i]) * term_d;

        // --- phi_3_base Calculation (using double for precision) ---
        double log_term_d = static_cast<double>(log_term);
        phi3_base_val -= make_vec3(u_i[i].x, u_i[i].y, u_i[i].z) * log_term_d;
    }

    // 4. Finalize and store results
    pots.phi_0 = phi0_val;
    pots.phi_1_base = phi1_base_val * 0.5f;
    pots.phi_2 = phi2_val;
    pots.phi_3_base = make_vec3(phi3_base_val.x, phi3_base_val.y, phi3_base_val.z);

    return pots;
}
#endif // SEMI_ANALYTIC_INTEGRALS_CUH
