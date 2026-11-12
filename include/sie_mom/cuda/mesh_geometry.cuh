#ifndef MESH_GEOMETRY_CUH
#define MESH_GEOMETRY_CUH

#include "platform.cuh"
#include "math_utils.cuh"

__host__ __device__ inline real_t getTriangleArea(const vec3_t tri_vertices[3])
{
    const vec3_t v0 = tri_vertices[0];
    const vec3_t v1 = tri_vertices[1];
    const vec3_t v2 = tri_vertices[2];
    
    const vec3_t edge1 = v1 - v0;
    const vec3_t edge2 = v2 - v0;
    
    return real_t(0.5) * length(cross(edge1, edge2));
}

/**
 * @brief Computes the signed edge length and free vertex for a triangle edge.
 *
 * This overload operates purely on local triangle data. The caller provides the
 * three global vertex indices that define the triangle and the already loaded
 * vertex positions, so this helper no longer depends on global mesh arrays.
 *
 * The local edge indexing follows the usual convention for vertices `[v0, v1, v2]`:
 * @verbatim
 *                              v2 (free for edge 0)
 *                             /  \
 *                            /    \
 *                    edge 2 /      \ edge 1
 *                          /        \
 *                         /__________\
 *    (free for edge 1) v0 --edge 0-- v1 (free for edge 2)
 * @endverbatim
 * - `local_edge = 0`: corresponds to the edge between `v0` and `v1`.
 * - `local_edge = 1`: corresponds to the edge between `v1` and `v2`.
 * - `local_edge = 2`: corresponds to the edge between `v2` and `v0`.
 *
 * The "signed length" is a convention used in RWG basis functions (see \f$e_{ni}\f$ in the theory).
 * It is the geometric length of the edge multiplied with a sign determined by the global
 * indices of its endpoints (`vA`, `vB`): `sign = +1` if `vA > vB`, and `-1` otherwise.
 * 
 * @param tri_vertex_indices Global vertex indices of the triangle in CCW order.
 * @param tri_vertices        Vertex positions corresponding to @p tri_vertex_indices.
 * @param local_edge          Local edge index (0, 1, or 2).
 * @param[out] r_free_out     Free vertex position for the requested edge.
 * @return Signed edge length, following the RWG sign convention (positive if the
 *         first endpoint index is greater than the second, negative otherwise).
 */
__host__ __device__ inline real_t getEdgeInfo(
    const int tri_vertex_indices[3],
    const vec3_t tri_vertices[3],
    int local_edge,
    vec3_t& r_free_out)
{
    int idxA, idxB, idxFree;
    if (local_edge == 0)      { idxA = 0; idxB = 1; idxFree = 2; }
    else if (local_edge == 1) { idxA = 1; idxB = 2; idxFree = 0; }
    else                      { idxA = 2; idxB = 0; idxFree = 1; }

    const vec3_t& rA = tri_vertices[idxA];
    const vec3_t& rB = tri_vertices[idxB];
    r_free_out = tri_vertices[idxFree];

    const vec3_t edge_vec = make_vec3(rB.x - rA.x, rB.y - rA.y, rB.z - rA.z);
    const real_t length = sqrt(edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y + edge_vec.z * edge_vec.z);

    const int vA = tri_vertex_indices[idxA];
    const int vB = tri_vertex_indices[idxB];
    // Compute sign
    // The absolute sign convention is arbitrary. What is crucial is that for any
    // global edge, the two faces that share it will traverse it in opposite
    // directions (e.g., vA->vB vs vB->vA) due to CCW face orientation. This
    // guarantees that one face gets a +1 sign and the other gets a -1,
    // which is the required behavior for the RWG basis function.
    const real_t sign = (vA > vB) ? real_t(1) : real_t(-1);

    return sign * length;
}

/**
 * @brief Computes the signed edge length and provides the local index of the free vertex.
 *
 * This is an overload of `getEdgeInfo` that returns the local index (0, 1, or 2)
 * of the free vertex instead of its 3D coordinates. This is useful when the
 * index is needed for subsequent logic, rather than the vertex position itself.
 *
 * @param tri_vertex_indices Global vertex indices of the triangle in CCW order.
 * @param tri_vertices        Vertex positions corresponding to @p tri_vertex_indices.
 * @param local_edge          Local edge index (0, 1, or 2).
 * @param[out] free_vertex_idx_out The local index of the free vertex for the requested edge.
 * @return Signed edge length, following the RWG sign convention.
 */
__host__ __device__ inline real_t getEdgeInfo(
    const int tri_vertex_indices[3],
    const vec3_t tri_vertices[3],
    int local_edge,
    int& free_vertex_idx_out)
{
    int idxA, idxB, idxFree;
    if (local_edge == 0)      { idxA = 0; idxB = 1; idxFree = 2; }
    else if (local_edge == 1) { idxA = 1; idxB = 2; idxFree = 0; }
    else                      { idxA = 2; idxB = 0; idxFree = 1; }

    free_vertex_idx_out = idxFree;

    const vec3_t& rA = tri_vertices[idxA];
    const vec3_t& rB = tri_vertices[idxB];
    const vec3_t edge_vec = make_vec3(rB.x - rA.x, rB.y - rA.y, rB.z - rA.z);
    const real_t length = sqrt(edge_vec.x * edge_vec.x + edge_vec.y * edge_vec.y + edge_vec.z * edge_vec.z);

    const int vA = tri_vertex_indices[idxA];
    const int vB = tri_vertex_indices[idxB];
    const real_t sign = (vA > vB) ? real_t(1) : real_t(-1);

    return sign * length;
}

__host__ __device__ inline bool shareVertex(const int test_tri_indices[3], const int source_tri_indices[3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (test_tri_indices[i] == source_tri_indices[j]) {
                return true;
            }
        }
    }
    return false;
}

enum AdjacencyType {
    COINCIDENT,
    EDGE_ADJACENT,
    VERTEX_ADJACENT,
    NON_ADJACENT
};

__host__ __device__ inline AdjacencyType getAdjacencyInfo(const int test_tri_indices[3], const int source_tri_indices[3], int& local_edge_i, int& local_edge_j, int& local_vertex_i, int& local_vertex_j)
{
    if (test_tri_indices[0] == source_tri_indices[0] &&
        test_tri_indices[1] == source_tri_indices[1] &&
        test_tri_indices[2] == source_tri_indices[2]) {
        return COINCIDENT;
    }

    int shared_vertices_indices_i[3];
    int shared_vertices_indices_j[3];
    int shared_count = 0;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (test_tri_indices[i] == source_tri_indices[j]) {
                if (shared_count < 3) {
                    shared_vertices_indices_i[shared_count] = i;
                    shared_vertices_indices_j[shared_count] = j;
                }
                shared_count++;
            }
        }
    }

    if (shared_count >= 2) { // Should be exactly 2 for a conforming mesh
        // To find the local edge index, we need the one vertex NOT in the shared list.
        // The local edge index is opposite the free vertex.
        // e.g., if shared vertices are v0 and v1, the free vertex is v2, and the edge is local edge 2.
        // A bit of logic to find the missing index (0, 1, or 2)
        int s0_i = shared_vertices_indices_i[0];
        int s1_i = shared_vertices_indices_i[1];
        // The missing index is 3 - s0 - s1. (e.g., if s0=0, s1=1, missing=2. if s0=0, s1=2, missing=1)
        local_edge_i = 3 - s0_i - s1_i;

        // Repeat for face_j
        int s0_j = shared_vertices_indices_j[0];
        int s1_j = shared_vertices_indices_j[1];
        local_edge_j = 3 - s0_j - s1_j;
        
        return EDGE_ADJACENT;
    }
    if (shared_count == 1) {
        local_vertex_i = shared_vertices_indices_i[0];
        local_vertex_j = shared_vertices_indices_j[0];
        return VERTEX_ADJACENT;
    }
    
    return NON_ADJACENT;
}


/**
 * @brief Maps a 2D quadrature point from a reference triangle to a 3D physical triangle.
 *
 * This function takes a quadrature point defined by barycentric coordinates `(xi, eta)`
 * on a 2D reference triangle and maps it to its corresponding location on a
 * physical triangle in 3D space. The physical triangle is defined by its three
 * vertices, making this function self-contained and independent of global mesh arrays.
 *
 * This is the primary method for converting quadrature coordinates to world-space
 * positions for integration.
 *
 * @param tri An array containing the three `vec3_t` vertices of the physical triangle, `[v0, v1, v2]`.
 * @param xi The barycentric coordinate corresponding to vertex `v1` (often called beta).
 * @param eta The barycentric coordinate corresponding to vertex `v2` (often called gamma).
 * @param[out] r_out A reference to a `vec3_t` where the mapped 3D point will be stored.
 *
 * @note The barycentric coordinate for the first vertex `v0` (alpha or zeta) is implicitly
 *       calculated as `zeta = 1.0 - xi - eta`. The mapping is performed as:
 *       `P = v0*zeta + v1*xi + v2*eta`. This convention must be consistent with
 *       the quadrature point generation logic (e.g., the output of Dunavant rules).
 */
__host__ __device__ inline void mapTriQuadPoint(const vec3_t tri[3], real_t xi, real_t eta,
                                                vec3_t& r_out)
{
    const vec3_t& v0 = tri[0];
    const vec3_t& v1 = tri[1];
    const vec3_t& v2 = tri[2];

    const real_t zeta = real_t(1) - xi - eta;
    r_out.x = v0.x * zeta + v1.x * xi + v2.x * eta;
    r_out.y = v0.y * zeta + v1.y * xi + v2.y * eta;
    r_out.z = v0.z * zeta + v1.z * xi + v2.z * eta;
}

__host__ __device__ inline void mapEdgeQuadPoint(const vec3_t& vA, const vec3_t& vB, real_t t, vec3_t& r_out)
{
    // The Gauss-Legendre quadrature is defined on the reference interval [0, 1],
    // so t directly serves as the interpolation parameter.
    r_out.x = vA.x + t * (vB.x - vA.x);
    r_out.y = vA.y + t * (vB.y - vA.y);
    r_out.z = vA.z + t * (vB.z - vA.z);
}

#endif // MESH_GEOMETRY_CUH
