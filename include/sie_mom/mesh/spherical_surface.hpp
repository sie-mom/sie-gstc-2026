#ifndef SPHERICAL_SURFACE_HPP
#define SPHERICAL_SURFACE_HPP

#include <vector>
#include "sie_mom/cuda/platform.cuh" // For float3, int2, int3

namespace sie_mom {
namespace mesh {

struct MeshData {
    std::vector<vec3_t> vertices;      // (nV, 3)
    std::vector<int3>   faces;         // (nF, 3) vertex indices
    std::vector<int2>   edges;         // (nE, 2) unique edges
    std::vector<int3>   faces_edges;   // (nF, 3) edge indices
};

class SphericalSurface {
public:
    SphericalSurface(float radius, int refinement_level);

    const MeshData& getMeshData() const;

private:
    void generate();
    void orientFacesCcwOutward();
    void buildEdgesAndMapping();

    float radius_;
    int refinement_level_;
    MeshData data_;
};

} // namespace mesh
} // namespace sie_mom

#endif // SPHERICAL_SURFACE_HPP
