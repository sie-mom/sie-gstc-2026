#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/cuda/math_utils.cuh"
#include <map>
#include <cmath>
#include <algorithm>

namespace sie_mom {
namespace mesh {

SphericalSurface::SphericalSurface(float radius, int refinement_level)
    : radius_(radius), refinement_level_(refinement_level) {
    generate();
}

const MeshData& SphericalSurface::getMeshData() const {
    return data_;
}

void SphericalSurface::generate() {
    // Canonical icosahedron (12 unique vertices) around origin before scaling.
    const real_t phi = (real_t(1) + sqrt(real_t(5))) / real_t(2);
    std::vector<vec3_t> V = {
        make_vec3(-1,  phi,  0), make_vec3(1,  phi,  0), make_vec3(-1, -phi,  0), make_vec3(1, -phi,  0),
        make_vec3(0, -1,  phi), make_vec3(0, 1,  phi), make_vec3(0, -1, -phi), make_vec3(0, 1, -phi),
        make_vec3(phi,  0, -1), make_vec3(phi,  0, 1), make_vec3(-phi,  0, -1), make_vec3(-phi,  0, 1)
    };

    for (auto& v : V) {
        v = normalize(v);
    }

    std::vector<int3> F = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };

    // Subdivide
    for (int level = 0; level < refinement_level_; ++level) {
        std::map<std::pair<int, int>, int> midpoint_cache;
        std::vector<int3> new_faces;
        
        auto midpoint = [&](int i, int j) -> int {
            auto key = std::minmax(i, j);
            if (midpoint_cache.count(key)) {
                return midpoint_cache[key];
            }
            vec3_t m = (V[i] + V[j]) * real_t(0.5);
            m = normalize(m);
            int idx = V.size();
            V.push_back(m);
            midpoint_cache[key] = idx;
            return idx;
        };

        for (const auto& face : F) {
            int a = face.x;
            int b = face.y;
            int c = face.z;
            int ab = midpoint(a, b);
            int bc = midpoint(b, c);
            int ca = midpoint(c, a);
            new_faces.push_back({a, ab, ca});
            new_faces.push_back({ab, b, bc});
            new_faces.push_back({ca, bc, c});
            new_faces.push_back({ab, bc, ca});
        }
        F = new_faces;
    }

    for (auto& v : V) {
        v = v * radius_;
    }

    data_.vertices = V;
    data_.faces = F;

    orientFacesCcwOutward();
    buildEdgesAndMapping();
}

void SphericalSurface::orientFacesCcwOutward() {
    vec3_t center = make_vec3(0, 0, 0); // Assuming centered at origin
    for (auto& face : data_.faces) {
        vec3_t& v0 = data_.vertices[face.x];
        vec3_t& v1 = data_.vertices[face.y];
        vec3_t& v2 = data_.vertices[face.z];
        
        vec3_t n = cross(v1 - v0, v2 - v0);
        vec3_t centroid = (v0 + v1 + v2) / real_t(3);
        vec3_t outward = centroid - center;
        
        if (dot(n, outward) < 0) {
            std::swap(face.y, face.z);
        }
    }
}

void SphericalSurface::buildEdgesAndMapping() {
    std::map<std::pair<int, int>, int> edge_dict;
    data_.faces_edges.resize(data_.faces.size());
    int next_id = 0;

    for (size_t fi = 0; fi < data_.faces.size(); ++fi) {
        const auto& face = data_.faces[fi];
        int tri_verts[] = {face.x, face.y, face.z};
        int3 face_edges_indices;
        
        for (int ei = 0; ei < 3; ++ei) {
            int u = tri_verts[ei];
            int v = tri_verts[(ei + 1) % 3];
            auto key = std::minmax(u, v);
            
            auto it = edge_dict.find(key);
            if (it == edge_dict.end()) {
                int edge_idx = next_id++;
                edge_dict[key] = edge_idx;
                if (ei == 0) face_edges_indices.x = edge_idx;
                else if (ei == 1) face_edges_indices.y = edge_idx;
                else face_edges_indices.z = edge_idx;
            } else {
                if (ei == 0) face_edges_indices.x = it->second;
                else if (ei == 1) face_edges_indices.y = it->second;
                else face_edges_indices.z = it->second;
            }
        }
        data_.faces_edges[fi] = face_edges_indices;
    }

    data_.edges.resize(edge_dict.size());
    for (const auto& pair : edge_dict) {
        data_.edges[pair.second] = {pair.first.first, pair.first.second};
    }
}

} // namespace mesh
} // namespace sie_mom
