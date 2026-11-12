#ifndef SIE_MOM_MESH_UTILS_HPP
#define SIE_MOM_MESH_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "sie_mom/mesh/spherical_surface.hpp" // For MeshData, vec3_t
#include "sie_mom/cuda/platform.cuh" // For vec3_t

namespace sie_mom {
namespace mesh {

/**
 * @brief Exports face centroids (theta, phi) to a generic CSV file.
 * Format: theta, phi
 * Used for external processing of spatially varying properties.
 * 
 * @param filename The output file path.
 * @param mesh The mesh data containing vertices and faces.
 */
inline void export_mesh_centroids(
    const std::string& filename, 
    const sie_mom::mesh::MeshData& mesh
) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::ofstream out(filename);
    if(!out.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    // Set precision high enough for float round-tripping
    out << std::scientific << std::setprecision(8);
    
    // Header for clarity
    out << "theta,phi\n";
    
    for(const auto& face : mesh.faces) {
        // Compute Centroid
        vec3_t v0 = mesh.vertices[face.x];
        vec3_t v1 = mesh.vertices[face.y];
        vec3_t v2 = mesh.vertices[face.z];
        
        double cx = (v0.x + v1.x + v2.x) / 3.0;
        double cy = (v0.y + v1.y + v2.y) / 3.0;
        double cz = (v0.z + v1.z + v2.z) / 3.0;
        
        double r = sqrt(cx*cx + cy*cy + cz*cz);
        
        // Compute Spherical Coords
        // Protect against numerical noise at poles/origin
        if(r < 1e-12) r = 1e-12;
        
        double val = cz / r;
        if(val > 1.0) val = 1.0;
        if(val < -1.0) val = -1.0;
        double theta = acos(val);
        
        double phi = atan2(cy, cx);
        if(phi < 0) phi += 2.0 * pi; // Normalize 0 to 2pi
        
        out << theta << "," << phi << "\n";
    }
    out.close();
}

/**
 * @brief Exports the mesh vertices and faces to CSV files.
 * Useful for plotting the mesh geometry in external tools (e.g., Python).
 * 
 * @param prefix The output file prefix (e.g., "sphere" creates "sphere_vertices.csv").
 * @param mesh The mesh data containing vertices and faces.
 */
inline void export_mesh_data(
    const std::string& prefix, 
    const sie_mom::mesh::MeshData& mesh
) {
    // Export Vertices
    std::ofstream v_out(prefix + "_vertices.csv");
    if(v_out.is_open()) {
        v_out << std::scientific << std::setprecision(6);
        v_out << "x,y,z\n";
        for(const auto& v : mesh.vertices) {
            v_out << v.x << "," << v.y << "," << v.z << "\n";
        }
    }

    // Export Faces
    std::ofstream f_out(prefix + "_faces.csv");
    if(f_out.is_open()) {
        f_out << "v0,v1,v2\n";
        for(const auto& f : mesh.faces) {
            f_out << f.x << "," << f.y << "," << f.z << "\n";
        }
    }
}

/**
 * @brief Computes the centroid for every triangular face in the mesh.
 */
inline std::vector<vec3_t> compute_face_centroids(const sie_mom::mesh::MeshData& mesh) {
    std::vector<vec3_t> centroids;
    centroids.reserve(mesh.faces.size());
    for (const auto& face : mesh.faces) {
        const auto& v0 = mesh.vertices[face.x];
        const auto& v1 = mesh.vertices[face.y];
        const auto& v2 = mesh.vertices[face.z];
        centroids.push_back(make_vec3(
            (v0.x + v1.x + v2.x) / real_t(3),
            (v0.y + v1.y + v2.y) / real_t(3),
            (v0.z + v1.z + v2.z) / real_t(3)
        ));
    }
    return centroids;
}

/**
 * @brief Offsets face centroids along their radial direction by a given distance.
 * Positive offsets point outward; negative offsets point inward.
 */
inline std::vector<vec3_t> offset_points(const sie_mom::mesh::MeshData& mesh, real_t offset) {
    std::vector<vec3_t> points;
    points.reserve(mesh.faces.size());
    for (const auto& face : mesh.faces) {
        const auto& v0 = mesh.vertices[face.x];
        const auto& v1 = mesh.vertices[face.y];
        const auto& v2 = mesh.vertices[face.z];
        const vec3_t centroid = make_vec3(
            (v0.x + v1.x + v2.x) / real_t(3),
            (v0.y + v1.y + v2.y) / real_t(3),
            (v0.z + v1.z + v2.z) / real_t(3)
        );
        const real_t radius = std::sqrt(
            centroid.x * centroid.x + centroid.y * centroid.y + centroid.z * centroid.z
        );
        if (!(radius > std::abs(offset))) {
            throw std::invalid_argument("offset must be smaller than every centroid radius");
        }
        const real_t scale = real_t(1) + offset / radius;
        points.push_back(make_vec3(
            centroid.x * scale,
            centroid.y * scale,
            centroid.z * scale
        ));
    }
    return points;
}

/**
 * @brief Infers the spherical radius of a mesh and verifies vertices lie on a common sphere.
 */
inline real_t infer_sphere_radius(
    const sie_mom::mesh::MeshData& mesh,
    real_t max_relative_spread = real_t(1e-4)
) {
    long double radius_sum = 0.0L;
    real_t minimum_radius = std::numeric_limits<real_t>::max();
    real_t maximum_radius = real_t(0);
    for (const auto& vertex : mesh.vertices) {
        const real_t radius = std::sqrt(
            vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z
        );
        if (!(radius > real_t(0)) || !std::isfinite(static_cast<double>(radius))) {
            throw std::runtime_error("sphere mesh contains a non-finite or zero-radius vertex");
        }
        radius_sum += static_cast<long double>(radius);
        minimum_radius = std::min(minimum_radius, radius);
        maximum_radius = std::max(maximum_radius, radius);
    }
    const real_t radius = static_cast<real_t>(
        radius_sum / static_cast<long double>(mesh.vertices.size())
    );
    const real_t relative_spread = (maximum_radius - minimum_radius) / radius;
    if (relative_spread > max_relative_spread) {
        throw std::runtime_error("mesh vertices do not lie on a common spherical radius");
    }
    return radius;
}

} // namespace mesh
} // namespace sie_mom

#endif // SIE_MOM_MESH_UTILS_HPP
