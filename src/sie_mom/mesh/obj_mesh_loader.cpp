#include "sie_mom/mesh/obj_mesh_loader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sie_mom {
namespace mesh {

namespace {

struct EdgeOccurrence {
    int face;
    int direction;
};

int parse_vertex_index(
    const std::string& token,
    std::size_t vertices_seen,
    const std::string& filepath,
    std::size_t line_number
) {
    const std::size_t slash = token.find('/');
    const std::string index_token = token.substr(0, slash);
    if (index_token.empty()) {
        throw std::runtime_error(
            filepath + ":" + std::to_string(line_number) +
            ": face has an empty vertex index"
        );
    }

    std::size_t parsed = 0;
    int raw_index = 0;
    try {
        raw_index = std::stoi(index_token, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            filepath + ":" + std::to_string(line_number) +
            ": invalid face vertex index '" + index_token + "'"
        );
    }
    if (parsed != index_token.size() || raw_index == 0) {
        throw std::runtime_error(
            filepath + ":" + std::to_string(line_number) +
            ": invalid face vertex index '" + index_token + "'"
        );
    }

    if (raw_index > 0) {
        return raw_index - 1;
    }
    const long long resolved =
        static_cast<long long>(vertices_seen) + static_cast<long long>(raw_index);
    if (resolved < 0 || resolved >= static_cast<long long>(vertices_seen)) {
        throw std::runtime_error(
            filepath + ":" + std::to_string(line_number) +
            ": relative face vertex index is out of range"
        );
    }
    return static_cast<int>(resolved);
}

void append_problem(std::ostringstream& problems, bool& first, const std::string& problem) {
    if (!first) {
        problems << "; ";
    }
    problems << problem;
    first = false;
}

} // namespace

ObjMeshLoader::ObjMeshLoader(const std::string& filepath) {
    parseFile(filepath);
    validateMesh(filepath);
    buildEdgesAndMapping();
}

void ObjMeshLoader::parseFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open OBJ file: " + filepath);
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        std::istringstream stream(line);
        std::string prefix;
        stream >> prefix;
        if (prefix.empty() || prefix.front() == '#') {
            continue;
        }

        if (prefix == "v") {
            vec3_t vertex;
            if (!(stream >> vertex.x >> vertex.y >> vertex.z)) {
                throw std::runtime_error(
                    filepath + ":" + std::to_string(line_number) +
                    ": vertex requires three coordinates"
                );
            }
            mesh_data_.vertices.push_back(vertex);
        } else if (prefix == "f") {
            std::vector<std::string> tokens;
            std::string token;
            while (stream >> token) {
                if (!token.empty() && token.front() == '#') {
                    break;
                }
                tokens.push_back(token);
            }
            if (tokens.size() != 3) {
                throw std::runtime_error(
                    filepath + ":" + std::to_string(line_number) +
                    ": only triangular faces are supported"
                );
            }
            mesh_data_.faces.push_back({
                parse_vertex_index(tokens[0], mesh_data_.vertices.size(), filepath, line_number),
                parse_vertex_index(tokens[1], mesh_data_.vertices.size(), filepath, line_number),
                parse_vertex_index(tokens[2], mesh_data_.vertices.size(), filepath, line_number),
            });
        }
    }
}

void ObjMeshLoader::validateMesh(const std::string& filepath) const {
    const auto& vertices = mesh_data_.vertices;
    const auto& faces = mesh_data_.faces;
    if (vertices.empty()) {
        throw std::runtime_error("Invalid OBJ mesh " + filepath + ": no vertices");
    }
    if (faces.empty()) {
        throw std::runtime_error("Invalid OBJ mesh " + filepath + ": no faces");
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    for (const auto& vertex : vertices) {
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
            !std::isfinite(vertex.z)) {
            throw std::runtime_error(
                "Invalid OBJ mesh " + filepath + ": non-finite vertex coordinate"
            );
        }
        min_x = std::min(min_x, static_cast<double>(vertex.x));
        min_y = std::min(min_y, static_cast<double>(vertex.y));
        min_z = std::min(min_z, static_cast<double>(vertex.z));
        max_x = std::max(max_x, static_cast<double>(vertex.x));
        max_y = std::max(max_y, static_cast<double>(vertex.y));
        max_z = std::max(max_z, static_cast<double>(vertex.z));
    }
    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;
    double scale = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (scale == 0.0) {
        scale = 1.0;
    }
    const double area_tolerance = 1e-10 * scale * scale;
    const double volume_tolerance = 1e-10 * scale * scale * scale;

    std::set<std::array<int, 3>> unique_faces;
    std::vector<bool> vertex_used(vertices.size(), false);
    std::map<std::pair<int, int>, std::vector<EdgeOccurrence>> edge_occurrences;
    int degenerate_faces = 0;
    int duplicate_faces = 0;
    double signed_volume_six = 0.0;

    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
        const int3& face = faces[face_index];
        const int indices[3] = {face.x, face.y, face.z};
        for (const int index : indices) {
            if (index < 0 || static_cast<std::size_t>(index) >= vertices.size()) {
                throw std::runtime_error(
                    "Invalid OBJ mesh " + filepath +
                    ": face contains an out-of-range vertex index"
                );
            }
            vertex_used[static_cast<std::size_t>(index)] = true;
        }

        std::array<int, 3> face_key = {face.x, face.y, face.z};
        std::sort(face_key.begin(), face_key.end());
        if (!unique_faces.insert(face_key).second) {
            ++duplicate_faces;
        }

        const vec3_t& v0 = vertices[static_cast<std::size_t>(face.x)];
        const vec3_t& v1 = vertices[static_cast<std::size_t>(face.y)];
        const vec3_t& v2 = vertices[static_cast<std::size_t>(face.z)];
        const double e1x = static_cast<double>(v1.x) - v0.x;
        const double e1y = static_cast<double>(v1.y) - v0.y;
        const double e1z = static_cast<double>(v1.z) - v0.z;
        const double e2x = static_cast<double>(v2.x) - v0.x;
        const double e2y = static_cast<double>(v2.y) - v0.y;
        const double e2z = static_cast<double>(v2.z) - v0.z;
        const double cross_x = e1y * e2z - e1z * e2y;
        const double cross_y = e1z * e2x - e1x * e2z;
        const double cross_z = e1x * e2y - e1y * e2x;
        const double doubled_area =
            std::sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
        if (face.x == face.y || face.y == face.z || face.z == face.x ||
            doubled_area <= area_tolerance) {
            ++degenerate_faces;
        }

        signed_volume_six +=
            static_cast<double>(v0.x) *
                (static_cast<double>(v1.y) * v2.z -
                 static_cast<double>(v1.z) * v2.y) -
            static_cast<double>(v0.y) *
                (static_cast<double>(v1.x) * v2.z -
                 static_cast<double>(v1.z) * v2.x) +
            static_cast<double>(v0.z) *
                (static_cast<double>(v1.x) * v2.y -
                 static_cast<double>(v1.y) * v2.x);

        for (int local_edge = 0; local_edge < 3; ++local_edge) {
            const int start = indices[local_edge];
            const int end = indices[(local_edge + 1) % 3];
            const std::pair<int, int> edge = std::minmax(start, end);
            const int direction = (start == edge.first) ? 1 : -1;
            edge_occurrences[edge].push_back(
                {static_cast<int>(face_index), direction}
            );
        }
    }

    const int orphan_vertices = static_cast<int>(
        std::count(vertex_used.begin(), vertex_used.end(), false)
    );
    int boundary_edges = 0;
    int nonmanifold_edges = 0;
    int orientation_failures = 0;
    std::vector<std::vector<int>> adjacency(faces.size());
    for (const auto& entry : edge_occurrences) {
        const auto& occurrences = entry.second;
        if (occurrences.size() == 1) {
            ++boundary_edges;
        } else if (occurrences.size() > 2) {
            ++nonmanifold_edges;
        }
        if (occurrences.size() == 2) {
            if (occurrences[0].direction == occurrences[1].direction) {
                ++orientation_failures;
            }
            const int first_face = occurrences[0].face;
            const int second_face = occurrences[1].face;
            adjacency[static_cast<std::size_t>(first_face)].push_back(second_face);
            adjacency[static_cast<std::size_t>(second_face)].push_back(first_face);
        }
    }

    int connected_components = 0;
    std::vector<bool> visited(faces.size(), false);
    for (std::size_t start = 0; start < faces.size(); ++start) {
        if (visited[start]) {
            continue;
        }
        ++connected_components;
        std::queue<int> pending;
        pending.push(static_cast<int>(start));
        visited[start] = true;
        while (!pending.empty()) {
            const int face = pending.front();
            pending.pop();
            for (const int neighbor : adjacency[static_cast<std::size_t>(face)]) {
                if (!visited[static_cast<std::size_t>(neighbor)]) {
                    visited[static_cast<std::size_t>(neighbor)] = true;
                    pending.push(neighbor);
                }
            }
        }
    }

    std::ostringstream problems;
    bool first_problem = true;
    if (orphan_vertices != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(orphan_vertices) + " unreferenced vertex/vertices"
        );
    }
    if (degenerate_faces != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(degenerate_faces) + " degenerate face(s)"
        );
    }
    if (duplicate_faces != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(duplicate_faces) + " duplicate face(s)"
        );
    }
    if (boundary_edges != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(boundary_edges) + " boundary edge(s)"
        );
    }
    if (nonmanifold_edges != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(nonmanifold_edges) + " non-manifold edge(s)"
        );
    }
    if (connected_components != 1) {
        append_problem(
            problems,
            first_problem,
            std::to_string(connected_components) +
                " connected component(s), expected 1"
        );
    }
    if (orientation_failures != 0) {
        append_problem(
            problems,
            first_problem,
            std::to_string(orientation_failures) +
                " inconsistently oriented interior edge(s)"
        );
    }
    if (signed_volume_six / 6.0 <= volume_tolerance) {
        append_problem(
            problems,
            first_problem,
            "signed volume is not positive; face winding is not outward"
        );
    }
    if (!first_problem) {
        throw std::runtime_error(
            "Invalid OBJ mesh " + filepath + ": " + problems.str()
        );
    }
}

void ObjMeshLoader::buildEdgesAndMapping() {
    std::map<std::pair<int, int>, int> edge_map;
    int edge_count = 0;
    mesh_data_.faces_edges.resize(mesh_data_.faces.size());

    for (std::size_t face_index = 0; face_index < mesh_data_.faces.size(); ++face_index) {
        const int3& face = mesh_data_.faces[face_index];
        const int face_vertices[3] = {face.x, face.y, face.z};
        for (int local_edge = 0; local_edge < 3; ++local_edge) {
            const std::pair<int, int> edge = std::minmax(
                face_vertices[local_edge], face_vertices[(local_edge + 1) % 3]
            );
            const auto existing = edge_map.find(edge);
            if (existing == edge_map.end()) {
                edge_map[edge] = edge_count;
                mesh_data_.edges.push_back({edge.first, edge.second});
                (&mesh_data_.faces_edges[face_index].x)[local_edge] = edge_count;
                ++edge_count;
            } else {
                (&mesh_data_.faces_edges[face_index].x)[local_edge] = existing->second;
            }
        }
    }
}

} // namespace mesh
} // namespace sie_mom
