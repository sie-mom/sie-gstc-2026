#include "catch.hpp"

#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/mesh/spherical_surface.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

namespace {

using sie_mom::mesh::SphericalSurface;
using sie_mom::mesh::MeshData;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegPerRad = 180.0f / kPi;

float angle_degrees(const vec3_t& u, const vec3_t& v)
{
    const float nu = length(u);
    const float nv = length(v);
    if (nu == 0.0f || nv == 0.0f)
    {
        return 0.0f;
    }
    float cosang = dot(u, v) / (nu * nv);
    cosang = std::max(-1.0f, std::min(1.0f, cosang));
    return std::acos(cosang) * kDegPerRad;
}

vec3_t normalize_checked(const vec3_t& v)
{
    const float len = length(v);
    if (len == 0.0f)
    {
        return make_vec3(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / len);
}

} // namespace

TEST_CASE("SphericalSurface basic invariants", "[Mesh][Icosphere]")
{
    SphericalSurface sphere(/*radius=*/1.0f, /*refinement_level=*/2);
    const MeshData& mesh = sphere.getMeshData();

    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;
    const auto& edges = mesh.edges;
    const auto& faces_edges = mesh.faces_edges;

    REQUIRE(!vertices.empty());
    REQUIRE(!faces.empty());
    REQUIRE(!edges.empty());
    REQUIRE(vertices.size() >= 12);

    vec3_t center = make_vec3(0.0f, 0.0f, 0.0f);
    for (const auto& v : vertices)
    {
        const bool has_nan = std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z);
        REQUIRE_FALSE(has_nan);
        center += v;
    }
    center = center * (1.0f / static_cast<float>(vertices.size()));

    SECTION("Vertices lie on radius within tolerance")
    {
        for (const auto& v : vertices)
        {
            const float radial = length(v - center);
            CHECK(radial == Approx(1.0f).margin(5e-4f));
        }
    }

    SECTION("Euler characteristic equals 2")
    {
        const long euler = static_cast<long>(vertices.size())
                          - static_cast<long>(edges.size())
                          + static_cast<long>(faces.size());
        CHECK(euler == 2);
    }

    SECTION("Each edge shared by two faces")
    {
        std::vector<int> incidence(edges.size(), 0);
        for (const auto& fe : faces_edges)
        {
            ++incidence[fe.x];
            ++incidence[fe.y];
            ++incidence[fe.z];
        }
        for (const int count : incidence)
        {
            CHECK(count == 2);
        }
    }

    SECTION("Triangle angles lie in near-equilateral range")
    {
        float min_angle = 180.0f;
        float max_angle = 0.0f;
        for (const auto& tri : faces)
        {
            const vec3_t& a = vertices[tri.x];
            const vec3_t& b = vertices[tri.y];
            const vec3_t& c = vertices[tri.z];

            const vec3_t ab = b - a;
            const vec3_t bc = c - b;
            const vec3_t ca = a - c;

            const float A = angle_degrees(-ab, ca);
            const float B = angle_degrees(-bc, ab);
            const float C = 180.0f - A - B;

            min_angle = std::min({min_angle, A, B, C});
            max_angle = std::max({max_angle, A, B, C});
        }
        CHECK(min_angle > 50.0f);
        CHECK(max_angle < 75.0f);
    }
}

TEST_CASE("SphericalSurface faces oriented outward", "[Mesh][Icosphere]")
{
    SphericalSurface sphere(1.0f, 2);
    const MeshData& mesh = sphere.getMeshData();
    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;

    vec3_t center = make_vec3(0.0f, 0.0f, 0.0f);
    for (const auto& v : vertices)
    {
        center += v;
    }
    center = center * (1.0f / static_cast<float>(vertices.size()));

    for (const auto& tri : faces)
    {
        const vec3_t& a = vertices[tri.x];
        const vec3_t& b = vertices[tri.y];
        const vec3_t& c = vertices[tri.z];
        const vec3_t normal = normalize_checked(cross(b - a, c - a));
        const vec3_t centroid = (a + b + c) * (1.0f / 3.0f);
        const vec3_t outward = normalize_checked(centroid - center);
        CHECK(dot(normal, outward) > 0.99f);
    }
}

TEST_CASE("SphericalSurface connectivity has no duplicates", "[Mesh][Icosphere]")
{
    SphericalSurface sphere(1.0f, 2);
    const MeshData& mesh = sphere.getMeshData();

    const auto& edges = mesh.edges;
    const auto& faces = mesh.faces;

    std::set<std::pair<int, int>> edge_set;
    for (const auto& e : edges)
    {
        const auto key = std::minmax(e.x, e.y);
        const bool inserted = edge_set.insert(key).second;
        CHECK(inserted);
    }

    std::set<std::array<int, 3>> face_set;
    for (const auto& tri : faces)
    {
        std::array<int, 3> key = {tri.x, tri.y, tri.z};
        std::sort(key.begin(), key.end());
        const bool inserted = face_set.insert(key).second;
        CHECK(inserted);
    }
}

TEST_CASE("RWG edge orientation is opposite across adjacent faces", "[Mesh][Icosphere]")
{
    SphericalSurface sphere(1.0f, 2);
    const MeshData& mesh = sphere.getMeshData();

    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;
    const auto& edges = mesh.edges;
    const auto& faces_edges = mesh.faces_edges;

    (void)vertices; // unused but kept for symmetry with Python test

    std::vector<std::vector<int>> edge_to_faces(edges.size());
    for (size_t fi = 0; fi < faces.size(); ++fi)
    {
        const int3& fe = faces_edges[fi];
        edge_to_faces[fe.x].push_back(static_cast<int>(fi));
        edge_to_faces[fe.y].push_back(static_cast<int>(fi));
        edge_to_faces[fe.z].push_back(static_cast<int>(fi));
    }

    for (size_t e_idx = 0; e_idx < edges.size(); ++e_idx)
    {
        const int va = edges[e_idx].x;
        const int vb = edges[e_idx].y;
        const auto& adjacent_faces = edge_to_faces[e_idx];
        REQUIRE(adjacent_faces.size() == 2);

        std::array<int, 2> signs = {0, 0};
        for (size_t k = 0; k < adjacent_faces.size(); ++k)
        {
            const int face_idx = adjacent_faces[k];
            const int3& tri = faces[face_idx];
            const int cyclic[3] = {tri.x, tri.y, tri.z};
            bool found = false;
            for (int i = 0; i < 3; ++i)
            {
                const int u = cyclic[i];
                const int v = cyclic[(i + 1) % 3];
                if (u == va && v == vb)
                {
                    signs[k] = +1;
                    found = true;
                    break;
                }
                if (u == vb && v == va)
                {
                    signs[k] = -1;
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        }
        CHECK(signs[0] == -signs[1]);
    }
}
