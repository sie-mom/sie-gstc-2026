#include "catch.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers used across multiple test cases
// ---------------------------------------------------------------------------
namespace {

// Write a clean tetrahedron .obj to disk and return the filename.
// The tetrahedron is the smallest possible closed manifold: V=4, E=6, F=4.
void write_tetrahedron(const std::string& filename)
{
    std::ofstream out(filename);
    out << "v 0.0 0.0 0.0\n";
    out << "v 1.0 0.0 0.0\n";
    out << "v 0.0 1.0 0.0\n";
    out << "v 0.0 0.0 1.0\n";
    out << "f 1 3 2\n";
    out << "f 1 2 4\n";
    out << "f 2 3 4\n";
    out << "f 3 1 4\n";
}

} // namespace

// ---------------------------------------------------------------------------
// TEST 1 – Happy path: correct topology on a clean synthetic mesh
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: clean tetrahedron builds correct topology",
          "[mesh][obj_mesh_loader]")
{
    const std::string filename = "test_tet_clean.obj";
    write_tetrahedron(filename);

    sie_mom::mesh::ObjMeshLoader loader(filename);
    const auto& mesh = loader.getMeshData();

    SECTION("Vertex, face, edge counts and Euler characteristic")
    {
        REQUIRE(mesh.vertices.size()    == 4);
        REQUIRE(mesh.faces.size()       == 4);
        // V - E + F = 2  =>  E = 6
        REQUIRE(mesh.edges.size()       == 6);
        REQUIRE(mesh.faces_edges.size() == 4);

        const long euler = static_cast<long>(mesh.vertices.size())
                         - static_cast<long>(mesh.edges.size())
                         + static_cast<long>(mesh.faces.size());
        CHECK(euler == 2);
    }

    SECTION("OBJ face order and outward winding are preserved")
    {
        CHECK(mesh.faces[0].x == 0);
        CHECK(mesh.faces[0].y == 2);
        CHECK(mesh.faces[0].z == 1);
        CHECK(mesh.faces[1].x == 0);
        CHECK(mesh.faces[1].y == 1);
        CHECK(mesh.faces[1].z == 3);
    }

    SECTION("Every edge is shared by exactly two faces")
    {
        std::vector<int> incidence(mesh.edges.size(), 0);
        for (const auto& fe : mesh.faces_edges)
        {
            ++incidence[fe.x];
            ++incidence[fe.y];
            ++incidence[fe.z];
        }
        for (const int count : incidence)
            CHECK(count == 2);
    }

    SECTION("No duplicate edges or faces in output")
    {
        std::set<std::pair<int,int>> edge_set;
        for (const auto& e : mesh.edges)
        {
            const auto key = std::minmax(e.x, e.y);
            CHECK(edge_set.insert(key).second); // false => duplicate
        }

        std::set<std::array<int,3>> face_set;
        for (const auto& tri : mesh.faces)
        {
            std::array<int,3> key = {tri.x, tri.y, tri.z};
            std::sort(key.begin(), key.end());
            CHECK(face_set.insert(key).second);
        }
    }

    SECTION("RWG: each edge traversed in opposite directions by its two faces")
    {
        std::vector<std::vector<int>> edge_to_faces(mesh.edges.size());
        for (size_t fi = 0; fi < mesh.faces.size(); ++fi)
        {
            const int3& fe = mesh.faces_edges[fi];
            edge_to_faces[fe.x].push_back(static_cast<int>(fi));
            edge_to_faces[fe.y].push_back(static_cast<int>(fi));
            edge_to_faces[fe.z].push_back(static_cast<int>(fi));
        }

        for (size_t ei = 0; ei < mesh.edges.size(); ++ei)
        {
            const int va = mesh.edges[ei].x;
            const int vb = mesh.edges[ei].y;
            REQUIRE(edge_to_faces[ei].size() == 2);

            std::array<int,2> signs = {0, 0};
            for (size_t k = 0; k < 2; ++k)
            {
                const int3& tri = mesh.faces[edge_to_faces[ei][k]];
                const int cyclic[3] = {tri.x, tri.y, tri.z};
                for (int i = 0; i < 3; ++i)
                {
                    if (cyclic[i] == va && cyclic[(i+1)%3] == vb) { signs[k] = +1; break; }
                    if (cyclic[i] == vb && cyclic[(i+1)%3] == va) { signs[k] = -1; break; }
                }
            }
            CHECK(signs[0] == -signs[1]);
        }
    }

    std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// TEST 2 – An OBJ requiring vertex repair is rejected rather than rewritten
//
// One vertex is repeated with a tiny spatial offset (well within tolerance).
// The true apex is left unreferenced when faces use its near-copy.
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: mesh requiring vertex repair is rejected",
          "[mesh][obj_mesh_loader]")
{
    const std::string filename = "test_tet_neardup_verts.obj";
    {
        std::ofstream out(filename);
        // Vertex 4 is the apex at (0,0,1); vertex 5 is a near-copy within tolerance
        out << "v 0.0 0.0 0.0\n";   // 1
        out << "v 1.0 0.0 0.0\n";   // 2
        out << "v 0.0 1.0 0.0\n";   // 3
        out << "v 0.0 0.0 1.0\n";   // 4  true apex
        out << "v 0.0 0.0 1.000005\n"; // 5  near-duplicate of 4 (within 1e-5 tolerance)
        // Faces reference the near-copy, leaving vertex 4 orphaned.
        out << "f 1 3 2\n";
        out << "f 1 2 5\n";
        out << "f 2 3 5\n";
        out << "f 3 1 5\n";
    }

    REQUIRE_THROWS(sie_mom::mesh::ObjMeshLoader(filename));

    std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// TEST 3 – A reversed duplicate face is rejected rather than removed
//
// This reproduces the Gmsh single-pass 2π revolve artefact: the exporter emits
// the same geometric triangle twice with opposite winding at the rotation axis.
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: reversed duplicate face is rejected",
          "[mesh][obj_mesh_loader]")
{
    const std::string filename = "test_tet_rev_dup_face.obj";
    {
        std::ofstream out(filename);
        out << "v 0.0 0.0 0.0\n";
        out << "v 1.0 0.0 0.0\n";
        out << "v 0.0 1.0 0.0\n";
        out << "v 0.0 0.0 1.0\n";
        out << "f 1 3 2\n";
        out << "f 1 2 4\n";
        out << "f 2 3 4\n";
        out << "f 3 1 4\n";
        out << "f 4 1 3\n"; // reversed duplicate of face 4
    }

    REQUIRE_THROWS(sie_mom::mesh::ObjMeshLoader(filename));

    std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// TEST 4 – A degenerate face is rejected rather than removed
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: degenerate face with repeated vertex is rejected",
          "[mesh][obj_mesh_loader]")
{
    const std::string filename = "test_tet_degen_face.obj";
    {
        std::ofstream out(filename);
        out << "v 0.0 0.0 0.0\n";
        out << "v 1.0 0.0 0.0\n";
        out << "v 0.0 1.0 0.0\n";
        out << "v 0.0 0.0 1.0\n";
        out << "f 1 3 2\n";
        out << "f 1 2 4\n";
        out << "f 2 3 4\n";
        out << "f 3 1 4\n";
        out << "f 1 1 2\n";
    }

    REQUIRE_THROWS(sie_mom::mesh::ObjMeshLoader(filename));

    std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// TEST 5 – Inconsistent winding is rejected rather than reoriented
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: inconsistent face winding is rejected",
          "[mesh][obj_mesh_loader]")
{
    const std::string filename = "test_tet_bad_winding.obj";
    {
        std::ofstream out(filename);
        out << "v 0.0 0.0 0.0\n";
        out << "v 1.0 0.0 0.0\n";
        out << "v 0.0 1.0 0.0\n";
        out << "v 0.0 0.0 1.0\n";
        out << "f 1 3 2\n";
        out << "f 1 2 4\n";
        out << "f 2 4 3\n"; // reversed from the outward orientation
        out << "f 3 1 4\n";
    }

    REQUIRE_THROWS(sie_mom::mesh::ObjMeshLoader(filename));
    std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// TEST 6 – Error handling: exception on missing file
// ---------------------------------------------------------------------------
TEST_CASE("ObjMeshLoader: throws on invalid file path",
          "[mesh][obj_mesh_loader]")
{
    REQUIRE_THROWS(sie_mom::mesh::ObjMeshLoader("non_existent_file_12345.obj"));
}
