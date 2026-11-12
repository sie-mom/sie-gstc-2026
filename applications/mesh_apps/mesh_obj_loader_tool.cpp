/**
 * Mesh health report tool for OBJ files.
 *
 * Loads an .obj file through the strict, non-mutating ObjMeshLoader and prints
 * a full topological health report. Exits with code 0 only if every check
 * passes so it can be used as a post-generation gate.
 *
 * Usage:
 *   ./mesh_obj_loader_tool [path_to_obj_file]
 *
 * Checks performed
 * ----------------
 *  1. Euler characteristic  V - E + F = 2   (closed genus-0 surface)
 *  2. Zero boundary edges   (no holes)
 *  3. Zero non-manifold edges  (no edge shared by 3+ faces)
 *  4. Single connected component
 *  5. Every edge traversed in exactly opposite directions by its two
 *     adjacent faces  (RWG orientation invariant)
 *
 * All counts refer to the OBJ exactly as written. The loader rejects meshes
 * that would require vertex merging, face removal, or winding correction.
 *
 * Note: automated unit tests for the ObjMeshLoader class live in
 *       tests/cpp/obj_mesh_loader_test.cpp.
 */

#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <vector>

// ── Topology helpers ──────────────────────────────────────────────────────────

struct TopologyReport
{
    int boundary_edges    = 0;  // edge incident to exactly 1 face
    int nonmanifold_edges = 0;  // edge incident to 3+ faces
    int components        = 0;  // connected components (face adjacency)
    int rwg_failures      = 0;  // edges where the two faces traverse it in the same direction
};

static TopologyReport analyse(const sie_mom::mesh::MeshData& mesh)
{
    const auto& faces       = mesh.faces;
    const auto& edges       = mesh.edges;
    const auto& faces_edges = mesh.faces_edges;

    // ── Edge incidence count ──────────────────────────────────────────────────
    std::vector<int> incidence(edges.size(), 0);
    std::vector<std::vector<int>> edge_to_faces(edges.size());
    for (size_t fi = 0; fi < faces.size(); ++fi)
    {
        const int3& fe = faces_edges[fi];
        for (const int ei : {fe.x, fe.y, fe.z})
        {
            ++incidence[ei];
            edge_to_faces[ei].push_back(static_cast<int>(fi));
        }
    }

    TopologyReport r;
    for (size_t ei = 0; ei < edges.size(); ++ei)
    {
        if (incidence[ei] == 1) ++r.boundary_edges;
        if (incidence[ei] >  2) ++r.nonmanifold_edges;
    }

    // ── Connected components (BFS over face adjacency) ────────────────────────
    // Two faces are adjacent if they share an edge with incidence == 2.
    std::vector<std::vector<int>> face_adj(faces.size());
    for (size_t ei = 0; ei < edges.size(); ++ei)
    {
        if (edge_to_faces[ei].size() == 2)
        {
            const int fa = edge_to_faces[ei][0];
            const int fb = edge_to_faces[ei][1];
            face_adj[fa].push_back(fb);
            face_adj[fb].push_back(fa);
        }
    }

    std::vector<bool> visited(faces.size(), false);
    for (size_t start = 0; start < faces.size(); ++start)
    {
        if (visited[start]) continue;
        ++r.components;
        std::queue<int> q;
        q.push(static_cast<int>(start));
        visited[start] = true;
        while (!q.empty())
        {
            const int cur = q.front(); q.pop();
            for (const int nb : face_adj[cur])
            {
                if (!visited[nb]) { visited[nb] = true; q.push(nb); }
            }
        }
    }

    // ── RWG orientation: each edge traversed in opposite directions ───────────
    for (size_t ei = 0; ei < edges.size(); ++ei)
    {
        if (edge_to_faces[ei].size() != 2) continue; // boundary/nonmanifold already counted

        const int va = edges[ei].x;
        const int vb = edges[ei].y;
        int signs[2] = {0, 0};
        for (int k = 0; k < 2; ++k)
        {
            const int3& tri = faces[edge_to_faces[ei][k]];
            const int cyc[3] = {tri.x, tri.y, tri.z};
            for (int i = 0; i < 3; ++i)
            {
                if (cyc[i] == va && cyc[(i+1)%3] == vb) { signs[k] = +1; break; }
                if (cyc[i] == vb && cyc[(i+1)%3] == va) { signs[k] = -1; break; }
            }
        }
        if (signs[0] != -signs[1]) ++r.rwg_failures;
    }

    return r;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    const std::string filename = (argc > 1) ? argv[1] : "blunted_cone_sphere.obj";

    try
    {
        sie_mom::mesh::ObjMeshLoader loader(filename);
        const auto& mesh = loader.getMeshData();

        const int V     = static_cast<int>(mesh.vertices.size());
        const int F     = static_cast<int>(mesh.faces.size());
        const int E     = static_cast<int>(mesh.edges.size());
        const int euler = V - E + F;

        std::cout << "\n=== Mesh Health Report: " << filename << " ===\n\n";
        std::cout << "  Vertices : " << V << "\n";
        std::cout << "  Faces    : " << F << "\n";
        std::cout << "  Edges    : " << E << "\n";
        std::cout << "  V-E+F    : " << euler << "\n\n";

        const TopologyReport rep = analyse(mesh);

        bool ok = true;

        auto check = [&](bool pass, const char* label, const char* detail = "")
        {
            std::cout << "  [" << (pass ? "PASS" : "FAIL") << "] " << label;
            if (*detail) std::cout << "  (" << detail << ")";
            std::cout << "\n";
            if (!pass) ok = false;
        };

        check(euler == 2,
              "Euler characteristic V-E+F = 2");
        check(rep.boundary_edges == 0,
              "No boundary edges (no holes)",
              (std::to_string(rep.boundary_edges) + " found").c_str());
        check(rep.nonmanifold_edges == 0,
              "No non-manifold edges (no 3+-incident edges)",
              (std::to_string(rep.nonmanifold_edges) + " found").c_str());
        check(rep.components == 1,
              "Single connected component",
              (std::to_string(rep.components) + " found").c_str());
        check(rep.rwg_failures == 0,
              "RWG orientation: edges traversed in opposite directions by adjacent faces",
              (std::to_string(rep.rwg_failures) + " failures").c_str());

        std::cout << "\n  Overall: " << (ok ? "HEALTHY -- ready for MoM assembly."
                                            : "UNHEALTHY -- do not pass to solver.") << "\n\n";

        return ok ? 0 : 1;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
