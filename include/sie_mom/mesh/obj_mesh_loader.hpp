#ifndef SIE_MOM_OBJ_MESH_LOADER_HPP
#define SIE_MOM_OBJ_MESH_LOADER_HPP

#include "sie_mom/mesh/spherical_surface.hpp" // Re-using MeshData, float3, int3, int2
#include <string>

namespace sie_mom {
namespace mesh {

/**
 * @brief Strict OBJ loader for triangular closed surface meshes.
 *
 * Vertex order, face order, and face winding are preserved exactly. Invalid
 * meshes are rejected rather than repaired, so per-face data remains aligned
 * with the OBJ face indices used to produce it.
 */
class ObjMeshLoader {
public:
    /**
     * @brief Parse and validate the OBJ, then build RWG edge connectivity.
     * 
     * @param filepath Path to the .obj file.
     */
    explicit ObjMeshLoader(const std::string& filepath);

    /**
     * @brief Get the unchanged vertices/faces and derived RWG connectivity.
     * 
     * @return const MeshData& 
     */
    const MeshData& getMeshData() const { return mesh_data_; }

private:
    MeshData mesh_data_;

    void parseFile(const std::string& filepath);
    void validateMesh(const std::string& filepath) const;
    void buildEdgesAndMapping();
};

} // namespace mesh
} // namespace sie_mom

#endif // SIE_MOM_OBJ_MESH_LOADER_HPP
