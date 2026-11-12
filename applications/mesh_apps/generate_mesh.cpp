#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/mesh/mesh_utils.hpp"
#include <iostream>
#include <string>

// This application generates a spherical mesh and exports the vertices and faces to CSV files. Example usage: ./generate_mesh 3 (where 3 is the refinement level)
// Is going to be deleted in the future, since it is not needed.

int main(int argc, char* argv[]) {
    int refinement_level = 2; // Default value
    if (argc > 1) {
        try {
            refinement_level = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Usage: " << argv[0] << " [refinement_level]\n";
            std::cerr << "Invalid argument. Using default refinement_level = 2.\n";
            refinement_level = 2;
        }
    }

    std::cout << "Generating mesh with refinement level: " << refinement_level << "\n";

    // Generate a spherical mesh (radius 1.0)
    sie_mom::mesh::SphericalSurface sphere(1.0f, refinement_level);
    
    // Export to CSV
    std::string prefix = "applications/mesh_plotting_apps/sphere";
    sie_mom::mesh::export_mesh_data(prefix, sphere.getMeshData());
    
    std::cout << "Mesh exported to " << prefix << "_vertices.csv and " << prefix << "_faces.csv\n";
    return 0;
}
