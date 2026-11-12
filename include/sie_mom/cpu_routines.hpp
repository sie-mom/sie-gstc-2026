#ifndef SIE_MOM_CPU_ROUTINES_HPP
#define SIE_MOM_CPU_ROUTINES_HPP

#include "sie_mom/mesh/spherical_surface.hpp"
#include "sie_mom/cuda/dunavant_quadrature.cuh"
#include <iostream>
#include <omp.h>
#include <vector>

namespace sie_mom {

template <typename ExcitationSource>
void assemble_V_vector_cpu(
    std::vector<complex__float>& global_V_vector,
    const mesh::MeshData& mesh,
    const std::vector<float>& quad_points,
    const std::vector<float>& quad_weights,
    int order_num,
    const ExcitationSource& excitation_source)
{
    std::cout << "Assembling V vector on CPU..." << std::endl;

    const auto& vertices = mesh.vertices;
    const auto& faces = mesh.faces;
    const auto& faces_edges = mesh.faces_edges;
    const size_t num_edges = mesh.edges.size();

    #pragma omp parallel for schedule(dynamic)
    for (size_t m = 0; m < faces.size(); ++m) {
        const auto& face_m = faces[m];
        const int tri_indices[3] = {face_m.x, face_m.y, face_m.z};
        vec3_t tri_v[3] = {vertices[face_m.x], vertices[face_m.y], vertices[face_m.z]};

        complex__float V_E_local[3];
        complex__float V_H_local[3];

        calculateExcitationQuadrature(tri_v, tri_indices,
                                        quad_points.data(), quad_weights.data(),
                                        order_num, excitation_source,
                                        V_E_local, V_H_local);

        const auto& face_edges = faces_edges[m];
        const int local_edge_indices[3] = {face_edges.x, face_edges.y, face_edges.z};

        auto accumulate_global = [&](size_t idx, const complex__float& value) {
#ifdef _OPENMP
            #pragma omp atomic
            global_V_vector[idx].x += value.x;
            #pragma omp atomic
            global_V_vector[idx].y += value.y;
#else
            complexAddInPlace(global_V_vector[idx], value);
#endif
        };

        for (int j = 0; j < 3; ++j) {
            const int global_edge_idx = local_edge_indices[j];
            accumulate_global(global_edge_idx, V_E_local[j]);
            accumulate_global(global_edge_idx + num_edges, V_H_local[j]);
        }
    }
}

} // namespace sie_mom

#endif // SIE_MOM_CPU_ROUTINES_HPP