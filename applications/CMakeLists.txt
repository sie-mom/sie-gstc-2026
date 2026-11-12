# add_subdirectory(dielectric_sphere)       # TODO: migrate float3 -> vec3_t

add_subdirectory(mesh_apps)
if(CUDAToolkit_FOUND)
    add_subdirectory(omega_biisotropic_sphere)
    add_subdirectory(anisotropic_gstc_sphere)
    add_subdirectory(gstc_surface)
    add_subdirectory(pmchwt_dielectric_sphere)
    add_subdirectory(pec_sphere)
endif()
# add_subdirectory(pmchwt_closed_dielectrics) # TODO: migrate float3 -> vec3_t
