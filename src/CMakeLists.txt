# This file builds the 'sie_mom' library.

# Create a library target named 'sie_mom'.
# This can be STATIC, SHARED, or empty. Let's start with STATIC.
add_library(sie_mom STATIC)

# List all the implementation files (.cpp, .cu) that are part of your library.
# You will need to add all your source files here.
target_sources(sie_mom PRIVATE
    sie_mom/quadrature/triangle_dunavant_rule.cpp
    sie_mom/quadrature/gauss_legendre_rule.cpp
    sie_mom/mesh/spherical_surface.cpp
    sie_mom/mesh/obj_mesh_loader.cpp
    # Add other .cpp and .cu files here as you create them
)

if(CUDAToolkit_FOUND)
    target_sources(sie_mom PRIVATE
        sie_mom/excitation/excitation.cu
        sie_mom/solvers/gstc_sie.cu
        sie_mom/solvers/dielectric/dielectric_solver.cu
        sie_mom/solvers/dielectric/dielectric_kernels.cu
        sie_mom/solvers/pec_solver.cu
        sie_mom/solvers/surface_current_fields.cu
    )
    target_link_libraries(sie_mom PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
    )
endif()

# Define the include directories for the library.
target_include_directories(sie_mom
    PUBLIC
        # The 'include' directory is public, so anyone linking to sie_mom gets it.
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        # The 'src' directory is private, only for building sie_mom itself.
        ${CMAKE_CURRENT_SOURCE_DIR}
)
