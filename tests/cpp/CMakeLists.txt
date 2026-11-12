# This CMake file builds the C++ host-side tests.

# This file is now included from the root CMakeLists.txt

# List of C++ test sources
set(TEST_SOURCES
    main.cpp
    semi_analytic_test.cpp
    double_quadrature_test.cpp
    edge_duffy_reference_test.cpp
    ntl_form_a_line_geometry_test.cpp
    ntl_form_b_reference_test.cpp
    triangle_dunavant_rule_test.cpp
    gauss_legendre_rule_test.cpp
    singular_quadrature_test.cpp
    spherical_surface_mesh_test.cpp
    tl_matrix_assembly_test.cpp
    sie_dielectric_sphere_test.cpp
    obj_mesh_loader_test.cpp
    sie_scaling_test.cpp
    math_utils_test.cpp
)

# If CUDA is available, add CUDA test sources
if(CUDAToolkit_FOUND)
    list(APPEND TEST_SOURCES excitation_test.cu)
    list(APPEND TEST_SOURCES gstc_sie_contract_test.cpp)
    list(APPEND TEST_SOURCES face_basis_gpu_test.cu)
    list(APPEND TEST_SOURCES singular_quadrature_test.cu)
    list(APPEND TEST_SOURCES test_matvec_consistency.cu)
    list(APPEND TEST_SOURCES test_gmres_solver.cu)
endif()

# Define the test executable by listing all your test files.
add_executable(run_cpp_tests ${TEST_SOURCES})

# Link the test executable against our main 'sie_mom' library.
# This automatically handles include directories and ensures the library
# is built before the tests.
target_link_libraries(run_cpp_tests PRIVATE sie_mom CUDA::cublas)

# Define the test for CTest.
# This is now enabled in the root CMakeLists.txt.
add_test(NAME cpp_unit_tests COMMAND run_cpp_tests)
