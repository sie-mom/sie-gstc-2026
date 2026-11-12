# Root CMakeLists.txt for the sie_mom project

cmake_minimum_required(VERSION 3.18)

project(sie_mom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC exposes math constants such as M_PI only when this is defined before
# the first math header. Set it on the command line to avoid include-order bugs.
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Find and enable OpenMP for all C++ targets in the project.
# This ensures that both libraries and executables are compiled with OpenMP support.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find CUDA and enable it if available
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found. Enabling CUDA language support.")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
endif()

# Precision control: set ON to compile all CUDA/C++ code in double precision.
option(SIE_MOM_USE_DOUBLE_PRECISION "Compile with double-precision floating point" ON)
if(SIE_MOM_USE_DOUBLE_PRECISION)
    message(STATUS "Double precision enabled (-DUSE_DOUBLE_PRECISION)")
    add_compile_definitions(USE_DOUBLE_PRECISION)
endif()

# Add the include directory for CUDA files
include_directories(include)

# Add vendor directory for third-party libraries
include_directories(vendor/eigen)

# Add the 'src' directory, which will build our main library
add_subdirectory(src)

# Add the 'applications' directory, which will build our applications
add_subdirectory(applications)

# Enable testing for the project
enable_testing()

# Add the 'tests' directory, which will build our test executable
add_subdirectory(tests)
