# This file builds the unified PMCHWT dielectric sphere application.

add_executable(pmchwt_dielectric_sphere_app
    pmchwt_app.cpp
)

target_link_libraries(pmchwt_dielectric_sphere_app PRIVATE
    sie_mom
)
