add_executable(omega_sphere_sie
    omega_sphere_sie.cpp
)

target_link_libraries(omega_sphere_sie PRIVATE
    sie_mom
)
