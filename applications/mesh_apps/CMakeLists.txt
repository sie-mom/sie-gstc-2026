
set(APPLICATION_EXECUTABLES
    generate_mesh
    mesh_obj_loader_tool
)

# Loop through the list and create an executable for each one,
# applying the same linking rules.
if(OpenMP_CXX_FOUND)
    separate_arguments(OPENMP_CXX_FLAGS_LIST NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

foreach(APP_EXE ${APPLICATION_EXECUTABLES})
    set(SOURCE_FILES "")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP_EXE}.cu" AND CUDAToolkit_FOUND)
        # If a .cu file exists and CUDA is available, use it.
        set(SOURCE_FILES ${APP_EXE}.cu)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP_EXE}.cpp")
        # Otherwise, if a .cpp file exists, use that.
        set(SOURCE_FILES ${APP_EXE}.cpp)
    endif()

    # Only create the executable if source files were found and assigned
    if(SOURCE_FILES)
        add_executable(${APP_EXE} ${SOURCE_FILES})
        target_link_libraries(${APP_EXE} PRIVATE sie_mom)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${APP_EXE} PUBLIC OpenMP::OpenMP_CXX)
            foreach(omp_flag IN LISTS OPENMP_CXX_FLAGS_LIST)
                target_compile_options(${APP_EXE} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${omp_flag}>
                )
            endforeach()
        endif()

        # If this is a CUDA executable, link the CUDA runtime
        if("${CMAKE_CURRENT_SOURCE_DIR}/${APP_EXE}.cu" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILES}")
            target_link_libraries(${APP_EXE} PRIVATE CUDA::cudart CUDA::cusolver)
        endif()
    endif()
endforeach()
