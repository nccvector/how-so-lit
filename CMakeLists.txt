cmake_minimum_required(VERSION 3.0.0)

project(optfs)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Find OptiX
find_package(OptiX REQUIRED)
include_directories(${OptiX_INCLUDE})

# Find OpenGL and Glfw
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Include optix macros
include(${CMAKE_MODULE_PATH}/configure_optix.cmake)

# Include common
include_directories(common)

# Set libraries to be linked
set(LIBS ${optix_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

# Add ptx files
cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

# Add executable
add_executable(${PROJECT_NAME}
        ${embedded_ptx_code}
        launchParams.h
        common/cudaMathHelper.h
        main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${LIBS}
    ${OPENGL_LIBRARIES}
    glfw)
