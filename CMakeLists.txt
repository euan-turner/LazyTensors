cmake_minimum_required(VERSION 3.18)
project(tensor LANGUAGES CXX CUDA)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)

# Always build with debug symbols for tests
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# -----------------------
# --- Build Options ---
# -----------------------
option(USE_CUDA "Enable CUDA support" ON)

# -----------------------
# --- Compiler Settings ---
# -----------------------
set(CMAKE_CXX_STANDARD 17)

if(USE_CUDA)
    set(CMAKE_CUDA_COMPILER "/vol/cuda/12.4.0/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CMAKE_CUDA_ARCHITECTURES 61) # Titan XP


    # Ensure CUDAToolkit path is set
    if(NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "/vol/cuda/12.4.0")
    endif()

    # Find CUDA Toolkit
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA toolkit found at ${CUDAToolkit_ROOT}")
        add_compile_definitions(CUDA_AVAILABLE)
    else()
        message(WARNING "CUDA toolkit not found; disabling CUDA")
        set(USE_CUDA OFF)
    endif()
else()
    message(STATUS "CUDA support disabled by user")
endif()

# -----------------------
# --- Function to create a library from a module ---
# -----------------------
function(add_module name)
    file(GLOB MODULE_HEADERS "include/${name}/*.hpp")
    file(GLOB MODULE_SOURCES "src/${name}/*.cpp")

    if(${name} STREQUAL "tensor" AND USE_CUDA)
        file(GLOB MODULE_CUDA_HEADERS "include/${name}/*.cuh")
        file(GLOB MODULE_CUDA_SOURCES "src/${name}/*.cu")
        add_library(${name} STATIC
            ${MODULE_SOURCES} ${MODULE_HEADERS}
            ${MODULE_CUDA_SOURCES} ${MODULE_CUDA_HEADERS}
        )
        target_include_directories(${name} PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
        )
        target_link_libraries(${name} PUBLIC CUDA::cudart)
        set_target_properties(${name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 61
        )
    else()
        add_library(${name} STATIC ${MODULE_SOURCES} ${MODULE_HEADERS})
        target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif()

    target_compile_options(${name} PRIVATE -g)
endfunction()

# -----------------------
# --- Function to create a test executable for a module ---
# -----------------------
function(add_module_test name)
    file(GLOB TEST_SOURCES "test/${name}/*.cpp")
    if(TEST_SOURCES)
        add_executable(${name}Tests
            ${TEST_SOURCES}
            test/catch_amalgamated.cpp
        )
        target_include_directories(${name}Tests PRIVATE include)
        target_link_libraries(${name}Tests PRIVATE ${name} tensor)
        add_test(NAME ${name}Tests COMMAND ${name}Tests)
    else()
        message(WARNING "No test sources found for module '${name}'")
    endif()
endfunction()

# -----------------------
# --- Define modules ---
# -----------------------
add_module(tensor)
add_module(init)
add_module(loss)

# -----------------------
# --- Define tests ---
# -----------------------
add_module_test(tensor)
add_module_test(init)
add_module_test(loss)
