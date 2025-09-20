cmake_minimum_required(VERSION 3.18)
project(tensor LANGUAGES CXX CUDA)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)

# -----------------------
# --- Build Options ---
# -----------------------
option(USE_CUDA "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA device debug symbols (-G -g). Slows kernels a lot!" OFF)
option(HOST_DEBUG "Enable host debug symbols" OFF)

# -----------------------
# --- Compiler Settings ---
# -----------------------
set(CMAKE_CXX_STANDARD 17)

# Default build type = Release if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------
# --- CUDA setup ---
# -----------------------
if(USE_CUDA)
    set(CMAKE_CUDA_COMPILER "/vol/cuda/12.4.0/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CMAKE_CUDA_ARCHITECTURES 61) # Titan XP

    if(NOT DEFINED CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "/vol/cuda/12.4.0")
    endif()

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

    # --- Apply debug flags conditionally ---
    if(HOST_DEBUG)
        message(STATUS "Enabling host debug symbols for module ${name}")
        target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
    endif()

    if(USE_CUDA)
        if(CUDA_DEBUG)
            message(STATUS "Enabling CUDA device debug (-G -g) for module ${name}")
            target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G;-g>)
        elseif(HOST_DEBUG)
            message(STATUS "Using line info only for CUDA for module ${name}")
            target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-g>)
        endif()
    endif()
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
    target_include_directories(${name}Tests PRIVATE include test)

        # Gather dependencies (default: module itself + tensor)
        set(default_deps ${name} tensor)

        # Merge any additional dependencies passed to the function
        if(ARGN)
            list(APPEND default_deps ${ARGN})
        endif()

        # Link the test executable
        target_link_libraries(${name}Tests PRIVATE ${default_deps})
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
add_module(module)
add_module(optim)

# -----------------------
# --- Define tests ---
# -----------------------
add_module_test(tensor init)
add_module_test(init)
add_module_test(loss)
add_module_test(module)
add_module_test(optim)
