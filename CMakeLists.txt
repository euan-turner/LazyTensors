cmake_minimum_required(VERSION 3.18)
project(LazyTensors LANGUAGES CXX CUDA)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)

# -----------------------
# --- Build Options ---
# -----------------------
option(USE_CUDA "Enable CUDA support" ON)
option(CUDA_DEBUG "Enable CUDA device debug symbols (-G -g). Slows kernels a lot!" OFF)
option(HOST_DEBUG "Enable host debug symbols" OFF)
option(ENABLE_DEBUG_SYMBOLS "High-level option to enable debug symbols (sets HOST_DEBUG and CMAKE_BUILD_TYPE=Debug)" OFF)

# If user requested the high-level switch, set host debug and build type unless
# the user explicitly set them already.
if(ENABLE_DEBUG_SYMBOLS)
    message(STATUS "ENABLE_DEBUG_SYMBOLS=ON: enabling host debug symbols and Debug build type")
    if(NOT HOST_DEBUG)
        set(HOST_DEBUG ON CACHE BOOL "Enable host debug symbols" FORCE)
    endif()
    if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif()
endif()

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
    # Keep an overridable default path for environments with custom CUDA installs
    set(CMAKE_CUDA_COMPILER "/vol/cuda/12.4.0/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures")

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
# --- Build tensor library ---
# -----------------------
file(GLOB TENSOR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/tensor/*.hpp")
file(GLOB TENSOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")

if(USE_CUDA)
    file(GLOB TENSOR_CU_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/tensor/*.cuh")
    file(GLOB TENSOR_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cu")
    add_library(tensor STATIC
        ${TENSOR_SOURCES} ${TENSOR_HEADERS}
        ${TENSOR_CU_SOURCES} ${TENSOR_CU_HEADERS}
    )
    target_include_directories(tensor PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        $<$<BOOL:${CUDAToolkit_FOUND}>:${CUDAToolkit_INCLUDE_DIRS}>
    )
    target_link_libraries(tensor PUBLIC CUDA::cudart)
    set_target_properties(tensor PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
else()
    add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
    target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# --- Apply debug flags conditionally ---
if(HOST_DEBUG)
    message(STATUS "Enabling host debug symbols for tensor")
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
endif()

if(USE_CUDA)
    if(CUDA_DEBUG)
        message(STATUS "Enabling CUDA device debug (-G -g) for tensor")
        target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G;-g>)
    elseif(HOST_DEBUG)
        message(STATUS "Using line info only for CUDA for tensor")
        target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info;-g>)
    endif()
endif()

# -----------------------
# --- Tests (tensor) ---
# -----------------------
file(GLOB TENSOR_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test/tensor/*.cpp")
if(TENSOR_TEST_SOURCES)
    add_executable(tensorTests
        ${TENSOR_TEST_SOURCES}
        test/catch_amalgamated.cpp
    )
    target_include_directories(tensorTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_link_libraries(tensorTests PRIVATE tensor)
    add_test(NAME tensorTests COMMAND tensorTests)
else()
    message(WARNING "No test sources found for 'tensor'")
endif()

# Ensure tests are also compiled with host debug symbols when requested.
if(TARGET tensorTests)
    if(HOST_DEBUG)
        message(STATUS "Enabling host debug symbols for tensorTests")
        target_compile_options(tensorTests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g>)
    endif()
endif()

