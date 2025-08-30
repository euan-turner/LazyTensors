# Always build with debug symbols for tests
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# TensorTests target

# TensorTests target (link with Catch2 amalgamated source)
add_executable(TensorTests
	test/tensor/cpu_impl_test.cpp
	test/catch_amalgamated.cpp
	# Add more tensor test files here as needed
)
target_compile_options(TensorTests PRIVATE -g)
target_include_directories(TensorTests PRIVATE include)

target_link_libraries(TensorTests PRIVATE tensor)

# Register TensorTests with CTest
add_test(NAME TensorTests COMMAND TensorTests)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)

cmake_minimum_required(VERSION 3.10)
project(tensor)

file(GLOB TENSOR_HEADERS "include/tensor/*.hpp")
file(GLOB TENSOR_SOURCES "src/tensor/*.cpp")

add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
target_compile_options(tensor PRIVATE -g)
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
