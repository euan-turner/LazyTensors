
cmake_minimum_required(VERSION 3.10)
project(tensor)

file(GLOB TENSOR_HEADERS "include/tensor/*.hpp")
file(GLOB TENSOR_SOURCES "src/tensor/*.cpp")

add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
