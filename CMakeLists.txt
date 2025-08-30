cmake_minimum_required(VERSION 3.10)
project(tensor)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)

# Always build with debug symbols for tests
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# ---- Function to create a library from a module ----
function(add_module name)
    file(GLOB MODULE_HEADERS "include/${name}/*.hpp")
    file(GLOB MODULE_SOURCES "src/${name}/*.cpp")
    add_library(${name} STATIC ${MODULE_SOURCES} ${MODULE_HEADERS})
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(${name} PRIVATE -g)
endfunction()

# ---- Function to create a test executable for a module ----
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

# ---- Define modules ----
add_module(tensor)
add_module(init)
add_module(loss)

# ---- Define tests ----
add_module_test(tensor)
add_module_test(init)
add_module_test(loss)
