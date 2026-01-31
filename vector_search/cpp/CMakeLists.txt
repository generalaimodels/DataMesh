cmake_minimum_required(VERSION 3.15)
project(vectorsearch_core VERSION 1.0.0 LANGUAGES CXX)

# C++17 required for structured bindings, constexpr if
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =============================================================================
# SIMD Detection
# =============================================================================
include(CheckCXXCompilerFlag)

# Check for AVX2
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2_MSVC)

# Check for AVX-512
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512_MSVC)

# Check for FMA
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

# Set SIMD flags
if(MSVC)
    if(COMPILER_SUPPORTS_AVX512_MSVC)
        set(SIMD_FLAGS "/arch:AVX512")
        add_compile_definitions(VECTORSEARCH_AVX512=1)
    elseif(COMPILER_SUPPORTS_AVX2_MSVC)
        set(SIMD_FLAGS "/arch:AVX2")
        add_compile_definitions(VECTORSEARCH_AVX2=1)
    endif()
else()
    if(COMPILER_SUPPORTS_AVX512)
        set(SIMD_FLAGS "-mavx512f -mavx512dq -mavx512bw -mavx512vl")
        add_compile_definitions(VECTORSEARCH_AVX512=1)
    elseif(COMPILER_SUPPORTS_AVX2)
        set(SIMD_FLAGS "-mavx2")
        add_compile_definitions(VECTORSEARCH_AVX2=1)
    endif()
    
    if(COMPILER_SUPPORTS_FMA)
        set(SIMD_FLAGS "${SIMD_FLAGS} -mfma")
        add_compile_definitions(VECTORSEARCH_FMA=1)
    endif()
endif()

# =============================================================================
# Compiler Flags
# =============================================================================
if(MSVC)
    add_compile_options(/W4 /O2 /fp:fast ${SIMD_FLAGS})
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -O3 -ffast-math -funroll-loops
        ${SIMD_FLAGS}
    )
endif()

# =============================================================================
# pybind11
# =============================================================================
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Fallback: fetch pybind11
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# =============================================================================
# Core Library
# =============================================================================
set(CORE_SOURCES
    src/simd_ops.cpp
    src/hnsw.cpp
    src/quantization.cpp
)

set(CORE_HEADERS
    include/common.hpp
    include/simd_ops.hpp
    include/hnsw.hpp
    include/quantization.hpp
)

# Static library
add_library(vectorsearch_core_static STATIC ${CORE_SOURCES})
target_include_directories(vectorsearch_core_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# =============================================================================
# Python Module
# =============================================================================
pybind11_add_module(_vectorsearch_core
    src/bindings.cpp
    ${CORE_SOURCES}
)
target_include_directories(_vectorsearch_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =============================================================================
# Testing
# =============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    
    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    add_executable(test_simd tests/test_simd.cpp)
    target_link_libraries(test_simd vectorsearch_core_static GTest::gtest_main)
    target_include_directories(test_simd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    add_executable(test_hnsw tests/test_hnsw.cpp)
    target_link_libraries(test_hnsw vectorsearch_core_static GTest::gtest_main)
    target_include_directories(test_hnsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    
    include(GoogleTest)
    gtest_discover_tests(test_simd)
    gtest_discover_tests(test_hnsw)
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS vectorsearch_core_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${CORE_HEADERS}
    DESTINATION include/vectorsearch
)

# Print configuration summary
message(STATUS "=== VectorSearch Core Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "SIMD flags: ${SIMD_FLAGS}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
