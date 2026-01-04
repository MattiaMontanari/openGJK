#                            _____      _ _  __                                    #
#                           / ____|    | | |/ /                                    #
#     ___  _ __   ___ _ __ | |  __     | | ' /                                     #
#    / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                      #
#   | (_) | |_) |  __/ | | | |__| | |__| | . \                                     #
#    \___/| .__/ \___|_| |_|\_____|\_____/|_|\_\                                   #
#         | |                                                                      #
#         |_|                                                                      #
#                                                                                  #
#  Copyright 2022-2026 Mattia Montanari, University of Oxford                      #
#                                                                                  #
#  This program is free software: you can redistribute it and/or modify it under  #
#  the terms of the GNU General Public License as published by the Free Software  #
#  Foundation, either version 3 of the License. You should have received a copy   #
#  of the GNU General Public License along with this program. If not, visit       #
#                                                                                  #
#      https://www.gnu.org/licenses/                                               #
#                                                                                  #
#  This program is distributed in the hope that it will be useful, but WITHOUT    #
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  #
#  FOR A PARTICULAR PURPOSE. See GNU General Public License for details.          #

cmake_minimum_required(VERSION 3.16)

project(opengjk_simd
    VERSION 1.0.0
    DESCRIPTION "SIMD implementation of the GJK algorithm using Google Highway"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(USE_32BITS "Use float instead of double" ON)
option(USE_MINIMAL_SIMD "Use smallest viable SIMD width instead of widest" OFF)

# Find Highway - prefer local ext directory, then system, then FetchContent
set(HWY_EXT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/highway")
if(EXISTS "${HWY_EXT_PATH}/CMakeLists.txt")
    message(STATUS "[opengjk_simd] Using local Highway from ${HWY_EXT_PATH}")
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    if(NOT TARGET hwy)
        add_subdirectory(${HWY_EXT_PATH} ${CMAKE_CURRENT_BINARY_DIR}/highway EXCLUDE_FROM_ALL)
    endif()
    set(HWY_LIBRARY hwy)
else()
    # Try system Highway first
    find_package(hwy CONFIG QUIET)
    if(hwy_FOUND)
        message(STATUS "[opengjk_simd] Using system Highway")
        set(HWY_LIBRARY hwy::hwy)
    else()
        # Fall back to FetchContent
        message(STATUS "[opengjk_simd] Fetching Highway via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            highway
            GIT_REPOSITORY https://github.com/google/highway.git
            GIT_TAG 1.2.0
            GIT_SHALLOW TRUE
        )
        set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
        set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(highway)
        set(HWY_LIBRARY hwy)
    endif()
endif()

# Precision definition - enables float instead of double
# Required for SSE4 targets which only have 2 lanes with double
if(USE_32BITS)
    add_compile_definitions(OPENGJK_SIMD_USE_FLOAT)
endif()

# SIMD width policy configuration (runtime-based, future-proof)
#
# USE_MINIMAL_SIMD=OFF (default): Use widest available SIMD (Highway auto-selects best)
# USE_MINIMAL_SIMD=ON: Use smallest viable SIMD width at runtime
#   - float (USE_32BITS=ON):  prefers 128-bit (SSE4/NEON) - 4 floats per register
#   - double (USE_32BITS=OFF): prefers 256-bit (AVX2) - 4 doubles per register
#
# Target filtering is done at runtime via InitSIMDTargets() in opengjk_simd_init.h
# This approach is future-proof and works on x86, ARM, and future architectures.
if(USE_MINIMAL_SIMD)
    add_compile_definitions(OPENGJK_SIMD_MINIMAL_WIDTH)
    if(USE_32BITS)
        message(STATUS "[opengjk_simd] Minimal SIMD: runtime preference for 128-bit (SSE4/NEON)")
    else()
        message(STATUS "[opengjk_simd] Minimal SIMD: runtime preference for 256-bit (AVX2)")
    endif()
else()
    message(STATUS "[opengjk_simd] SIMD width: widest available (runtime auto-select)")
endif()

# Library target
add_library(opengjk_simd
    opengjk.cc
)

target_include_directories(opengjk_simd
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(opengjk_simd
    PUBLIC ${HWY_LIBRARY}
)

# Compiler flags for Highway
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(opengjk_simd PRIVATE
        -Wall -Wextra -Wpedantic
        -fno-exceptions
    )
    # Suppress warnings from external Highway library code
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(opengjk_simd PRIVATE
            -Wno-gnu-zero-variadic-macro-arguments # Highway uses GNU extension
        )
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(opengjk_simd PRIVATE
            -Wno-pedantic # Highway uses __int128 which triggers -Wpedantic
        )
    endif()
endif()

if(MSVC)
    target_compile_options(opengjk_simd PRIVATE /W4)
endif()

# Alias for consistent naming
add_library(opengjk::simd ALIAS opengjk_simd)

# Tests
if(BUILD_TESTS)
    # Add scalar library for cross-validation tests (only if not already added)
    set(SCALAR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../scalar")
    if(NOT TARGET opengjk_scalar_static)
        if(EXISTS "${SCALAR_PATH}/CMakeLists.txt")
            message(STATUS "Adding scalar library from ${SCALAR_PATH}")
            set(OPENGJK_SCALAR_BUILD_SHARED OFF CACHE BOOL "" FORCE)
            if(USE_32BITS)
                set(OPENGJK_SCALAR_SINGLE_PRECISION ON CACHE BOOL "" FORCE)
            endif()
            add_subdirectory(${SCALAR_PATH} ${CMAKE_CURRENT_BINARY_DIR}/scalar EXCLUDE_FROM_ALL)
            # Create alias that tests expect
            if(NOT TARGET opengjk_scalar)
                add_library(opengjk_scalar ALIAS opengjk_scalar_static)
            endif()
        else()
            message(WARNING "Scalar library not found at ${SCALAR_PATH}, crossval tests will be skipped")
        endif()
    else()
        message(STATUS "Using existing scalar library target")
    endif()

    add_subdirectory(tests)
endif()

# Installation (disabled when using local Highway)
if(TARGET hwy::hwy)
    include(GNUInstallDirs)

    install(TARGETS opengjk_simd
        EXPORT opengjk_simd-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(EXPORT opengjk_simd-targets
        FILE opengjk_simd-targets.cmake
        NAMESPACE opengjk::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opengjk_simd
    )
endif()

