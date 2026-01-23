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
#  Root CMakeLists.txt for OpenGJK - builds both scalar and SIMD variants          #
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

cmake_minimum_required(VERSION 3.14)

project(openGJK
    VERSION 3.0.0
    DESCRIPTION "GJK algorithm for computing minimum distance between convex shapes"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_SCALAR "Build scalar (C) implementation" ON)
option(BUILD_SIMD "Build SIMD (C++ Highway) implementation" ON)
option(BUILD_GPU "Build GPU (CUDA) implementation" OFF)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(USE_32BITS "Use float instead of double" ON)

# Precision definition (shared between scalar and SIMD)
if(USE_32BITS)
    add_compile_definitions(USE_32BITS)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Building scalar: ${BUILD_SCALAR}")
message(STATUS "Building SIMD: ${BUILD_SIMD}")
message(STATUS "Building GPU: ${BUILD_GPU}")
message(STATUS "Building tests: ${BUILD_TESTS}")
message(STATUS "Building examples: ${BUILD_EXAMPLES}")
message(STATUS "Using 32-bit floats: ${USE_32BITS}")

# ============================================================================
# Testing (before subdirectories so both scalar and SIMD can register tests)
# ============================================================================

if(BUILD_TESTS)
    enable_testing()
    include(CTest)
    message(STATUS "Tests enabled. Run 'ctest' or 'make test' to run tests.")
endif()

# ============================================================================
# Scalar Implementation
# ============================================================================

if(BUILD_SCALAR)
    add_subdirectory(scalar)
endif()

# ============================================================================
# SIMD Implementation
# ============================================================================

if(BUILD_SIMD)
    add_subdirectory(simd)
endif()

# ============================================================================
# GPU Implementation
# ============================================================================

if(BUILD_GPU)
    enable_language(CUDA)
    add_subdirectory(gpu)
endif()


# ============================================================================
# Package Configuration
# ============================================================================

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/opengjk-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/opengjk-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/opengjk-config.cmake"
    @ONLY
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "OpenGJK v${PROJECT_VERSION} Configuration Summary:")
message(STATUS "  - Scalar library: ${BUILD_SCALAR}")
message(STATUS "  - SIMD library: ${BUILD_SIMD}")
message(STATUS "  - GPU library: ${BUILD_GPU}")
message(STATUS "  - Tests: ${BUILD_TESTS}")
message(STATUS "  - Examples: ${BUILD_EXAMPLES}")
message(STATUS "  - Precision: ${USE_32BITS} (32-bit float)")
message(STATUS "")
