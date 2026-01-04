#                            _____      _ _  __                                    #
#                           / ____|    | | |/ /                                    #
#     ___  _ __   ___ _ __ | |  __     | | ' /                                     #
#    / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                      #
#   | (_) | |_) |  __/ | | | |__| | |__| | . \                                     #
#    \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                    #
#         | |                                                                      #
#         |_|                                                                      #
#                                                                                  #
#  Copyright 2022 Mattia Montanari, University of Oxford                           #
#                                                                                  #
#  This program is free software: you can redistribute it and/or modify it under   #
#  the terms of the GNU General Public License as published by the Free Software   #
#  Foundation, either version 3 of the License. You should have received a copy    #
#  of the GNU General Public License along with this program. If not, visit        #
#                                                                                  #
#      https://www.gnu.org/licenses/                                               #
#                                                                                  #
#  This program is distributed in the hope that it will be useful, but WITHOUT     #
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS   #
#  FOR A PARTICULAR PURPOSE. See GNU General Public License for details.           #

# Scalar openGJK implementation
# This builds the original C-based scalar version

cmake_minimum_required(VERSION 3.14)

project(opengjk_scalar
    LANGUAGES C CXX
    DESCRIPTION "openGJK scalar implementation (CE)"
    VERSION 3.0.3
)

# User options
option(OPENGJK_SCALAR_BUILD_SHARED "Build shared library" ON)
option(OPENGJK_SCALAR_SINGLE_PRECISION "Use single precision (float)" OFF)

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")
endif()

# Precision definition
if(OPENGJK_SCALAR_SINGLE_PRECISION)
    add_compile_definitions(USE_32BITS)
endif()

add_compile_definitions(INCLUDE_CMAKE_HEADER)

# Object library for the scalar implementation
add_library(opengjk_scalar_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/openGJK.c
    ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
)

target_include_directories(opengjk_scalar_obj
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if(NOT WIN32)
    target_link_libraries(opengjk_scalar_obj PRIVATE m)
endif()

set_target_properties(opengjk_scalar_obj
    PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
    POSITION_INDEPENDENT_CODE ON
)

# Generate export header
include(GenerateExportHeader)

# Shared library
if(OPENGJK_SCALAR_BUILD_SHARED)
    add_library(opengjk_scalar SHARED $<TARGET_OBJECTS:opengjk_scalar_obj>)
    
    generate_export_header(opengjk_scalar
        BASE_NAME OPENGJK
        EXPORT_FILE_NAME opengjk_export.h
    )
    
    target_include_directories(opengjk_scalar PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    
    # Alias for easy linking
    add_library(openGJK::scalar ALIAS opengjk_scalar)
endif()

# Static library (always built for testing/linking)
add_library(opengjk_scalar_static STATIC $<TARGET_OBJECTS:opengjk_scalar_obj>)

target_include_directories(opengjk_scalar_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Generate export header for static lib too
generate_export_header(opengjk_scalar_static
    BASE_NAME OPENGJK
    EXPORT_FILE_NAME opengjk_export.h
)

add_library(openGJK::scalar_static ALIAS opengjk_scalar_static)

message(STATUS "[opengjk_scalar] Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "[opengjk_scalar] Single precision: ${OPENGJK_SCALAR_SINGLE_PRECISION}")

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

