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
#  Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund                            #
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

# GPU openGJK implementation (CUDA)
# Based on OpenGJK-GPU: https://github.com/vismaychuriwala/OpenGJK-GPU
#
# Files:
#   gjk_kernel.cu - CUDA kernel implementation with warp-level parallelism
#   opengjk_gpu.cu - Public API implementation (memory management & kernel launches)
#   include/opengjk_gpu.h - Public API header

cmake_minimum_required(VERSION 3.18)

project(opengjk_gpu
    LANGUAGES C CXX CUDA
    DESCRIPTION "openGJK GPU implementation (CUDA)"
    VERSION 3.0.0
)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wunused-macros")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wunused-macros")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
    # Intel compiler - use defaults
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    # MSVC - use defaults
endif()

set(headers
    "include/opengjk_gpu.h"
)

set(sources
    "gjk_kernel.cu"
    "opengjk_gpu.cu"
)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

# Build GPU library
add_library(opengjk_gpu ${sources} ${headers})

target_include_directories(opengjk_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(UNIX)
    target_link_libraries(opengjk_gpu PRIVATE m)
endif()

# CUDA architecture handling (same as OpenGJK-GPU)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(opengjk_gpu PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(opengjk_gpu PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(opengjk_gpu PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Debug flags for CUDA
target_compile_options(opengjk_gpu PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Alias for consistency with scalar/simd
add_library(openGJK::gpu ALIAS opengjk_gpu)

# Build example
add_subdirectory(examples/gpu)

message(STATUS "[opengjk_gpu] CUDA enabled")
message(STATUS "[opengjk_gpu] Using 32-bit floats: ${USE_32BITS}")
message(STATUS "[opengjk_gpu] Build type: ${CMAKE_BUILD_TYPE}")
