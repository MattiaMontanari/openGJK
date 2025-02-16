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

# User options . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
option(BUILD_EXAMPLE "Build demo"                        ON )
option(BUILD_MONO    "Build C# example"                  OFF)
option(BUILD_CTYPES  "Expose symbols for Python cytpes"  OFF)

# CMake setup  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
cmake_minimum_required(VERSION          3.5) # You can downgrade to 2.8, but I don't maintain it anymore
cmake_policy(SET CMP0048                NEW) 
cmake_policy(SET CMP0063                NEW) # Visibility of preset and hidden lines on shared libs

set( CMAKE_C_VISIBILITY_PRESET           hidden)
set( CMAKE_VISIBILITY_INLINES_HIDDEN     ON)
set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS    ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS       ON)

# Project setup  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
set(GK_VERSION_MAJOR                      3)
set(GK_VERSION_MINOR                      0)
set(GK_VERSION_PATCH                      3)

project(lib_opengjk_ce 
    LANGUAGES C
	DESCRIPTION "openGJK library community edition (CE)"
    HOMEPAGE_URL "www.mattiamontanari.com/opengjk/"
    VERSION ${GK_VERSION_MAJOR}.${GK_VERSION_MINOR}.${GK_VERSION_PATCH}
)

# Compiler flags  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
message(STATUS "Compiler in use: ${CMAKE_C_COMPILER_ID}")

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set( CMAKE_C_FLAGS_DEBUG                 "-O0 -g -Wall")
    set( CMAKE_C_FLAGS_RELEASE               "-O3 -Werror")
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set( CMAKE_C_FLAGS_DEBUG                 "-O0 -g -Wall")
    set( CMAKE_C_FLAGS_RELEASE               "-O3 -Werror")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wunused-macros -Wno-unused-command-line-argument")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
endif()

# Ends up being a bit messy, but supporting more languages it's easier this way
add_compile_definitions(INCLUDE_CMAKE_HEADER)
# Target . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
include(GenerateExportHeader)

if(BUILD_CTYPES)
    set( CMAKE_C_VISIBILITY_PRESET           default)
endif(BUILD_CTYPES)

if(BUILD_MONO)
    set( CMAKE_C_VISIBILITY_PRESET           default)
    add_compile_definitions(CS_MONO_BUILD)
endif(BUILD_MONO)

if(SINGLE_PRECISION)
    add_compile_definitions(USE_32BITS)
endif(SINGLE_PRECISION)

add_library( obj_openGJK 
    OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/openGJK.c 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
)

target_include_directories( obj_openGJK 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_BINARY_DIR}
)

IF (NOT WIN32)
target_link_libraries(obj_openGJK PRIVATE m)
ENDIF()

set_target_properties(obj_openGJK 
    PROPERTIES 
        C_STANDARD 11
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
        POSITION_INDEPENDENT_CODE 1
)

add_library(opengjk_ce SHARED $<TARGET_OBJECTS:obj_openGJK>)

if (BUILD_CTYPES)
    add_custom_command(
        TARGET opengjk_ce
        COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:opengjk_ce>"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/python_ctypes/src/pyopengjk"
    )
endif(BUILD_CTYPES)

generate_export_header(opengjk_ce
    BASE_NAME OPENGJK
    EXPORT_FILE_NAME opengjk_export.h
)

target_include_directories(opengjk_ce PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

# Build demo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
if(BUILD_EXAMPLE)
    add_subdirectory(examples/c)
endif(BUILD_EXAMPLE)

# Install  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
if (UNIX)
    set(DESTDIR "/usr/local")
    INSTALL(TARGETS opengjk_ce 
        LIBRARY DESTINATION "${DESTDIR}/lib"
        PERMISSIONS WORLD_READ GROUP_READ OWNER_READ
    )
    INSTALL(TARGETS obj_openGJK
        PUBLIC_HEADER DESTINATION "${DESTDIR}/include/openGJK"
        PERMISSIONS WORLD_READ GROUP_READ OWNER_READ
    )
endif (UNIX)

# Wrap up  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
message(STATUS "Version     : " ${CMAKE_PROJECT_VERSION}  )
message(STATUS "Build type  : " ${CMAKE_BUILD_TYPE} )
