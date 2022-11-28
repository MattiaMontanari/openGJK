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
option(BUILD_EXAMPLE "Build demo"        ON)

# CMake setup  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
cmake_minimum_required(VERSION          3.0) # You can downgrade to 2.8, but I don't maintain it anymore
cmake_policy(SET CMP0048                NEW) 
cmake_policy(SET CMP0063                NEW) # Visibility of preset and hidden lines on shared libs

set( CMAKE_C_VISIBILITY_PRESET           hidden)
set( CMAKE_VISIBILITY_INLINES_HIDDEN     TRUE)
set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS    ON)

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
set( CMAKE_C_FLAGS_DEBUG                 "-O0 -g -Wall -Wunused")
set( CMAKE_C_FLAGS_RELEASE               "-O3 -Werror")

if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wunused-macros -Wno-unused-command-line-argument")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
endif()

# Target . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
add_library( ${PROJECT_NAME} 
    STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/openGJK.c 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
)

target_include_directories( ${PROJECT_NAME} 
PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/include
${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE m)

set_target_properties(${PROJECT_NAME} 
    PROPERTIES 
        C_STANDARD 11
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
)

# Build demo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
if(BUILD_EXAMPLE)
    add_subdirectory(examples/c)
endif(BUILD_EXAMPLE)

# Install  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
if (UNIX)
    install(TARGETS ${PROJECT_NAME} PERMISSIONS WORLD_WRITE )
    set(DESTDIR "/usr")
    INSTALL(TARGETS ${PROJECT_NAME} 
        LIBRARY DESTINATION "${DESTDIR}/lib"
        PUBLIC_HEADER DESTINATION "${DESTDIR}/include"
    )
endif (UNIX)

# Wrap up  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
message(STATUS "Version     : " ${CMAKE_PROJECT_VERSION}  )
message(STATUS "Build type  : " ${CMAKE_BUILD_TYPE} )
