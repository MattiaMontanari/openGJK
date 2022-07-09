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

cmake_minimum_required(VERSION          2.8)
cmake_policy(SET CMP0048                NEW) 

option(BUILD_EXAMPLE "Build demo"        ON)
set(GK_VERSION_MAJOR                      3)
set(GK_VERSION_MINOR                      0)
set(GK_VERSION_PATCH                      0)

project(lib_opengjk_ce 
    LANGUAGES C
	DESCRIPTION "openGJK library community edition (CE)"
    HOMEPAGE_URL "www.mattiamontanari.com/opengjk/"
    VERSION ${GK_VERSION_MAJOR}.${GK_VERSION_MINOR}.${GK_VERSION_PATCH}
)

set( C_STANDARD                          99)
set( CMAKE_CXX_VISIBILITY_PRESET         hidden)
set( CMAKE_VISIBILITY_INLINES_HIDDEN     1)
set( CMAKE_POLICY_DEFAULT_CMP0079        NEW)
set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS    ON)
set( CMAKE_INCLUDE_CURRENT_DIR           TRUE)
    
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -pedantic -Wunused-macros")
set( CMAKE_C_FLAGS_DEBUG                 "-O0 -g -Wall -Wno-unused-command-line-argument")
set( CMAKE_C_FLAGS_RELEASE               "-O3 -Werror -Wno-unused-command-line-argument")

set( GK_PUBLIC_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} 
    STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/openGJK.c 
    ${GK_PUBLIC_HEADER_DIR}/openGJK/openGJK.h
)

target_link_libraries(${PROJECT_NAME}  m)

target_include_directories( 
    ${PROJECT_NAME} 
    PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(${PROJECT_NAME} 
    PROPERTIES 
        PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/openGJK/openGJK.h
)

if(BUILD_EXAMPLE)
    add_subdirectory(examples/c)
endif(BUILD_EXAMPLE)

if (UNIX)
    install(TARGETS ${PROJECT_NAME} PERMISSIONS WORLD_WRITE )
    set(DESTDIR "/usr")
    INSTALL(TARGETS ${PROJECT_NAME} 
        LIBRARY DESTINATION "${DESTDIR}/lib"
        PUBLIC_HEADER DESTINATION "${DESTDIR}/include"
    )
endif (UNIX)

# Wrap up feedback on setup
message(STATUS "Version     : " ${CMAKE_PROJECT_VERSION}  )
message(STATUS "Build type  : " ${CMAKE_BUILD_TYPE} )