# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                                    #####        # #    #                #
#        ####  #####  ###### #    # #     #       # #   #                 #
#       #    # #    # #      ##   # #             # #  #                  #
#       #    # #    # #####  # #  # #  ####       # ###                   #
#       #    # #####  #      #  # # #     # #     # #  #                  #
#       #    # #      #      #   ## #     # #     # #   #                 #
#        ####  #      ###### #    #  #####   #####  #    #                #
#                                                                         #
#   This file is part of openGJK.                                         #
#                                                                         #
#   openGJK is free software: you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by #
#    the Free Software Foundation, either version 3 of the License, or    #
#    any later version.                                                   #
#                                                                         #
#    openGJK is distributed in the hope that it will be useful,           #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        #
#    GNU General Public License for more details.                         #
#                                                                         #
#   You should have received a copy of the GNU General Public License     #
#    along with openGJK. If not, see <https://www.gnu.org/licenses/>.     #
#                                                                         #
#        openGJK: open-source Gilbert-Johnson-Keerthi algorithm           #
#             Copyright (C) Mattia Montanari 2018 - 2019                  #
#               http://iel.eng.ox.ac.uk/?page_id=504                      #
#                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

cmake_minimum_required(VERSION   3.13)

set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)
set(LIBRARY_VERSION              "2.0.3")

project(openGJKlib VERSION ${LIBRARY_VERSION} LANGUAGES C)

set(CMAKE_C_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakeDefaults)
include(CompilerFlags)
include(PlatformDefaults)

message( "[${PROJECT_NAME}] CMake setting ..")
message(STATUS "Version     : " ${CMAKE_PROJECT_VERSION}  )
message(STATUS "Build type  : " ${CMAKE_BUILD_TYPE} )

# Specify project specific and user custum options
include(CMakeProjectOptions)

set( SOURCE_FILES src/openGJK.c )
set( SOURCE_HEADS include/openGJK/openGJK.h)

IF(BUILD_STATIC_LIB)
    add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${SOURCE_HEADS})
    add_definitions(-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=FALSE)
ELSE()
    add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${SOURCE_HEADS})
    add_definitions(-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE)
ENDIF(BUILD_STATIC_LIB)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${SOURCE_HEADS})

# Add compiler flags
include(CompilerFlags)

# Install setup
install(TARGETS ${PROJECT_NAME} PERMISSIONS WORLD_WRITE )

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}") 
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Link include file
target_include_directories( ${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} ${CMOCKA_LIBRARY}  )

set(DESTDIR "/usr")
INSTALL(TARGETS ${PROJECT_NAME} 
        LIBRARY DESTINATION "${DESTDIR}/lib"
        PUBLIC_HEADER DESTINATION "${DESTDIR}/include"
)

if (WITH_EXAMPLES)
    add_subdirectory(examples/c)
endif (WITH_EXAMPLES)

message(STATUS "Completed CMake setting for ${PROJECT_NAME}" )