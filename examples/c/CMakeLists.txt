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

project(openGJKdemo VERSION 1.0.0 LANGUAGES C)

set(APPLICATION_NAME ${PROJECT_NAME})
set(CMAKE_C_STANDARD 11)
set(TEST_NAME ${PROJECT_NAME}_CTEST)

message( "[${PROJECT_NAME}] CMake setting ..")

# Set source file
set(SOURCE_FILES main.c )

# Create the executable
add_executable(demo ${SOURCE_FILES})

# Copy input files after build
add_custom_command(
        TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/c/userP.dat
                ${CMAKE_CURRENT_BINARY_DIR}/userP.dat
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/examples/c/userQ.dat
                ${CMAKE_CURRENT_BINARY_DIR}/userQ.dat)

# PLATFORM-SPECIFIC SETTING
if (UNIX)
    find_library(M_LIB m)
	# Link to openGJK and math library
	target_link_libraries(demo openGJKlib m)
else ()
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_link_libraries(demo openGJKlib)
endif ()


include(AddCMockaTest)
find_library(LIB_CMOCKA cmocka)
    target_link_libraries(demo ${CMOCKA_LIBRARY})
if(NOT LIB_CMOCKA)
  message(STATUS "${LIB_CMOCKA} library not found ")
endif() 

if (UNIT_TESTING)
    target_link_libraries(demo cmocka)
    add_test( ${TEST_NAME} demo)
endif (UNIT_TESTING)

message(STATUS "Completed CMake setting for ${PROJECT_NAME}" )