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

project(example_lib_opengjk_ce
    LANGUAGES C CXX
    VERSION 1.0.0
)

# Force C++ compiler for C files
if(FORCE_CXX_COMPILER)
    file(GLOB_RECURSE CFILES "${CMAKE_CURRENT_SOURCE_DIR}/main.c")
    SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX)
endif(FORCE_CXX_COMPILER)

add_executable(example_lib_opengjk_ce ${CMAKE_CURRENT_SOURCE_DIR}/main.c)

if(WIN32)
    target_link_libraries(example_lib_opengjk_ce opengjk_scalar_static)
else()
    target_link_libraries(example_lib_opengjk_ce opengjk_scalar_static m)
endif()

# Copy input files for this example after build
add_custom_command(
    TARGET example_lib_opengjk_ce POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/userP.dat
    ${CMAKE_CURRENT_BINARY_DIR}/userP.dat
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/userQ.dat
    ${CMAKE_CURRENT_BINARY_DIR}/userQ.dat
)
