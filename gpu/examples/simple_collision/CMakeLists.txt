#                            _____      _ _  __                                    #
#                           / ____|    | | |/ /                                    #
#     ___  _ __   ___ _ __ | |  __     | | ' /                                     #
#    / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                      #
#   | (_) | |_) |  __/ | | | |__| | |__| | . \                                     #
#    \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                    #
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

project(example_lib_openGJK_GPU
    LANGUAGES C CXX CUDA
    VERSION 1.0.0
)

add_executable(example_lib_openGJK_GPU ${CMAKE_CURRENT_SOURCE_DIR}/main.cu)

target_link_libraries(example_lib_openGJK_GPU openGJK_GPU)

# CUDA architecture handling (same as parent)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(example_lib_openGJK_GPU PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(example_lib_openGJK_GPU PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(example_lib_openGJK_GPU PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Copy input files to the executable directory (handles multi-config generators like Visual Studio)
add_custom_command(
    TARGET example_lib_openGJK_GPU POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/userP.dat
    $<TARGET_FILE_DIR:example_lib_openGJK_GPU>/userP.dat
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/userQ.dat
    $<TARGET_FILE_DIR:example_lib_openGJK_GPU>/userQ.dat
)
