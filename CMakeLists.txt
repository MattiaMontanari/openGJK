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
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.     #
#                                                                         #
#        openGJK: open-source Gilbert-Johnson-Keerthi algorithm           #
#             Copyright (C) Mattia Montanari 2018 - 2019                  #
#               http://iel.eng.ox.ac.uk/?page_id=504                      #
#                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

cmake_minimum_required(VERSION 3.5)
project(openGJK)

set(CMAKE_C_STANDARD 11)

message("[${CMAKE_PROJECT_NAME}] Welcome, please change user options if needed.")

# APPLY DEFAULT SETTINGS
if(NOT CMAKE_BUILD_TYPE)
    message("[${CMAKE_PROJECT_NAME}] Use default CMAKE_BUILD_TYPE")
    set(CMAKE_BUILD_TYPE Release)
endif()


# PLATFORM-SPECIFIC SETTING
if (UNIX)
    find_library(M_LIB m)
    set(CMAKE_C_FLAGS "-lm")
    set(CMAKE_CXX_FLAGS "-lm")
else ()
 	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

# COMPILER SETTING
IF(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_BUILD_TYPE Release)
ELSEIF(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_BUILD_TYPE Debug)
ENDIF()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -finline-functions")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra -finline-functions")
  set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
  set(CMAKE_C_FLAGS_RELEASE "-O3")

  add_compile_options(-static-libgcc -static-libstdc++ )
  add_definitions(-DMT)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4131 /wd4701 /wd4255 /wd4710 /wd4820 /wd4711 /wd5045")
  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG /D_DEBUG /MDd /Zi  /Ob0 /Od /RTC1")
  set(CMAKE_CXX_FLAGS_RELEASE "/Ox")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4131 /wd4701 /wd4255 /wd4710  /wd4820 /wd4711 /wd5045")
  set(CMAKE_C_FLAGS_DEBUG "-DDEBUG /D_DEBUG /MDd /Zi  /Ob0 /Od /RTC1")
  set(CMAKE_C_FLAGS_RELEASE "/Ox")

  set(CMAKE_SUPPRESS_REGENERATION true)

 endif()

# DEBUG FLAGS
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
ENDIF()

# INCLUDE LIBRARY AND EXAMPLE DIR
add_subdirectory(lib)
add_subdirectory(example1_c)
