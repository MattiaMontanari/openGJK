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


# PLATFORM-SPECIFIC SETTING
if (UNIX)
    find_library(M_LIB m)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm")
else ()
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra  -Werror")

  add_compile_options(-static-libgcc -static-libstdc++ )
  add_definitions(-DMT)

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4131 /wd4701 /wd4255 /wd4710  /wd4820 /wd4711 /wd5045")
  set(CMAKE_C_FLAGS_DEBUG "-DDEBUG /D_DEBUG /MDd /Zi  /Ob0 /Od /RTC1")
  set(CMAKE_C_FLAGS_RELEASE "/Ox")

  set(CMAKE_SUPPRESS_REGENERATION true)

endif()

if (UNIX AND NOT WIN32)

    # Activate with: -DCMAKE_BUILD_TYPE=Debug
    set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG -Wall -Wextra  -Werror" 
        CACHE STRING "Flags used by the C compiler during DEBUG builds.")

    # Activate with: -DCMAKE_BUILD_TYPE=Release
    set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -finline-functions -Wextra  -Werror"
        CACHE STRING "Flags used by the C compiler during RELEASE builds.")

    # Activate with: -DCMAKE_BUILD_TYPE=Profiling
    set(CMAKE_C_FLAGS_PROFILING "-O0 -g -fprofile-arcs -ftest-coverage"
        CACHE STRING "Flags used by the C compiler during PROFILING builds.")
    set(CMAKE_CXX_FLAGS_PROFILING "-O0 -g -fprofile-arcs -ftest-coverage"
        CACHE STRING "Flags used by the CXX compiler during PROFILING builds.")
    set(CMAKE_SHARED_LINKER_FLAGS_PROFILING "-fprofile-arcs -ftest-coverage"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during PROFILING builds.")
    set(CMAKE_MODULE_LINKER_FLAGS_PROFILING "-fprofile-arcs -ftest-coverage"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during PROFILING builds.")
    set(CMAKE_EXEC_LINKER_FLAGS_PROFILING "-fprofile-arcs -ftest-coverage"
        CACHE STRING "Flags used by the linker during PROFILING builds.")

    # Activate with: -DCMAKE_BUILD_TYPE=AddressSanitizer
    set(CMAKE_C_FLAGS_ADDRESSSANITIZER "-g -O1 -fsanitize=address -fno-omit-frame-pointer"
        CACHE STRING "Flags used by the C compiler during ADDRESSSANITIZER builds.")
    set(CMAKE_CXX_FLAGS_ADDRESSSANITIZER "-g -O1 -fsanitize=address -fno-omit-frame-pointer"
        CACHE STRING "Flags used by the CXX compiler during ADDRESSSANITIZER builds.")
    set(CMAKE_SHARED_LINKER_FLAGS_ADDRESSSANITIZER "-fsanitize=address"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during ADDRESSSANITIZER builds.")
    set(CMAKE_MODULE_LINKER_FLAGS_ADDRESSSANITIZER "-fsanitize=address"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during ADDRESSSANITIZER builds.")
    set(CMAKE_EXEC_LINKER_FLAGS_ADDRESSSANITIZER "-fsanitize=address"
        CACHE STRING "Flags used by the linker during ADDRESSSANITIZER builds.")

    # Activate with: -DCMAKE_BUILD_TYPE=MemorySanitizer
    set(CMAKE_C_FLAGS_MEMORYSANITIZER "-g -O2 -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer"
        CACHE STRING "Flags used by the C compiler during MEMORYSANITIZER builds.")
    set(CMAKE_CXX_FLAGS_MEMORYSANITIZER "-g -O2 -fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer"
        CACHE STRING "Flags used by the CXX compiler during MEMORYSANITIZER builds.")
    set(CMAKE_SHARED_LINKER_FLAGS_MEMORYSANITIZER "-fsanitize=memory"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during MEMORYSANITIZER builds.")
    set(CMAKE_MODULE_LINKER_FLAGS_MEMORYSANITIZER "-fsanitize=memory"
        CACHE STRING "Flags used by the linker during the creation of shared libraries during MEMORYSANITIZER builds.")
    set(CMAKE_EXEC_LINKER_FLAGS_MEMORYSANITIZER "-fsanitize=memory"
        CACHE STRING "Flags used by the linker during MEMORYSANITIZER builds.")

endif()
