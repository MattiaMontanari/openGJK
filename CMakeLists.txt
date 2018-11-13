# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                                    #####        # #    #                #
#        ####  #####  ###### #    # #     #       # #   #                 #
#       #    # #    # #      ##   # #             # #  #                  #
#       #    # #    # #####  # #  # #  ####       # ###                   #
#       #    # #####  #      #  # # #     # #     # #  #                  #
#       #    # #      #      #   ## #     # #     # #   #                 #
#        ####  #      ###### #    #  #####   #####  #    #                #
#                                                                         #
#           Mattia Montanari    |   University of Oxford 2018             #
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -finline-functions")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -finline-functions")
  set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -Werror")

  add_compile_options(-static-libgcc -static-libstdc++ )
  add_definitions(-DMT)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /WX /wd4131 /wd4701 /wd4255 /wd4710 /wd4820 /wd4711")
  set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG /D_DEBUG /MDd /Zi  /Ob0 /Od /RTC1")
  set(CMAKE_CXX_FLAGS_RELEASE "/Ox")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Wall /WX /wd4131 /wd4701 /wd4255 /wd4710  /wd4820 /wd4711")
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
