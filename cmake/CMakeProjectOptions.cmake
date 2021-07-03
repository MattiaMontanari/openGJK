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
#   openGJK is distributed in the hope that it will be useful,            #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        #
#    GNU General Public License for more details.                         #
#                                                                         #
#   You should have received a copy of the GNU General Public License     #
#    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.     #
#                                                                         #
#        openGJK: open-source Gilbert-Johnson-Keerthi algorithm           #
#             Copyright (C) Mattia Montanari 2018 - 2020                  #
#               http://iel.eng.ox.ac.uk/?page_id=504                      #
#                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

option(WITH_STATIC_LIB   "Build static lib"   OFF)
option(WITH_EXAMPLES 	 "Build C example" 	  ON)

# Default build type
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release")

# APPLY USER OPTIONS
IF (WITH_STATIC_LIB)
    set(BUILD_STATIC_LIB ON)
ENDIF (WITH_STATIC_LIB)

# FEEDBACK
message(STATUS "  Build static lib (ON): " ${WITH_STATIC_LIB})
message(STATUS "  Build C examples (ON): " ${WITH_EXAMPLES})