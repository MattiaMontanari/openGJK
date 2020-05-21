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
#   OpenGJK is free software: you can redistribute it and/or modify       #
#    it under the terms of the GNU General Public License as published by #
#    the Free Software Foundation, either version 3 of the License, or    #
#    any later version.                                                   #
#                                                                         #
#   OpenGJK is distributed in the hope that it will be useful,           #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        #
#    GNU General Public License for more details.                         #
#                                                                         #
#   You should have received a copy of the GNU General Public License     #
#    along with OpenGJK. If not, see <https://www.gnu.org/licenses/>.     #
#                                                                         #
#        openGJK: open-source Gilbert-Johnson-Keerthi algorithm           #
#             Copyright (C) Mattia Montanari 2018 - 2020                  #
#               http://iel.eng.ox.ac.uk/?page_id=504                      #
#                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

import numpy as np
import openGJK_cython as opengjk

a = np.array([[1.,1.,1.],[1.,1.,1.]])
b = np.array([[11.,1.,1.],[1.,1.,1.]])
d = opengjk.pygjk(a,b)

print(d)