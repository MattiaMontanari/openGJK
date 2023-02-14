#                           _____      _ _  __                                   #
#                          / ____|    | | |/ /                                   #
#    ___  _ __   ___ _ __ | |  __     | | ' /                                    #
#   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                     #
#  | (_) | |_) |  __/ | | | |__| | |__| | . \                                    #
#   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                   #
#        | |                                                                     #
#        |_|                                                                     #
#                                                                                #
# Copyright 2022 Mattia Montanari, University of Oxford                          #
#                                                                                #
# This program is free software: you can redistribute it and/or modify it under  #
# the terms of the GNU General Public License as published by the Free Software  #
# Foundation, either version 3 of the License. You should have received a copy   #
# of the GNU General Public License along with this program. If not, visit       #
#                                                                                #
#     https://www.gnu.org/licenses/                                              #
#                                                                                #
# This program is distributed in the hope that it will be useful, but WITHOUT    #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  #
# FOR A PARTICULAR PURPOSE. See GNU General Public License for details.          #

import numpy as np
import openGJK_cython as opengjk

a = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0]])
b = np.array([[-1.,-1.,-1.],[-1.,-1.,-1.]])
d = opengjk.pygjk(a,b)

print("Distance is:" , d)