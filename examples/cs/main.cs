/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
 *                                   #####        # #    #                *
 *       ####  #####  ###### #    # #     #       # #   #                 *
 *      #    # #    # #      ##   # #             # #  #                  *
 *      #    # #    # #####  # #  # #  ####       # ###                   *
 *      #    # #####  #      #  # # #     # #     # #  #                  *
 *      #    # #      #      #   ## #     # #     # #   #                 *
 *       ####  #      ###### #    #  #####   #####  #    #                *
 *                                                                        *
 *  This file is part of openGJK.                                         *
 *                                                                        *
 *  openGJK is free software: you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by *
 *   the Free Software Foundation, either version 3 of the License, or    *
 *   any later version.                                                   *
 *                                                                        *
 *   openGJK is distributed in the hope that it will be useful,           *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See The        *
 *   GNU General Public License for more details.                         *
 *                                                                        *
 *  You should have received a copy of the GNU General Public License     *
 *   along with Foobar.  If not, see <https://www.gnu.org/licenses/>.     *
 *                                                                        *
 *       openGJK: open-source Gilbert-Johnson-Keerthi algorithm           *
 *            Copyright (C) Mattia Montanari 2018 - 2019                  *
 *              http://iel.eng.ox.ac.uk/?page_id=504                      *
 *                                                                        *
 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

using System;

using System.Runtime.InteropServices;

public class Tester
{
#if UNIX
    [DllImport("libopengjk_ce.so", EntryPoint="csFunction", CallingConvention = CallingConvention.StdCall)]
#else 
    [DllImport("libopengjk_ce", EntryPoint = "csFunction", CallingConvention = CallingConvention.StdCall)]
#endif

    static extern float compute_minimum_distance(int na, float[,] ia, int nb, float[,] ib);

    public static void Main(string[] args)
    {
        float dist;
        // Define array A with coordinates
        int nCoordsA = 9;
        var inCoordsA = new float[3, 9] { { 0.0f, 2.3f, 8.1f, 4.3f, 2.5f, 7.1f, 1.0f, 3.3f, 6.0f }, { 5.5f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 1.5f, 0.5f, 1.4f }, { 0.0f, -2.0f, 2.4f, 2.2f, 2.3f, 2.4f, 0.3f, 0.3f, 0.2f } };

        // Define array B with coordinates 
        int nCoordsB = 9;
        var inCoordsB = new float[3, 9] { { -0.0f, -2.3f, -8.1f, -4.3f, -2.5f, -7.1f, -1.0f, -3.3f, -6.0f }, { -5.5f, -1.0f, -4.0f, -5.0f, -1.0f, -1.0f, -1.5f, -0.5f, -1.4f }, { -0.0f, 2.0f, -2.4f, -2.2f, -2.3f, -2.4f, -0.3f, -0.3f, -0.2f } };

        // Invoke GJK to compute distance
        dist = compute_minimum_distance(nCoordsA, inCoordsA, nCoordsB, inCoordsB);

        // Output results
        var s = string.Format("{0:0.##}", dist);
        var message = string.Format("The distance between {0} is {1}", "A and B", s);
        Console.WriteLine(message);

    }
}
