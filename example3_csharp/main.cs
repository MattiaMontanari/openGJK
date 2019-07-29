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
    [DllImport("libopenGJKlib.so", EntryPoint="csFunction", CallingConvention = CallingConvention.StdCall)]
#else 
    [DllImport("openGJKlib", EntryPoint = "csFunction", CallingConvention = CallingConvention.StdCall)]
#endif
    static extern double gjk(int na, double [,] ia, int nb, double [,] ib);

        public static void Main(string[] args)
        {
		double dist;
		// Define array A with coordinates
		int  nCoordsA = 9;
                var inCoordsA = new double[3,9] { {0.0 , 2.3 , 8.1 , 4.3  ,2.5 , 7.1 , 1.0 , 3.3 , 6.0} , { 5.5 , 1.0 , 4.0 , 5.0  ,1.0,  1.0,  1.5,  0.5 , 1.4} ,{ 0.0 , -2.0,  2.4,  2.2,  2.3 , 2.4 , 0.3 , 0.3 , 0.2} };

		// Define array B with coordinates 
		int nCoordsB = 9;
                var inCoordsB = new double[3,9] { {-0.0 , -2.3 , -8.1 , -4.3  ,-2.5 , -7.1 , -1.0 , -3.3 , -6.0} , { -5.5 , -1.0 ,- 4.0 ,- 5.0  ,-1.0,  -1.0,  -1.5,  -0.5 , -1.4} ,{ -0.0 , 2.0,  -2.4,  -2.2,  -2.3 , -2.4 , -0.3 , -0.3 , -0.2} };

		// Invoke GJK to compute distance
		dist = gjk( nCoordsA, inCoordsA, nCoordsB, inCoordsB );
 
		// Output results
		var s = string.Format("{0:0.##}", dist);
		var message = string.Format("The distance between {0} is {1}","A and B",s);
		Console.WriteLine(message);
		Console.WriteLine("Press any key to exit");
		Console.ReadLine();
        }
}
