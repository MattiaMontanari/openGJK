//                           _____      _ _  __                                   //
//                          / ____|    | | |/ /                                   //
//    ___  _ __   ___ _ __ | |  __     | | ' /                                    //
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                     //
//  | (_) | |_) |  __/ | | | |__| | |__| | . \                                    //
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                   //
//        | |                                                                     //
//        |_|                                                                     //
//                                                                                //
// Copyright 2022 Mattia Montanari, University of Oxford                          //
//                                                                                //
// This program is free software: you can redistribute it and/or modify it under  //
// the terms of the GNU General Public License as published by the Free Software  //
// Foundation, either version 3 of the License. You should have received a copy   //
// of the GNU General Public License along with this program. If not, visit       //
//                                                                                //
//     https://www.gnu.org/licenses/                                              //
//                                                                                //
// This program is distributed in the hope that it will be useful, but WITHOUT    //
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  //
// FOR A PARTICULAR PURPOSE. See GNU General Public License for details.          //

/// @author Mattia Montanari
/// @date July 2022

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "openGJK/openGJK.h"

#define fscanf_s fscanf

/// @brief Function for reading input file with body's coordinates.
int
readinput(const char* inputfile, gkFloat*** pts, int* out) {
  int npoints = 0;
  int idx = 0;
  FILE* fp;

  /* Open file. */
#ifdef WIN32
  errno_t err;
  if ((err = fopen_s(&fp, inputfile, "r")) != 0) {
#else
  if ((fp = fopen(inputfile, "r")) == NULL) {
#endif
    fprintf(stdout, "ERROR: input file %s not found!\n", inputfile);
    fprintf(stdout, "  -> The file must be in the folder from which this "
                    "program is launched\n\n");
    return 1;
  }

  /* Read number of input vertices. */
  if (fscanf_s(fp, "%d", &npoints) != 1) {
    return 1;
  }

  /* Allocate memory. */
  gkFloat** arr = (gkFloat**)malloc(npoints * sizeof(gkFloat*));
  for (int i = 0; i < npoints; i++) {
    arr[i] = (gkFloat*)malloc(3 * sizeof(gkFloat));
  }

  /* Read and store vertices' coordinates. */
  for (idx = 0; idx < npoints; idx++) {
#ifdef USE_32BITS
    if (fscanf_s(fp, "%f %f %f\n", &arr[idx][0], &arr[idx][1], &arr[idx][2]) != 3) {
      return 1;
    }
#else
    if (fscanf_s(fp, "%lf %lf %lf\n", &arr[idx][0], &arr[idx][1], &arr[idx][2]) != 3) {
      return 1;
    }
#endif
  }
  fclose(fp);

  *pts = arr;
  *out = idx;

  return (0);
}

/**
 * @brief Main program of example1_c (described in Section 3.1 of the paper).
 *
 */
int
main() {
  /* Squared distance computed by openGJK.                                 */
  gkFloat dd, dx, dy, dz;
  /* Structure of simplex used by openGJK.                                 */
  gkSimplex s;
  /* Number of vertices defining body 1 and body 2, respectively.          */
  int nvrtx1, nvrtx2;
  /* Structures of body 1 and body 2, respectively.                        */
  gkPolytope bd1;
  gkPolytope bd2;
  /* Specify name of input files for body 1 and body 2, respectively.      */
  char inputfileA[40] = "userP.dat", inputfileB[40] = "userQ.dat";
  /* Pointers to vertices' coordinates of body 1 and body 2, respectively. */
  gkFloat(**vrtx1) = NULL, (**vrtx2) = NULL;

  /* For importing openGJK this is Step 2: adapt the data structure for the
   * two bodies that will be passed to the GJK procedure. */

  /* Import coordinates of object 1. */
  if (readinput(inputfileA, &vrtx1, &nvrtx1)) {
    return (1);
  }
  bd1.coord = vrtx1;
  bd1.numpoints = nvrtx1;

  /* Import coordinates of object 2. */
  if (readinput(inputfileB, &vrtx2, &nvrtx2)) {
    return (1);
  }
  bd2.coord = vrtx2;
  bd2.numpoints = nvrtx2;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

  /* For importing openGJK this is Step 3: invoke the GJK procedure. */
  /* Compute squared distance using GJK algorithm. */
  dd = compute_minimum_distance(bd1, bd2, &s);

  /* Print distance between objects. */
  printf("Distance between bodies %f\n", dd);
  printf("Witnesses: (%f, %f, %f) and (%f, %f, %f)\n",
         s.witnesses[0][0], s.witnesses[0][1], s.witnesses[0][2],
         s.witnesses[1][0], s.witnesses[1][1], s.witnesses[1][2]);
  dx = s.witnesses[0][0] - s.witnesses[1][0];
  dy = s.witnesses[0][1] - s.witnesses[1][1];
  dz = s.witnesses[0][2] - s.witnesses[1][2];

  /* Free memory */
  for (int i = 0; i < bd1.numpoints; i++) {
    free(bd1.coord[i]);
  }
  free(bd1.coord);
  for (int i = 0; i < bd2.numpoints; i++) {
    free(bd2.coord[i]);
  }
  free(bd2.coord);

  return (0);
}
