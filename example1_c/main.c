/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
*                                    #####        # #    #                *
*        ####  #####  ###### #    # #     #       # #   #                 *
*       #    # #    # #      ##   # #             # #  #                  *
*       #    # #    # #####  # #  # #  ####       # ###                   *
*       #    # #####  #      #  # # #     # #     # #  #                  *
*       #    # #      #      #   ## #     # #     # #   #                 *
*        ####  #      ###### #    #  #####   #####  #    #                *
*                                                                         *
*           Mattia Montanari    |   University of Oxford 2018             *
* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
*                                                                         *
* This file runs an example to illustrate how to invoke the openGJK lib   *
*   within a C program. An executable called 'demo' can be compiled with  *
*   CMake. This reads the coordinates of two polytopes from the input     *
*   files userP.dat and userQ.dat, respectively, and returns the minimum  *
*   distance between them computed using the openGJK library.             *
*                                                                         *
* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

/**
 * @file main.c
 * @author Mattia Montanari
 * @date April 2018
 * @brief File illustrating an application that invokes openGJK.
 *
 */

#include <stdio.h>

/* For importing openGJK this is Step 1: include header in subfolder. */
#include "openGJK/openGJK.h"


#ifndef WIN32
#define fscanf_s fscanf
#endif

/**
* @brief Function for reading input file with body's coordinates.
*
*/
int readinput ( const char *inputfile, double ***pts, int * out ) {
  int npoints = 0;
  int idx = 0;
  FILE *fp;

  /* Open file. */
#ifdef WIN32
  errno_t err;
  if ((err = fopen_s(&fp, inputfile, "r")) != 0) {
#else
  if ((fp = fopen(inputfile, "r")) == NULL) {
#endif
    fprintf(stdout, "ERROR: input file %s not found!\n", inputfile);
    fprintf(stdout, "  -> The file must be in the folder from which this program is launched\n\n");
    return 1;
  }

  /* Read number of input vertices. */
  if (fscanf_s(fp, "%1d", &npoints) != 1)
    return 1;

  /* Allocate memory. */
  double **arr = (double **)malloc(npoints * sizeof(double *));
  for (int i=0; i<npoints; i++)
    arr[i] = (double *)malloc(3 * sizeof(double));

  /* Read and store vertices' coordinates. */
  for (idx = 0; idx < npoints; idx++)
  {
    if (fscanf_s(fp, "%lf %lf %lf\n", &arr[idx][0], &arr[idx][1], &arr[idx][2]) != 3 )
      return 1;
  }

  /* Close file. */
  fclose(fp);

  /* Pass pointers. */
  *pts = arr;
  *out = idx;

  return (0);
}


/**
* @brief Main program of example1_c (described in Section 3.1 of the paper).
*
*/
int main() {
  /* Squared distance computed by openGJK.                                 */
  double dd;
  /* Structure of simplex used by openGJK.                                 */
  struct simplex  s;
  /* Number of vertices defining body 1 and body 2, respectively.          */
  int    nvrtx1,
         nvrtx2;
  /* Structures of body 1 and body 2, respectively.                        */
  struct bd       bd1;
  struct bd       bd2;
  /* Specify name of input files for body 1 and body 2, respectively.      */
  char   inputfileA[40] = "userP.dat",
         inputfileB[40] = "userQ.dat";
  /* Pointers to vertices' coordinates of body 1 and body 2, respectively. */
  double (**vrtx1) = NULL,
         (**vrtx2) = NULL;

  /* For importing openGJK this is Step 2: adapt the data structure for the
   * two bodies that will be passed to the GJK procedure. */

  /* Import coordinates of object 1. */
  if (readinput ( inputfileA, &vrtx1, &nvrtx1 ))
    return (1);
  bd1.coord = vrtx1;
  bd1.numpoints = nvrtx1;

  /* Import coordinates of object 2. */
  if (readinput ( inputfileB, &vrtx2, &nvrtx2 ))
    return (1);
  bd2.coord = vrtx2;
  bd2.numpoints = nvrtx2;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

#ifdef DEBUG
  /* Verify input of body A. */
  for (int i = 0; i < bd1.numpoints; ++i) {
    printf ( "%.2f ", vrtx1[i][0]);
    printf ( "%.2f ", vrtx1[i][1]);
    printf ( "%.2f\n", bd1.coord[i][2]);
  }

  /* Verify input of body B. */
  for (int i = 0; i < bd2.numpoints; ++i) {
    printf ( "%.2f ", bd2.coord[i][0]);
    printf ( "%.2f ", bd2.coord[i][1]);
    printf ( "%.2f\n", bd2.coord[i][2]);
  }
#endif

  /* For importing openGJK this is Step 3: invoke the GJK procedure. */
  /* Compute squared distance using GJK algorithm. */
  dd = gjk (bd1, bd2, &s);

  /* Print distance between objects. */
  printf ("Distance between bodies %f\n", dd);

  /* Free memory */
  for (int i=0; i<bd1.numpoints; i++)
    free(bd1.coord[i]);
  free(bd1.coord);
  for (int i=0; i<bd2.numpoints; i++)
    free(bd2.coord[i]);
  free(bd2.coord);

  return (0);
}
