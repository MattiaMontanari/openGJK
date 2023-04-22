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
//                                                                               //
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

/**
 * @file openGJK.h
 * @author Mattia Montanari
 * @date 1 Jan 2023
 * @brief Main interface of OpenGJK containing quick reference and API documentation.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

#ifndef OPENGJK_H__
#define OPENGJK_H__

#include <float.h>
/*! @brief Precision of floating-point numbers.
 *
 * Default is set to 64-bit (Double). Change this to quickly play around with 16- and 32-bit. */
#ifdef USE_32BITS
#define gkFloat   float
#define gkEpsilon FLT_EPSILON
#else
#define gkFloat   double
#define gkEpsilon DBL_EPSILON
#endif

/*! @brief Data structure for convex polytopes.
   *
   * Polytopes are three-dimensional shapes and the GJK algorithm works directly on their convex-hull. However the convex-hull is never computed explicitly, instead each GJK-iteration employs a support function that has a cost linearly dependent on the number of points defining the polytope. */
typedef struct gkPolytope_ {
  int numpoints; /*!< Number of points defining the polytope. */
  gkFloat s
      [3]; /*!< Furthest point returned by the support function and updated at each GJK-iteration. For the first iteration this value is a guess - and this guess not irrelevant. */
  gkFloat**
      coord; /*!< Coordinates of the points of the polytope. This is owned by user who manages and garbage-collects the memory for these coordinates. */
} gkPolytope;

/*! @brief Data structure for simplex.
   *
   * The simplex is updated at each GJK-iteration. For the first iteration this value is a guess - and this guess not irrelevant. */
typedef struct gkSimplex_ {
  int nvrtx;          /*!< Number of points defining the simplex. */
  gkFloat vrtx[4][3]; /*!< Coordinates of the points of the simplex. */
} gkSimplex;

/*! @brief Invoke the GJK algorithm to compute the minimum distance between two polytopes.
   *
   * The simplex has to be initialised prior the call to this function. */
gkFloat compute_minimum_distance(const gkPolytope p_, const gkPolytope q_, gkSimplex* s_);

#endif // OPENGJK_H__
