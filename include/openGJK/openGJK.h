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

#ifndef OPENGJK_H__
#define OPENGJK_H__

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Use double as default precision
#define gkFloat double

/// @brief Structure of a body
typedef struct gkPolytope_ {
  int numpoints;        // Number of points defining the body
  gkFloat s[3];         // Support mapping computed last
  gkFloat **coord;      // Points' coordinates
} gkPolytope;

/// @brief Structure of the simplex
typedef struct gkSimplex_ {
  int nvrtx;            // Number of simplex's vertices
  int wids[4];          // Label of the simplex's vertices
  gkFloat lambdas[4];   // Barycentric coordiantes for each vertex
  gkFloat vrtx[4][3];   // Coordinates of simplex's vertices
} gkSimplex;

/// @brief Uses the GJK algorithm to compute the minimum distance between two bodies
gkFloat compute_minimum_distance(const gkPolytope p_, const gkPolytope q_, gkSimplex *s_);

#ifdef __cplusplus
}
#endif

#endif  // OPENGJK_H__
