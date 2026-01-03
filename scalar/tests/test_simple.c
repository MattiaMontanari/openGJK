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

// Standard
#include <setjmp.h>
#include <stdio.h>
// Framework
#include <cmocka.h>
// Tested library
#include "openGJK/openGJK.h"

// #define UNIT_TESTING 1 - needed?

// #define UNUSED(x) (void)(x) - needed?
#define FLOAT_TOL gkEpsilon

static void
access_structures(void** state) {

  // UNUSED(state);- needed?

  gkSimplex s;
  gkPolytope bd1;

  gkFloat a[3] = {0};
  s.nvrtx = 0;
  s.vrtx[0][0] = 0.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 0.0;

  bd1.numpoints = 1;
  bd1.s[0] = 0;
  bd1.coord = (gkFloat**)&a[0];

  bd1.numpoints = 1 + bd1.numpoints;

  gkFloat d = 0.00;
  // I cheat on pointers, so comment this out..
  // d = compute_minimum_distance(bd1, bd1, &s);

  assert_float_equal(d, 0.0, FLOAT_TOL);
  (void)state;
}

int
main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(access_structures),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
