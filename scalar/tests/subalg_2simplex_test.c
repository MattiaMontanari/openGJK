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

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <cmocka.h>
#include "../openGJK.c"
#include "openGJK/openGJK.h"

#define UNUSED(x) (void)(x)
#define FLOAT_TOL gkEpsilon

gkFloat v[3] = {1, 1, 1};
gkSimplex s;

static void
subalg_2simplex_tets_v1(void** state) {

  UNUSED(state);

  s.nvrtx = 3;

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = 2.0;
  s.vrtx[0][2] = 0.0;
  s.vrtx[1][0] = 2.0;
  s.vrtx[1][1] = -2.0;
  s.vrtx[1][2] = 0.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 1);
  assert_float_equal(v[0], 1.000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], 0.000000e+00, FLOAT_TOL);
  assert_float_equal(v[2], 0.000000e+00, FLOAT_TOL);

  s.nvrtx = 3;

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -4.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -4.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);
  assert_float_equal(v[0], 0.000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], -1.000000e+00, FLOAT_TOL);
  assert_float_equal(v[2], 1.000000e+00, FLOAT_TOL);

  assert_int_equal(s.nvrtx, 1);
}

static void
subalg_2simplex_tets_v12(void** state) {

  UNUSED(state);

  s.nvrtx = 3;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 1.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = 0.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 5.000000e-01, FLOAT_TOL);
  assert_float_equal(v[1], 5.000000e-01, FLOAT_TOL);
  assert_float_equal(v[2], 0.000000e+00, FLOAT_TOL);

  s.nvrtx = 3;

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -4.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -4.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 6.400000e-01, FLOAT_TOL);
  assert_float_equal(v[1], -4.800000e-01, FLOAT_TOL);
  assert_float_equal(v[2], 1.000000e+00, FLOAT_TOL);

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -4.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -4.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 0.000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], -4.000000e+00, FLOAT_TOL);
  assert_float_equal(v[2], 1.000000e+00, FLOAT_TOL);

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -4.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -4.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = -1.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 0.000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], -4.000000e+00, FLOAT_TOL);
  assert_float_equal(v[2], 1.000000e+00, FLOAT_TOL);
}

static void
subalg_2simplex_tets_v13(void** state) {

  UNUSED(state);

  s.nvrtx = 2;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 0.0;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
}

static void
subalg_2simplex_tets_v123(void** state) {

  UNUSED(state);

  s.nvrtx = 3;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);

  s.nvrtx = 3;

  s.vrtx[0][0] = 4.0;
  s.vrtx[0][1] = -3.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -4.0;
  s.vrtx[1][1] = -3.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);

  s.nvrtx = 3;

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -4.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -4.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);
}

int
main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(subalg_2simplex_tets_v1),
      cmocka_unit_test(subalg_2simplex_tets_v12),
      cmocka_unit_test(subalg_2simplex_tets_v13),
      cmocka_unit_test(subalg_2simplex_tets_v123),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
