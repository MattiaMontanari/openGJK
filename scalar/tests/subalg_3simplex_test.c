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

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <cmocka.h>
#include "../openGJK.c"
#include "openGJK/openGJK.h"

#define UNUSED(x) (void)(x)
#define FLOAT_TOL (gkFloat) gkEpsilon * 1e1f

gkFloat v[3] = {1, 1, 1};
gkSimplex s;

static void
subalg_3simplex_tets_v1(void** state) {

  UNUSED(state);

  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 1.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 0.5;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.5, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 1);

  /* Baseline test # 1*/
  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = -1.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = -1.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -2.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = -1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], -1.0, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 1);
}

static void
subalg_3simplex_tets_v12(void** state) {

  UNUSED(state);

  // Degenerate case
  s.nvrtx = 4;

  s.vrtx[3][0] = -2.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 0;
  s.vrtx[2][0] = -1.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = -1000000;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -1000000;
  s.vrtx[0][0] = -3.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -1000000;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -1.99999999, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], -1.99999999e-06, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = -1.0;
  s.vrtx[0][2] = 0.5;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = 0.5;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = 0.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.5, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.5, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 3*/
  s.nvrtx = 4;

  s.vrtx[0][0] = -7.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = 7.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.5;
  s.vrtx[3][0] = -2.0;
  s.vrtx[3][1] = 0.5;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.0733137829, FLOAT_TOL);
  assert_float_equal(v[1], 0.3929618768, FLOAT_TOL);
  assert_float_equal(v[2], -0.4281524926, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 7 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -2.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = 2.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.5;
  s.vrtx[3][0] = -0.25;
  s.vrtx[3][1] = 0.2;
  s.vrtx[3][2] = 0.2;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.014080965551923591, FLOAT_TOL);
  assert_float_equal(v[1], 0.17902941916017101, FLOAT_TOL);
  assert_float_equal(v[2], -0.030676389238119162, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 8 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -2.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = 2.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.5;
  s.vrtx[3][0] = 0.25;
  s.vrtx[3][1] = 0.2;
  s.vrtx[3][2] = 0.2;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.014080965551923591, FLOAT_TOL);
  assert_float_equal(v[1], 0.17902941916017101, FLOAT_TOL);
  assert_float_equal(v[2], -0.030676389238119162, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 12 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 5.0;
  s.vrtx[0][1] = -3.0;
  s.vrtx[0][2] = -1.0;
  s.vrtx[1][0] = -5.0;
  s.vrtx[1][1] = -3.0;
  s.vrtx[1][2] = -1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 3.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = -1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], -0.058823529411764719, FLOAT_TOL);
  assert_float_equal(v[2], -0.23529411764705882, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 14 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = -5.0;
  s.vrtx[0][2] = -0.1;
  s.vrtx[1][0] = -5.0;
  s.vrtx[1][1] = -5.0;
  s.vrtx[1][2] = -0.1;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = 0.15;
  s.vrtx[3][1] = -0.5;
  s.vrtx[3][2] = -0.1;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.11678832116788321, FLOAT_TOL);
  assert_float_equal(v[1], -0.16788321167883213, FLOAT_TOL);
  assert_float_equal(v[2], -0.29927007299270075, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 17 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2.5;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -5000;
  s.vrtx[1][1] = 0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = -1.0;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = -0.5;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.21171708236380071, FLOAT_TOL);
  assert_float_equal(v[1], 0.15480471529393666, FLOAT_TOL);
  assert_float_equal(v[2], -0.057037364666064239, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 18 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 3.1;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -1000;
  s.vrtx[1][1] = 0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = 2.1;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = -0.22;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 20 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 4.5;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -2.5;
  s.vrtx[1][1] = -0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = 0.5;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = 1.5;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 1.0025615782e+00, FLOAT_TOL);
  assert_float_equal(v[1], 1.0647413809e-01, FLOAT_TOL);
  assert_float_equal(v[2], -3.6795566008e-01, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 1.0025615782e+00, FLOAT_TOL);
  assert_float_equal(v[1], 1.0647413809e-01, FLOAT_TOL);
  assert_float_equal(v[2], -3.6795566008e-01, FLOAT_TOL);

  /* Baseline test # 22 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2.5;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -4.5;
  s.vrtx[1][1] = -0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = 0.5;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = -1.5;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -1.0038776664, FLOAT_TOL);
  assert_float_equal(v[1], 0.15620152, FLOAT_TOL);
  assert_float_equal(v[2], -0.364336417, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 23 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -4;
  s.vrtx[1][0] = -2;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -4;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -4;
  s.vrtx[3][0] = 0;
  s.vrtx[3][1] = 1;
  s.vrtx[3][2] = 1;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.29411764705882359, FLOAT_TOL);
  assert_float_equal(v[2], -0.17647058823529416, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 26 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -3;
  s.vrtx[1][0] = -2;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -3;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -3;
  s.vrtx[3][0] = 0;
  s.vrtx[3][1] = 1;
  s.vrtx[3][2] = -2.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.09766925638179802, FLOAT_TOL);
  assert_float_equal(v[2], -2.9300776914539401, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 30 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -1;
  s.vrtx[0][1] = 5;
  s.vrtx[0][2] = -5;
  s.vrtx[1][0] = 1;
  s.vrtx[1][1] = -2;
  s.vrtx[1][2] = -5;
  s.vrtx[2][0] = -3;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -5;
  s.vrtx[3][0] = -1;
  s.vrtx[3][1] = 2;
  s.vrtx[3][2] = -4.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.049475262368815498, FLOAT_TOL);
  assert_float_equal(v[1], 0.098950524737630996, FLOAT_TOL);
  assert_float_equal(v[2], -4.947526236881559, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  /* Baseline test # 31 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 1;
  s.vrtx[0][1] = 5;
  s.vrtx[0][2] = -5;
  s.vrtx[1][0] = 3;
  s.vrtx[1][1] = -2;
  s.vrtx[1][2] = -5;
  s.vrtx[2][0] = -1;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -5;
  s.vrtx[3][0] = 1;
  s.vrtx[3][1] = 2;
  s.vrtx[3][2] = -4.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.049475262368815498, FLOAT_TOL);
  assert_float_equal(v[1], 0.098950524737630996, FLOAT_TOL);
  assert_float_equal(v[2], -4.9475262368815596, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);
}

static void
subalg_3simplex_tets_v13(void** state) {

  UNUSED(state);

  // Degenerate case
  s.nvrtx = 4;

  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = -10.0;
  s.vrtx[3][2] = 10;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = -9.0;
  s.vrtx[2][2] = -1000000;
  s.vrtx[1][0] = 0.0;
  s.vrtx[1][1] = -10.0;
  s.vrtx[1][2] = -1000000;
  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = -10.0;
  s.vrtx[0][2] = -1000000;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 9.999910000e-06, FLOAT_TOL);
  assert_float_equal(v[1], -9.99999, FLOAT_TOL);
  assert_float_equal(v[2], -9.99988e-06, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = -1.0;
  s.vrtx[0][2] = 0.5;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = 0.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = 0.5;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.5, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.5, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);
}

static void
subalg_3simplex_tets_v14(void** state) {

  UNUSED(state);

  // Degenerate case
  s.nvrtx = 4;

  s.vrtx[3][0] = 8.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 0;
  s.vrtx[2][0] = 9.0;
  s.vrtx[2][1] = 0.0;
  s.vrtx[2][2] = -1000000;
  s.vrtx[1][0] = 8.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -1000000;
  s.vrtx[0][0] = 7.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -1000000;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 7.99999999, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], -7.99999999e-06, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);

  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 0.0;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = 0.5;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 0.5;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.5, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.5, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 2);
}

static void
subalg_3simplex_tets_v123(void** state) {

  UNUSED(state);

  // Degenerate case
  s.nvrtx = 4;
  s.vrtx[3][0] = -2.0;
  s.vrtx[3][1] = -10.0;
  s.vrtx[3][2] = 0;
  s.vrtx[2][0] = 1000.0;
  s.vrtx[2][1] = -10.0;
  s.vrtx[2][2] = 0;
  s.vrtx[1][0] = -2.0;
  s.vrtx[1][1] = -10.0;
  s.vrtx[1][2] = -1000000;
  s.vrtx[0][0] = -3.0;
  s.vrtx[0][1] = -10.0;
  s.vrtx[0][2] = -1000000;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0, FLOAT_TOL);
  assert_float_equal(v[1], -10, FLOAT_TOL);
  assert_float_equal(v[2], 0, FLOAT_TOL);
  // assert_int_equal(s.nvrtx, 3); - EG fails this

  s.nvrtx = 4;

  s.vrtx[0][0] = 10;
  s.vrtx[0][1] = 10;
  s.vrtx[0][2] = 10;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 1.0;
  s.vrtx[3][0] = 1.0;
  s.vrtx[3][1] = 1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.3333333333333333, FLOAT_TOL);
  assert_float_equal(v[1], 0.3333333333333333, FLOAT_TOL);
  assert_float_equal(v[2], 0.6666666666666666, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);
}

static void
subalg_3simplex_tets_v124(void** state) {

  UNUSED(state);

  s.nvrtx = 4;

  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = 1.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = 10;
  s.vrtx[1][1] = 10;
  s.vrtx[1][2] = 10;
  s.vrtx[2][0] = 1.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = 1.0;
  s.vrtx[3][0] = 1.0;
  s.vrtx[3][1] = 1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.3333333333333333, FLOAT_TOL);
  assert_float_equal(v[1], 0.3333333333333333, FLOAT_TOL);
  assert_float_equal(v[2], 0.6666666666666666, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 9 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -2.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = 2.0;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.5;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.800000000000, FLOAT_TOL);
  assert_float_equal(v[2], -0.40000000000, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 10 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -5.0;
  s.vrtx[0][1] = -5.0;
  s.vrtx[0][2] = -0.1;
  s.vrtx[1][0] = 5.0;
  s.vrtx[1][1] = -3.0;
  s.vrtx[1][2] = -0.1;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 3.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = -0.3;
  s.vrtx[3][1] = -1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 11 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -5.0;
  s.vrtx[0][1] = -5.0;
  s.vrtx[0][2] = -0.1;
  s.vrtx[1][0] = 5.0;
  s.vrtx[1][1] = -3.0;
  s.vrtx[1][2] = -0.1;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 3.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = 0.3;
  s.vrtx[3][1] = -1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.052826334609174221, FLOAT_TOL);
  assert_float_equal(v[1], -0.06327154167962456, FLOAT_TOL);
  assert_float_equal(v[2], -0.2689340671012505, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 12 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2.0;
  s.vrtx[0][1] = -5.0;
  s.vrtx[0][2] = -0.1;
  s.vrtx[1][0] = -5.0;
  s.vrtx[1][1] = -5.0;
  s.vrtx[1][2] = -0.1;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = 0.3;
  s.vrtx[3][1] = -0.4;
  s.vrtx[3][2] = -0.1;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.14219589627767243, FLOAT_TOL);
  assert_float_equal(v[1], -0.16383440223297041, FLOAT_TOL);
  assert_float_equal(v[2], -0.30225214667717809, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 15 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 3.1;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -1000;
  s.vrtx[1][1] = 0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = 2.1;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = -0.1;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 3.8091191626003488e-05, FLOAT_TOL);
  assert_float_equal(v[1], 0.12723727709472699, FLOAT_TOL);
  assert_float_equal(v[2], -0.0028841380592822302, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 16 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 3.1;
  s.vrtx[0][1] = 0.2;
  s.vrtx[0][2] = -10;
  s.vrtx[1][0] = -1000;
  s.vrtx[1][1] = 0.2;
  s.vrtx[1][2] = -10;
  s.vrtx[2][0] = 2.1;
  s.vrtx[2][1] = -0.1;
  s.vrtx[2][2] = -10;
  s.vrtx[3][0] = -0.3;
  s.vrtx[3][1] = 0.15;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.04148884032254542, FLOAT_TOL);
  assert_float_equal(v[1], 0.13829613440848473, FLOAT_TOL);
  assert_float_equal(v[2], -0.01219520457965729, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 19 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 10;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -4;
  s.vrtx[1][0] = -10;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -4;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -4;
  s.vrtx[3][0] = 0.02;
  s.vrtx[3][1] = 0.12;
  s.vrtx[3][2] = 0.2;
  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.00752917555, FLOAT_TOL);
  assert_float_equal(v[1], 0.01882293888, FLOAT_TOL);
  assert_float_equal(v[2], -0.0095369557, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 21 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 10;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -4;
  s.vrtx[1][0] = -10;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -4;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -4;
  s.vrtx[3][0] = -0.02;
  s.vrtx[3][1] = 0.12;
  s.vrtx[3][2] = 0.2;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.0075291755, FLOAT_TOL);
  assert_float_equal(v[1], 0.018822938888191747, FLOAT_TOL);
  assert_float_equal(v[2], -0.0095369557033504852, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 24 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -3;
  s.vrtx[1][0] = -2;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -3;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -3;
  s.vrtx[3][0] = 0.3;
  s.vrtx[3][1] = 0.4;
  s.vrtx[3][2] = -2.9;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 27 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 1.25;
  s.vrtx[0][1] = 5;
  s.vrtx[0][2] = -5;
  s.vrtx[1][0] = -3.25;
  s.vrtx[1][1] = -2;
  s.vrtx[1][2] = -5;
  s.vrtx[2][0] = -0.75;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -5;
  s.vrtx[3][0] = 1.25;
  s.vrtx[3][1] = 1;
  s.vrtx[3][2] = -4.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.16426193118756879, FLOAT_TOL);
  assert_float_equal(v[2], -4.927857935627081, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 25 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -3;
  s.vrtx[1][0] = -2;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -3;
  s.vrtx[2][0] = 0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -3;
  s.vrtx[3][0] = -0.3;
  s.vrtx[3][1] = 0.4;
  s.vrtx[3][2] = -2.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.1944751381215, FLOAT_TOL);
  assert_float_equal(v[1], 0.09723756906077, FLOAT_TOL);
  assert_float_equal(v[2], -2.91712707182, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 28 */
  s.nvrtx = 4;

  s.vrtx[0][0] = -1.25;
  s.vrtx[0][1] = 5;
  s.vrtx[0][2] = -5;
  s.vrtx[1][0] = -3.25;
  s.vrtx[1][1] = -2;
  s.vrtx[1][2] = -5;
  s.vrtx[2][0] = -0.75;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -5;
  s.vrtx[3][0] = 1.25;
  s.vrtx[3][1] = 1;
  s.vrtx[3][2] = -4.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], -0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.16426193118756879, FLOAT_TOL);
  assert_float_equal(v[2], -4.927857935627, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 29 */
  s.nvrtx = 4;

  s.vrtx[0][0] = 2;
  s.vrtx[0][1] = 2;
  s.vrtx[0][2] = -3;
  s.vrtx[1][0] = -2;
  s.vrtx[1][1] = 2;
  s.vrtx[1][2] = -3;
  s.vrtx[2][0] = -0;
  s.vrtx[2][1] = -2;
  s.vrtx[2][2] = -3;
  s.vrtx[3][0] = 0.5;
  s.vrtx[3][1] = 0.25;
  s.vrtx[3][2] = -2.9;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.17997644287396949, FLOAT_TOL);
  assert_float_equal(v[1], 0.089988221436984747, FLOAT_TOL);
  assert_float_equal(v[2], -2.92461719670, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);
}

static void
subalg_3simplex_tets_v134(void** state) {

  UNUSED(state);

  // Degenerate case
  s.nvrtx = 4;

  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = -10.0;
  s.vrtx[3][2] = 10;
  s.vrtx[2][0] = 5.0;
  s.vrtx[2][1] = -11.0; // This to -10 makes EG fail!
  s.vrtx[2][2] = -10;
  s.vrtx[1][0] = 2.0;
  s.vrtx[1][1] = -10.0;
  s.vrtx[1][2] = -1000000;
  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = -10.0;
  s.vrtx[0][2] = -1000000;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], -10.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.0, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);

  /* Baseline test # 4*/
  s.nvrtx = 4;

  s.vrtx[0][0] = -17.0;
  s.vrtx[0][1] = 1.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = 7.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 4.0;
  s.vrtx[2][2] = 0.0;
  s.vrtx[3][0] = -1.0;
  s.vrtx[3][1] = 3.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 3);
  assert_float_equal(v[0], 0.0000000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], 1.6153846154e+00, FLOAT_TOL);
  assert_float_equal(v[2], -1.076923e+00, FLOAT_TOL);

  s.nvrtx = 4;

  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = 1.0;
  s.vrtx[0][2] = 1.0;
  s.vrtx[1][0] = 1.0;
  s.vrtx[1][1] = -1.0;
  s.vrtx[1][2] = 1.0;
  s.vrtx[2][0] = 10;
  s.vrtx[2][1] = 10;
  s.vrtx[2][2] = 10;
  s.vrtx[3][0] = 1.0;
  s.vrtx[3][1] = 1.0;
  s.vrtx[3][2] = 0.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.33333333333, FLOAT_TOL);
  assert_float_equal(v[1], 0.33333333333, FLOAT_TOL);
  assert_float_equal(v[2], 0.66666666666, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 3);
}

static void
subalg_3simplex_tets_v1234(void** state) {

  UNUSED(state);

  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 1.0;
  s.vrtx[0][2] = -1.0;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = -1.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = -1.0;
  s.vrtx[2][2] = -1.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.0, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 4);

  /* Baseline test # 2*/
  s.nvrtx = 4;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = -1.0;
  s.vrtx[0][2] = -2.0;
  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = -1.0;
  s.vrtx[1][2] = -2.0;
  s.vrtx[2][0] = 0.0;
  s.vrtx[2][1] = 1.0;
  s.vrtx[2][2] = -2.0;
  s.vrtx[3][0] = 0.0;
  s.vrtx[3][1] = 0.0;
  s.vrtx[3][2] = 1.0;

  subalgorithm(&s, v);

  assert_float_equal(v[0], 0.0, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.0, FLOAT_TOL);
  assert_int_equal(s.nvrtx, 4);
}

int
main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(subalg_3simplex_tets_v1),   cmocka_unit_test(subalg_3simplex_tets_v12),
      cmocka_unit_test(subalg_3simplex_tets_v13),  cmocka_unit_test(subalg_3simplex_tets_v14),
      cmocka_unit_test(subalg_3simplex_tets_v123), cmocka_unit_test(subalg_3simplex_tets_v124),
      cmocka_unit_test(subalg_3simplex_tets_v134), cmocka_unit_test(subalg_3simplex_tets_v1234),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
