/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
 *       | |
 *       |_|
 *
 * Copyright 2022-2026 Mattia Montanari, University of Oxford
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3. See https://www.gnu.org/licenses/
 */

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
subalg_1simplex_tets_v1(void** state) {

  UNUSED(state);

  s.nvrtx = 2;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 0.0;

  s.vrtx[1][0] = 0.1;
  s.vrtx[1][1] = 0.0;
  s.vrtx[1][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 1);
  assert_float_equal(v[0], 0.1, FLOAT_TOL);
  assert_float_equal(v[1], 0.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.0, FLOAT_TOL);

  s.nvrtx = 2;

  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = -3.0;
  s.vrtx[0][2] = 0.0;

  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = -1.0;
  s.vrtx[1][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 1);
  assert_float_equal(v[0], -1.0, FLOAT_TOL);
  assert_float_equal(v[1], -1.0, FLOAT_TOL);
  assert_float_equal(v[2], 0.0, FLOAT_TOL);
}

static void
subalg_1simplex_tets_v12(void** state) {

  UNUSED(state);

  s.nvrtx = 2;

  s.vrtx[0][0] = 1.0;
  s.vrtx[0][1] = 0.0;
  s.vrtx[0][2] = 0.0;

  s.vrtx[1][0] = 0.0;
  s.vrtx[1][1] = 0.1;
  s.vrtx[1][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], 9.900990e-03, FLOAT_TOL);
  assert_float_equal(v[1], 9.900990e-02, FLOAT_TOL);
  assert_float_equal(v[2], 0.000000e+00, FLOAT_TOL);

  s.nvrtx = 2;

  s.vrtx[0][0] = -1.0;
  s.vrtx[0][1] = -3.0;
  s.vrtx[0][2] = 0.0;

  s.vrtx[1][0] = -1.0;
  s.vrtx[1][1] = 1.0;
  s.vrtx[1][2] = 0.0;

  subalgorithm(&s, v);

  assert_int_equal(s.nvrtx, 2);
  assert_float_equal(v[0], -1.000000e+00, FLOAT_TOL);
  assert_float_equal(v[1], 0.000000e+00, FLOAT_TOL);
  assert_float_equal(v[2], 0.000000e+00, FLOAT_TOL);
}

int
main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(subalg_1simplex_tets_v1),
      cmocka_unit_test(subalg_1simplex_tets_v12),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
