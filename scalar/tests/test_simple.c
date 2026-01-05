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
#include <stdio.h>
#include <cmocka.h>
#include "openGJK/openGJK.h"

#define FLOAT_TOL gkEpsilon

static void
access_structures(void** state) {
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
