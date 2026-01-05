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

#define UNIT_TESTING 1
#define UNUSED(x)    (void)(x)

gkFloat p[3], q[3], r[3], s[3];

static void
HFF1_test(void** state) {

  UNUSED(state);

  p[0] = 1.0;
  p[1] = 1.0;
  p[2] = 1.0;

  q[0] = 2.0;
  q[1] = 2.0;
  q[2] = 2.0;
  assert_int_equal(hff1(p, q), 0);

  q[0] = 1.0;
  q[1] = 1.0;
  q[2] = 1.0;
  assert_int_equal(hff1(p, q), 0);

  q[0] = -1.0;
  q[1] = -1.0;
  q[2] = -1.0;
  assert_int_equal(hff1(p, q), 1);

  q[0] = 2.0;
  q[1] = 2.0;
  q[2] = 0.0;
  assert_int_equal(hff1(p, q), 0);
}

static void
HFF2_test(void** state) {
  UNUSED(state);

  p[0] = 1.0;
  p[1] = 0.0;
  p[2] = 0.0;
  q[0] = 0.0;
  q[1] = 1.0;
  q[2] = 0.0;

  r[0] = 1.0;
  r[1] = 1.0;
  r[2] = 1.0;
  assert_int_equal(hff2(p, q, r), 1);

  r[0] = -1.0;
  r[1] = -1.0;
  r[2] = -1.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 0.5;
  r[1] = 0.5;
  r[2] = 0.5;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 0.5;
  r[1] = 0.5;
  r[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 0.25;
  r[1] = 0.0;
  r[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 0.0;
  r[1] = 0.25;
  r[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 100000;
  r[1] = 100000;
  r[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 1);

  q[0] = 1000;
  q[1] = 0.0;
  q[2] = 0.0;
  r[0] = 100000;
  r[1] = 0.0;
  r[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  q[0] = 100000;
  q[1] = 0.0;
  q[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 1000;
  r[1] = 0.0;
  r[2] = 0.0;
  q[0] = 100000;
  q[1] = 0.0;
  q[2] = 0.0;
  assert_int_equal(hff2(p, q, r), 0);

  r[0] = 0.0;
  r[1] = 1.0;
  r[2] = 0.0;
  q[0] = -1.0;
  q[1] = -1.0;
  q[2] = -1.0;
  assert_int_equal(hff2(p, q, r), 0);
}

static void
HFF3_test(void** state) {

  UNUSED(state);

  p[0] = 1.0;
  p[1] = 0.0;
  p[2] = 0.0;
  q[0] = 0.0;
  q[1] = 1.0;
  q[2] = 0.0;
  r[0] = 1.0;
  r[1] = 1.0;
  r[2] = 0.0;
  assert_int_equal(hff3(p, q, r), 1);

  r[0] = 0.0;
  r[1] = 0.0;
  r[2] = 0.0;
  assert_int_equal(hff3(p, q, r), 1);

  p[2] = -1.0;
  q[2] = -1.0;
  r[2] = -1.0;
  assert_int_equal(hff3(p, q, r), 1);

  r[0] = 1.0;
  r[1] = 1.0;
  r[2] = 1.0;
  assert_int_equal(hff3(p, q, r), 0);
}

int
main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(HFF1_test),
      cmocka_unit_test(HFF2_test),
      cmocka_unit_test(HFF3_test),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}