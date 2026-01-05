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

// Test file for memory leak detection with valgrind.
// This file intentionally leaks memory to verify that valgrind
// correctly detects and reports memory leaks.

#include <stdio.h>
#include <stdlib.h>

int
main() {
  /* ALLOCATE MEMORY - - - - - - - */
  int* array = (int*)malloc(2 * sizeof(int));
  if (array == NULL) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  array[1] = 1 + array[0];

  /* DO NOT FREE MEMORY - - - - - - - - - */
  // free(array);

  return 0;
}
