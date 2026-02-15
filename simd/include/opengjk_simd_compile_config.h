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

#ifndef OPENGJK_SIMD_CONFIG_H_
#define OPENGJK_SIMD_CONFIG_H_

#include "hwy/detect_compiler_arch.h"

// Disable SVE targets on Apple Silicon (M1/M2/M3/M4).
// Apple Silicon only supports NEON, not SVE. Additionally, SVE uses "sizeless
// types" which cannot be used in arrays (e.g., V (&S)[4] in our simplex code).
// Highway normally marks SVE as broken on Apple, but we explicitly disable it
// here to ensure compile-time safety regardless of Highway version.
#if HWY_ARCH_ARM_A64 && HWY_OS_APPLE
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SVE | HWY_SVE2 | HWY_SVE_256 | HWY_SVE2_128)
#endif
#endif

#endif // OPENGJK_SIMD_CONFIG_H_
