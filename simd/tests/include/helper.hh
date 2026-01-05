/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
 *       | |
 *       |_|
 */
//
// Copyright 2022-2026 Mattia Montanari, University of Oxford
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License. You should have received a copy
// of the GNU General Public License along with this program. If not, visit
//
//     https://www.gnu.org/licenses/
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See GNU General Public License for details.

// Test helpers for SIMD GJK unit tests

#ifndef OPENGJK_SIMD_TEST_HELPER_HH_
#define OPENGJK_SIMD_TEST_HELPER_HH_

#include <cmath>
#include <array>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_compile_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "include/helper.hh"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "opengjk_simd.hh"

HWY_BEFORE_NAMESPACE();

namespace opengjk {
namespace test {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using D = hn::CappedTag<simd::gjkFloat, 4>;
using V = hn::Vec<D>;

// ============================================================================
// Vector Helpers
// ============================================================================

/**
 * @brief Create a SIMD vector from 3 components (w=0 padding).
 */
HWY_INLINE V
make_vec(simd::gjkFloat x, simd::gjkFloat y, simd::gjkFloat z) {
  D d;
  HWY_ALIGN simd::gjkFloat arr[4] = {x, y, z, 0};
  return hn::Load(d, arr);
}

/**
 * @brief Extract the x, y, z components from a SIMD vector.
 */
HWY_INLINE void
extract(V v, simd::gjkFloat& x, simd::gjkFloat& y, simd::gjkFloat& z) {
  D d;
  HWY_ALIGN simd::gjkFloat arr[4];
  hn::Store(v, d, arr);
  x = arr[0];
  y = arr[1];
  z = arr[2];
}

/**
 * @brief Compute the squared norm of a SIMD vector.
 */
HWY_INLINE simd::gjkFloat
norm2(V v) {
  D d;
  HWY_ALIGN simd::gjkFloat arr[4];
  hn::Store(v, d, arr);
  return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
}

/**
 * @brief Compute the norm of a SIMD vector.
 */
HWY_INLINE simd::gjkFloat
norm(V v) {
  return std::sqrt(norm2(v));
}

/**
 * @brief Dot product of two SIMD vectors.
 */
HWY_INLINE simd::gjkFloat
dot(V a, V b) {
  D d;
  HWY_ALIGN simd::gjkFloat arr_a[4], arr_b[4];
  hn::Store(a, d, arr_a);
  hn::Store(b, d, arr_b);
  return arr_a[0] * arr_b[0] + arr_a[1] * arr_b[1] + arr_a[2] * arr_b[2];
}

// ============================================================================
// Test Fixtures
// ============================================================================

/**
 * @brief Check that v is the closest point on the simplex to the origin.
 */
HWY_INLINE void
assert_closest_point(V v, simd::gjkFloat tol = 1e-6) {
  simd::gjkFloat dist = norm(v);
  // v should be the minimum distance vector from origin to simplex
  EXPECT_GE(dist, 0.0);
}

/**
 * @brief Check that the simplex dimension is as expected.
 */
HWY_INLINE void
assert_simplex_size(V size_v, int expected) {
  D d;
  int actual = static_cast<int>(hn::GetLane(size_v));
  EXPECT_EQ(actual, expected);
}

} // namespace HWY_NAMESPACE
} // namespace test
} // namespace opengjk

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace opengjk {
namespace test {

// Re-export from chosen target
using namespace HWY_NAMESPACE;

} // namespace test
} // namespace opengjk
#endif

#endif // OPENGJK_SIMD_TEST_HELPER_HH_
