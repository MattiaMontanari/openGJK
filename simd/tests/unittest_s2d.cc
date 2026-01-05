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

// Unit tests for S2D sub-algorithm
// Tests stability and correctness of S2D implementation
//
// VERTEX CONVENTION: S[0] = p (newest), S[1] = q, S[2] = r
// In Progressive search, p must be part of the solution.
//
// NOTE: For verifying SIMD matches scalar exactly, use unittest_s2d_crossval.cc.
// These tests verify stability, non-NaN output, and valid simplex sizes.
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.
//
// DEGENERATE CASES:
// For nearly degenerate simplices (very thin triangles, near-collinear points),
// different SIMD architectures may produce different but equally valid results:
//   - The simplex size may differ (e.g., size=2 vs size=3)
//   - The selected support vertices may differ
// However, the output distance vector v (closest point to origin) MUST be
// numerically equivalent across all architectures. Tests for degenerate cases
// only validate the distance, not the simplex structure.
//
// SHARED HELPERS: See include/helper.hh for common test utilities.

#include <cmath>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_compile_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "unittest_s2d.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "opengjk_simd.hh"
#include "../opengjk-inl.h"

HWY_BEFORE_NAMESPACE();

namespace opengjk {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using D = hn::CappedTag<gjkFloat, 4>;
using V = hn::Vec<D>;

HWY_INLINE V
make_vec(gjkFloat x, gjkFloat y, gjkFloat z) {
  D d;
  HWY_ALIGN gjkFloat arr[4] = {x, y, z, 0};
  return hn::Load(d, arr);
}

HWY_INLINE void
extract(V v, gjkFloat* arr) {
  D d;
  hn::Store(v, d, arr);
}

// Squared norm of a vector (matches norm2 in other test files)
gjkFloat
norm2(V v) {
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
}

// ============================================================================
// Test implementations (inside HWY_NAMESPACE for multi-target compilation)
// ============================================================================

void
TestStability_Triangle1() {
  D d;
  V S[4];
  S[0] = make_vec(0, 0, 1);
  S[1] = make_vec(2, 0, 1);
  S[2] = make_vec(1, 2, 1);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 3);
  HWY_ASSERT(norm2(v) >= 0.0);
}

void
TestStability_Triangle2() {
  D d;
  V S[4];
  S[0] = make_vec(1, 1, 0);
  S[1] = make_vec(5, 1, 0);
  S[2] = make_vec(3, 5, 0);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 3);
}

void
TestStability_Triangle3D() {
  D d;
  V S[4];
  S[0] = make_vec(1, 0, 2);
  S[1] = make_vec(0, 1, 2);
  S[2] = make_vec(-1, 0, 2);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));
}

void
TestStability_Degenerate() {
  D d;
  // Collinear points
  V S[4];
  S[0] = make_vec(1, 0, 0);
  S[1] = make_vec(2, 0, 0);
  S[2] = make_vec(3, 0, 0);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size <= 2); // Should reduce to line or point
}

void
TestStability_LargeCoordinates() {
  D d;
  V S[4];
  S[0] = make_vec(0, 0, 1000);
  S[1] = make_vec(2000, 0, 1000);
  S[2] = make_vec(1000, 2000, 1000);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));
}

void
TestStability_MultipleConfigs() {
  D d;

  struct TestCase {
    gjkFloat p[3], q[3], r[3];
  } cases[] = {
      {{0, 0, 1}, {2, 0, 1}, {1, 2, 1}},  {{1, 1, 0}, {5, 1, 0}, {3, 5, 0}},  {{2, 0, 0}, {-2, 0, 0}, {0, 3, 0}},
      {{1, 0, 2}, {0, 1, 2}, {-1, 0, 2}}, {{-1, 1, 0}, {1, 1, 0}, {0, 5, 0}}, {{1, -1, 0}, {5, 0, 0}, {1, 1, 0}},
  };

  for (const auto& tc : cases) {
    V S[4];
    S[0] = make_vec(tc.p[0], tc.p[1], tc.p[2]);
    S[1] = make_vec(tc.q[0], tc.q[1], tc.q[2]);
    S[2] = make_vec(tc.r[0], tc.r[1], tc.r[2]);

    V size_v = hn::Set(d, static_cast<gjkFloat>(3));
    V v;
    S2D_vector<ProgressiveSearch>(S, v, size_v);

    HWY_ALIGN gjkFloat arr[4];
    extract(v, arr);
    HWY_ASSERT(!std::isnan(arr[0]));
    HWY_ASSERT(!std::isnan(arr[1]));
    HWY_ASSERT(!std::isnan(arr[2]));

    int size = static_cast<int>(hn::GetLane(size_v));
    HWY_ASSERT(size >= 1 && size <= 3);
    HWY_ASSERT(norm2(v) >= 0.0);
  }
}

void
TestExhaustive_OriginClosestToVertexQ() {
  D d;
  // Place q closest to origin, p and r far away
  V S[4];
  S[0] = make_vec(5, 5, 0);  // p (newest) - far
  S[1] = make_vec(1, 0, 0);  // q - closest
  S[2] = make_vec(5, -5, 0); // r

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 0.01);
  HWY_ASSERT(std::abs(arr[1]) < 0.01);
}

void
TestExhaustive_OriginClosestToVertexR() {
  D d;
  V S[4];
  S[0] = make_vec(5, 5, 0);  // p (newest) - far
  S[1] = make_vec(5, -5, 0); // q - far
  S[2] = make_vec(1, 0, 0);  // r - closest

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 0.01);
  HWY_ASSERT(std::abs(arr[1]) < 0.01);
}

void
TestExhaustive_OriginClosestToEdgeQR() {
  D d;
  // q at (-1, 1, 0), r at (1, 1, 0), p far
  // Origin projects to (0, 1, 0) on edge q-r
  V S[4];
  S[0] = make_vec(0, 10, 0); // p (newest) - far
  S[1] = make_vec(-1, 1, 0); // q
  S[2] = make_vec(1, 1, 0);  // r

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
  HWY_ASSERT(std::abs(arr[0]) < 0.01);
  HWY_ASSERT(std::abs(arr[1] - 1.0) < 0.01);
}

void
TestExhaustive_FaceProjection() {
  D d;
  // Triangle at z=2, origin projects onto face
  V S[4];
  S[0] = make_vec(2, 0, 2);
  S[1] = make_vec(-1, 2, 2);
  S[2] = make_vec(-1, -2, 2);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 3);
  HWY_ASSERT(std::abs(arr[2] - 2.0) < 0.01);
}

void
TestExhaustive_OriginInsideTriangle() {
  D d;
  // Origin is inside the triangle at z=0
  V S[4];
  S[0] = make_vec(2, 0, 0);
  S[1] = make_vec(-1, 1.732, 0);
  S[2] = make_vec(-1, -1.732, 0);

  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v;
  S2D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));

  // Origin on plane, projection at/near origin
  HWY_ASSERT(std::abs(arr[0]) < 0.1);
  HWY_ASSERT(std::abs(arr[1]) < 0.1);
  HWY_ASSERT(std::abs(arr[2]) < 0.1);
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace opengjk

HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic dispatch exports and GTest wrappers (compiled once)
// ============================================================================
#if HWY_ONCE

namespace opengjk {
namespace simd {

// Export function pointers for dynamic dispatch
HWY_EXPORT(TestStability_Triangle1);
HWY_EXPORT(TestStability_Triangle2);
HWY_EXPORT(TestStability_Triangle3D);
HWY_EXPORT(TestStability_Degenerate);
HWY_EXPORT(TestStability_LargeCoordinates);
HWY_EXPORT(TestStability_MultipleConfigs);
HWY_EXPORT(TestExhaustive_OriginClosestToVertexQ);
HWY_EXPORT(TestExhaustive_OriginClosestToVertexR);
HWY_EXPORT(TestExhaustive_OriginClosestToEdgeQR);
HWY_EXPORT(TestExhaustive_FaceProjection);
HWY_EXPORT(TestExhaustive_OriginInsideTriangle);

} // namespace simd
} // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

// GTest wrappers - one test per function, uses dynamic dispatch
TEST(S2D_Progressive, Stability_Triangle1) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_Triangle1)(); }

TEST(S2D_Progressive, Stability_Triangle2) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_Triangle2)(); }

TEST(S2D_Progressive, Stability_Triangle3D) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_Triangle3D)(); }

TEST(S2D_Progressive, Stability_Degenerate) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_Degenerate)(); }

TEST(S2D_Progressive, Stability_LargeCoordinates) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_LargeCoordinates)();
}

TEST(S2D_Progressive, Stability_MultipleConfigs) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestStability_MultipleConfigs)();
}

TEST(S2D_Exhaustive, OriginClosestToVertexQ) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestExhaustive_OriginClosestToVertexQ)();
}

TEST(S2D_Exhaustive, OriginClosestToVertexR) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestExhaustive_OriginClosestToVertexR)();
}

TEST(S2D_Exhaustive, OriginClosestToEdgeQR) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestExhaustive_OriginClosestToEdgeQR)();
}

TEST(S2D_Exhaustive, FaceProjection) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestExhaustive_FaceProjection)(); }

TEST(S2D_Exhaustive, OriginInsideTriangle) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestExhaustive_OriginInsideTriangle)();
}

#endif // HWY_ONCE
