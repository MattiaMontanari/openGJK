// Unit tests for S1D sub-algorithm
// Tests Voronoi regions of a 1-simplex (line segment) with both search policies
//
// VERTEX CONVENTION: S[0] = p (newest), S[1] = q
// In Progressive search, p must be part of the solution.
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.
//
// DEGENERATE CASES:
// For nearly degenerate simplices (very short edges, near-collinear points),
// different SIMD architectures may produce different but equally valid results:
//   - The simplex size may differ (e.g., size=1 vs size=2)
//   - The selected support vertices may differ
// However, the output distance vector v (closest point to origin) MUST be
// numerically equivalent across all architectures. Tests for degenerate cases
// only validate the distance, not the simplex structure.
//
// SHARED HELPERS: See include/helper.hh for common test utilities.

#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "unittest_s1d.cc"
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

// Helper to create a vector
HWY_INLINE V
make_vec(gjkFloat x, gjkFloat y, gjkFloat z) {
  D d;
  HWY_ALIGN gjkFloat arr[4] = {x, y, z, 0};
  return hn::Load(d, arr);
}

// Helper to extract components
HWY_INLINE void
extract(V v, gjkFloat* arr) {
  D d;
  hn::Store(v, d, arr);
}

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
TestOriginProjectsOntoEdge_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 0, 0);
  S[1] = make_vec(-1, 0, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  gjkFloat dist = norm2(v);
  HWY_ASSERT(dist < 1e-6);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestOriginClosestToNewestVertexP_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 1, 0);
  S[1] = make_vec(3, 1, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 1e-6);
  HWY_ASSERT(std::abs(arr[1] - 1.0) < 1e-6);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
}

void
TestOriginOnEdge3D_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 1, 1);
  S[1] = make_vec(-1, -1, -1);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  gjkFloat dist = norm2(v);
  HWY_ASSERT(dist < 1e-6);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestEdgePerpendicular_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 0, 1);
  S[1] = make_vec(1, 0, -1);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 1e-6);
  HWY_ASSERT(std::abs(arr[1]) < 1e-6);
  HWY_ASSERT(std::abs(arr[2]) < 1e-6);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestOriginClosestToVertexQ_Exhaustive() {
  D d;
  V S[4];
  S[0] = make_vec(3, 1, 0);
  S[1] = make_vec(1, 1, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ExhaustiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 1e-6);
  HWY_ASSERT(std::abs(arr[1] - 1.0) < 1e-6);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
}

void
TestLargeEdgeSymmetric() {
  D d;
  V S[4];
  S[0] = make_vec(1000, 0, 0);
  S[1] = make_vec(-1000, 0, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0]) < 1e-3);
  HWY_ASSERT(std::abs(arr[1]) < 1e-3);
  HWY_ASSERT(std::abs(arr[2]) < 1e-3);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestVeryLargeDiagonal() {
  D d;
  V S[4];
  S[0] = make_vec(1e6f, 1e6f, 1e6f);
  S[1] = make_vec(-1e6f, -1e6f, -1e6f);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0]) < 1.0);
  HWY_ASSERT(std::abs(arr[1]) < 1.0);
  HWY_ASSERT(std::abs(arr[2]) < 1.0);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestRotatedEdge45deg() {
  D d;
  const gjkFloat c = 0.7071067811865476f;
  V S[4];
  S[0] = make_vec(c, c, 0);
  S[1] = make_vec(-c, -c, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0]) < 1e-5);
  HWY_ASSERT(std::abs(arr[1]) < 1e-5);
  HWY_ASSERT(std::abs(arr[2]) < 1e-5);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestArbitraryOrientation() {
  D d;
  V S[4];
  S[0] = make_vec(3, 4, 5);
  S[1] = make_vec(-3, -4, -5);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0]) < 1e-5);
  HWY_ASSERT(std::abs(arr[1]) < 1e-5);
  HWY_ASSERT(std::abs(arr[2]) < 1e-5);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
}

void
TestNearlyDegenerateEdge() {
  // DEGENERATE CASE: Very short edge. Simplex size may vary by architecture,
  // but the distance vector v must be valid and close to zero.
  D d;
  V S[4];
  S[0] = make_vec(0.0001f, 0, 0);
  S[1] = make_vec(-0.0001f, 0, 0);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  // Only check distance vector - simplex size may vary across architectures
  HWY_ASSERT(std::abs(arr[0]) < 0.001);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));
  // NOTE: size is NOT checked - may be 1 or 2 depending on architecture
}

void
TestEdgeAlongAxes() {
  D d;
  gjkFloat axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int axis = 0; axis < 3; ++axis) {
    V S[4];
    S[0] = make_vec(axes[axis][0], axes[axis][1], axes[axis][2]);
    S[1] = make_vec(-axes[axis][0], -axes[axis][1], -axes[axis][2]);
    V size_v = hn::Set(d, 2.0);
    V v;
    S1D_vector<ProgressiveSearch>(S, &size_v, v);
    HWY_ALIGN gjkFloat arr[4];
    extract(v, arr);
    HWY_ASSERT(std::abs(arr[0]) < 1e-5);
    HWY_ASSERT(std::abs(arr[1]) < 1e-5);
    HWY_ASSERT(std::abs(arr[2]) < 1e-5);
    int size = static_cast<int>(hn::GetLane(size_v));
    HWY_ASSERT(size == 2);
  }
}

void
TestNegativeCoordinates() {
  D d;
  V S[4];
  S[0] = make_vec(-1, -1, -1);
  S[1] = make_vec(-3, -3, -3);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0] - (-1.0)) < 1e-5);
  HWY_ASSERT(std::abs(arr[1] - (-1.0)) < 1e-5);
  HWY_ASSERT(std::abs(arr[2] - (-1.0)) < 1e-5);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
}

void
TestMixedSignCoordinates() {
  D d;
  V S[4];
  S[0] = make_vec(2, -1, 3);
  S[1] = make_vec(-2, 1, -3);
  V size_v = hn::Set(d, 2.0);
  V v;
  S1D_vector<ProgressiveSearch>(S, &size_v, v);
  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0]) < 1e-5);
  HWY_ASSERT(std::abs(arr[1]) < 1e-5);
  HWY_ASSERT(std::abs(arr[2]) < 1e-5);
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 2);
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
HWY_EXPORT(TestOriginProjectsOntoEdge_Progressive);
HWY_EXPORT(TestOriginClosestToNewestVertexP_Progressive);
HWY_EXPORT(TestOriginOnEdge3D_Progressive);
HWY_EXPORT(TestEdgePerpendicular_Progressive);
HWY_EXPORT(TestOriginClosestToVertexQ_Exhaustive);
HWY_EXPORT(TestLargeEdgeSymmetric);
HWY_EXPORT(TestVeryLargeDiagonal);
HWY_EXPORT(TestRotatedEdge45deg);
HWY_EXPORT(TestArbitraryOrientation);
HWY_EXPORT(TestNearlyDegenerateEdge);
HWY_EXPORT(TestEdgeAlongAxes);
HWY_EXPORT(TestNegativeCoordinates);
HWY_EXPORT(TestMixedSignCoordinates);

} // namespace simd
} // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

// Diagnostic test to verify SIMD target configuration
// This test initializes SIMD targets based on compile-time config and verifies the result
TEST(SIMD_Target, ReportActiveTarget) {
  using namespace opengjk::simd;

  // Get CPU capabilities BEFORE any filtering
  int64_t cpu_all = hwy::SupportedTargets();
  int64_t cpu_best = cpu_all & -cpu_all;

  std::cout << "\n=== Highway SIMD Target Report ===" << std::endl;
  std::cout << "Configuration: " << GetSIMDConfigDescription() << std::endl;

  std::cout << "\nCPU capabilities:" << std::endl;
  std::cout << "  Best target: " << hwy::TargetName(cpu_best) << std::endl;
  std::cout << "  All supported: ";
  for (int64_t t = cpu_all; t != 0;) {
    int64_t target = t & -t;
    std::cout << hwy::TargetName(target);
    t &= t - 1;
    if (t != 0) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;

  // Initialize runtime target filtering based on compile-time config
  InitSIMDTargetsFromConfig();

  // Get filtered targets AFTER initialization
  int64_t filtered = hwy::SupportedTargets();
  int64_t active = filtered & -filtered;

  std::cout << "\nAfter runtime filtering:" << std::endl;
  std::cout << "  Active target: " << hwy::TargetName(active) << std::endl;
  std::cout << "  Available: ";
  for (int64_t t = filtered; t != 0;) {
    int64_t target = t & -t;
    std::cout << hwy::TargetName(target);
    t &= t - 1;
    if (t != 0) {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;

  // Validate the active target meets our requirements
  ASSERT_NE(active, HWY_SCALAR) << "ERROR: Only SCALAR available - no SIMD!";
  ASSERT_NE(active, HWY_EMU128) << "ERROR: Only EMU128 available - emulated SIMD!";

#ifndef OPENGJK_SIMD_USE_FLOAT
  // Double precision requires 256-bit minimum
  ASSERT_TRUE(IsAtLeast256Bit(active)) << "ERROR: Double precision requires 256-bit SIMD (AVX2+)!\n"
                                       << "Active target: " << hwy::TargetName(active) << "\n"
                                       << "Use USE_32BITS=ON for 128-bit SIMD support.";
  std::cout << "\nOK: 256-bit+ SIMD active for double precision!" << std::endl;
#else
  // Float precision works with 128-bit+
  ASSERT_TRUE(IsAtLeast128Bit(active)) << "ERROR: Float precision requires 128-bit SIMD (SSE4+)!";
  std::cout << "\nOK: Native SIMD active for float precision!" << std::endl;
#endif

#ifdef OPENGJK_SIMD_MINIMAL_WIDTH
  // Verify minimal width preference was applied
  std::cout << "\nMinimal width policy active:" << std::endl;
#ifdef OPENGJK_SIMD_USE_FLOAT
  if (IsAtLeast128Bit(active) && !IsAtLeast256Bit(active)) {
    std::cout << "  OK: Using 128-bit target as preferred" << std::endl;
  } else if (IsAtLeast256Bit(active)) {
    std::cout << "  NOTE: 256-bit target active (128-bit may not be available)" << std::endl;
  }
#else
  if (active == HWY_AVX2) {
    std::cout << "  OK: Using AVX2 (256-bit) as preferred" << std::endl;
  } else if (Is512BitOrMore(active)) {
    std::cout << "  NOTE: 512-bit target active (AVX2 filtering may not apply)" << std::endl;
  }
#endif
#endif

  // Reset for other tests (they may want widest available)
  ResetSIMDTargets();
}

// GTest wrappers - one test per function, uses dynamic dispatch
TEST(S1D_Progressive, OriginProjectsOntoEdge) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginProjectsOntoEdge_Progressive)();
}

TEST(S1D_Progressive, OriginClosestToNewestVertexP) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToNewestVertexP_Progressive)();
}

TEST(S1D_Progressive, OriginOnEdge3D) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginOnEdge3D_Progressive)(); }

TEST(S1D_Progressive, EdgePerpendicular) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgePerpendicular_Progressive)(); }

TEST(S1D_Exhaustive, OriginClosestToVertexQ) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToVertexQ_Exhaustive)();
}

TEST(S1D_LargeScale, LargeEdgeSymmetric) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestLargeEdgeSymmetric)(); }

TEST(S1D_LargeScale, VeryLargeDiagonal) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestVeryLargeDiagonal)(); }

TEST(S1D_Orientation, RotatedEdge45deg) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestRotatedEdge45deg)(); }

TEST(S1D_Orientation, ArbitraryOrientation) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestArbitraryOrientation)(); }

TEST(S1D_EdgeCases, NearlyDegenerateEdge) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestNearlyDegenerateEdge)(); }

TEST(S1D_EdgeCases, EdgeAlongAxes) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgeAlongAxes)(); }

TEST(S1D_EdgeCases, NegativeCoordinates) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestNegativeCoordinates)(); }

TEST(S1D_EdgeCases, MixedSignCoordinates) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestMixedSignCoordinates)(); }

#endif // HWY_ONCE
