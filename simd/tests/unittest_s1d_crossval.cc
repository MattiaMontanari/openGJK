// S1D Cross-validation: SIMD vs Scalar
// Tests that SIMD S1D produces same distance vector as scalar S1D
//
// NOTE: Vertex ordering convention differs:
//   SIMD:   S[0] = p (newest), S[1] = q
//   Scalar: vrtx[1] = p (newest), vrtx[0] = q
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.

#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_compile_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "unittest_s1d_crossval.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "opengjk_simd.hh"
#include "../opengjk-inl.h"

// Scalar implementation
extern "C" {
#include "openGJK/openGJK.h"
}

HWY_BEFORE_NAMESPACE();
namespace opengjk {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using D = hn::CappedTag<gjkFloat, 4>;
using V = hn::Vec<D>;

// Helper to create SIMD vector
HWY_INLINE V make_vec(gjkFloat x, gjkFloat y, gjkFloat z) {
  D d;
  HWY_ALIGN gjkFloat arr[4] = {x, y, z, 0};
  return hn::Load(d, arr);
}

// Helper to extract components
HWY_INLINE void extract(V v, gjkFloat* arr) {
  D d;
  hn::Store(v, d, arr);
}

// ============================================================================
// Helper to call scalar S1D with correct vertex ordering
// SIMD:   S[0] = p (newest), S[1] = q
// Scalar: vrtx[1] = p (newest), vrtx[0] = q
// ============================================================================
void call_scalar_s1d(gkFloat p[3], gkFloat q[3], gkFloat v[3]) {
  gkSimplex s;
  s.nvrtx = 2;
  // Map SIMD ordering to scalar ordering
  for (int i = 0; i < 3; i++) {
    s.vrtx[1][i] = p[i];  // newest
    s.vrtx[0][i] = q[i];  // oldest
  }
  opengjk_test_S1D(&s, v);
}

// ============================================================================
// Test implementations
// ============================================================================

void TestEdgeSymmetric() {
  D d;
  gjkFloat px = 1, py = 0, pz = 0;
  gjkFloat qx = -1, qy = 0, qz = 0;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(2));
  V v_simd;
  S1D_vector<ProgressiveSearch>(S, &size_v, v_simd);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat v_scalar[3];
  call_scalar_s1d(p_sc, q_sc, v_scalar);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-5);
}

void TestOriginClosestToP() {
  D d;
  gjkFloat px = 1, py = 1, pz = 0;
  gjkFloat qx = 3, qy = 1, qz = 0;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(2));
  V v_simd;
  S1D_vector<ProgressiveSearch>(S, &size_v, v_simd);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat v_scalar[3];
  call_scalar_s1d(p_sc, q_sc, v_scalar);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[0] - 1.0) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[1] - 1.0) < 1e-5);
}

void TestEdgeDiagonal3D() {
  D d;
  gjkFloat px = 1, py = 1, pz = 1;
  gjkFloat qx = -1, qy = -1, qz = -1;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(2));
  V v_simd;
  S1D_vector<ProgressiveSearch>(S, &size_v, v_simd);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat v_scalar[3];
  call_scalar_s1d(p_sc, q_sc, v_scalar);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[0]) < 1e-5);
}

void TestEdgePerpendicular() {
  D d;
  gjkFloat px = 1, py = 0, pz = 1;
  gjkFloat qx = 1, qy = 0, qz = -1;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(2));
  V v_simd;
  S1D_vector<ProgressiveSearch>(S, &size_v, v_simd);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat v_scalar[3];
  call_scalar_s1d(p_sc, q_sc, v_scalar);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-5);
  HWY_ASSERT(std::abs(simd_arr[0] - 1.0) < 1e-5);
}

void TestRandomEdges() {
  D d;
  struct TestCase {
    gjkFloat p[3];
    gjkFloat q[3];
  } cases[] = {
    {{2, 0, 0}, {-2, 0, 0}},
    {{0, 2, 0}, {0, -2, 0}},
    {{0, 0, 2}, {0, 0, -2}},
    {{1, 2, 3}, {-1, -2, -3}},
    {{5, 0, 0}, {3, 0, 0}},
    {{0.5, 0.5, 0}, {-0.5, -0.5, 0}},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const auto& tc = cases[i];

    V S[4];
    S[0] = make_vec(tc.p[0], tc.p[1], tc.p[2]);
    S[1] = make_vec(tc.q[0], tc.q[1], tc.q[2]);
    V size_v = hn::Set(d, static_cast<gjkFloat>(2));
    V v_simd;
    S1D_vector<ProgressiveSearch>(S, &size_v, v_simd);

    HWY_ALIGN gjkFloat simd_arr[4];
    extract(v_simd, simd_arr);

    gkFloat p_sc[3] = {static_cast<gkFloat>(tc.p[0]),
                       static_cast<gkFloat>(tc.p[1]),
                       static_cast<gkFloat>(tc.p[2])};
    gkFloat q_sc[3] = {static_cast<gkFloat>(tc.q[0]),
                       static_cast<gkFloat>(tc.q[1]),
                       static_cast<gkFloat>(tc.q[2])};
    gkFloat v_scalar[3];
    call_scalar_s1d(p_sc, q_sc, v_scalar);

    HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
    HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
    HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace opengjk
HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic dispatch exports and GTest wrappers
// ============================================================================
#if HWY_ONCE

namespace opengjk {
namespace simd {

HWY_EXPORT(TestEdgeSymmetric);
HWY_EXPORT(TestOriginClosestToP);
HWY_EXPORT(TestEdgeDiagonal3D);
HWY_EXPORT(TestEdgePerpendicular);
HWY_EXPORT(TestRandomEdges);

}  // namespace simd
}  // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

TEST(S1D_Crossval, EdgeSymmetric) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgeSymmetric)();
}

TEST(S1D_Crossval, OriginClosestToP) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToP)();
}

TEST(S1D_Crossval, EdgeDiagonal3D) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgeDiagonal3D)();
}

TEST(S1D_Crossval, EdgePerpendicular) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgePerpendicular)();
}

TEST(S1D_Crossval, RandomEdges) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestRandomEdges)();
}

#endif  // HWY_ONCE
