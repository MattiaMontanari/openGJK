// S2D Cross-validation: SIMD vs Scalar
// Tests that SIMD S2D produces same distance vector as scalar S2D
//
// NOTE: Vertex ordering convention differs:
//   SIMD:   S[0] = p (newest), S[1] = q, S[2] = r
//   Scalar: vrtx[2] = p (newest), vrtx[1] = q, vrtx[0] = r
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.

#include <cmath>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "unittest_s2d_crossval.cc"
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

// ============================================================================
// Helper to call scalar S2D with correct vertex ordering
// SIMD:   S[0] = p (newest), S[1] = q, S[2] = r
// Scalar: vrtx[2] = p (newest), vrtx[1] = q, vrtx[0] = r
// ============================================================================
void
call_scalar_s2d(gkFloat p[3], gkFloat q[3], gkFloat r[3], gkFloat v[3]) {
  gkSimplex s;
  s.nvrtx = 3;
  // Map SIMD ordering to scalar ordering
  for (int i = 0; i < 3; i++) {
    s.vrtx[2][i] = p[i]; // newest
    s.vrtx[1][i] = q[i];
    s.vrtx[0][i] = r[i]; // oldest
  }
  opengjk_test_S2D(&s, v);
}

// ============================================================================
// Test implementations
// ============================================================================

void
TestTriangleFace() {
  D d;
  // Triangle in z=1 plane, origin at (0,0,0) projects to face
  gkFloat px = 0, py = 0, pz = 1; // newest
  gkFloat qx = 2, qy = 0, qz = 1;
  gkFloat rx = 1, ry = 2, rz = 1;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestTriangleFace: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestVertexP() {
  D d;
  // Origin closest to newest vertex p
  gkFloat px = 1, py = 1, pz = 0; // newest, closest to origin
  gkFloat qx = 5, qy = 1, qz = 0;
  gkFloat rx = 3, ry = 5, rz = 0;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestVertexP: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestEdgePQ() {
  D d;
  // Origin closest to edge p-q
  gkFloat px = -1, py = 1, pz = 0;
  gkFloat qx = 1, qy = 1, qz = 0;
  gkFloat rx = 0, ry = 5, rz = 0; // far from origin

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestEdgePQ: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2], v_scalar[0],
          v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestEdgePR() {
  D d;
  // Origin closest to edge p-r
  gkFloat px = 1, py = -1, pz = 0; // newest
  gkFloat qx = 5, qy = 0, qz = 0;  // far
  gkFloat rx = 1, ry = 1, rz = 0;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestEdgePR: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2], v_scalar[0],
          v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestOriginAtCenter() {
  D d;
  // Equilateral triangle centered at origin
  gkFloat px = 1, py = 0, pz = 0;
  gkFloat qx = -0.5, qy = 0.866, qz = 0;
  gkFloat rx = -0.5, ry = -0.866, rz = 0;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestOriginAtCenter: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestTriangle3D() {
  D d;
  // Triangle at z=2 plane
  gkFloat px = 1, py = 0, pz = 2;
  gkFloat qx = 0, qy = 1, qz = 2;
  gkFloat rx = -1, ry = 0, rz = 2;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestTriangle3D: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestLargeScale() {
  D d;
  // Large triangle coordinates
  gkFloat px = 0, py = 0, pz = 1000;
  gkFloat qx = 2000, qy = 0, qz = 1000;
  gkFloat rx = 1000, ry = 2000, rz = 1000;

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(3));
  V v_simd;
  S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat v_scalar[3];
  call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

  fprintf(stderr, "TestLargeScale: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  // Larger tolerance for large coordinates
  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1.0);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1.0);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1.0);
}

void
TestMultipleConfigs() {
  D d;

  struct TestCase {
    gkFloat p[3], q[3], r[3];
  } cases[] = {
      {{2, 0, 0}, {-2, 0, 0}, {0, 3, 0}},
      {{1, -1, 0}, {5, 0, 0}, {1, 1, 0}},
      {{0, 0, 5}, {3, 0, 5}, {1.5, 3, 5}},
      {{10, 10, 0}, {15, 10, 0}, {12.5, 15, 0}},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const auto& tc = cases[i];

    V S[4];
    S[0] = make_vec(tc.p[0], tc.p[1], tc.p[2]);
    S[1] = make_vec(tc.q[0], tc.q[1], tc.q[2]);
    S[2] = make_vec(tc.r[0], tc.r[1], tc.r[2]);
    V size_v = hn::Set(d, static_cast<gjkFloat>(3));
    V v_simd;
    S2D_vector<ProgressiveSearch>(S, v_simd, size_v);

    HWY_ALIGN gjkFloat simd_arr[4];
    extract(v_simd, simd_arr);

    gkFloat p_sc[3] = {tc.p[0], tc.p[1], tc.p[2]};
    gkFloat q_sc[3] = {tc.q[0], tc.q[1], tc.q[2]};
    gkFloat r_sc[3] = {tc.r[0], tc.r[1], tc.r[2]};
    gkFloat v_scalar[3];
    call_scalar_s2d(p_sc, q_sc, r_sc, v_scalar);

    fprintf(stderr, "TestMultiple[%zu]: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", i, simd_arr[0], simd_arr[1], simd_arr[2],
            v_scalar[0], v_scalar[1], v_scalar[2]);

    HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
    HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
    HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
  }
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace opengjk

HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic dispatch exports and GTest wrappers
// ============================================================================
#if HWY_ONCE

namespace opengjk {
namespace simd {

HWY_EXPORT(TestTriangleFace);
HWY_EXPORT(TestVertexP);
HWY_EXPORT(TestEdgePQ);
HWY_EXPORT(TestEdgePR);
HWY_EXPORT(TestOriginAtCenter);
HWY_EXPORT(TestTriangle3D);
HWY_EXPORT(TestLargeScale);
HWY_EXPORT(TestMultipleConfigs);

} // namespace simd
} // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

TEST(S2D_Crossval, TriangleFace) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestTriangleFace)(); }

TEST(S2D_Crossval, VertexP) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestVertexP)(); }

TEST(S2D_Crossval, EdgePQ) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgePQ)(); }

TEST(S2D_Crossval, EdgePR) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgePR)(); }

TEST(S2D_Crossval, OriginAtCenter) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginAtCenter)(); }

TEST(S2D_Crossval, Triangle3D) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestTriangle3D)(); }

TEST(S2D_Crossval, LargeScale) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestLargeScale)(); }

TEST(S2D_Crossval, MultipleConfigs) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestMultipleConfigs)(); }

#endif // HWY_ONCE
