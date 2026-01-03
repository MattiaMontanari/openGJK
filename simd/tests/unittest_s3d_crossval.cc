// S3D Cross-validation: SIMD vs Scalar
// Tests that SIMD S3D produces same distance vector as scalar S3D
//
// NOTE: Vertex ordering for S3D:
//   SIMD:   S[0] = p, S[1] = q, S[2] = r, S[3] = s (newest)
//   Scalar: vrtx[0], vrtx[1], vrtx[2], vrtx[3] = newest
//   (Same ordering - vrtx[3] is newest in both)
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.

#include <cmath>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

// Must be included before Highway headers to configure target selection
#include "opengjk_simd_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "unittest_s3d_crossval.cc"
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
// Helper to call scalar S3D with correct vertex ordering
// SIMD:   S[0] = p, S[1] = q, S[2] = r, S[3] = s (newest)
// Scalar: vrtx[3] = newest, vrtx[0] = oldest
// For S3D the indices happen to align
// ============================================================================
void
call_scalar_s3d(gkFloat p[3], gkFloat q[3], gkFloat r[3], gkFloat s[3], gkFloat v[3]) {
  gkSimplex simplex;
  simplex.nvrtx = 4;
  for (int i = 0; i < 3; i++) {
    simplex.vrtx[0][i] = p[i]; // oldest
    simplex.vrtx[1][i] = q[i];
    simplex.vrtx[2][i] = r[i];
    simplex.vrtx[3][i] = s[i]; // newest
  }
  opengjk_test_S3D(&simplex, v);
}

// ============================================================================
// Test implementations
// ============================================================================

void
TestOriginInside() {
  D d;
  // Regular tetrahedron centered at origin
  gkFloat px = 1, py = 1, pz = 1;
  gkFloat qx = 1, qy = -1, qz = -1;
  gkFloat rx = -1, ry = 1, rz = -1;
  gkFloat sx = -1, sy = -1, sz = 1; // newest

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

  fprintf(stderr, "TestOriginInside: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  // Origin inside tetrahedron -> v should be zero
  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestVertexS() {
  D d;
  // Origin closest to newest vertex s
  gkFloat px = 5, py = 5, pz = 5;
  gkFloat qx = 7, qy = 5, qz = 5;
  gkFloat rx = 6, ry = 7, rz = 5;
  gkFloat sx = 1, sy = 1, sz = 1; // newest, closest to origin

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

  fprintf(stderr, "TestVertexS: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestFacePQR() {
  D d;
  // Face p-q-r at z=1, origin projects onto it
  gkFloat px = 1, py = 0, pz = 1;
  gkFloat qx = -1, qy = 1, qz = 1;
  gkFloat rx = -1, ry = -1, rz = 1;
  gkFloat sx = 0, sy = 0, sz = 5; // newest, far from origin

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

  fprintf(stderr, "TestFacePQR: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestEdgeWithS() {
  D d;
  // Edge involving newest vertex s
  gkFloat px = 2, py = 0, pz = 0;
  gkFloat qx = 2, qy = 2, qz = 0;
  gkFloat rx = 4, ry = 0, rz = 0;
  gkFloat sx = 2, sy = 0, sz = 2; // newest

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

  fprintf(stderr, "TestEdgeWithS: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestFaceWithS() {
  D d;
  // Face containing newest vertex s
  // Tetrahedron with one face at z=2 containing the newest vertex
  // and origin clearly outside (below)
  gkFloat px = 2, py = 0, pz = 2; // oldest
  gkFloat qx = -1, qy = 2, qz = 2;
  gkFloat rx = -1, ry = -2, rz = 2;
  gkFloat sx = 0, sy = 0, sz = 5; // newest, at apex

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

  fprintf(stderr, "TestFaceWithS: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", simd_arr[0], simd_arr[1], simd_arr[2],
          v_scalar[0], v_scalar[1], v_scalar[2]);

  HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-4);
  HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-4);
}

void
TestLargeScale() {
  D d;
  // Large coordinates
  gkFloat px = 0, py = 0, pz = 1000;
  gkFloat qx = 2000, qy = 0, qz = 1000;
  gkFloat rx = 1000, ry = 2000, rz = 1000;
  gkFloat sx = 1000, sy = 1000, sz = 3000; // newest

  V S[4];
  S[0] = make_vec(px, py, pz);
  S[1] = make_vec(qx, qy, qz);
  S[2] = make_vec(rx, ry, rz);
  S[3] = make_vec(sx, sy, sz);
  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v_simd;
  S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

  HWY_ALIGN gjkFloat simd_arr[4];
  extract(v_simd, simd_arr);

  gkFloat p_sc[3] = {px, py, pz};
  gkFloat q_sc[3] = {qx, qy, qz};
  gkFloat r_sc[3] = {rx, ry, rz};
  gkFloat s_sc[3] = {sx, sy, sz};
  gkFloat v_scalar[3];
  call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

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
    gkFloat p[3], q[3], r[3], s[3];
  } cases[] = {
      // Standard tetrahedron
      {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}},
      // Tetrahedron with origin outside
      {{2, 2, 2}, {4, 2, 2}, {3, 4, 2}, {3, 3, 4}},
      // Elongated tetrahedron
      {{0, 0, 10}, {1, 0, 10}, {0.5, 1, 10}, {0.5, 0.5, 11}},
      // Flat tetrahedron (nearly coplanar)
      {{0, 0, 1}, {2, 0, 1}, {1, 2, 1}, {1, 1, 1.01}},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const auto& tc = cases[i];

    V S[4];
    S[0] = make_vec(tc.p[0], tc.p[1], tc.p[2]);
    S[1] = make_vec(tc.q[0], tc.q[1], tc.q[2]);
    S[2] = make_vec(tc.r[0], tc.r[1], tc.r[2]);
    S[3] = make_vec(tc.s[0], tc.s[1], tc.s[2]);
    V size_v = hn::Set(d, static_cast<gjkFloat>(4));
    V v_simd;
    S3D_vector<ProgressiveSearch>(S, v_simd, size_v);

    HWY_ALIGN gjkFloat simd_arr[4];
    extract(v_simd, simd_arr);

    gkFloat p_sc[3] = {tc.p[0], tc.p[1], tc.p[2]};
    gkFloat q_sc[3] = {tc.q[0], tc.q[1], tc.q[2]};
    gkFloat r_sc[3] = {tc.r[0], tc.r[1], tc.r[2]};
    gkFloat s_sc[3] = {tc.s[0], tc.s[1], tc.s[2]};
    gkFloat v_scalar[3];
    call_scalar_s3d(p_sc, q_sc, r_sc, s_sc, v_scalar);

    fprintf(stderr, "TestMultiple[%zu]: SIMD=[%g,%g,%g] Scalar=[%g,%g,%g]\n", i, simd_arr[0], simd_arr[1], simd_arr[2],
            v_scalar[0], v_scalar[1], v_scalar[2]);

    HWY_ASSERT(std::abs(simd_arr[0] - static_cast<gjkFloat>(v_scalar[0])) < 1e-3);
    HWY_ASSERT(std::abs(simd_arr[1] - static_cast<gjkFloat>(v_scalar[1])) < 1e-3);
    HWY_ASSERT(std::abs(simd_arr[2] - static_cast<gjkFloat>(v_scalar[2])) < 1e-3);
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

HWY_EXPORT(TestOriginInside);
HWY_EXPORT(TestVertexS);
HWY_EXPORT(TestFacePQR);
HWY_EXPORT(TestEdgeWithS);
HWY_EXPORT(TestFaceWithS);
HWY_EXPORT(TestLargeScale);
HWY_EXPORT(TestMultipleConfigs);

} // namespace simd
} // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

TEST(S3D_Crossval, OriginInside) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginInside)(); }

TEST(S3D_Crossval, VertexS) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestVertexS)(); }

TEST(S3D_Crossval, FacePQR) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestFacePQR)(); }

TEST(S3D_Crossval, EdgeWithS) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestEdgeWithS)(); }

TEST(S3D_Crossval, FaceWithS) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestFaceWithS)(); }

TEST(S3D_Crossval, LargeScale) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestLargeScale)(); }

TEST(S3D_Crossval, MultipleConfigs) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestMultipleConfigs)(); }

#endif // HWY_ONCE
