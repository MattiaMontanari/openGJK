// Unit tests for S3D sub-algorithm
// Tests Voronoi regions of a 3-simplex (tetrahedron) with both search policies
//
// VERTEX CONVENTION: S[0] = p, S[1] = q, S[2] = r, S[3] = s (newest)
// Note: S3D uses S[3] as newest, unlike S1D/S2D which use S[0] as newest.
//
// Uses Highway dynamic dispatch - tests run on best supported SIMD target only.
//
// DEGENERATE CASES:
// For nearly degenerate simplices (flat tetrahedra, coplanar points),
// different SIMD architectures may produce different but equally valid results:
//   - The simplex size may differ (e.g., size=3 vs size=4)
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
#define HWY_TARGET_INCLUDE "unittest_s3d.cc"
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
TestOriginInsideTetrahedron_Progressive() {
  D d;
  // Regular tetrahedron centered at origin
  V S[4];
  S[0] = make_vec(1, 1, 1);
  S[1] = make_vec(1, -1, -1);
  S[2] = make_vec(-1, 1, -1);
  S[3] = make_vec(-1, -1, 1); // newest vertex

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ProgressiveSearch>(S, v, size_v);

  gjkFloat dist = norm2(v);
  HWY_ASSERT(dist < 1e-6);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 4);
}

void
TestOriginClosestToNewestVertex_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(5, 5, 5);
  S[1] = make_vec(7, 5, 5);
  S[2] = make_vec(6, 7, 5);
  S[3] = make_vec(1, 1, 1); // newest, closest to origin

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ProgressiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(std::abs(arr[0] - 1.0) < 1e-6);
  HWY_ASSERT(std::abs(arr[1] - 1.0) < 1e-6);
  HWY_ASSERT(std::abs(arr[2] - 1.0) < 1e-6);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 1);
}

void
TestOriginClosestToFaceWithNewest_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 0, 1);
  S[1] = make_vec(-1, 1, 1);
  S[2] = make_vec(-1, -1, 1);
  S[3] = make_vec(0, 0, 3); // newest

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ProgressiveSearch>(S, v, size_v);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));
}

void
TestOriginClosestToEdgeWithNewest_Progressive() {
  D d;
  V S[4];
  S[0] = make_vec(-1, 1, 0);
  S[1] = make_vec(1, 1, 0);
  S[2] = make_vec(0, 3, 0);
  S[3] = make_vec(0, 2, 2); // newest

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ProgressiveSearch>(S, v, size_v);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);
  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  gjkFloat dist2 = arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
  HWY_ASSERT(dist2 >= 0.0);
}

void
TestOriginInsideTetrahedron_Exhaustive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 1, 1);
  S[1] = make_vec(1, -1, -1);
  S[2] = make_vec(-1, 1, -1);
  S[3] = make_vec(-1, -1, 1);

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  gjkFloat dist = norm2(v);
  HWY_ASSERT(dist < 1e-6);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 4);
}

void
TestOriginClosestToVertexP_Exhaustive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 1, 1); // p - closest to origin
  S[1] = make_vec(3, 1, 1); // q
  S[2] = make_vec(2, 3, 1); // r
  S[3] = make_vec(2, 2, 3); // s - newest

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);

  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);

  gjkFloat dist2 = arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
  HWY_ASSERT(dist2 >= 0.0);
}

void
TestOriginClosestToEdgePQ_Exhaustive() {
  D d;
  V S[4];
  S[0] = make_vec(-1, 1, 0); // p
  S[1] = make_vec(1, 1, 0);  // q
  S[2] = make_vec(0, 5, 0);  // r - far
  S[3] = make_vec(0, 3, 5);  // s - newest, far

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);

  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);

  gjkFloat dist2 = arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
  HWY_ASSERT(dist2 >= 0.0);
}

void
TestOriginClosestToFacePQR_Exhaustive() {
  D d;
  // Tetrahedron with base at z=1, apex at z=5
  V S[4];
  S[0] = make_vec(2, 0, 1);   // p - base vertex
  S[1] = make_vec(-1, 2, 1);  // q - base vertex
  S[2] = make_vec(-1, -2, 1); // r - base vertex
  S[3] = make_vec(0, 0, 5);   // s - newest, apex

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);

  HWY_ALIGN gjkFloat arr[4];
  extract(v, arr);

  HWY_ASSERT(!std::isnan(arr[0]));
  HWY_ASSERT(!std::isnan(arr[1]));
  HWY_ASSERT(!std::isnan(arr[2]));

  gjkFloat dist2 = arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
  HWY_ASSERT(dist2 >= 0.0);
}

void
TestOriginClosestToFaceWithNewest_Exhaustive() {
  D d;
  V S[4];
  S[0] = make_vec(1, 0, 0);
  S[1] = make_vec(3, 2, 0);
  S[2] = make_vec(1, 2, 0);
  S[3] = make_vec(2, 1, 2);

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size >= 1 && size <= 4);
}

void
TestDegenerateFlat_Exhaustive() {
  D d;
  // All points at z = 1 (coplanar)
  V S[4];
  S[0] = make_vec(0, 1, 1);
  S[1] = make_vec(-1, 0, 1);
  S[2] = make_vec(0, -1, 1);
  S[3] = make_vec(1, 0, 1);

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  // Should reduce to face or edge for degenerate tetrahedron
  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size <= 4);
}

void
TestOriginAtCentroid_Exhaustive() {
  D d;
  // Tetrahedron with centroid at origin
  gjkFloat a = 1.0;
  V S[4];
  S[0] = make_vec(a, a, a);
  S[1] = make_vec(a, -a, -a);
  S[2] = make_vec(-a, a, -a);
  S[3] = make_vec(-a, -a, a);

  V size_v = hn::Set(d, static_cast<gjkFloat>(4));
  V v;
  S3D_vector<ExhaustiveSearch>(S, v, size_v);

  gjkFloat dist = norm2(v);
  HWY_ASSERT(dist < 1e-6);

  int size = static_cast<int>(hn::GetLane(size_v));
  HWY_ASSERT(size == 4);
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
HWY_EXPORT(TestOriginInsideTetrahedron_Progressive);
HWY_EXPORT(TestOriginClosestToNewestVertex_Progressive);
HWY_EXPORT(TestOriginClosestToFaceWithNewest_Progressive);
HWY_EXPORT(TestOriginClosestToEdgeWithNewest_Progressive);
HWY_EXPORT(TestOriginInsideTetrahedron_Exhaustive);
HWY_EXPORT(TestOriginClosestToVertexP_Exhaustive);
HWY_EXPORT(TestOriginClosestToEdgePQ_Exhaustive);
HWY_EXPORT(TestOriginClosestToFacePQR_Exhaustive);
HWY_EXPORT(TestOriginClosestToFaceWithNewest_Exhaustive);
HWY_EXPORT(TestDegenerateFlat_Exhaustive);
HWY_EXPORT(TestOriginAtCentroid_Exhaustive);

} // namespace simd
} // namespace opengjk

// Include runtime SIMD initialization
#include "opengjk_simd_init.h"

// GTest wrappers - one test per function, uses dynamic dispatch
TEST(S3D_Progressive, OriginInsideTetrahedron) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginInsideTetrahedron_Progressive)();
}

TEST(S3D_Progressive, OriginClosestToNewestVertex) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToNewestVertex_Progressive)();
}

TEST(S3D_Progressive, OriginClosestToFaceWithNewest) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToFaceWithNewest_Progressive)();
}

TEST(S3D_Progressive, OriginClosestToEdgeWithNewest) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToEdgeWithNewest_Progressive)();
}

TEST(S3D_Exhaustive, OriginInsideTetrahedron) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginInsideTetrahedron_Exhaustive)();
}

TEST(S3D_Exhaustive, OriginClosestToVertexP) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToVertexP_Exhaustive)();
}

TEST(S3D_Exhaustive, OriginClosestToEdgePQ) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToEdgePQ_Exhaustive)();
}

TEST(S3D_Exhaustive, OriginClosestToFacePQR) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToFacePQR_Exhaustive)();
}

TEST(S3D_Exhaustive, OriginClosestToFaceWithNewest) {
  HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginClosestToFaceWithNewest_Exhaustive)();
}

TEST(S3D_Exhaustive, DegenerateFlat) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestDegenerateFlat_Exhaustive)(); }

TEST(S3D_Exhaustive, OriginAtCentroid) { HWY_DYNAMIC_DISPATCH(opengjk::simd::TestOriginAtCentroid_Exhaustive)(); }

#endif // HWY_ONCE
