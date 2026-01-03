//                           _____      _ _  __
//                          / ____|    | | |/ /
//    ___  _ __   ___ _ __ | |  __     | | ' /
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
//  | (_) | |_) |  __/ | | | |__| | |__| | . \
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
//        | |
//        |_|
//
// Copyright 2022-2026 Mattia Montanari, University of Oxford
//
// SIMD kernels for GJK algorithm using Google Highway.
// This file uses Highway's per-target include guards and must be included
// after foreach_target.h with HWY_TARGET_INCLUDE defined.

#if defined(OPENGJK_SIMD_INL_H_TARGET) == defined(HWY_TARGET_TOGGLE)
#ifdef OPENGJK_SIMD_INL_H_TARGET
#undef OPENGJK_SIMD_INL_H_TARGET
#else
#define OPENGJK_SIMD_INL_H_TARGET
#endif

// Must be included before highway.h to configure target selection
#include "include/opengjk_simd_config.h"

#include <hwy/highway.h>

#include "include/opengjk_simd.hh"

HWY_BEFORE_NAMESPACE();

namespace opengjk {
namespace simd {
namespace HWY_NAMESPACE {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

// Use capped tag to fit 4 elements (3D vector + padding)
using D = hn::CappedTag<gjkFloat, 4>;
using V = hn::Vec<D>;

// ============================================================================
// Search Policies for Voronoi Region Testing
// ============================================================================

/**
 * @brief Progressive search policy for GJK-style simplex iteration.
 *
 * In GJK, the simplex grows monotonically toward the origin. The newest
 * vertex is guaranteed to be in the direction of the origin, so we only
 * need to test Voronoi regions reachable from that vertex. This allows
 * skipping "backward" region checks for improved performance.
 */
struct ProgressiveSearch {
  static constexpr bool kTestAllRegions = false;
};

/**
 * @brief Exhaustive search policy for EPA-style distance queries.
 *
 * In EPA and general closest-point queries, the simplex may need to
 * reduce to any vertex or edge, not just those adjacent to the newest
 * vertex. This policy enables testing of all Voronoi regions.
 */
struct ExhaustiveSearch {
  static constexpr bool kTestAllRegions = true;
};

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * @brief Compute dot product of two 3D vectors stored in SIMD registers.
 *
 * Uses ReduceSum for portable horizontal reduction across all lane counts.
 * Works correctly whether the target has 2, 4, or more lanes.
 * Result is broadcast to all lanes.
 */
[[nodiscard]] HWY_INLINE V
dot_vector(const V a, const V b) {
  D d;
  const auto prod = a * b;
  // ReduceSum handles any lane count correctly
  const auto sum = hn::ReduceSum(d, prod);
  return hn::Set(d, sum);
}

/**
 * @brief Compute cross product of two 3D vectors.
 *
 * Uses shuffle operations to compute: c = a × b
 */
[[nodiscard]] HWY_INLINE V
cross_vector(V q, V p) {
  const auto t0 = hn::Per4LaneBlockShuffle<3, 0, 2, 1>(q);
  const auto t1 = hn::Per4LaneBlockShuffle<3, 1, 0, 2>(p);
  const auto t2 = t0 * p;
  const auto t4 = hn::Per4LaneBlockShuffle<3, 0, 2, 1>(t2);
  return hn::MulSub(t0, t1, t4);
}

/**
 * @brief Compute determinant det(p, q, r) = r · (p × q).
 */
[[nodiscard]] HWY_INLINE V
determinant(V p, V q, V r) {
  return dot_vector(r, cross_vector(p, q));
}

/**
 * @brief Project origin onto line through p and q, return closest point.
 *
 * Computes: v = q - pq * (q · pq) / (pq · pq)
 * where pq = q - p
 */
[[nodiscard]] HWY_INLINE V
projectOnLine_vector(const V q, const V /*p*/, const V pq) {
  const auto nom = dot_vector(q, pq);
  const auto den = dot_vector(pq, pq);
  const auto t2 = nom / den;
  return hn::MulAdd(hn::Neg(pq), t2, q);
}

/**
 * @brief Project origin onto plane defined by points r, q, p.
 *
 * Computes the projection using cross product normal.
 */
[[nodiscard]] HWY_INLINE V
projectOnPlane_vector(V r, V q, V p) {
  const auto qr = r - q;
  const auto qp = r - p;
  const auto t0 = cross_vector(qr, qp);
  const auto t1 = dot_vector(t0, r) / dot_vector(t0, t0);
  return t0 * t1;
}

// ============================================================================
// Half-Face Functions (hff)
// ============================================================================

/**
 * @brief Half-face test 1: Is origin past q along edge p-q?
 *
 * Returns true if q · (q - p) > 0, meaning keep q in simplex.
 */
[[nodiscard]] HWY_INLINE hn::Mask<D>
hff1(const V q, const V /*p*/, const V pq) {
  D d;
  return dot_vector(q, pq) > hn::Zero(d);
}

/**
 * @brief Half-face test 2: Is origin on the q-side of edge p-r?
 *
 * Returns true if p · ((q-p) × ((q-p) × (r-p))) < 0, meaning discard r.
 */
[[nodiscard]] HWY_INLINE hn::Mask<D>
hff2(V r, V q, V p) {
  D d;
  const auto rq = r - q;
  const auto rp = r - p;
  const auto t0 = cross_vector(rq, rp);
  const auto t1 = cross_vector(rq, t0);
  return dot_vector(r, t1) > hn::Zero(d);
}

/**
 * @brief Half-face test 3: Is origin on the correct side of triangle face?
 *
 * Returns the signed volume (not a mask).
 */
[[nodiscard]] HWY_INLINE V
hff3(V s, V sr, V sp) {
  const auto t0 = cross_vector(sr, sp);
  return dot_vector(s, t0);
}

// ============================================================================
// Simplex Sub-Algorithms
// ============================================================================

/*
 * VERTEX NAMING CONVENTION (applies to all sub-algorithms):
 *
 *   S1D (line):        S[0] = p (newest),  S[1] = q
 *   S2D (triangle):    S[0] = p (newest),  S[1] = q,  S[2] = r
 *   S3D (tetrahedron): S[0] = p (newest),  S[1] = q,  S[2] = r,  S[3] = t
 *
 * The newest vertex (p) is always at index 0 and was just added to the
 * simplex in the direction of the origin. In GJK's progressive construction,
 * the solution MUST include vertex p. In EPA's exhaustive search, the
 * solution may reduce to any subset of vertices.
 */

/**
 * @brief S1D: Find closest point to origin on a 1-simplex (line segment).
 *
 * @tparam Policy  Search policy (ProgressiveSearch or ExhaustiveSearch).
 * @param[in,out] S     Simplex vertices: S[0] = p (newest), S[1] = q.
 * @param[out]    size  Updated number of vertices.
 * @param[out]    v     Closest point to origin.
 *
 * Voronoi regions:
 *   - Region {p}: origin closest to vertex p
 *   - Region {q}: origin closest to vertex q
 *   - Region {p,q}: origin projects onto edge interior
 *
 * For ProgressiveSearch (GJK): p is newest, so solution is {p} or {p,q}.
 * For ExhaustiveSearch (EPA): Tests all three regions {p}, {q}, {p,q}.
 */
template <typename Policy = ProgressiveSearch>
HWY_INLINE void
S1D_vector(V* HWY_RESTRICT S, V* HWY_RESTRICT size, V& v) {
  D d;
  const V Ones = hn::Set(d, 1);

  const V p = S[0]; // newest vertex
  const V q = S[1];
  const V qp = p - q; // direction from q toward p (toward newest)

  // Test: is origin past p (in direction away from q)?
  // hff1(p, q, qp) checks if p · qp > 0
  const auto hff1_p = hff1(p, q, qp);

  if (hn::AllFalse(d, hff1_p)) {
    // Region {p}: origin is closest to newest vertex p
    *size = Ones;
    // S[0] already contains p
    v = p;
    return;
  }

  // For exhaustive search, also test if origin is past q (away from p)
  if constexpr (Policy::kTestAllRegions) {
    const V pq = q - p;
    const auto hff1_q = hff1(q, p, pq);

    if (hn::AllFalse(d, hff1_q)) {
      // Region {q}: origin is closest to vertex q
      *size = Ones;
      S[0] = q;
      v = q;
      return;
    }
  }

  // Region {p,q}: project origin onto line segment
  // v = p - qp * (p · qp) / (qp · qp)
  v = projectOnLine_vector(p, q, qp);
}

/**
 * @brief S2D: Find closest point to origin on a 2-simplex (triangle).
 *
 * This implementation matches the scalar openGJK.c S2D exactly.
 *
 * @tparam Policy  Search policy (ProgressiveSearch or ExhaustiveSearch).
 * @param[in,out] S     Simplex vertices: S[0] = p (newest), S[1] = q, S[2] = r.
 * @param[out]    v     Closest point to origin.
 * @param[out]    size  Updated number of vertices.
 *
 * For ProgressiveSearch (GJK): p is newest, tests regions {p}, {p,q}, {p,r}, {p,q,r}
 * For ExhaustiveSearch (EPA): Tests all 7 Voronoi regions.
 */
template <typename Policy = ProgressiveSearch>
HWY_INLINE void
S2D_vector(V (&S)[4], V& v, V& size) {
  D d;

  const V Zeros = hn::Zero(d);
  const V Ones = hn::Set(d, 1);
  const V twos = hn::Set(d, 2);
  const V threes = hn::Set(d, 3);

  // Vertex naming matches scalar: s1p = newest, s2p, s3p = oldest
  const V s1p = S[0]; // newest (p)
  const V s2p = S[1]; // q
  const V s3p = S[2]; // r (oldest)

  // Degenerate triangle check
  const V s1s2 = s2p - s1p;
  const V s1s3 = s3p - s1p;
  const V eps = hn::Set(d, static_cast<gjkFloat>(1e-10));
  const V signArea = cross_vector(s1s2, s1s3);
  const V area = dot_vector(signArea, signArea);
  if (hn::AllTrue(d, eps >= area)) {
    // Degenerate: fall back to S1D on edge s1-s2
    size = twos;
    S1D_vector<Policy>(S, &size, v);
    return;
  }

  // Scalar hff1: s1 · (s1 - s2) > 0
  auto hff1_local = [&](const V& s1, const V& s2) -> hn::Mask<D> {
    const V diff = s1 - s2;
    return dot_vector(s1, diff) > Zeros;
  };

  // Scalar hff2: p · (pq × (pq × pr)) < 0, where pq = q - p, pr = r - p
  // Returns true if we should discard r (project onto edge p-q instead)
  auto hff2_local = [&](const V& p, const V& q, const V& r) -> hn::Mask<D> {
    const V pq = q - p;
    const V pr = r - p;
    const V n = cross_vector(pq, pr);
    const V t1 = cross_vector(pq, n);
    return dot_vector(p, t1) < Zeros; // Note: < 0, not > 0
  };

  const auto hff1f_s12 = hff1_local(s1p, s2p); // s1 · (s1 - s2) > 0
  const auto hff1f_s13 = hff1_local(s1p, s3p); // s1 · (s1 - s3) > 0

  if (hn::AllTrue(d, hff1f_s12)) {
    // Origin is past s1 toward s2
    const auto hff2f_23 = hn::Not(hff2_local(s1p, s2p, s3p)); // !hff2(s1, s2, s3)

    if (hn::AllTrue(d, hff2f_23)) {
      // Don't project onto edge s1-s2
      if (hn::AllTrue(d, hff1f_s13)) {
        // Origin is also past s1 toward s3
        const auto hff2f_32 = hn::Not(hff2_local(s1p, s3p, s2p)); // !hff2(s1, s3, s2)

        if (hn::AllTrue(d, hff2f_32)) {
          // Region {s1,s2,s3}: face
          size = threes;
          v = projectOnPlane_vector(s1p, s2p, s3p);
        } else {
          // Region {s1,s3}: edge s1-s3
          size = twos;
          S[1] = s3p;
          v = projectOnLine_vector(s1p, s3p, s1s3);
        }
      } else {
        // Origin is NOT past s1 toward s3
        // Region {s1,s2,s3}: face
        size = threes;
        v = projectOnPlane_vector(s1p, s2p, s3p);
      }
    } else {
      // Region {s1,s2}: edge s1-s2
      size = twos;
      S[1] = s2p;
      v = projectOnLine_vector(s1p, s2p, s1s2);
    }
  } else if (hn::AllTrue(d, hff1f_s13)) {
    // Origin is past s1 toward s3 (but NOT toward s2)
    const auto hff2f_32 = hn::Not(hff2_local(s1p, s3p, s2p));

    if (hn::AllTrue(d, hff2f_32)) {
      // Region {s1,s2,s3}: face
      size = threes;
      v = projectOnPlane_vector(s1p, s2p, s3p);
    } else {
      // Region {s1,s3}: edge s1-s3
      size = twos;
      S[1] = s3p;
      v = projectOnLine_vector(s1p, s3p, s1s3);
    }
  } else {
    // ProgressiveSearch: Origin is NOT past s1 toward s2 or s3
    // Region {s1}: vertex s1
    if constexpr (!Policy::kTestAllRegions) {
      size = Ones;
      // S[0] already contains s1p
      v = s1p;
      return;
    }
  }

  // ===========================================================================
  // ExhaustiveSearch: Test regions not containing the newest vertex p
  // These are {q}, {r}, and {q,r}
  // ===========================================================================
  if constexpr (Policy::kTestAllRegions) {
    // Test vertex q region
    const auto hff1_q_to_p = hff1_local(s2p, s1p); // q · (q - p) > 0
    const auto hff1_q_to_r = hff1_local(s2p, s3p); // q · (q - r) > 0

    if (hn::AllFalse(d, hff1_q_to_p) && hn::AllFalse(d, hff1_q_to_r)) {
      // Origin is closest to vertex q
      size = Ones;
      S[0] = s2p;
      v = s2p;
      return;
    }

    // Test vertex r region
    const auto hff1_r_to_p = hff1_local(s3p, s1p); // r · (r - p) > 0
    const auto hff1_r_to_q = hff1_local(s3p, s2p); // r · (r - q) > 0

    if (hn::AllFalse(d, hff1_r_to_p) && hn::AllFalse(d, hff1_r_to_q)) {
      // Origin is closest to vertex r
      size = Ones;
      S[0] = s3p;
      v = s3p;
      return;
    }

    // Test edge q-r region (only if origin is past both q and r along their edge)
    if (hn::AllTrue(d, hff1_q_to_r) && hn::AllTrue(d, hff1_r_to_q)) {
      // Origin projects onto edge q-r - check if it's on the q-r side of the triangle
      const auto hff2_qr = hff2_local(s2p, s3p, s1p); // Should we discard p?
      if (hn::AllTrue(d, hff2_qr)) {
        // Origin is closest to edge q-r
        size = twos;
        S[0] = s2p; // q
        S[1] = s3p; // r
        const V s2s3 = s3p - s2p;
        v = projectOnLine_vector(s2p, s3p, s2s3);
        return;
      }
    }

    // If we get here, origin must be in face region or we've found edge p-q or p-r
    // Check if origin is between all three edges (face region)
    // We already know it's not in vertex regions {q}, {r} or edge {q,r}
    // So project onto face
    size = threes;
    v = projectOnPlane_vector(s1p, s2p, s3p);
  }
}

/**
 * @brief S3D: Find closest point to origin on a 3-simplex (tetrahedron).
 *
 * @tparam Policy  Search policy (ProgressiveSearch or ExhaustiveSearch).
 * @param[in,out] S     Simplex vertices (S[0]=p, S[1]=q, S[2]=r, S[3]=s where s is newest).
 * @param[out]    v     Closest point to origin.
 * @param[out]    size  Updated number of vertices.
 *
 * For ProgressiveSearch (GJK): Tests regions reachable from newest vertex s.
 * For ExhaustiveSearch (EPA): Tests all 15 Voronoi regions.
 */
template <typename Policy = ProgressiveSearch>
HWY_INLINE void
S3D_vector(V (&S)[4], V& v, V& size) {
  D d;

  const V Zeros = hn::Zero(d);
  const V Ones = hn::Set(d, 1);
  const V twos = hn::Set(d, 2);
  const V threes = hn::Set(d, 3);

  const V p = S[0];
  const V q = S[1];
  const V r = S[2];
  const V s = S[3];

  const V rs = s - r;
  const V qs = s - q;
  const V ps = s - p;

  const hn::Mask<D> hff1_tests[3] = {hff1(s, p, ps), hff1(s, q, qs), hff1(s, r, rs)};
  const auto testLineThree = hn::IfThenElse(hff1_tests[1], Ones, Zeros);
  const auto testLineFour = hn::IfThenElse(hff1_tests[0], Ones, Zeros);

  const auto dotTotal = testLineFour + testLineThree + hn::IfThenElse(hff1_tests[2], Ones, Zeros);

  if (hn::AllTrue(d, dotTotal == Zeros)) {
    // Region {1}: only s
    size = Ones;
    S[0] = s;
    v = s;
    return;
  }

  const auto det = determinant(rs, qs, ps);

  const auto testPlane2 = hn::IfThenElse((det * hff3(s, qs, ps) > Zeros), Ones, Zeros);
  const auto testPlane3 = hn::IfThenElse((det * hff3(s, ps, rs) > Zeros), Ones, Zeros);
  const auto testPlane4 = hn::IfThenElse((det * hff3(s, rs, qs) > Zeros), Ones, Zeros);

  const auto swith = testPlane2 + testPlane3 + testPlane4;

  if (hn::AllTrue(d, swith == hn::Set(d, 3))) {
    // Region {1,2,3,4}: origin inside tetrahedron
    v = Zeros;
  } else if (hn::AllTrue(d, swith == hn::Set(d, 2))) {
    // One face visible: call S2D
    size = twos + Ones;
    S[2] = s;
    if (hn::AllFalse(d, testPlane3 > Zeros)) {
      S[1] = r;
    } else if (hn::AllFalse(d, testPlane4 > Zeros)) {
      S[1] = r;
      S[0] = q;
    }
    S2D_vector<Policy>(S, v, size);
  } else if (hn::AllTrue(d, swith == hn::Set(d, 1))) {
    // Two faces visible: select appropriate region
    size_t i, j, k;
    if (hn::AllTrue(d, testPlane2 > Zeros)) {
      k = 2;
      i = 1;
      j = 0;
    } else if (hn::AllTrue(d, testPlane3 > Zeros)) {
      k = 1;
      i = 0;
      j = 2;
    } else {
      k = 0;
      i = 2;
      j = 1;
    }

    const auto si = S[i];
    const auto sj = S[j];
    const auto sk = S[k];

    const auto hff2_ik = hff2(s, si, sk);
    const auto hff2_jk = hff2(s, sj, sk);
    const auto hff2_ki = hff2(s, sk, si);
    const auto hff2_kj = hff2(s, sk, sj);

    if (hn::AllTrue(d, dotTotal == hn::Set(d, 1))) {
      if (hn::AllTrue(d, hff1_tests[k])) {
        if (hn::AllFalse(d, hff2_ki)) {
          S[2] = s;
          S[1] = si;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, si, sk);
        } else if (hn::AllFalse(d, hff2_kj)) {
          S[2] = s;
          S[1] = sj;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, sj, sk);
        } else {
          S[1] = s;
          S[0] = sk;
          size = twos;
          v = projectOnLine_vector(s, sk, sk - s);
        }
      } else if (hn::AllTrue(d, hff1_tests[i])) {
        if (hn::AllFalse(d, hff2_ik)) {
          S[2] = s;
          S[1] = si;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, sk, si);
        } else {
          S[1] = s;
          S[0] = sk;
          size = twos;
          v = projectOnLine_vector(s, si, si - s);
        }
      } else {
        if (hn::AllFalse(d, hff2_jk)) {
          S[2] = s;
          S[1] = si;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, sj, sk);
        } else {
          S[1] = s;
          S[0] = sj;
          size = twos;
          v = projectOnLine_vector(s, sj, sj - s);
        }
      }
    } else if (hn::AllTrue(d, dotTotal == hn::Set(d, 2))) {
      if (hn::AllTrue(d, hff1_tests[i])) {
        if (hn::AllFalse(d, hff2_ki)) {
          if (hn::AllFalse(d, hff2_ik)) {
            S[2] = s;
            S[1] = si;
            S[0] = sk;
            size = threes;
            v = projectOnPlane_vector(s, si, sk);
          } else {
            S[1] = s;
            S[0] = sk;
            size = twos;
            v = projectOnLine_vector(s, sk, sk - s);
          }
        } else {
          if (hn::AllFalse(d, hff2_kj)) {
            S[2] = s;
            S[1] = sj;
            S[0] = sk;
            size = threes;
            v = projectOnPlane_vector(s, sj, sk);
          } else {
            S[1] = s;
            S[0] = sk;
            size = twos;
            v = projectOnLine_vector(s, sk, sk - s);
          }
        }
      } else if (hn::AllTrue(d, hff1_tests[j])) {
        if (hn::AllFalse(d, hff2_kj)) {
          if (hn::AllFalse(d, hff2_jk)) {
            S[2] = s;
            S[1] = sj;
            S[0] = sk;
            size = threes;
            v = projectOnPlane_vector(s, sj, sk);
          } else {
            S[1] = s;
            S[0] = sj;
            size = twos;
            v = projectOnLine_vector(s, sj, sj - s);
          }
        } else {
          if (hn::AllFalse(d, hff2_ki)) {
            S[2] = s;
            S[1] = si;
            S[0] = sk;
            size = threes;
            v = projectOnPlane_vector(s, si, sk);
          } else {
            S[1] = s;
            S[0] = sk;
            size = twos;
            v = projectOnLine_vector(s, sk, sk - s);
          }
        }
      }
    } else if (hn::AllTrue(d, dotTotal == hn::Set(d, 3))) {
      if (hn::AllFalse(d, hn::Or(hff2_ki, hff2_kj))) {
        // Unexpected
      } else if (hn::AllTrue(d, hn::And(hff2_ki, hff2_kj))) {
        S[1] = s;
        S[0] = sk;
        size = twos;
        v = projectOnLine_vector(s, sk, sk - s);
      } else if (hn::AllTrue(d, hff2_ki)) {
        if (hn::AllTrue(d, hff2_jk)) {
          S[1] = s;
          S[0] = sj;
          size = twos;
          v = projectOnLine_vector(s, sj, sj - s);
        } else {
          S[2] = s;
          S[1] = sj;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, sj, sk);
        }
      } else {
        if (hn::AllTrue(d, hff2_ik)) {
          S[1] = s;
          S[0] = sk;
          size = twos;
          v = projectOnLine_vector(s, si, si - s);
        } else {
          S[2] = s;
          S[1] = si;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, si, sk);
        }
      }
    }
  } else if (hn::AllTrue(d, swith == hn::Set(d, 0))) {
    // Origin outside all faces
    if (hn::AllTrue(d, dotTotal == hn::Set(d, 1))) {
      size_t i, j, k;
      if (hn::AllTrue(d, testLineThree > Zeros)) {
        k = 2;
        i = 1;
        j = 0;
      } else if (hn::AllTrue(d, testLineFour > Zeros)) {
        k = 1;
        i = 0;
        j = 2;
      } else {
        k = 0;
        i = 2;
        j = 1;
      }
      const auto si = S[i];
      const auto sj = S[j];
      const auto sk = S[k];

      if (hn::AllFalse(d, hff2(s, si, sj))) {
        S[2] = s;
        S[1] = si;
        S[0] = sj;
        size = threes;
        v = projectOnPlane_vector(s, sj, si);
      } else if (hn::AllFalse(d, hff2(s, si, sk))) {
        S[2] = s;
        S[1] = si;
        S[0] = sk;
        size = threes;
        v = projectOnPlane_vector(s, si, sk);
      } else {
        S[1] = s;
        S[0] = sk;
        size = twos;
        v = projectOnLine_vector(s, si, si - s);
      }
    } else if (hn::AllTrue(d, dotTotal == hn::Set(d, 2))) {
      size_t i, j, k = 99;
      if (hn::AllFalse(d, testLineThree > Zeros)) {
        k = 2;
        i = 1;
        j = 0;
      } else if (hn::AllFalse(d, testLineFour > Zeros)) {
        k = 1;
        i = 0;
        j = 2;
      } else {
        k = 0;
        i = 2;
        j = 1;
      }
      const auto si = S[i];
      const auto sj = S[j];
      const auto sk = S[k];

      if (hn::AllFalse(d, hff2(s, sj, sk))) {
        if (hn::AllFalse(d, hff2(s, sk, sj))) {
          S[2] = s;
          S[1] = sj;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, sj, sk);
        } else if (hn::AllFalse(d, hff2(s, sk, si))) {
          S[2] = s;
          S[1] = si;
          S[0] = sk;
          size = threes;
          v = projectOnPlane_vector(s, si, sk);
        } else {
          S[1] = s;
          S[0] = sk;
          size = twos;
          v = projectOnLine_vector(s, sk, sk - s);
        }
      } else if (hn::AllFalse(d, hff2(s, sj, si))) {
        S[2] = s;
        S[1] = si;
        S[0] = sj;
        size = threes;
        v = projectOnPlane_vector(s, sj, si);
      } else {
        S[1] = s;
        S[0] = sj;
        size = twos;
        v = projectOnLine_vector(s, sj, sj - s);
      }
    }
  }
}

} // namespace
} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace opengjk

HWY_AFTER_NAMESPACE();

#endif // OPENGJK_SIMD_INL_H_TARGET toggle
