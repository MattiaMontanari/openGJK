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

#include <cmath>
#include <cstdio>

// MUST be included BEFORE any Highway headers to configure HWY_DISABLED_TARGETS
#include "include/opengjk_simd_compile_config.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "opengjk.cc"
#include <hwy/foreach_target.h>

#include <hwy/highway.h>
#include <hwy/cache_control.h>

#include "include/opengjk_simd.hh"
#include "opengjk-inl.h"

HWY_BEFORE_NAMESPACE();

namespace opengjk {
namespace simd {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

using D = hn::CappedTag<gjkFloat, 4>;
using V = hn::Vec<D>;

// ============================================================================
// Support Function
// ============================================================================

/**
 * @brief Find the furthest point along direction v using SIMD.
 *
 * Uses prefetch hints to improve cache behavior for large polytopes.
 * For reads, false sharing is not a concern (reads don't cause cache
 * invalidation). The function uses non-temporal hints where beneficial.
 *
 * @param coords_x  X coordinates (SoA layout).
 * @param coords_y  Y coordinates.
 * @param coords_z  Z coordinates.
 * @param num_items Number of points.
 * @param v         Search direction.
 * @return Index of the furthest point.
 */
HWY_INLINE size_t
furthest_point_along_v(const gjkFloat* HWY_RESTRICT coords_x, const gjkFloat* HWY_RESTRICT coords_y,
                       const gjkFloat* HWY_RESTRICT coords_z, const size_t num_items, const gjkFloat v[3]) {
  const hn::ScalableTag<gjkFloat> d;
  const size_t N = hn::Lanes(d);

  // Prefetch distance: typically 2-4 cache lines ahead
  constexpr size_t kPrefetchDistance = 256 / sizeof(gjkFloat);

  const auto vx = hn::Set(d, v[0]);
  const auto vy = hn::Set(d, v[1]);
  const auto vz = hn::Set(d, v[2]);

  auto global_max = hn::Set(d, static_cast<gjkFloat>(-1e30));
  auto global_max_idx = hn::Set(d, static_cast<gjkFloat>(-1));

  size_t i = 0;
  for (; i + N <= num_items; i += N) {
    // Prefetch next iterations' data to hide memory latency
    if (i + kPrefetchDistance < num_items) {
      hwy::Prefetch(coords_x + i + kPrefetchDistance);
      hwy::Prefetch(coords_y + i + kPrefetchDistance);
      hwy::Prefetch(coords_z + i + kPrefetchDistance);
    }

    const auto px = hn::Load(d, coords_x + i);
    const auto py = hn::Load(d, coords_y + i);
    const auto pz = hn::Load(d, coords_z + i);

    const auto dot = hn::MulAdd(px, vx, hn::MulAdd(py, vy, pz * vz));
    const auto is_greater = hn::Gt(dot, global_max);
    const auto iota = hn::Iota(d, static_cast<gjkFloat>(i));

    global_max = hn::IfThenElse(is_greater, dot, global_max);
    global_max_idx = hn::IfThenElse(is_greater, iota, global_max_idx);
  }

  // Handle remainder with scalar loop
  gjkFloat max_val = hn::ReduceMax(d, global_max);
  size_t max_idx = static_cast<size_t>(hn::GetLane(global_max_idx));

  // Find which lane had the maximum
  HWY_ALIGN gjkFloat idx_arr[hn::MaxLanes(d)];
  HWY_ALIGN gjkFloat val_arr[hn::MaxLanes(d)];
  hn::Store(global_max_idx, d, idx_arr);
  hn::Store(global_max, d, val_arr);

  for (size_t j = 0; j < N; ++j) {
    if (val_arr[j] == max_val && idx_arr[j] >= 0) {
      max_idx = static_cast<size_t>(idx_arr[j]);
      break;
    }
  }

  // Scalar remainder
  for (; i < num_items; ++i) {
    gjkFloat dot = coords_x[i] * v[0] + coords_y[i] * v[1] + coords_z[i] * v[2];
    if (dot > max_val) {
      max_val = dot;
      max_idx = i;
    }
  }

  return max_idx;
}

/**
 * @brief Support function for AoS layout polytopes.
 */
HWY_INLINE void
support_aos(gjkFloat** coord, int numpoints, const gjkFloat* v, gjkFloat* s, int* s_idx) {
  gjkFloat max_dot = coord[0][0] * v[0] + coord[0][1] * v[1] + coord[0][2] * v[2];
  int best = 0;

  for (int i = 1; i < numpoints; ++i) {
    gjkFloat dot = coord[i][0] * v[0] + coord[i][1] * v[1] + coord[i][2] * v[2];
    if (dot > max_dot) {
      max_dot = dot;
      best = i;
    }
  }

  s[0] = coord[best][0];
  s[1] = coord[best][1];
  s[2] = coord[best][2];
  *s_idx = best;
}

// ============================================================================
// Witness Computation
// ============================================================================

HWY_INLINE void
W0D(const Polytope& bd1, const Polytope& bd2, Simplex* smp) {
  const gjkFloat* w00 = bd1.coord[smp->vrtx_idx[0][0]];
  const gjkFloat* w01 = bd2.coord[smp->vrtx_idx[0][1]];
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t];
    smp->witnesses[1][t] = w01[t];
  }
}

HWY_INLINE void
W1D(const Polytope& bd1, const Polytope& bd2, Simplex* smp) {
  gjkFloat pq[3], po[3];

  const gjkFloat* p = smp->vrtx[0];
  const gjkFloat* q = smp->vrtx[1];

  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    po[t] = -p[t];
  }

  const gjkFloat det = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
  if (det == 0.0) {
    W0D(bd1, bd2, smp);
    return;
  }

  const gjkFloat a1 = (pq[0] * po[0] + pq[1] * po[1] + pq[2] * po[2]) / det;
  const gjkFloat a0 = 1.0 - a1;

  const gjkFloat* w00 = bd1.coord[smp->vrtx_idx[0][0]];
  const gjkFloat* w01 = bd2.coord[smp->vrtx_idx[0][1]];
  const gjkFloat* w10 = bd1.coord[smp->vrtx_idx[1][0]];
  const gjkFloat* w11 = bd2.coord[smp->vrtx_idx[1][1]];

  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t] * a0 + w10[t] * a1;
    smp->witnesses[1][t] = w01[t] * a0 + w11[t] * a1;
  }
}

HWY_INLINE void
W2D(const Polytope& bd1, const Polytope& bd2, Simplex* smp) {
  gjkFloat pq[3], pr[3], po[3];

  const gjkFloat* p = smp->vrtx[0];
  const gjkFloat* q = smp->vrtx[1];
  const gjkFloat* r = smp->vrtx[2];

  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    pr[t] = r[t] - p[t];
    po[t] = -p[t];
  }

  const gjkFloat T00 = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
  const gjkFloat T01 = pq[0] * pr[0] + pq[1] * pr[1] + pq[2] * pr[2];
  const gjkFloat T11 = pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2];
  const gjkFloat det = T00 * T11 - T01 * T01;

  if (det == 0.0) {
    W1D(bd1, bd2, smp);
    return;
  }

  const gjkFloat b0 = pq[0] * po[0] + pq[1] * po[1] + pq[2] * po[2];
  const gjkFloat b1 = pr[0] * po[0] + pr[1] * po[1] + pr[2] * po[2];
  const gjkFloat a1 = (T11 * b0 - T01 * b1) / det;
  const gjkFloat a2 = (-T01 * b0 + T00 * b1) / det;
  const gjkFloat a0 = 1.0 - a1 - a2;

  const gjkFloat* w00 = bd1.coord[smp->vrtx_idx[0][0]];
  const gjkFloat* w01 = bd2.coord[smp->vrtx_idx[0][1]];
  const gjkFloat* w10 = bd1.coord[smp->vrtx_idx[1][0]];
  const gjkFloat* w11 = bd2.coord[smp->vrtx_idx[1][1]];
  const gjkFloat* w20 = bd1.coord[smp->vrtx_idx[2][0]];
  const gjkFloat* w21 = bd2.coord[smp->vrtx_idx[2][1]];

  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t] * a0 + w10[t] * a1 + w20[t] * a2;
    smp->witnesses[1][t] = w01[t] * a0 + w11[t] * a1 + w21[t] * a2;
  }
}

HWY_INLINE void
W3D(const Polytope& bd1, const Polytope& bd2, Simplex* smp) {
  gjkFloat pq[3], pr[3], ps[3], po[3];

  const gjkFloat* p = smp->vrtx[0];
  const gjkFloat* q = smp->vrtx[1];
  const gjkFloat* r = smp->vrtx[2];
  const gjkFloat* s = smp->vrtx[3];

  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    pr[t] = r[t] - p[t];
    ps[t] = s[t] - p[t];
    po[t] = -p[t];
  }

  const gjkFloat T00 = pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2];
  const gjkFloat T01 = pq[0] * pr[0] + pq[1] * pr[1] + pq[2] * pr[2];
  const gjkFloat T02 = pq[0] * ps[0] + pq[1] * ps[1] + pq[2] * ps[2];
  const gjkFloat T11 = pr[0] * pr[0] + pr[1] * pr[1] + pr[2] * pr[2];
  const gjkFloat T12 = pr[0] * ps[0] + pr[1] * ps[1] + pr[2] * ps[2];
  const gjkFloat T22 = ps[0] * ps[0] + ps[1] * ps[1] + ps[2] * ps[2];

  const gjkFloat det00 = T11 * T22 - T12 * T12;
  const gjkFloat det01 = T01 * T22 - T02 * T12;
  const gjkFloat det02 = T01 * T12 - T02 * T11;
  const gjkFloat det = T00 * det00 - T01 * det01 + T02 * det02;

  if (det == 0.0) {
    W2D(bd1, bd2, smp);
    return;
  }

  const gjkFloat b0 = pq[0] * po[0] + pq[1] * po[1] + pq[2] * po[2];
  const gjkFloat b1 = pr[0] * po[0] + pr[1] * po[1] + pr[2] * po[2];
  const gjkFloat b2 = ps[0] * po[0] + ps[1] * po[1] + ps[2] * po[2];

  const gjkFloat det11 = T00 * T22 - T02 * T02;
  const gjkFloat det12 = T00 * T12 - T01 * T02;
  const gjkFloat det22 = T00 * T11 - T01 * T01;

  const gjkFloat a1 = (det00 * b0 - det01 * b1 + det02 * b2) / det;
  const gjkFloat a2 = (-det01 * b0 + det11 * b1 - det12 * b2) / det;
  const gjkFloat a3 = (det02 * b0 - det12 * b1 + det22 * b2) / det;
  const gjkFloat a0 = 1.0 - a1 - a2 - a3;

  const gjkFloat* w00 = bd1.coord[smp->vrtx_idx[0][0]];
  const gjkFloat* w01 = bd2.coord[smp->vrtx_idx[0][1]];
  const gjkFloat* w10 = bd1.coord[smp->vrtx_idx[1][0]];
  const gjkFloat* w11 = bd2.coord[smp->vrtx_idx[1][1]];
  const gjkFloat* w20 = bd1.coord[smp->vrtx_idx[2][0]];
  const gjkFloat* w21 = bd2.coord[smp->vrtx_idx[2][1]];
  const gjkFloat* w30 = bd1.coord[smp->vrtx_idx[3][0]];
  const gjkFloat* w31 = bd2.coord[smp->vrtx_idx[3][1]];

  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t] * a0 + w10[t] * a1 + w20[t] * a2 + w30[t] * a3;
    smp->witnesses[1][t] = w01[t] * a0 + w11[t] * a1 + w21[t] * a2 + w31[t] * a3;
  }
}

HWY_INLINE void
compute_witnesses(const Polytope& bd1, const Polytope& bd2, Simplex* smp) {
  switch (smp->nvrtx) {
    case 4:
      W3D(bd1, bd2, smp);
      break;
    case 3:
      W2D(bd1, bd2, smp);
      break;
    case 2:
      W1D(bd1, bd2, smp);
      break;
    case 1:
      W0D(bd1, bd2, smp);
      break;
    default:
      break;
  }
}

// ============================================================================
// Main GJK Algorithm
// ============================================================================

gjkFloat
gjk_impl(const Polytope& bd1, const Polytope& bd2, Simplex* s) {
  D d;
  unsigned int k = 0;
  const unsigned int mk = kMaxIterations;
  const gjkFloat eps_rel = kEpsilonRel;
  const gjkFloat eps_rel2 = eps_rel * eps_rel;
  gjkFloat norm2Wmax = 0;

  // Local copies for modification
  gjkFloat s1[4] = {bd1.coord[0][0], bd1.coord[0][1], bd1.coord[0][2], 0};
  gjkFloat s2[4] = {bd2.coord[0][0], bd2.coord[0][1], bd2.coord[0][2], 0};
  int s1_idx = 0;
  int s2_idx = 0;

  // Initialize search direction
  HWY_ALIGN gjkFloat v[4] = {bd1.coord[0][0] - bd2.coord[0][0], bd1.coord[0][1] - bd2.coord[0][1],
                             bd1.coord[0][2] - bd2.coord[0][2], 0};

  // Initialize simplex with first point
  s->nvrtx = 1;
  for (int t = 0; t < 3; ++t) {
    s->vrtx[0][t] = v[t];
  }
  s->vrtx[0][3] = 0;
  s->vrtx_idx[0][0] = 0;
  s->vrtx_idx[0][1] = 0;

  // SIMD simplex storage and index tracking
  V S[4];
  S[0] = hn::Load(d, v);

  // Parallel index arrays to track original polytope vertex indices
  int idx0[4] = {0, 0, 0, 0}; // indices into bd1
  int idx1[4] = {0, 0, 0, 0}; // indices into bd2

  V size_v = hn::Set(d, static_cast<gjkFloat>(1));
  V v_vec = hn::Load(d, v);

  do {
    k++;

    // Negate search direction for first body
    HWY_ALIGN gjkFloat vminus[4] = {-v[0], -v[1], -v[2], 0};

    // Support function calls
    support_aos(bd1.coord, bd1.numpoints, vminus, s1, &s1_idx);
    support_aos(bd2.coord, bd2.numpoints, v, s2, &s2_idx);

    // Compute Minkowski difference point
    HWY_ALIGN gjkFloat w[4] = {s1[0] - s2[0], s1[1] - s2[1], s1[2] - s2[2], 0};

    // Test termination: can we make progress?
    gjkFloat norm2_v = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    gjkFloat v_dot_w = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    gjkFloat exceed_tol = norm2_v - v_dot_w;

    if (exceed_tol <= eps_rel * norm2_v || exceed_tol < kEpsilonAbs) {
      break;
    }

    if (norm2_v < eps_rel2) {
      break;
    }

    // Add new vertex to simplex
    int i = s->nvrtx;
    for (int t = 0; t < 3; ++t) {
      s->vrtx[i][t] = w[t];
    }
    s->vrtx[i][3] = 0;
    s->vrtx_idx[i][0] = s1_idx;
    s->vrtx_idx[i][1] = s2_idx;
    s->nvrtx++;

    // Load into SIMD registers and track indices
    S[i] = hn::Load(d, w);
    idx0[i] = s1_idx;
    idx1[i] = s2_idx;
    size_v = hn::Set(d, static_cast<gjkFloat>(s->nvrtx));

    // Save pre-sub-algorithm state for index matching
    HWY_ALIGN gjkFloat saved_verts[4][4];
    int saved_idx0[4], saved_idx1[4];
    for (int j = 0; j < s->nvrtx; ++j) {
      hn::Store(S[j], d, saved_verts[j]);
      saved_idx0[j] = idx0[j];
      saved_idx1[j] = idx1[j];
    }
    int saved_nvrtx = s->nvrtx;

    // Invoke sub-algorithm
    // NOTE: S1D/S2D expect newest vertex at S[0], S3D expects newest at S[3]
    switch (s->nvrtx) {
      case 4:
        // S3D: newest at S[3] - already correct (added at position 3)
        S3D_vector(S, v_vec, size_v);
        break;
      case 3: {
        // S2D: newest at S[0] - rotate so S[2] (newest) becomes S[0]
        V tmp = S[0];
        S[0] = S[2]; // newest
        S[2] = tmp;
        // Also rotate the saved state
        gjkFloat tv[4];
        for (int t = 0; t < 4; ++t) {
          tv[t] = saved_verts[0][t];
        }
        for (int t = 0; t < 4; ++t) {
          saved_verts[0][t] = saved_verts[2][t];
          saved_verts[2][t] = tv[t];
        }
        int ti0 = saved_idx0[0], ti1 = saved_idx1[0];
        saved_idx0[0] = saved_idx0[2];
        saved_idx1[0] = saved_idx1[2];
        saved_idx0[2] = ti0;
        saved_idx1[2] = ti1;
        S2D_vector(S, v_vec, size_v);
        break;
      }
      case 2: {
        // S1D: newest at S[0] - swap so S[1] (newest) becomes S[0]
        V tmp = S[0];
        S[0] = S[1]; // newest
        S[1] = tmp;
        // Also swap the saved state
        gjkFloat tv[4];
        for (int t = 0; t < 4; ++t) {
          tv[t] = saved_verts[0][t];
        }
        for (int t = 0; t < 4; ++t) {
          saved_verts[0][t] = saved_verts[1][t];
          saved_verts[1][t] = tv[t];
        }
        int ti0 = saved_idx0[0], ti1 = saved_idx1[0];
        saved_idx0[0] = saved_idx0[1];
        saved_idx1[0] = saved_idx1[1];
        saved_idx0[1] = ti0;
        saved_idx1[1] = ti1;
        S1D_vector(S, &size_v, v_vec);
        break;
      }
    }

    // Store results back
    hn::Store(v_vec, d, v);
    s->nvrtx = static_cast<int>(hn::GetLane(size_v));

    // Match output vertices to saved input vertices and update indices
    for (int j = 0; j < s->nvrtx; ++j) {
      HWY_ALIGN gjkFloat out_vert[4];
      hn::Store(S[j], d, out_vert);

      // Find matching input vertex by coordinate comparison
      for (int k_match = 0; k_match < saved_nvrtx; ++k_match) {
        gjkFloat dx = out_vert[0] - saved_verts[k_match][0];
        gjkFloat dy = out_vert[1] - saved_verts[k_match][1];
        gjkFloat dz = out_vert[2] - saved_verts[k_match][2];
        if (dx * dx + dy * dy + dz * dz < static_cast<gjkFloat>(1e-20)) {
          idx0[j] = saved_idx0[k_match];
          idx1[j] = saved_idx1[k_match];
          break;
        }
      }
    }

    // Update simplex vertices and indices from SIMD registers
    for (int j = 0; j < s->nvrtx; ++j) {
      HWY_ALIGN gjkFloat tmp[4];
      hn::Store(S[j], d, tmp);
      for (int t = 0; t < 3; ++t) {
        s->vrtx[j][t] = tmp[t];
      }
      s->vrtx_idx[j][0] = idx0[j];
      s->vrtx_idx[j][1] = idx1[j];
    }

    // Track maximum norm for termination test
    for (int jj = 0; jj < s->nvrtx; jj++) {
      gjkFloat testnorm =
          s->vrtx[jj][0] * s->vrtx[jj][0] + s->vrtx[jj][1] * s->vrtx[jj][1] + s->vrtx[jj][2] * s->vrtx[jj][2];
      if (testnorm > norm2Wmax) {
        norm2Wmax = testnorm;
      }
    }

    gjkFloat norm2_v_new = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (norm2_v_new <= kEpsilonAbs * kEpsilonAbs * norm2Wmax) {
      break;
    }

  } while (s->nvrtx != 4 && k != mk);

  // Compute witness points
  compute_witnesses(bd1, bd2, s);

  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

} // namespace HWY_NAMESPACE
} // namespace simd
} // namespace opengjk

HWY_AFTER_NAMESPACE();

// ============================================================================
// Dynamic Dispatch
// ============================================================================

#if HWY_ONCE
namespace opengjk {
namespace simd {

namespace {
HWY_EXPORT(gjk_impl);
}

gjkFloat
compute_minimum_distance(const Polytope& bd1, const Polytope& bd2, Simplex* s) {
  return HWY_DYNAMIC_DISPATCH(gjk_impl)(bd1, bd2, s);
}

} // namespace simd
} // namespace opengjk
#endif
