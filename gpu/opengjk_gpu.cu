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
 * Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3. See https://www.gnu.org/licenses/
 */

/**
 * @file openGJK_GPU.cu
 * @author Mattia Montanari, Vismay Churiwala, Marcus Hedlund
 * @date 22 Jan 2026
 * @brief GPU (CUDA) implementation of OpenGJK - Kernel implementations
 *
 * CUDA kernel and host API implementation with warp-level parallelism for
 * high-performance collision detection on NVIDIA GPUs.
 *
 * @see https://github.com/vismaychuriwala/OpenGJK-GPU
 * @see https://www.mattiamontanari.com/opengjk/
 */

#include "openGJK_GPU.h"

#include <stdio.h>
#include <stdlib.h>

#include "math.h"

#define eps_rel22 (gkFloat) gkEpsilon * 1e4f
#define eps_tot22 (gkFloat) gkEpsilon * 1e2f

// Threads per computation for parallel kernels
#define THREADS_PER_COMPUTATION 16  // GJK uses 16 threads (half-warp)
#define THREADS_PER_EPA 32           // EPA uses 32 threads (full warp)

// Maximum number of faces in the EPA polytope
#define MAX_EPA_FACES 128
#define MAX_EPA_VERTICES (MAX_EPA_FACES + 4)

#define getCoord(body, index, component) body->coord[(index) * 3 + (component)]

#define norm2(a) (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
#define dotProduct(a, b) (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

#define S3Dregion1234() \
  v[0] = 0;             \
  v[1] = 0;             \
  v[2] = 0;             \
  s->nvrtx = 4;

#define select_1ik()                                             \
  s->nvrtx = 3;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = si[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = si_idx[t];         \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

#define select_1ij()                                             \
  s->nvrtx = 3;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = si[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = si_idx[t];         \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = sj[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sj_idx[t];

#define select_1jk()                                             \
  s->nvrtx = 3;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = sj[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = sj_idx[t];         \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

#define select_1i()                                              \
  s->nvrtx = 2;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = si[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = si_idx[t];

#define select_1j()                                              \
  s->nvrtx = 2;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = sj[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sj_idx[t];

#define select_1k()                                              \
  s->nvrtx = 2;                                                  \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  _Pragma("unroll") for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  _Pragma("unroll") for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

#define getvrtx(point, location)   \
  point[0] = s->vrtx[location][0]; \
  point[1] = s->vrtx[location][1]; \
  point[2] = s->vrtx[location][2];

#define getvrtxidx(point, index, location) \
  point[0] = s->vrtx[location][0];         \
  point[1] = s->vrtx[location][1];         \
  point[2] = s->vrtx[location][2];         \
  index[0] = s->vrtx_idx[location][0];     \
  index[1] = s->vrtx_idx[location][1];

#define calculateEdgeVector(p1p2, p2) \
  p1p2[0] = p2[0] - s->vrtx[3][0];    \
  p1p2[1] = p2[1] - s->vrtx[3][1];    \
  p1p2[2] = p2[2] - s->vrtx[3][2];

#define S1Dregion1()                     \
  v[0] = s->vrtx[1][0];                  \
  v[1] = s->vrtx[1][1];                  \
  v[2] = s->vrtx[1][2];                  \
  s->nvrtx = 1;                          \
  s->vrtx[0][0] = s->vrtx[1][0];         \
  s->vrtx[0][1] = s->vrtx[1][1];         \
  s->vrtx[0][2] = s->vrtx[1][2];         \
  s->vrtx_idx[0][0] = s->vrtx_idx[1][0]; \
  s->vrtx_idx[0][1] = s->vrtx_idx[1][1];

#define S2Dregion1()                     \
  v[0] = s->vrtx[2][0];                  \
  v[1] = s->vrtx[2][1];                  \
  v[2] = s->vrtx[2][2];                  \
  s->nvrtx = 1;                          \
  s->vrtx[0][0] = s->vrtx[2][0];         \
  s->vrtx[0][1] = s->vrtx[2][1];         \
  s->vrtx[0][2] = s->vrtx[2][2];         \
  s->vrtx_idx[0][0] = s->vrtx_idx[2][0]; \
  s->vrtx_idx[0][1] = s->vrtx_idx[2][1];

#define S2Dregion12()                    \
  s->nvrtx = 2;                          \
  s->vrtx[0][0] = s->vrtx[2][0];         \
  s->vrtx[0][1] = s->vrtx[2][1];         \
  s->vrtx[0][2] = s->vrtx[2][2];         \
  s->vrtx_idx[0][0] = s->vrtx_idx[2][0]; \
  s->vrtx_idx[0][1] = s->vrtx_idx[2][1];

#define S2Dregion13()                    \
  s->nvrtx = 2;                          \
  s->vrtx[1][0] = s->vrtx[2][0];         \
  s->vrtx[1][1] = s->vrtx[2][1];         \
  s->vrtx[1][2] = s->vrtx[2][2];         \
  s->vrtx_idx[1][0] = s->vrtx_idx[2][0]; \
  s->vrtx_idx[1][1] = s->vrtx_idx[2][1];

#define S3Dregion1()             \
  v[0] = s1[0];                  \
  v[1] = s1[1];                  \
  v[2] = s1[2];                  \
  s->nvrtx = 1;                  \
  s->vrtx[0][0] = s1[0];         \
  s->vrtx[0][1] = s1[1];         \
  s->vrtx[0][2] = s1[2];         \
  s->vrtx_idx[0][0] = s1_idx[0]; \
  s->vrtx_idx[0][1] = s1_idx[1];




__device__ inline static gkFloat determinant(const gkFloat* p,
  const gkFloat* q,
  const gkFloat* r) {
  return p[0] * ((q[1] * r[2]) - (r[1] * q[2])) -
    p[1] * (q[0] * r[2] - r[0] * q[2]) +
    p[2] * (q[0] * r[1] - r[0] * q[1]);
}

__device__ inline static void crossProduct(const gkFloat* a,
  const gkFloat* b,
  gkFloat* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline static void projectOnLine(const gkFloat* p,
  const gkFloat* q,
  gkFloat* v) {
  gkFloat pq[3];
  pq[0] = p[0] - q[0];
  pq[1] = p[1] - q[1];
  pq[2] = p[2] - q[2];

  const gkFloat tmp = dotProduct(p, pq) / dotProduct(pq, pq);

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    v[i] = p[i] - pq[i] * tmp;
  }
}

__device__ inline static void projectOnPlane(const gkFloat* p,
  const gkFloat* q,
  const gkFloat* r,
  gkFloat* v) {
  gkFloat n[3], pq[3], pr[3];

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pq[i] = p[i] - q[i];
  }
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pr[i] = p[i] - r[i];
  }

  crossProduct(pq, pr, n);
  const gkFloat tmp = dotProduct(n, p) / dotProduct(n, n);

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    v[i] = n[i] * tmp;
  }
}

__device__ inline static int hff1(const gkFloat* p, const gkFloat* q) {
  gkFloat tmp = 0;

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    tmp += (p[i] * p[i] - p[i] * q[i]);
  }

  if (tmp > 0) {
    return 1;  // keep q
  }
  return 0;
}

__device__ inline static int hff2(const gkFloat* p, const gkFloat* q,
  const gkFloat* r) {
  gkFloat ntmp[3];
  gkFloat n[3], pq[3], pr[3];

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pq[i] = q[i] - p[i];
  }
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pr[i] = r[i] - p[i];
  }

  crossProduct(pq, pr, ntmp);
  crossProduct(pq, ntmp, n);

  return dotProduct(p, n) < 0;  // Discard r if true
}

__device__ inline static int hff3(const gkFloat* p, const gkFloat* q,
  const gkFloat* r) {
  gkFloat n[3], pq[3], pr[3];

  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pq[i] = q[i] - p[i];
  }
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    pr[i] = r[i] - p[i];
  }

  crossProduct(pq, pr, n);
  return dotProduct(p, n) <= 0;  // discard s if true
}

__device__ inline static void S1D(gkSimplex* s, gkFloat* v) {
  const gkFloat* s1p = s->vrtx[1];
  const gkFloat* s2p = s->vrtx[0];

  if (hff1(s1p, s2p)) {
    projectOnLine(s1p, s2p, v);  // Update v, no need to update s
    return;                      // Return V{1,2}
  }
  else {
    S1Dregion1();  // Update v and s
    return;        // Return V{1}
  }
}

__device__ inline static void S2D(gkSimplex* s, gkFloat* v) {
  const gkFloat* s1p = s->vrtx[2];
  const gkFloat* s2p = s->vrtx[1];
  const gkFloat* s3p = s->vrtx[0];
  const int hff1f_s12 = hff1(s1p, s2p);
  const int hff1f_s13 = hff1(s1p, s3p);

  if (hff1f_s12) {
    const int hff2f_23 = !hff2(s1p, s2p, s3p);
    if (hff2f_23) {
      if (hff1f_s13) {
        const int hff2f_32 = !hff2(s1p, s3p, s2p);
        if (hff2f_32) {
          projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
          return;                            // Return V{1,2,3}
        }
        else {
          projectOnLine(s1p, s3p, v);  // Update v
          S2Dregion13();               // Update s
          return;                      // Return V{1,3}
        }
      }
      else {
        projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
        return;                            // Return V{1,2,3}
      }
    }
    else {
      projectOnLine(s1p, s2p, v);  // Update v
      S2Dregion12();               // Update s
      return;                      // Return V{1,2}
    }
  }
  else if (hff1f_s13) {
    const int hff2f_32 = !hff2(s1p, s3p, s2p);
    if (hff2f_32) {
      projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update v
      return;                            // Return V{1,2,3}
    }
    else {
      projectOnLine(s1p, s3p, v);  // Update v
      S2Dregion13();               // Update s
      return;                      // Return V{1,3}
    }
  }
  else {
    S2Dregion1();  // Update s and v
    return;        // Return V{1}
  }
}

// Warp-parallel variants of the distance sub-algorithm pieces.
// These use a small number of lanes in the half-warp to evaluate
// independent predicates (hff1 / hff3) in parallel

__device__ inline static void S1D_warp_parallel(
  gkSimplex* s,
  gkFloat* v,
  int half_lane_idx) {
  // There is only a single hff1 test in S1D, so we can just execute previous alg on lead lane
  if (half_lane_idx == 0) {
    S1D(s, v);
  }
}

__device__ inline static void S2D_warp_parallel(
  gkSimplex* s,
  gkFloat* v,
  int half_lane_idx,
  unsigned int half_warp_mask,
  int half_warp_base_thread_idx) {

  const gkFloat* s1p = s->vrtx[2];
  const gkFloat* s2p = s->vrtx[1];
  const gkFloat* s3p = s->vrtx[0];

  // do hff1 and hff2 in parallel using two lanes in the half-warp.
  int hff1f[2] = { 0, 0 };
  const gkFloat* hff1_pts[2] = { s2p, s3p };

  if (half_lane_idx < 2) {
    hff1f[half_lane_idx] = hff1(s1p, hff1_pts[half_lane_idx]);
  }

  int hff2f[2] = { 0, 0 };
  const gkFloat* hff2_q[2] = { s2p, s3p };
  const gkFloat* hff2_r[2] = { s3p, s2p };

  if (half_lane_idx < 2) {
    hff2f[half_lane_idx] = !hff2(s1p, hff2_q[half_lane_idx], hff2_r[half_lane_idx]);
  }

  // Broadcast results from the lanes that computed them.
  const int lane0 = half_warp_base_thread_idx + 0;
  const int lane1 = half_warp_base_thread_idx + 1;
  int hff1f_s12 = __shfl_sync(half_warp_mask, hff1f[0], lane0);
  int hff1f_s13 = __shfl_sync(half_warp_mask, hff1f[1], lane1);
  int hff2f_23 = __shfl_sync(half_warp_mask, hff2f[0], lane0);
  int hff2f_32 = __shfl_sync(half_warp_mask, hff2f[1], lane1);

  // Only the lead lane runs if-else logic to determine region
  if (half_lane_idx != 0) {
    return;
  }

  if (hff1f_s12) {
    if (hff2f_23) {
      if (hff1f_s13) {
        if (hff2f_32) {
          projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
          return;                            // Return V{1,2,3}
        }
        else {
          projectOnLine(s1p, s3p, v);  // Update v
          S2Dregion13();               // Update s
          return;                      // Return V{1,3}
        }
      }
      else {
        projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
        return;                            // Return V{1,2,3}
      }
    }
    else {
      projectOnLine(s1p, s2p, v);  // Update v
      S2Dregion12();               // Update s
      return;                      // Return V{1,2}
    }
  }
  else if (hff1f_s13) {
    const int hff2f_32 = !hff2(s1p, s3p, s2p);
    if (hff2f_32) {
      projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update v
      return;                            // Return V{1,2,3}
    }
    else {
      projectOnLine(s1p, s3p, v);  // Update v
      S2Dregion13();               // Update s
      return;                      // Return V{1,3}
    }
  }
  else {
    S2Dregion1();  // Update s and v
    return;        // Return V{1}
  }
}

__device__ inline static void S3D_warp_parallel(
  gkSimplex* s,
  gkFloat* v,
  int half_lane_idx,
  unsigned int half_warp_mask,
  int half_warp_base_thread_idx) {

  gkFloat s1[3], s2[3], s3[3], s4[3], s1s2[3], s1s3[3], s1s4[3];
  gkFloat si[3], sj[3], sk[3];
  int s1_idx[2], s2_idx[2], s3_idx[2];
  int si_idx[2], sj_idx[2], sk_idx[2];
  int testLineThree, testLineFour, testPlaneTwo, testPlaneThree, testPlaneFour,
    dotTotal;
  int i, j, k, t;

  getvrtxidx(s1, s1_idx, 3);
  getvrtxidx(s2, s2_idx, 2);
  getvrtxidx(s3, s3_idx, 1);
  getvrtx(s4, 0);
  calculateEdgeVector(s1s2, s2);
  calculateEdgeVector(s1s3, s3);
  calculateEdgeVector(s1s4, s4);

  // Parallel evaluation of the thff1 and hff3 predicaes
  int hff1_s[3] = { 0, 0, 0 };
  int hff3_s[3] = { 0, 0, 0 };

  const gkFloat* hff1_pts[3] = { s2, s3, s4 };
  const gkFloat* hff3_q[3] = { s3, s4, s2 };  // second argument
  const gkFloat* hff3_r[3] = { s4, s2, s3 };  // third argument

  if (half_lane_idx < 3) {
    hff1_s[half_lane_idx] = hff1(s1, hff1_pts[half_lane_idx]);
    hff3_s[half_lane_idx] = hff3(s1, hff3_q[half_lane_idx], hff3_r[half_lane_idx]);
  }

  int hff1_s12 = 0, hff1_s13 = 0, hff1_s14 = 0;
  int hff3_134 = 0, hff3_142 = 0, hff3_123 = 0;

  const int lane0 = half_warp_base_thread_idx + 0;
  const int lane1 = half_warp_base_thread_idx + 1;
  const int lane2 = half_warp_base_thread_idx + 2;

  // Gather each lane's result on all threads.
  hff1_s12 = __shfl_sync(half_warp_mask, hff1_s[0], lane0);
  hff1_s13 = __shfl_sync(half_warp_mask, hff1_s[1], lane1);
  hff1_s14 = __shfl_sync(half_warp_mask, hff1_s[2], lane2);
  hff3_134 = __shfl_sync(half_warp_mask, hff3_s[0], lane0);
  hff3_142 = __shfl_sync(half_warp_mask, hff3_s[1], lane1);
  hff3_123 = __shfl_sync(half_warp_mask, hff3_s[2], lane2);

  // Only the lead lane executes the rest of if-else logic to determine region
  if (half_lane_idx != 0) {
    return;
  }

  int hff1_tests[3];
  hff1_tests[2] = hff1_s12;
  hff1_tests[1] = hff1_s13;
  hff1_tests[0] = hff1_s14;
  testLineThree = hff1_s13;
  testLineFour = hff1_s14;

  dotTotal = hff1_s12 + testLineThree + testLineFour;
  if (dotTotal == 0) { /* case 0.0 -------------------------------------- */
    S3Dregion1();
    return;
  }

  const gkFloat det134 = determinant(s1s3, s1s4, s1s2);
  const int sss = (det134 <= 0);

  testPlaneTwo = hff3_134 - sss;
  testPlaneTwo = testPlaneTwo * testPlaneTwo;
  testPlaneThree = hff3_142 - sss;
  testPlaneThree = testPlaneThree * testPlaneThree;
  testPlaneFour = hff3_123 - sss;
  testPlaneFour = testPlaneFour * testPlaneFour;

  switch (testPlaneTwo + testPlaneThree + testPlaneFour) {
  case 3:
    S3Dregion1234();
    break;

  case 2:
    // Only one facing the oring
    // 1,i,j, are the indices of the points on the triangle and remove k from
    // simplex
    s->nvrtx = 3;
    if (!testPlaneTwo) {  // k = 2;   removes s2
      #pragma unroll
      for (i = 0; i < 3; i++) {
        s->vrtx[2][i] = s->vrtx[3][i];
      }
      #pragma unroll
      for (i = 0; i < 2; i++) {
        s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
      }
    }
    else if (!testPlaneThree) {  // k = 1; // removes s3
      #pragma unroll
      for (i = 0; i < 3; i++) {
        s->vrtx[1][i] = s2[i];
        s->vrtx[2][i] = s->vrtx[3][i];
      }
      #pragma unroll
      for (i = 0; i < 2; i++) {
        s->vrtx_idx[1][i] = s2_idx[i];
        s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
      }
    }
    else if (!testPlaneFour) {  // k = 0; // removes s4  and no need to
      // reorder
      #pragma unroll
      for (i = 0; i < 3; i++) {
        s->vrtx[0][i] = s3[i];
        s->vrtx[1][i] = s2[i];
        s->vrtx[2][i] = s->vrtx[3][i];
      }
      #pragma unroll
      for (i = 0; i < 2; i++) {
        s->vrtx_idx[0][i] = s3_idx[i];
        s->vrtx_idx[1][i] = s2_idx[i];
        s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
      }
    }
    // Call S2D
    S2D(s, v);
    break;
  case 1:
    // Two triangles face the origins:
    //    The only positive hff3 is for triangle 1,i,j, therefore k must be in
    //    the solution as it supports the the point of minimum norm.

    // 1,i,j, are the indices of the points on the triangle and remove k from
    // simplex
    s->nvrtx = 3;
    if (testPlaneTwo) {
      k = 2;  // s2
      i = 1;
      j = 0;
    }
    else if (testPlaneThree) {
      k = 1;  // s3
      i = 0;
      j = 2;
    }
    else {
      k = 0;  // s4
      i = 2;
      j = 1;
    }

    getvrtxidx(si, si_idx, i);
    getvrtxidx(sj, sj_idx, j);
    getvrtxidx(sk, sk_idx, k);

    if (dotTotal == 1) {
      if (hff1_tests[k]) {
        if (!hff2(s1, sk, si)) {
          select_1ik();
          projectOnPlane(s1, si, sk, v);
        }
        else if (!hff2(s1, sk, sj)) {
          select_1jk();
          projectOnPlane(s1, sj, sk, v);
        }
        else {
          select_1k();  // select region 1i
          projectOnLine(s1, sk, v);
        }
      }
      else if (hff1_tests[i]) {
        if (!hff2(s1, si, sk)) {
          select_1ik();
          projectOnPlane(s1, si, sk, v);
        }
        else {
          select_1i();  // select region 1i
          projectOnLine(s1, si, v);
        }
      }
      else {
        if (!hff2(s1, sj, sk)) {
          select_1jk();
          projectOnPlane(s1, sj, sk, v);
        }
        else {
          select_1j();  // select region 1i
          projectOnLine(s1, sj, v);
        }
      }
    }
    else if (dotTotal == 2) {
      // Two edges have positive hff1, meaning that for two edges the origin's
      // project fall on the segement.
      //  Certainly the edge 1,k supports the the point of minimum norm, and
      //  so hff1_1k is positive

      if (hff1_tests[i]) {
        if (!hff2(s1, sk, si)) {
          if (!hff2(s1, si, sk)) {
            select_1ik();  // select region 1ik
            projectOnPlane(s1, si, sk, v);
          }
          else {
            select_1k();  // select region 1k
            projectOnLine(s1, sk, v);
          }
        }
        else {
          if (!hff2(s1, sk, sj)) {
            select_1jk();  // select region 1jk
            projectOnPlane(s1, sj, sk, v);
          }
          else {
            select_1k();  // select region 1k
            projectOnLine(s1, sk, v);
          }
        }
      }
      else if (hff1_tests[j]) {  //  there is no other choice
        if (!hff2(s1, sk, sj)) {
          if (!hff2(s1, sj, sk)) {
            select_1jk();  // select region 1jk
            projectOnPlane(s1, sj, sk, v);
          }
          else {
            select_1j();  // select region 1j
            projectOnLine(s1, sj, v);
          }
        }
        else {
          if (!hff2(s1, sk, si)) {
            select_1ik();  // select region 1ik
            projectOnPlane(s1, si, sk, v);
          }
          else {
            select_1k();  // select region 1k
            projectOnLine(s1, sk, v);
          }
        }
      }
      else {
        // ERROR;
      }

    }
    else if (dotTotal == 3) {
      // MM : ALL THIS HYPHOTESIS IS FALSE
      // sk is s.t. hff3 for sk < 0. So, sk must support the origin because
      // there are 2 triangles facing the origin.

      int hff2_ik = hff2(s1, si, sk);
      int hff2_jk = hff2(s1, sj, sk);
      int hff2_ki = hff2(s1, sk, si);
      int hff2_kj = hff2(s1, sk, sj);

      if (hff2_ki == 0 && hff2_kj == 0) {
        //   mexPrintf("\n\n UNEXPECTED VALUES!!! \n\n");
      }
      if (hff2_ki == 1 && hff2_kj == 1) {
        select_1k();
        projectOnLine(s1, sk, v);
      }
      else if (hff2_ki) {
        // discard i
        if (hff2_jk) {
          // discard k
          select_1j();
          projectOnLine(s1, sj, v);
        }
        else {
          select_1jk();
          projectOnPlane(s1, sk, sj, v);
        }
      }
      else {
        // discard j
        if (hff2_ik) {
          // discard k
          select_1i();
          projectOnLine(s1, si, v);
        }
        else {
          select_1ik();
          projectOnPlane(s1, sk, si, v);
        }
      }
    }
    break;

  case 0:
    // The origin is outside all 3 triangles
    if (dotTotal == 1) {
      // Here si is set such that hff(s1,si) > 0
      if (testLineThree) {
        k = 2;
        i = 1;  // s3
        j = 0;
      }
      else if (testLineFour) {
        k = 1;  // s3
        i = 0;
        j = 2;
      }
      else {
        k = 0;
        i = 2;  // s2
        j = 1;
      }
      getvrtxidx(si, si_idx, i);
      getvrtxidx(sj, sj_idx, j);
      getvrtxidx(sk, sk_idx, k);

      if (!hff2(s1, si, sj)) {
        select_1ij();
        projectOnPlane(s1, si, sj, v);
      }
      else if (!hff2(s1, si, sk)) {
        select_1ik();
        projectOnPlane(s1, si, sk, v);
      }
      else {
        select_1i();
        projectOnLine(s1, si, v);
      }
    }
    else if (dotTotal == 2) {
      // Here si is set such that hff(s1,si) < 0
      s->nvrtx = 3;
      if (!testLineThree) {
        k = 2;
        i = 1;  // s3
        j = 0;
      }
      else if (!testLineFour) {
        k = 1;
        i = 0;  // s4
        j = 2;
      }
      else {
        k = 0;
        i = 2;  // s2
        j = 1;
      }
      getvrtxidx(si, si_idx, i);
      getvrtxidx(sj, sj_idx, j);
      getvrtxidx(sk, sk_idx, k);

      if (!hff2(s1, sj, sk)) {
        if (!hff2(s1, sk, sj)) {
          select_1jk();  // select region 1jk
          projectOnPlane(s1, sj, sk, v);
        }
        else if (!hff2(s1, sk, si)) {
          select_1ik();
          projectOnPlane(s1, sk, si, v);
        }
        else {
          select_1k();
          projectOnLine(s1, sk, v);
        }
      }
      else if (!hff2(s1, sj, si)) {
        select_1ij();
        projectOnPlane(s1, si, sj, v);
      }
      else {
        select_1j();
        projectOnLine(s1, sj, v);
      }
    }
    break;
  default: {
    //   mexPrintf("\nERROR:\tunhandled");
  }
  }
}

// Warp-parallel wrapper for the distance sub-algorithm.
// Uses the warp-parallel S1D/S2D/S3D which evaluate the
// independent predicates (hff1/hff3) on a few lanes in parallel
__device__ inline static void subalgorithm_warp_parallel(
  gkSimplex* s,
  gkFloat* v,
  int half_lane_idx,
  unsigned int half_warp_mask,
  int half_warp_base_thread_idx) {

  __syncwarp(half_warp_mask);

  switch (s->nvrtx) {
  case 4:
    S3D_warp_parallel(s, v, half_lane_idx, half_warp_mask,
      half_warp_base_thread_idx);
    break;
  case 3:
    S2D_warp_parallel(s, v, half_lane_idx, half_warp_mask,
      half_warp_base_thread_idx);
    break;
  case 2:
    S1D_warp_parallel(s, v, half_lane_idx);
    break;
  default: {
    //   mexPrintf("\nERROR:\t invalid simplex\n");
  }
  }

  // Ensure subalgorithm has finished before broadcasting results.
  __syncwarp(half_warp_mask);

  // Broadcast updated search direction.
  v[0] = __shfl_sync(half_warp_mask, v[0], half_warp_base_thread_idx);
  v[1] = __shfl_sync(half_warp_mask, v[1], half_warp_base_thread_idx);
  v[2] = __shfl_sync(half_warp_mask, v[2], half_warp_base_thread_idx);

  // Broadcast simplex data
  s->nvrtx =
    __shfl_sync(half_warp_mask, s->nvrtx, half_warp_base_thread_idx);

  #pragma unroll 4
  for (int vtx = 0; vtx < 4; ++vtx) {
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      s->vrtx[vtx][t] =
        __shfl_sync(half_warp_mask, s->vrtx[vtx][t], half_warp_base_thread_idx);
    }
    s->vrtx_idx[vtx][0] =
      __shfl_sync(half_warp_mask, s->vrtx_idx[vtx][0], half_warp_base_thread_idx);
    s->vrtx_idx[vtx][1] =
      __shfl_sync(half_warp_mask, s->vrtx_idx[vtx][1], half_warp_base_thread_idx);
  }
}

__device__ inline static void W0D(const gkPolytope* bd1, const gkPolytope* bd2,
  gkSimplex* smp) {
  const int idx00 = smp->vrtx_idx[0][0];
  const int idx01 = smp->vrtx_idx[0][1];
  #pragma unroll
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = getCoord(bd1, idx00, t);
    smp->witnesses[1][t] = getCoord(bd2, idx01, t);
  }
}

__device__ inline static void W1D(const gkPolytope* bd1, const gkPolytope* bd2,
  gkSimplex* smp) {
  gkFloat pq[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];

  #pragma unroll
  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    po[t] = -p[t];
  }

  // Compute barycentric coordinates via matrix inversion
  // (in the linear case the matrix is 1x1 thus simplified)
  const gkFloat det = dotProduct(pq, pq);
  if (det == 0.0) {
    // Degenerate case
    W0D(bd1, bd2, smp);
  }

  const gkFloat a1 = dotProduct(pq, po) / det;
  const gkFloat a0 = 1.0 - a1;

  // Compute witness points
  const int idx00 = smp->vrtx_idx[0][0];
  const int idx01 = smp->vrtx_idx[0][1];
  const int idx10 = smp->vrtx_idx[1][0];
  const int idx11 = smp->vrtx_idx[1][1];
  #pragma unroll
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = getCoord(bd1, idx00, t) * a0 + getCoord(bd1, idx10, t) * a1;
    smp->witnesses[1][t] = getCoord(bd2, idx01, t) * a0 + getCoord(bd2, idx11, t) * a1;
  }
}

__device__ inline static void W2D(const gkPolytope* bd1, const gkPolytope* bd2,
  gkSimplex* smp) {
  gkFloat pq[3], pr[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];
  const gkFloat* r = smp->vrtx[2];

  #pragma unroll
  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    pr[t] = r[t] - p[t];
    po[t] = -p[t];
  }

  /**
   *  Compute barycentric coordinates via matrix inversion
   *  Given the points $P$, $Q$, and $R$ forming a triangle
   *  we want to find the barycentric coordinates of the origin
   *  projected onto the triangle. We can do this
   *  by inverting $\mathbf{T}$ in the linear equation below:
   *
   *  \begin{align*}
   *  \mathbf{T}
   *  \begin{bmatrix}
   *  \lambda_q \\
   *  \lambda_r
   *  \end{bmatrix} &= \begin{bmatrix}
   *  \overrightarrow{PQ}\cdot\overrightarrow{PO} \\
   *  \overrightarrow{PR}\cdot\overrightarrow{PO}
   *  \end{bmatrix} \\
   *  \lambda_p &= 1 - \lambda_q - \lambda_r \\
   *  \mathbf{T} &= \begin{bmatrix}
   *  \overrightarrow{PQ}\cdot\overrightarrow{PQ} &
   * \overrightarrow{PR}\cdot\overrightarrow{PQ} \\
   *  \overrightarrow{PR}\cdot\overrightarrow{PQ} &
   * \overrightarrow{PR}\cdot\overrightarrow{PR}
   *  \end{bmatrix}
   *  \end{align*}
   */
  const gkFloat T00 = dotProduct(pq, pq);
  const gkFloat T01 = dotProduct(pq, pr);
  const gkFloat T11 = dotProduct(pr, pr);
  const gkFloat det = T00 * T11 - T01 * T01;
  if (det == 0.0) {
    // Degenerate case
    W1D(bd1, bd2, smp);
  }

  const gkFloat b0 = dotProduct(pq, po);
  const gkFloat b1 = dotProduct(pr, po);
  const gkFloat I00 = T11 / det;
  const gkFloat I01 = -T01 / det;
  const gkFloat I11 = T00 / det;
  const gkFloat a1 = I00 * b0 + I01 * b1;
  const gkFloat a2 = I01 * b0 + I11 * b1;
  const gkFloat a0 = 1.0 - a1 - a2;

  // check if the origin is very close to one of the edges of the
  // simplex. In this case, a 1D projection will be more accurate.
  if (a0 < gkEpsilon) {
    smp->nvrtx = 2;
    smp->vrtx[0][0] = smp->vrtx[2][0];
    smp->vrtx[0][1] = smp->vrtx[2][1];
    smp->vrtx[0][2] = smp->vrtx[2][2];
    smp->vrtx_idx[0][0] = smp->vrtx_idx[2][0];
    smp->vrtx_idx[0][1] = smp->vrtx_idx[2][1];
    W1D(bd1, bd2, smp);
  }
  else if (a1 < gkEpsilon) {
    smp->nvrtx = 2;
    smp->vrtx[1][0] = smp->vrtx[2][0];
    smp->vrtx[1][1] = smp->vrtx[2][1];
    smp->vrtx[1][2] = smp->vrtx[2][2];
    smp->vrtx_idx[1][0] = smp->vrtx_idx[2][0];
    smp->vrtx_idx[1][1] = smp->vrtx_idx[2][1];
    W1D(bd1, bd2, smp);
  }
  else if (a2 < gkEpsilon) {
    smp->nvrtx = 2;
    W1D(bd1, bd2, smp);
  }

  // Compute witness points
  // This is done by blending the source points using
  // the barycentric coordinates
  const int idx00 = smp->vrtx_idx[0][0];
  const int idx01 = smp->vrtx_idx[0][1];
  const int idx10 = smp->vrtx_idx[1][0];
  const int idx11 = smp->vrtx_idx[1][1];
  const int idx20 = smp->vrtx_idx[2][0];
  const int idx21 = smp->vrtx_idx[2][1];
  #pragma unroll
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = getCoord(bd1, idx00, t) * a0 + getCoord(bd1, idx10, t) * a1 + getCoord(bd1, idx20, t) * a2;
    smp->witnesses[1][t] = getCoord(bd2, idx01, t) * a0 + getCoord(bd2, idx11, t) * a1 + getCoord(bd2, idx21, t) * a2;
  }
}

__device__ inline static void W3D(const gkPolytope* bd1, const gkPolytope* bd2,
  gkSimplex* smp) {
  gkFloat pq[3], pr[3], ps[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];
  const gkFloat* r = smp->vrtx[2];
  const gkFloat* s = smp->vrtx[3];

  #pragma unroll
  for (int t = 0; t < 3; t++) {
    pq[t] = q[t] - p[t];
    pr[t] = r[t] - p[t];
    ps[t] = s[t] - p[t];
    po[t] = -p[t];
  }

  /**
   *  Compute barycentric coordinates via matrix inversion
   *  Given the points $P$, $Q$, and $R$, and $S$ forming a
   *  tetrahedron we want to find the barycentric coordinates of
   *  the origin. We can do this by inverting $\mathbf{T}$ in the
   *  linear equation below:
   *
   *  \begin{align*}
   *  \mathbf{T}
   *  \begin{bmatrix}
   *  \lambda_q \\
   *  \lambda_r \\
   *  \lambda_s
   *  \end{bmatrix} &= \begin{bmatrix}
   *  \overrightarrow{PQ}\cdot\overrightarrow{PO} \\
   *  \overrightarrow{PR}\cdot\overrightarrow{PO} \\
   *  \overrightarrow{PS}\cdot\overrightarrow{PO}
   *  \end{bmatrix} \\
   *  \lambda_p &= 1 - \lambda_q - \lambda_r - \lambda_s \\
   *  \mathbf{T} &= \begin{bmatrix}
   *  \overrightarrow{PQ}\cdot\overrightarrow{PQ} &
   * \overrightarrow{PQ}\cdot\overrightarrow{PR} &
   * \overrightarrow{PQ}\cdot\overrightarrow{PS}\\
   *  \overrightarrow{PR}\cdot\overrightarrow{PQ} & \overrightarrow{PR} \cdot
   * \overrightarrow{PR} & \overrightarrow{PR}\cdot\overrightarrow{PS} \\
   *  \overrightarrow{PS}\cdot\overrightarrow{PQ} &
   * \overrightarrow{PS}\cdot\overrightarrow{PR} &
   * \overrightarrow{PS}\cdot\overrightarrow{PS}
   *  \end{bmatrix}
   *  \end{align*}
   */

  const gkFloat T00 = dotProduct(pq, pq);
  const gkFloat T01 = dotProduct(pq, pr);
  const gkFloat T02 = dotProduct(pq, ps);
  const gkFloat T11 = dotProduct(pr, pr);
  const gkFloat T12 = dotProduct(pr, ps);
  const gkFloat T22 = dotProduct(ps, ps);
  const gkFloat det00 = T11 * T22 - T12 * T12;
  const gkFloat det01 = T01 * T22 - T02 * T12;
  const gkFloat det02 = T01 * T12 - T02 * T11;
  const gkFloat det = T00 * det00 - T01 * det01 + T02 * det02;
  if (det == 0.0) {
    // Degenerate case
    W2D(bd1, bd2, smp);
  }

  const gkFloat b0 = dotProduct(pq, po);
  const gkFloat b1 = dotProduct(pr, po);
  const gkFloat b2 = dotProduct(ps, po);

  // inverse matrix
  // (the matrix is symmetric, so we can use the cofactor matrix)
  const gkFloat det11 = T00 * T22 - T02 * T02;
  const gkFloat det12 = T00 * T12 - T01 * T02;
  const gkFloat det22 = T00 * T11 - T01 * T01;
  const gkFloat I00 = det00 / det;
  const gkFloat I01 = -det01 / det;
  const gkFloat I02 = det02 / det;
  const gkFloat I11 = det11 / det;
  const gkFloat I12 = -det12 / det;
  const gkFloat I22 = det22 / det;

  const gkFloat a1 = I00 * b0 + I01 * b1 + I02 * b2;
  const gkFloat a2 = I01 * b0 + I11 * b1 + I12 * b2;
  const gkFloat a3 = I02 * b0 + I12 * b1 + I22 * b2;
  const gkFloat a0 = 1.0 - a1 - a2 - a3;

  // check if the origin is very close to one of the faces of the
  // simplex. In this case, a 2D projection will be more accurate.
  if (a0 < gkEpsilon) {
    smp->nvrtx = 3;
    smp->vrtx[0][0] = smp->vrtx[3][0];
    smp->vrtx[0][1] = smp->vrtx[3][1];
    smp->vrtx[0][2] = smp->vrtx[3][2];
    smp->vrtx_idx[0][0] = smp->vrtx_idx[3][0];
    smp->vrtx_idx[0][1] = smp->vrtx_idx[3][1];
    W2D(bd1, bd2, smp);
  }
  else if (a1 < gkEpsilon) {
    smp->nvrtx = 3;
    smp->vrtx[1][0] = smp->vrtx[3][0];
    smp->vrtx[1][1] = smp->vrtx[3][1];
    smp->vrtx[1][2] = smp->vrtx[3][2];
    smp->vrtx_idx[1][0] = smp->vrtx_idx[3][0];
    smp->vrtx_idx[1][1] = smp->vrtx_idx[3][1];
    W2D(bd1, bd2, smp);
  }
  else if (a2 < gkEpsilon) {
    smp->nvrtx = 3;
    smp->vrtx[2][0] = smp->vrtx[3][0];
    smp->vrtx[2][1] = smp->vrtx[3][1];
    smp->vrtx[2][2] = smp->vrtx[3][2];
    smp->vrtx_idx[2][0] = smp->vrtx_idx[3][0];
    smp->vrtx_idx[2][1] = smp->vrtx_idx[3][1];
    W2D(bd1, bd2, smp);
  }
  else if (a3 < gkEpsilon) {
    smp->nvrtx = 3;
    W2D(bd1, bd2, smp);
  }

  // Compute witness points
  // This is done by blending the original points using
  // the barycentric coordinates
  const int idx00 = smp->vrtx_idx[0][0];
  const int idx01 = smp->vrtx_idx[0][1];
  const int idx10 = smp->vrtx_idx[1][0];
  const int idx11 = smp->vrtx_idx[1][1];
  const int idx20 = smp->vrtx_idx[2][0];
  const int idx21 = smp->vrtx_idx[2][1];
  const int idx30 = smp->vrtx_idx[3][0];
  const int idx31 = smp->vrtx_idx[3][1];
  #pragma unroll
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] =
      getCoord(bd1, idx00, t) * a0 + getCoord(bd1, idx10, t) * a1 + getCoord(bd1, idx20, t) * a2 + getCoord(bd1, idx30, t) * a3;
    smp->witnesses[1][t] =
      getCoord(bd2, idx01, t) * a0 + getCoord(bd2, idx11, t) * a1 + getCoord(bd2, idx21, t) * a2 + getCoord(bd2, idx31, t) * a3;
  }
}

__device__ inline static void compute_witnesses(const gkPolytope* bd1,
  const gkPolytope* bd2, gkSimplex* smp) {
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
  {
    //   mexPrintf("\nERROR:\t invalid simplex\n");
  }
  }
}

//*******************************************************************************************
// Warp Parallel GJK Implementation
//*******************************************************************************************

// Half a warp per polytope-polytope collision
// Have first thread in group lead the GJK iteration, others for parallel support function calls

// Parallel version of support function using all threads in a half-warp
__device__ inline static void support_parallel(gkPolytope* body,
  const gkFloat* v, int half_lane_idx, unsigned int half_warp_mask, int half_warp_base_thread_idx) {

  // Each thread searches a subset of the total points in the body so they can compute in parallel
  const int points_per_thread = (body->numpoints + THREADS_PER_COMPUTATION - 1) / THREADS_PER_COMPUTATION;
  const int start_idx = half_lane_idx * points_per_thread;
  const int end_idx = (start_idx + points_per_thread < body->numpoints) ?
    (start_idx + points_per_thread) : body->numpoints;

  // Initialize each thead with its current best point
  gkFloat local_maxs = dotProduct(body->s, v);
  int local_better = -1;
  gkFloat vrt[3];

  // Each thread searches its assigned range of points
  for (int i = start_idx; i < end_idx; ++i) {
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
      vrt[j] = getCoord(body, i, j);
    }
    gkFloat s = dotProduct(vrt, v);
    if (s > local_maxs) {
      local_maxs = s;
      local_better = i;
    }
  }

  // Parallel reduction to find global maximum across all threads in the half-warp
  gkFloat global_maxs = local_maxs;
  int global_better = local_better;

  // Reduction tree compare with threads at increasing offsets (power of 2)
  #pragma unroll
  for (int offset = THREADS_PER_COMPUTATION / 2; offset > 0; offset /= 2) {
    // Get maxs and better index from thread at offset distance
    gkFloat other_maxs = __shfl_down_sync(half_warp_mask, global_maxs, offset);
    int other_better = __shfl_down_sync(half_warp_mask, global_better, offset);

    // Update values if other thread found a better point
    if (other_maxs > global_maxs ||
      (other_maxs == global_maxs && other_better != -1 && global_better == -1)) {
      global_maxs = other_maxs;
      global_better = other_better;
    }
  }

  // Broadcast the best result to all threads (from thread 0 of our half-warp) to make sure all threads agree
  global_maxs = __shfl_sync(half_warp_mask, global_maxs, half_warp_base_thread_idx);
  global_better = __shfl_sync(half_warp_mask, global_better, half_warp_base_thread_idx);

  // All threads update their local copy (all have identical global_better)
  if (global_better != -1) {
    body->s[0] = getCoord(body, global_better, 0);
    body->s[1] = getCoord(body, global_better, 1);
    body->s[2] = getCoord(body, global_better, 2);
    body->s_idx = global_better;
  }
}

__global__ void compute_minimum_distance_kernel(
  const gkPolytope* polytopes1,
  const gkPolytope* polytopes2,
  gkSimplex* simplices,
  gkFloat* distances,
  const int n) {

  // Calculate which collision this half-warp handles
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int half_warp_idx = index / THREADS_PER_COMPUTATION;

  // Get thread indexwithin the warp (0-31) and within the half-warp (0-15)
  int warp_lane_idx = threadIdx.x % 32;  // Lane ID within warp
  int half_warp_in_warp = warp_lane_idx / THREADS_PER_COMPUTATION;  // Which half-warp iwe are in (0 or 1)
  int half_lane_idx = warp_lane_idx % THREADS_PER_COMPUTATION;  // thread idx within half-warp (0-15)

  // Calculate the lead thread index for our half-warp within the warp (0 or 16)
  // This is the thread that will coordinate the half warp and complete and broadcast computations we only need one thread to do
  int half_warp_base_thread_idx = half_warp_in_warp * THREADS_PER_COMPUTATION;

  // Create mask for our half-warp (0xFFFF for first half, 0xFFFF0000 for second half) to determine which threads to sync with
  unsigned int half_warp_mask = (half_warp_in_warp == 0) ? 0xFFFF : 0xFFFF0000;

  if (half_warp_idx >= n) {
    return;
  }

  // Each half-warp (16 threads) handles a single GJK computation
  // all threads in the half-warp work on the same collision

  // Copy to local memory for fast access during iteration
  gkPolytope bd1 = polytopes1[half_warp_idx];
  gkPolytope bd2 = polytopes2[half_warp_idx];
  gkSimplex s = simplices[half_warp_idx];

  unsigned int k = 0;                /**< Iteration counter                 */
  const int mk = 25;                 /**< Maximum number of GJK iterations  */
  const gkFloat eps_rel = eps_rel22; /**< Tolerance on relative             */
  const gkFloat eps_tot = eps_tot22; /**< Tolerance on absolute distance    */

  const gkFloat eps_rel2 = eps_rel * eps_rel;
  unsigned int i;
  gkFloat w[3];
  int w_idx[2];
  gkFloat v[3];
  gkFloat vminus[3];
  gkFloat norm2Wmax = 0;

  // Synchronize all threads in the half-warp before starting
  __syncwarp(half_warp_mask);

  // Initialize search direction
  v[0] = bd1.coord[0] - bd2.coord[0];
  v[1] = bd1.coord[1] - bd2.coord[1];
  v[2] = bd1.coord[2] - bd2.coord[2];

  // Initialize simplex - all threads compute identical values, no broadcast needed
  s.nvrtx = 1;
  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    s.vrtx[0][t] = v[t];
  }
  s.vrtx_idx[0][0] = 0;
  s.vrtx_idx[0][1] = 0;

  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    bd1.s[t] = bd1.coord[t];
  }
  bd1.s_idx = 0;

  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    bd2.s[t] = bd2.coord[t];
  }
  bd2.s_idx = 0;

  /* Begin GJK iteration */
  // Thread 0 controls the loop but all threads participate in parallel operations
  bool continue_iteration = true;

  while (continue_iteration) {
    // Broadcast k and s.nvrtx to all threads for loop condition check
    k = __shfl_sync(half_warp_mask, k, half_warp_base_thread_idx);
    s.nvrtx = __shfl_sync(half_warp_mask, s.nvrtx, half_warp_base_thread_idx);

    // Check loop conditions
    if (s.nvrtx == 4 || k == mk) {
      continue_iteration = false;
      break;
    }

    k++;

    /* Update negative search direction - all threads compute*/
    // Note: v is already the same on all threads from previous iteration
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      vminus[t] = -v[t];
    }

    /* Support function - parallelized using all threads */
    // All threads participate in finding support points for speedup but only thread 0 updates the body
    support_parallel(&bd1, vminus, half_lane_idx, half_warp_mask, half_warp_base_thread_idx);
    support_parallel(&bd2, v, half_lane_idx, half_warp_mask, half_warp_base_thread_idx);

    // all threads compute w for witness point computation
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      w[t] = bd1.s[t] - bd2.s[t];
    }
    w_idx[0] = bd1.s_idx;
    w_idx[1] = bd2.s_idx;

    /* Test first exit condition (new point already in simplex/can't move
     * further) */
    gkFloat norm2_v = norm2(v);
    gkFloat exeedtol_rel = (norm2_v - dotProduct(v, w));

    //check exit conditions
    bool should_break = false;
    if (exeedtol_rel <= (eps_rel * norm2_v) || exeedtol_rel < eps_tot22) {
      should_break = true;
    }
    if (norm2_v < eps_rel2) {
      should_break = true;
    }
    // if any thread should break, all threads break at same spot
    if (should_break) {
      continue_iteration = false;
      break;
    }

    /* Add new vertex to simplex - all threads compute identical values */
    i = s.nvrtx;
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      s.vrtx[i][t] = w[t];
    }
    s.vrtx_idx[i][0] = w_idx[0];
    s.vrtx_idx[i][1] = w_idx[1];
    s.nvrtx++;

    /* Invoke distance sub-algorithm (warp-parallel wrapper).*/
    subalgorithm_warp_parallel(&s, v, half_lane_idx, half_warp_mask,
      half_warp_base_thread_idx);

    /* Test */
    // All threads compute the same value since s.vrtx is the same on all threads (broadcast earlier)
    #pragma unroll 4
    for (int jj = 0; jj < 4; jj++) {
      if (jj < s.nvrtx) {
        norm2Wmax = gkFmax(norm2Wmax, norm2(s.vrtx[jj]));
      }
    }

    // Check exit condition
    norm2_v = norm2(v);
    if ((norm2_v <= (eps_tot * eps_tot * norm2Wmax))) {
      continue_iteration = false;
    }

    __syncwarp(half_warp_mask);
  }

  if (half_lane_idx == 0 && k == mk) {
    // mexPrintf(
    //     "\n * * * * * * * * * * * * MAXIMUM ITERATION NUMBER REACHED!!!  "
    //     " * * * * * * * * * * * * * * \n");
  }

  // Compute witnesses and final distance on first thread only
  if (half_lane_idx == 0) {
    compute_witnesses(&bd1, &bd2, &s);
    distances[half_warp_idx] = gkSqrt(norm2(v));
    // Write back updated simplex
    simplices[half_warp_idx] = s;
  }
}

__global__ void compute_minimum_distance_indexed_kernel(
    const gkPolytope* polytopes,
    const gkCollisionPair* pairs,
    gkSimplex* simplices,
    gkFloat* distances,
    const int n
) {// Calculate which collision this half-warp handles
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int half_warp_idx = index / THREADS_PER_COMPUTATION;

  // Get thread indexwithin the warp (0-31) and within the half-warp (0-15)
  int warp_lane_idx = threadIdx.x % 32;  // Lane ID within warp
  int half_warp_in_warp = warp_lane_idx / THREADS_PER_COMPUTATION;  // Which half-warp iwe are in (0 or 1)
  int half_lane_idx = warp_lane_idx % THREADS_PER_COMPUTATION;  // thread idx within half-warp (0-15)

  // Calculate the lead thread index for our half-warp within the warp (0 or 16)
  // This is the thread that will coordinate the half warp and complete and broadcast computations we only need one thread to do
  int half_warp_base_thread_idx = half_warp_in_warp * THREADS_PER_COMPUTATION;

  // Create mask for our half-warp (0xFFFF for first half, 0xFFFF0000 for second half) to determine which threads to sync with
  unsigned int half_warp_mask = (half_warp_in_warp == 0) ? 0xFFFF : 0xFFFF0000;

  if (half_warp_idx >= n) {
    return;
  }

  // Each half-warp (16 threads) handles a single GJK computation
  // all threads in the half-warp work on the same collision

  // Copy to local memory for fast access during iteration
  gkCollisionPair pair = pairs[half_warp_idx];
  gkPolytope bd1 = polytopes[pair.idx1];
  gkPolytope bd2 = polytopes[pair.idx2];
  gkSimplex s = simplices[half_warp_idx];

  unsigned int k = 0;                /**< Iteration counter                 */
  const int mk = 25;                 /**< Maximum number of GJK iterations  */
  const gkFloat eps_rel = eps_rel22; /**< Tolerance on relative             */
  const gkFloat eps_tot = eps_tot22; /**< Tolerance on absolute distance    */

  const gkFloat eps_rel2 = eps_rel * eps_rel;
  unsigned int i;
  gkFloat w[3];
  int w_idx[2];
  gkFloat v[3];
  gkFloat vminus[3];
  gkFloat norm2Wmax = 0;

  // Synchronize all threads in the half-warp before starting
  __syncwarp(half_warp_mask);

  // Initialize search direction
  v[0] = bd1.coord[0] - bd2.coord[0];
  v[1] = bd1.coord[1] - bd2.coord[1];
  v[2] = bd1.coord[2] - bd2.coord[2];

  // Initialize simplex - all threads compute identical values, no broadcast needed
  s.nvrtx = 1;
  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    s.vrtx[0][t] = v[t];
  }
  s.vrtx_idx[0][0] = 0;
  s.vrtx_idx[0][1] = 0;

  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    bd1.s[t] = bd1.coord[t];
  }
  bd1.s_idx = 0;

  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    bd2.s[t] = bd2.coord[t];
  }
  bd2.s_idx = 0;

  /* Begin GJK iteration */
  // Thread 0 controls the loop but all threads participate in parallel operations
  bool continue_iteration = true;

  while (continue_iteration) {
    // Broadcast k and s.nvrtx to all threads for loop condition check
    k = __shfl_sync(half_warp_mask, k, half_warp_base_thread_idx);
    s.nvrtx = __shfl_sync(half_warp_mask, s.nvrtx, half_warp_base_thread_idx);

    // Check loop conditions
    if (s.nvrtx == 4 || k == mk) {
      continue_iteration = false;
      break;
    }

    k++;

    /* Update negative search direction - all threads compute*/
    // Note: v is already the same on all threads from previous iteration
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      vminus[t] = -v[t];
    }

    /* Support function - parallelized using all threads */
    // All threads participate in finding support points for speedup but only thread 0 updates the body
    support_parallel(&bd1, vminus, half_lane_idx, half_warp_mask, half_warp_base_thread_idx);
    support_parallel(&bd2, v, half_lane_idx, half_warp_mask, half_warp_base_thread_idx);

    // all threads compute w for witness point computation
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      w[t] = bd1.s[t] - bd2.s[t];
    }
    w_idx[0] = bd1.s_idx;
    w_idx[1] = bd2.s_idx;

    /* Test first exit condition (new point already in simplex/can't move
     * further) */
    gkFloat norm2_v = norm2(v);
    gkFloat exeedtol_rel = (norm2_v - dotProduct(v, w));

    //check exit conditions
    bool should_break = false;
    if (exeedtol_rel <= (eps_rel * norm2_v) || exeedtol_rel < eps_tot22) {
      should_break = true;
    }
    if (norm2_v < eps_rel2) {
      should_break = true;
    }
    // if any thread should break, all threads break at same spot
    if (should_break) {
      continue_iteration = false;
      break;
    }

    /* Add new vertex to simplex - all threads compute identical values */
    i = s.nvrtx;
    #pragma unroll
    for (int t = 0; t < 3; ++t) {
      s.vrtx[i][t] = w[t];
    }
    s.vrtx_idx[i][0] = w_idx[0];
    s.vrtx_idx[i][1] = w_idx[1];
    s.nvrtx++;

    /* Invoke distance sub-algorithm (warp-parallel wrapper).*/
    subalgorithm_warp_parallel(&s, v, half_lane_idx, half_warp_mask,
      half_warp_base_thread_idx);

    /* Test */
    // All threads compute the same value since s.vrtx is the same on all threads (broadcast earlier)
    #pragma unroll 4
    for (int jj = 0; jj < 4; jj++) {
      if (jj < s.nvrtx) {
        norm2Wmax = gkFmax(norm2Wmax, norm2(s.vrtx[jj]));
      }
    }

    // Check exit condition
    norm2_v = norm2(v);
    if ((norm2_v <= (eps_tot * eps_tot * norm2Wmax))) {
      continue_iteration = false;
    }

    __syncwarp(half_warp_mask);
  }

  if (half_lane_idx == 0 && k == mk) {
    // mexPrintf(
    //     "\n * * * * * * * * * * * * MAXIMUM ITERATION NUMBER REACHED!!!  "
    //     " * * * * * * * * * * * * * * \n");
  }

  // Compute witnesses and final distance on first thread only
  if (half_lane_idx == 0) {
    compute_witnesses(&bd1, &bd2, &s);
    distances[half_warp_idx] = gkSqrt(norm2(v));
    // Write back updated simplex
    simplices[half_warp_idx] = s;
  }
}

//*******************************************************************************************
// WarpParallel EPA Implementation
//*******************************************************************************************

// Entry point to EPA implementation. 1 Warp per collision.

// Face structure for EPA polytope
// Each face is a triangle with 3 vertex indices
typedef struct {
  int v[3];           // Vertex indices in the polytope
  int v_idx[3][2];    // Original vertex indices from original polytopes for witness computation [vertex][body]
  gkFloat normal[3];  // Face normal (pointing outward from origin)
  gkFloat distance;   // Distance from origin to face plane
  bool valid;         // Whether this face is still valid (not removed)
} EPAFace;

// Polytope structure for EPA
typedef struct {
  gkFloat vertices[MAX_EPA_FACES + 4][3];  // Vertex coordinates in the Minkowski difference
  int vertex_indices[MAX_EPA_FACES + 4][2]; // Original vertex indices [vertex][body]
  int num_vertices;
  EPAFace faces[MAX_EPA_FACES];
  int max_face_index; // Highest face index in use (for iteration bounds)
} EPAPolytope;

// Compute face normal and distance of face from origin
__device__ inline static void compute_face_normal_distance(EPAPolytope* poly, int face_idx, const gkFloat* centroid) {
  EPAFace* face = &poly->faces[face_idx];
  if (!face->valid) return;

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat* v1 = poly->vertices[face->v[1]];
  gkFloat* v2 = poly->vertices[face->v[2]];

  // Compute edge vectors
  gkFloat e0[3], e1[3];
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
  }

  // Compute normal
  crossProduct(e0, e1, face->normal);
  gkFloat norm_sq = norm2(face->normal);

  if (norm_sq > gkEpsilon * gkEpsilon) {
    gkFloat norm = gkSqrt(norm_sq);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      face->normal[i] /= norm;
    }

    // Ensure normal points away from centroid (outward from polytope interior)
    gkFloat to_centroid[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      to_centroid[i] = centroid[i] - v0[i];
    }

    // If normal points toward centroid flip it
    if (dotProduct(face->normal, to_centroid) > 0) {
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        face->normal[i] = -face->normal[i];
      }
    }

    face->distance = dotProduct(face->normal, v0);

    if (face->distance < 0) {
      // Flip normal and distance
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        face->normal[i] = -face->normal[i];
      }
      face->distance = -face->distance;
    }
  }
  else {
    // Degenerate face
    face->valid = false;
    face->distance = 1e10f;
  }
}

// Check if a face is visible from a point (point is on positive side of face) Needed to determine which faces to restructure when vertex is added
__device__ inline static bool is_face_visible(EPAPolytope* poly, int face_idx, const gkFloat* point) {
  EPAFace* face = &poly->faces[face_idx];
  if (!face->valid) return false;

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat diff[3];
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    diff[i] = point[i] - v0[i];
  }

  return dotProduct(face->normal, diff) > gkEpsilon;
}

// Parallel support function for EPA basicallly GJK one but only care about minkowski difference point
__device__ inline static void support_epa_parallel(gkPolytope* body1, gkPolytope* body2,
  const gkFloat* direction, gkFloat* result, int* result_idx,
  int warp_lane_idx, unsigned int warp_mask) {

  // Each thread searches a subset of points
  const int max_points = (body1->numpoints > body2->numpoints) ?
    body1->numpoints : body2->numpoints;
  const int points_per_thread = (max_points + 31) / 32;
  const int start_idx = warp_lane_idx * points_per_thread;
  const int end_idx = (start_idx + points_per_thread < max_points) ?
    (start_idx + points_per_thread) : max_points;

  gkFloat local_max1 = -1e10f;
  gkFloat local_max2 = -1e10f;
  int local_best1 = -1;
  int local_best2 = -1;
  gkFloat vrt[3];

  // Search body1
  for (int i = start_idx; i < body1->numpoints && i < end_idx; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      vrt[j] = getCoord(body1, i, j);
    }
    gkFloat s = dotProduct(vrt, direction);
    if (s > local_max1) {
      local_max1 = s;
      local_best1 = i;
    }
  }

  // Search body2 opposite direction
  gkFloat neg_dir[3] = { -direction[0], -direction[1], -direction[2] };
  for (int i = start_idx; i < body2->numpoints && i < end_idx; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      vrt[j] = getCoord(body2, i, j);
    }
    gkFloat s = dotProduct(vrt, neg_dir);
    if (s > local_max2) {
      local_max2 = s;
      local_best2 = i;
    }
  }

  // Parallel reduction for body1
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    gkFloat other_max = __shfl_down_sync(warp_mask, local_max1, offset);
    int other_best = __shfl_down_sync(warp_mask, local_best1, offset);
    if (other_max > local_max1) {
      local_max1 = other_max;
      local_best1 = other_best;
    }
  }

  // Parallel reduction for body2
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    gkFloat other_max = __shfl_down_sync(warp_mask, local_max2, offset);
    int other_best = __shfl_down_sync(warp_mask, local_best2, offset);
    if (other_max > local_max2) {
      local_max2 = other_max;
      local_best2 = other_best;
    }
  }

  // Broadcast results from thread 0
  local_best1 = __shfl_sync(warp_mask, local_best1, 0);
  local_best2 = __shfl_sync(warp_mask, local_best2, 0);

  // Compute Minkowski difference point
  if (warp_lane_idx == 0 && local_best1 >= 0 && local_best2 >= 0) {
    gkFloat p1[3], p2[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      p1[i] = getCoord(body1, local_best1, i);
      p2[i] = getCoord(body2, local_best2, i);
      result[i] = p1[i] - p2[i];
    }
    result_idx[0] = local_best1;
    result_idx[1] = local_best2;
  }

  // Broadcast result to all threads
  result[0] = __shfl_sync(warp_mask, result[0], 0);
  result[1] = __shfl_sync(warp_mask, result[1], 0);
  result[2] = __shfl_sync(warp_mask, result[2], 0);
  result_idx[0] = __shfl_sync(warp_mask, result_idx[0], 0);
  result_idx[1] = __shfl_sync(warp_mask, result_idx[1], 0);
}

// Initialize EPA polytope from GJK simplex (should be a tetrahedron)
__device__ inline static void init_epa_polytope(EPAPolytope* poly, const gkSimplex* simplex, gkFloat* centroid) {
  // Clear all faces first
  #pragma unroll
  for (int i = 0; i < MAX_EPA_FACES; ++i) {
    poly->faces[i].valid = false;
    poly->faces[i].distance = 1e10f;
  }

  // Copy vertices from simplex
  poly->num_vertices = 4;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      poly->vertices[i][j] = simplex->vrtx[i][j];
    }
    poly->vertex_indices[i][0] = simplex->vrtx_idx[i][0];
    poly->vertex_indices[i][1] = simplex->vrtx_idx[i][1];
  }

  // Compute centroid of the tetrahedron
  centroid[0] = centroid[1] = centroid[2] = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      centroid[j] += poly->vertices[i][j];
    }
  }
  #pragma unroll
  for (int j = 0; j < 3; j++) {
    centroid[j] /= 4.0f;
  }

  // Create 4 faces of tetrahedron
  // set up the faces and then fix winding based on normal direction
  // Face 0: vertices 0, 1, 2
  poly->faces[0].v[0] = 0;
  poly->faces[0].v[1] = 1;
  poly->faces[0].v[2] = 2;
  poly->faces[0].valid = true;

  // Face 1: vertices 0, 3, 1
  poly->faces[1].v[0] = 0;
  poly->faces[1].v[1] = 3;
  poly->faces[1].v[2] = 1;
  poly->faces[1].valid = true;

  // Face 2: vertices 0, 2, 3
  poly->faces[2].v[0] = 0;
  poly->faces[2].v[1] = 2;
  poly->faces[2].v[2] = 3;
  poly->faces[2].valid = true;

  // Face 3: vertices 1, 3, 2
  poly->faces[3].v[0] = 1;
  poly->faces[3].v[1] = 3;
  poly->faces[3].v[2] = 2;
  poly->faces[3].valid = true;

  // Copy vertex indices for witness point computation
  #pragma unroll
  for (int f = 0; f < 4; f++) {
    #pragma unroll
    for (int v = 0; v < 3; v++) {
      int vi = poly->faces[f].v[v];
      poly->faces[f].v_idx[v][0] = simplex->vrtx_idx[vi][0];
      poly->faces[f].v_idx[v][1] = simplex->vrtx_idx[vi][1];
    }
  }

  // Compute normals and fix winding
  #pragma unroll
  for (int f = 0; f < 4; f++) {
    gkFloat* v0 = poly->vertices[poly->faces[f].v[0]];
    gkFloat* v1 = poly->vertices[poly->faces[f].v[1]];
    gkFloat* v2 = poly->vertices[poly->faces[f].v[2]];

    gkFloat e0[3], e1[3], normal[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      e0[i] = v1[i] - v0[i];
      e1[i] = v2[i] - v0[i];
    }
    crossProduct(e0, e1, normal);

    // Vector from face to centroid
    gkFloat to_centroid[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      to_centroid[i] = centroid[i] - v0[i];
    }

    // If normal points toward centroid need to flip the winding
    if (dotProduct(normal, to_centroid) > 0) {
      int tmp = poly->faces[f].v[1];
      poly->faces[f].v[1] = poly->faces[f].v[2];
      poly->faces[f].v[2] = tmp;

      int tmp_idx0 = poly->faces[f].v_idx[1][0];
      int tmp_idx1 = poly->faces[f].v_idx[1][1];
      poly->faces[f].v_idx[1][0] = poly->faces[f].v_idx[2][0];
      poly->faces[f].v_idx[1][1] = poly->faces[f].v_idx[2][1];
      poly->faces[f].v_idx[2][0] = tmp_idx0;
      poly->faces[f].v_idx[2][1] = tmp_idx1;
    }
  }

  poly->max_face_index = 4;
}

// barycentric coordinate compute closest point on triangle to origin
__device__ inline static void compute_barycentric_origin(
  const gkFloat* v0, const gkFloat* v1, const gkFloat* v2,
  gkFloat* a0, gkFloat* a1, gkFloat* a2) {

  // Compute vectors
  gkFloat e0[3], e1[3], v0_neg[3];
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
    v0_neg[i] = -v0[i];
  }

  // Compute dot products for barycentric coords
  gkFloat d00 = dotProduct(e0, e0);
  gkFloat d01 = dotProduct(e0, e1);
  gkFloat d11 = dotProduct(e1, e1);
  gkFloat d20 = dotProduct(v0_neg, e0);
  gkFloat d21 = dotProduct(v0_neg, e1);

  gkFloat denom = d00 * d11 - d01 * d01;

  if (fabs(denom) < gkEpsilon) {
    // Degenerate
    *a0 = *a1 = *a2 = (gkFloat)1.0 / (gkFloat)3.0;
    return;
  }

  gkFloat inv_denom = (gkFloat)1.0 / denom;
  gkFloat u = (d11 * d20 - d01 * d21) * inv_denom;
  gkFloat v = (d00 * d21 - d01 * d20) * inv_denom;
  gkFloat w = (gkFloat)1.0 - u - v;

  // Clamp to triangle
  if (w < 0) {
    // Origin projects outside edge v1-v2
    // Project onto edge v1-v2
    gkFloat e12[3], v1_neg[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      e12[i] = v2[i] - v1[i];
      v1_neg[i] = -v1[i];
    }
    gkFloat t = dotProduct(v1_neg, e12) / dotProduct(e12, e12);
    t = fmax((gkFloat)0.0, fmin((gkFloat)1.0, t));
    *a0 = 0;
    *a1 = (gkFloat)1.0 - t;
    *a2 = t;
  }
  else if (u < 0) {
    // Origin projects outside edge v0-v2
    gkFloat t = dotProduct(v0_neg, e1) / dotProduct(e1, e1);
    t = fmax((gkFloat)0.0, fmin((gkFloat)1.0, t));
    *a0 = (gkFloat)1.0 - t;
    *a1 = 0;
    *a2 = t;
  }
  else if (v < 0) {
    // Origin projects outside edge v0-v1
    gkFloat t = dotProduct(v0_neg, e0) / dotProduct(e0, e0);
    t = fmax((gkFloat)0.0, fmin((gkFloat)1.0, t));
    *a0 = (gkFloat)1.0 - t;
    *a1 = t;
    *a2 = 0;
  }
  else {
    // Inside triangle
    *a0 = w;
    *a1 = u;
    *a2 = v;
  }
}

// Structure for horizon edge collection
typedef struct {
  int v1, v2;  // Vertex indices in polytope
  int v_idx1[2], v_idx2[2];  // Original vertex indices for witness computation
  bool valid;
} EPAEdge;

// ENTRY POINT TO EPA CALL
// Main EPA kernel - one warp per collision
__global__ void compute_epa_kernel(
  const gkPolytope* polytopes1,
  const gkPolytope* polytopes2,
  gkSimplex* simplices,
  gkFloat* distances,
  gkFloat* witness1,
  gkFloat* witness2,
  gkFloat* contact_normals,
  const int n) {

  // Calculate which collision this warp handles
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int warp_idx = index / 32;

  if (warp_idx >= n) {
    return;
  }

  // Get thread index within warp (0-31)
  int warp_lane_idx = threadIdx.x % 32;
  unsigned int warp_mask = 0xFFFFFFFF;

  // Copy polytopes to local memory
  gkPolytope bd1_local = polytopes1[warp_idx];
  gkPolytope bd2_local = polytopes2[warp_idx];
  gkPolytope* bd1 = &bd1_local;
  gkPolytope* bd2 = &bd2_local;
  gkSimplex simplex = simplices[warp_idx];
  gkFloat distance = distances[warp_idx];

  // if distance isn't 0 didn't detect collision - skip EPA
  if (distance > gkEpsilon) {
    if (warp_lane_idx == 0) {
      // Witness points already computed by GJK: todo: witness points seem to be off in testing right now
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        witness1[warp_idx * 3 + i] = simplex.witnesses[0][i];
        witness2[warp_idx * 3 + i] = simplex.witnesses[1][i];
      }
      // Compute contact normal from witness1 to witness2 (for non-colliding case)
      gkFloat w1_to_w2[3];
      gkFloat norm = 0.0f;
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        w1_to_w2[i] = witness2[warp_idx * 3 + i] - witness1[warp_idx * 3 + i];
        norm += w1_to_w2[i] * w1_to_w2[i];
      }
      norm = gkSqrt(norm);
      if (norm > gkEpsilon) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          contact_normals[warp_idx * 3 + i] = w1_to_w2[i] / norm;
        }
      } else {
        // Default normal if witnesses are too close
        contact_normals[warp_idx * 3 + 0] = 1.0f;
        contact_normals[warp_idx * 3 + 1] = 0.0f;
        contact_normals[warp_idx * 3 + 2] = 0.0f;
      }
    }
    return;
  }

  // If GJK returned a degenerate simplex, rebuild it properly for EPA
  if (simplex.nvrtx != 4) {
    // Need to get it up to 4 vertices
    if (simplex.nvrtx == 1) {
      // Grow simplex from a single point: fire a support in some direction.
      // We use current simplex point for new direction for the
      // support; if this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;
      bool terminate_epa = false;

      if (warp_lane_idx == 0) {
        dir[0] = simplex.vrtx[0][0];
        dir[1] = simplex.vrtx[0][1];
        dir[2] = simplex.vrtx[0][2];
      }

      // Broadcast direction from lane 0 to whole warp.
      dir[0] = __shfl_sync(warp_mask, dir[0], 0);
      dir[1] = __shfl_sync(warp_mask, dir[1], 0);
      dir[2] = __shfl_sync(warp_mask, dir[2], 0);

      // Parallel EPA support in that direction.
      support_epa_parallel(bd1, bd2, dir, new_vertex, new_vertex_idx,
        warp_lane_idx, warp_mask);

      __syncwarp(warp_mask);

      if (warp_lane_idx == 0) {
        // Check if this is a new point relative to the existing simplex vertex.
        bool is_new = true;
        gkFloat dx = new_vertex[0] - simplex.vrtx[0][0];
        gkFloat dy = new_vertex[1] - simplex.vrtx[0][1];
        gkFloat dz = new_vertex[2] - simplex.vrtx[0][2];
        gkFloat d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < eps_sq) {
          is_new = false;
        }

        if (is_new) {
          int idx = simplex.nvrtx;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            simplex.vrtx[idx][c] = new_vertex[c];
          }
          simplex.vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex.vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex.nvrtx = 2;
        }
        else {
          // No new support point means penetration depth effectively zero.
          distances[warp_idx] = 0.0f;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            gkFloat p1 = getCoord(bd1, new_vertex_idx[0], c);
            gkFloat p2 = getCoord(bd2, new_vertex_idx[1], c);
            witness1[warp_idx * 3 + c] = p1;
            witness2[warp_idx * 3 + c] = p2;
          }
          // Compute contact normal from witness1 to witness2
          gkFloat w1_to_w2[3];
          gkFloat norm = 0.0f;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            w1_to_w2[c] = witness2[warp_idx * 3 + c] - witness1[warp_idx * 3 + c];
            norm += w1_to_w2[c] * w1_to_w2[c];
          }
          norm = gkSqrt(norm);
          if (norm > gkEpsilon) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
              contact_normals[warp_idx * 3 + c] = w1_to_w2[c] / norm;
            }
          } else {
            contact_normals[warp_idx * 3 + 0] = 1.0f;
            contact_normals[warp_idx * 3 + 1] = 0.0f;
            contact_normals[warp_idx * 3 + 2] = 0.0f;
          }
          terminate_epa = true;
        }
      }

      int term_flag = __shfl_sync(warp_mask, terminate_epa ? 1 : 0, 0);
      if (term_flag) {
        return;
      }

      // Broadcast updated simplex
      simplex.nvrtx = __shfl_sync(warp_mask, simplex.nvrtx, 0);
      #pragma unroll 4
      for (int v = 0; v < simplex.nvrtx; v++) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          simplex.vrtx[v][c] = __shfl_sync(warp_mask, simplex.vrtx[v][c], 0);
        }
        simplex.vrtx_idx[v][0] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][0], 0);
        simplex.vrtx_idx[v][1] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][1], 0);
      }
    }
    if (simplex.nvrtx == 2) {
      // Grow simplex from an edge: fire a support in a direction perpendicular
      // to the edge. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;
      bool terminate_epa = false;

      if (warp_lane_idx == 0) {
        gkFloat p0[3], p1[3], edge[3];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
          p0[c] = simplex.vrtx[0][c];
          p1[c] = simplex.vrtx[1][c];
          edge[c] = p1[c] - p0[c];
        }

        // Build a perpindicular
        gkFloat axis[3] = { 1.0f, 0.0f, 0.0f };
        gkFloat edge_norm = gkSqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]);
        if (edge_norm > gkEpsilon && fabs(edge[0]) > 0.9f * edge_norm) {
          axis[0] = 0.0f; axis[1] = 1.0f; axis[2] = 0.0f;
        }

        // dir = edge x axis
        crossProduct(edge, axis, dir);
        gkFloat nrm2 = norm2(dir);
        if (nrm2 < gkEpsilon) {
          // Fallback axis
          axis[0] = 0.0f; axis[1] = 0.0f; axis[2] = 1.0f;
          crossProduct(edge, axis, dir);
        }
      }

      // Broadcast
      dir[0] = __shfl_sync(warp_mask, dir[0], 0);
      dir[1] = __shfl_sync(warp_mask, dir[1], 0);
      dir[2] = __shfl_sync(warp_mask, dir[2], 0);

      // Parallel EPA support in that direction.
      support_epa_parallel(bd1, bd2, dir, new_vertex, new_vertex_idx,
        warp_lane_idx, warp_mask);

      __syncwarp(warp_mask);

      if (warp_lane_idx == 0) {
        // Check if this is a new point relative to both existing simplex vertices.
        bool is_new = true;
        #pragma unroll 4
        for (int vtx = 0; vtx < simplex.nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex.vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex.vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex.vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex.nvrtx;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            simplex.vrtx[idx][c] = new_vertex[c];
          }
          simplex.vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex.vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex.nvrtx = 3;
        }
        else {
          // No new support point means penetration depth effectively zero.
          distances[warp_idx] = 0.0f;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            gkFloat p1 = getCoord(bd1, new_vertex_idx[0], c);
            gkFloat p2 = getCoord(bd2, new_vertex_idx[1], c);
            witness1[warp_idx * 3 + c] = p1;
            witness2[warp_idx * 3 + c] = p2;
          }
          // Compute contact normal from witness1 to witness2
          gkFloat w1_to_w2[3];
          gkFloat norm = 0.0f;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            w1_to_w2[c] = witness2[warp_idx * 3 + c] - witness1[warp_idx * 3 + c];
            norm += w1_to_w2[c] * w1_to_w2[c];
          }
          norm = gkSqrt(norm);
          if (norm > gkEpsilon) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
              contact_normals[warp_idx * 3 + c] = w1_to_w2[c] / norm;
            }
          } else {
            contact_normals[warp_idx * 3 + 0] = 1.0f;
            contact_normals[warp_idx * 3 + 1] = 0.0f;
            contact_normals[warp_idx * 3 + 2] = 0.0f;
          }
          terminate_epa = true;
        }
      }

      int term_flag = __shfl_sync(warp_mask, terminate_epa ? 1 : 0, 0);
      if (term_flag) {
        return;
      }

      // Broadcast updated simplex
      simplex.nvrtx = __shfl_sync(warp_mask, simplex.nvrtx, 0);
      #pragma unroll 4
      for (int v = 0; v < simplex.nvrtx; v++) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          simplex.vrtx[v][c] = __shfl_sync(warp_mask, simplex.vrtx[v][c], 0);
        }
        simplex.vrtx_idx[v][0] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][0], 0);
        simplex.vrtx_idx[v][1] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][1], 0);
      }
    }
    if (simplex.nvrtx == 3) {
      // Grow simplex from a triangle: fire a support in the direction of the
      // triangle normal. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;
      bool terminate_epa = false;

      if (warp_lane_idx == 0) {
        gkFloat p0[3], p1[3], p2[3];
        gkFloat e0[3], e1[3];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
          p0[c] = simplex.vrtx[0][c];
          p1[c] = simplex.vrtx[1][c];
          p2[c] = simplex.vrtx[2][c];
          e0[c] = p1[c] - p0[c];
          e1[c] = p2[c] - p0[c];
        }
        // dir = e0 x e1 (normal to the triangle)
        crossProduct(e0, e1, dir);
      }

      // Broadcast
      dir[0] = __shfl_sync(warp_mask, dir[0], 0);
      dir[1] = __shfl_sync(warp_mask, dir[1], 0);
      dir[2] = __shfl_sync(warp_mask, dir[2], 0);

      // Parallel EPA support in that direction.
      support_epa_parallel(bd1, bd2, dir, new_vertex, new_vertex_idx,
        warp_lane_idx, warp_mask);

      __syncwarp(warp_mask);

      if (warp_lane_idx == 0) {
        // Check if this is a new point relative to all three existing simplex vertices.
        bool is_new = true;
        #pragma unroll 4
        for (int vtx = 0; vtx < simplex.nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex.vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex.vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex.vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex.nvrtx;
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            simplex.vrtx[idx][c] = new_vertex[c];
          }
          simplex.vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex.vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex.nvrtx = 4;
        }
        else {
          // Try opposite direction
          dir[0] = -dir[0];
          dir[1] = -dir[1];
          dir[2] = -dir[2];
        }
      }

      // If first direction didn't work, try opposite
      int curr_nvrtx = __shfl_sync(warp_mask, simplex.nvrtx, 0);
      if (curr_nvrtx == 3) {
        dir[0] = __shfl_sync(warp_mask, dir[0], 0);
        dir[1] = __shfl_sync(warp_mask, dir[1], 0);
        dir[2] = __shfl_sync(warp_mask, dir[2], 0);

        support_epa_parallel(bd1, bd2, dir, new_vertex, new_vertex_idx,
          warp_lane_idx, warp_mask);

        __syncwarp(warp_mask);

        if (warp_lane_idx == 0) {
          bool is_new = true;
          #pragma unroll 4
          for (int vtx = 0; vtx < simplex.nvrtx; ++vtx) {
            gkFloat dx = new_vertex[0] - simplex.vrtx[vtx][0];
            gkFloat dy = new_vertex[1] - simplex.vrtx[vtx][1];
            gkFloat dz = new_vertex[2] - simplex.vrtx[vtx][2];
            gkFloat d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < eps_sq) {
              is_new = false;
              break;
            }
          }

          if (is_new) {
            int idx = simplex.nvrtx;
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
              simplex.vrtx[idx][c] = new_vertex[c];
            }
            simplex.vrtx_idx[idx][0] = new_vertex_idx[0];
            simplex.vrtx_idx[idx][1] = new_vertex_idx[1];
            simplex.nvrtx = 4;
          }
          else {
            distances[warp_idx] = 0.0f;
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
              gkFloat p1 = getCoord(bd1, new_vertex_idx[0], c);
              gkFloat p2 = getCoord(bd2, new_vertex_idx[1], c);
              witness1[warp_idx * 3 + c] = p1;
              witness2[warp_idx * 3 + c] = p2;
            }
            // Compute contact normal from witness1 to witness2
            gkFloat w1_to_w2[3];
            gkFloat norm = 0.0f;
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
              w1_to_w2[c] = witness2[warp_idx * 3 + c] - witness1[warp_idx * 3 + c];
              norm += w1_to_w2[c] * w1_to_w2[c];
            }
            norm = gkSqrt(norm);
            if (norm > gkEpsilon) {
              #pragma unroll
              for (int c = 0; c < 3; ++c) {
                contact_normals[warp_idx * 3 + c] = w1_to_w2[c] / norm;
              }
            } else {
              contact_normals[warp_idx * 3 + 0] = 1.0f;
              contact_normals[warp_idx * 3 + 1] = 0.0f;
              contact_normals[warp_idx * 3 + 2] = 0.0f;
            }
            terminate_epa = true;
          }
        }
        int term_flag = __shfl_sync(warp_mask, terminate_epa ? 1 : 0, 0);
        if (term_flag) {
          return;
        }
      }

      // Broadcast updated simplex
      simplex.nvrtx = __shfl_sync(warp_mask, simplex.nvrtx, 0);
      #pragma unroll 4
      for (int v = 0; v < simplex.nvrtx; v++) {
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          simplex.vrtx[v][c] = __shfl_sync(warp_mask, simplex.vrtx[v][c], 0);
        }
        simplex.vrtx_idx[v][0] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][0], 0);
        simplex.vrtx_idx[v][1] = __shfl_sync(warp_mask, simplex.vrtx_idx[v][1], 0);
      }
    }

    // If we still don't have 4 vertices, abort
    if (simplex.nvrtx != 4) {
      if (warp_lane_idx == 0) {
        distances[warp_idx] = 0.0f;
        // Compute contact normal from witness points if available
        gkFloat w1_to_w2[3];
        gkFloat norm = 0.0f;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
          w1_to_w2[c] = witness2[warp_idx * 3 + c] - witness1[warp_idx * 3 + c];
          norm += w1_to_w2[c] * w1_to_w2[c];
        }
        norm = gkSqrt(norm);
        if (norm > gkEpsilon) {
          #pragma unroll
          for (int c = 0; c < 3; ++c) {
            contact_normals[warp_idx * 3 + c] = w1_to_w2[c] / norm;
          }
        } else {
          contact_normals[warp_idx * 3 + 0] = 1.0f;
          contact_normals[warp_idx * 3 + 1] = 0.0f;
          contact_normals[warp_idx * 3 + 2] = 0.0f;
        }
      }
      return;
    }
  }

  // On to actual EPA alg with a valid tetrahedron simplex
  // Initialize EPA polytope from simplex
  EPAPolytope poly;
  gkFloat centroid[3];
  if (warp_lane_idx == 0) {
    init_epa_polytope(&poly, &simplex, centroid);
  }

  __syncwarp(warp_mask);

  // Broadcast polytope initialization to all threads
  poly.num_vertices = __shfl_sync(warp_mask, poly.num_vertices, 0);
  poly.max_face_index = __shfl_sync(warp_mask, poly.max_face_index, 0);
  centroid[0] = __shfl_sync(warp_mask, centroid[0], 0);
  centroid[1] = __shfl_sync(warp_mask, centroid[1], 0);
  centroid[2] = __shfl_sync(warp_mask, centroid[2], 0);

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 3; j++) {
      poly.vertices[i][j] = __shfl_sync(warp_mask, poly.vertices[i][j], 0);
    }
    poly.vertex_indices[i][0] = __shfl_sync(warp_mask, poly.vertex_indices[i][0], 0);
    poly.vertex_indices[i][1] = __shfl_sync(warp_mask, poly.vertex_indices[i][1], 0);
  }

  #pragma unroll
  for (int f = 0; f < 4; f++) {
    #pragma unroll
    for (int v = 0; v < 3; v++) {
      poly.faces[f].v[v] = __shfl_sync(warp_mask, poly.faces[f].v[v], 0);
      poly.faces[f].v_idx[v][0] = __shfl_sync(warp_mask, poly.faces[f].v_idx[v][0], 0);
      poly.faces[f].v_idx[v][1] = __shfl_sync(warp_mask, poly.faces[f].v_idx[v][1], 0);
    }
    poly.faces[f].valid = __shfl_sync(warp_mask, poly.faces[f].valid ? 1 : 0, 0) != 0;
  }

  // EPA iteration parameters
  const int max_iterations = 64;
  const gkFloat tolerance = eps_tot22;
  int iteration = 0;

  // Main EPA loop
  while (iteration < max_iterations && poly.num_vertices < MAX_EPA_VERTICES - 1) {
    iteration++;

    // Compute face normals and distances in parallel across warp
    // Each thread processes a subset of faces
    const int faces_per_thread = (poly.max_face_index + 31) / 32;
    const int start_face = warp_lane_idx * faces_per_thread;
    const int end_face = (start_face + faces_per_thread < poly.max_face_index) ? 
                         (start_face + faces_per_thread) : poly.max_face_index;

    // Recompute normals & distances for assigned faces
    for (int i = start_face; i < end_face; ++i) {
      if (poly.faces[i].valid) {
        compute_face_normal_distance(&poly, i, centroid);
      }
    }

    __syncwarp(warp_mask);

    // Broadcast updated face normals and distances to all threads
    // Each thread receives updates from the thread that computed each face
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (poly.faces[i].valid) {
        int source_thread = i / faces_per_thread;
        if (source_thread >= 32) source_thread = 31; // Clamp to valid thread index
        
        // Receive face data from the thread that computed it
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          poly.faces[i].normal[c] = __shfl_sync(warp_mask, poly.faces[i].normal[c], source_thread);
        }
        poly.faces[i].distance = __shfl_sync(warp_mask, poly.faces[i].distance, source_thread);
      }
    }

    // parallel reduction to find closest face
    // Each thread finds the closest face in its assigned range
    int local_closest_face = -1;
    gkFloat local_closest_distance = 1e10f;

    for (int i = start_face; i < end_face; ++i) {
      if (!poly.faces[i].valid) continue;
      if (poly.faces[i].distance >= 0.0f && poly.faces[i].distance < local_closest_distance) {
        local_closest_distance = poly.faces[i].distance;
        local_closest_face = i;
      }
    }

    // Parallel reduction across warp to find global minimum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      gkFloat other_dist = __shfl_down_sync(warp_mask, local_closest_distance, offset);
      int other_face = __shfl_down_sync(warp_mask, local_closest_face, offset);
      
      if (other_face >= 0 && (local_closest_face < 0 || other_dist < local_closest_distance)) {
        local_closest_distance = other_dist;
        local_closest_face = other_face;
      }
    }

    // Broadcast result from thread 0 to all threads
    int closest_face = __shfl_sync(warp_mask, local_closest_face, 0);
    gkFloat closest_distance = __shfl_sync(warp_mask, local_closest_distance, 0);
    
    // Read direction from closest face (all threads have same polytope data)
    gkFloat dir_x = 0, dir_y = 0, dir_z = 0;
    if (closest_face >= 0) {
      dir_x = poly.faces[closest_face].normal[0];
      dir_y = poly.faces[closest_face].normal[1];
      dir_z = poly.faces[closest_face].normal[2];
    }

    if (closest_face < 0) {
      break;
    }

    gkFloat direction[3] = { dir_x, dir_y, dir_z };
    EPAFace* closest = &poly.faces[closest_face];

    // Get support point in direction of closest face normal
    gkFloat new_vertex[3];
    int new_vertex_idx[2];
    support_epa_parallel(bd1, bd2, direction, new_vertex, new_vertex_idx,
      warp_lane_idx, warp_mask);

    __syncwarp(warp_mask);

    // Check termination condition: if distance to new vertex along normal is not more than tolerance further than closest face
    gkFloat dist_to_new = dir_x * new_vertex[0] + dir_y * new_vertex[1] + dir_z * new_vertex[2];
    gkFloat improvement = dist_to_new - closest_distance;

    if (improvement < tolerance) {
      // Converged, compute witness points with bary coords
      if (warp_lane_idx == 0) {
        gkFloat* v0 = poly.vertices[closest->v[0]];
        gkFloat* v1 = poly.vertices[closest->v[1]];
        gkFloat* v2 = poly.vertices[closest->v[2]];

        // bary computation
        gkFloat a0, a1, a2;
        compute_barycentric_origin(v0, v1, v2, &a0, &a1, &a2);

        // Compute witness points using barycentric coords
        int idx0[2] = { closest->v_idx[0][0], closest->v_idx[0][1] };
        int idx1[2] = { closest->v_idx[1][0], closest->v_idx[1][1] };
        int idx2[2] = { closest->v_idx[2][0], closest->v_idx[2][1] };

        #pragma unroll
        for (int i = 0; i < 3; i++) {
          witness1[warp_idx * 3 + i] =
            getCoord(bd1, idx0[0], i) * a0 +
            getCoord(bd1, idx1[0], i) * a1 +
            getCoord(bd1, idx2[0], i) * a2;
          witness2[warp_idx * 3 + i] =
            getCoord(bd2, idx0[1], i) * a0 +
            getCoord(bd2, idx1[1], i) * a1 +
            getCoord(bd2, idx2[1], i) * a2;
        }

        // Penetration depth is negative distance (objects overlap)
        distances[warp_idx] = -closest_distance;
        
        // Store contact normal (points from polytope1 to polytope2)
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          contact_normals[warp_idx * 3 + i] = closest->normal[i];
        }
      }
      break;
    }

    /// Check if new vertex is duplicate
    bool is_duplicate = false;
    if (warp_lane_idx == 0) {
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;
      for (int i = 0; i < poly.num_vertices; i++) {
        gkFloat dx = new_vertex[0] - poly.vertices[i][0];
        gkFloat dy = new_vertex[1] - poly.vertices[i][1];
        gkFloat dz = new_vertex[2] - poly.vertices[i][2];
        if (dx * dx + dy * dy + dz * dz < eps_sq) {
          is_duplicate = true;
          break;
        }
      }
    }

    is_duplicate = __shfl_sync(warp_mask, is_duplicate ? 1 : 0, 0) != 0;

    if (is_duplicate) {
      // Can't make progress, use current best
      if (warp_lane_idx == 0) {
        gkFloat* v0 = poly.vertices[closest->v[0]];
        gkFloat* v1 = poly.vertices[closest->v[1]];
        gkFloat* v2 = poly.vertices[closest->v[2]];

        gkFloat a0, a1, a2;
        compute_barycentric_origin(v0, v1, v2, &a0, &a1, &a2);

        int idx0[2] = { closest->v_idx[0][0], closest->v_idx[0][1] };
        int idx1[2] = { closest->v_idx[1][0], closest->v_idx[1][1] };
        int idx2[2] = { closest->v_idx[2][0], closest->v_idx[2][1] };

        #pragma unroll
        for (int i = 0; i < 3; i++) {
          witness1[warp_idx * 3 + i] =
            getCoord(bd1, idx0[0], i) * a0 +
            getCoord(bd1, idx1[0], i) * a1 +
            getCoord(bd1, idx2[0], i) * a2;
          witness2[warp_idx * 3 + i] =
            getCoord(bd2, idx0[1], i) * a0 +
            getCoord(bd2, idx1[1], i) * a1 +
            getCoord(bd2, idx2[1], i) * a2;
        }

        distances[warp_idx] = -closest_distance;
        
        // Store contact normal (points from polytope1 to polytope2)
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          contact_normals[warp_idx * 3 + i] = closest->normal[i];
        }
      }
      break;
    }

    // Add new vertex to polytope
    int new_vertex_id = -1;
    if (warp_lane_idx == 0) {
      new_vertex_id = poly.num_vertices;
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        poly.vertices[new_vertex_id][i] = new_vertex[i];
      }
      poly.vertex_indices[new_vertex_id][0] = new_vertex_idx[0];
      poly.vertex_indices[new_vertex_id][1] = new_vertex_idx[1];
      poly.num_vertices++;

      // Update centroid incrementally
      gkFloat n = (gkFloat)poly.num_vertices;
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        centroid[i] = centroid[i] * (n - 1.0f) / n + new_vertex[i] / n;
      }
    }

    __syncwarp(warp_mask);

    // Broadcast new vertex
    new_vertex_id = __shfl_sync(warp_mask, new_vertex_id, 0);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      poly.vertices[new_vertex_id][i] = __shfl_sync(warp_mask, poly.vertices[new_vertex_id][i], 0);
    }
    poly.vertex_indices[new_vertex_id][0] = __shfl_sync(warp_mask, poly.vertex_indices[new_vertex_id][0], 0);
    poly.vertex_indices[new_vertex_id][1] = __shfl_sync(warp_mask, poly.vertex_indices[new_vertex_id][1], 0);
    poly.num_vertices = __shfl_sync(warp_mask, poly.num_vertices, 0);
    centroid[0] = __shfl_sync(warp_mask, centroid[0], 0);
    centroid[1] = __shfl_sync(warp_mask, centroid[1], 0);
    centroid[2] = __shfl_sync(warp_mask, centroid[2], 0);

    // Find horizon edges (edges shared by exactly one invalid face and one valid face)
    // Only create new faces from horizon edges
    // Maybe some way to leverage multiple threads in warp to speed this up
    if (warp_lane_idx == 0) {
      // Mark visible faces
      for (int i = 0; i < poly.max_face_index; i++) {
        if (poly.faces[i].valid && is_face_visible(&poly, i, new_vertex)) {
          poly.faces[i].valid = false;
        }
      }

      // Collect horizon edges from invalid faces
      EPAEdge edges[MAX_EPA_FACES * 3];
      int num_edges = 0;

      for (int f = 0; f < poly.max_face_index; f++) {
        if (poly.faces[f].valid) continue;  // Skip valid faces

        // face was just invalidated so collect edges
        // But only if face was actually valid before this iteration
        // We need to check if it has valid vertex indices

        // Edge 0-1
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[0];
          edges[num_edges].v2 = poly.faces[f].v[1];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[0][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[0][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[1][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[1][1];
          edges[num_edges].valid = true;
          num_edges++;
        }

        // Edge 1-2
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[1];
          edges[num_edges].v2 = poly.faces[f].v[2];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[1][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[1][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[2][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[2][1];
          edges[num_edges].valid = true;
          num_edges++;
        }

        // Edge 2-0
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[2];
          edges[num_edges].v2 = poly.faces[f].v[0];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[2][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[2][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[0][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[0][1];
          edges[num_edges].valid = true;
          num_edges++;
        }
      }

      // Remove duplicate edges (edges shared by two removed faces)
      for (int i = 0; i < num_edges; i++) {
        if (!edges[i].valid) continue;

        for (int j = i + 1; j < num_edges; j++) {
          if (!edges[j].valid) continue;

          // Check if same edge (either direction)
          if ((edges[i].v1 == edges[j].v1 && edges[i].v2 == edges[j].v2) ||
            (edges[i].v1 == edges[j].v2 && edges[i].v2 == edges[j].v1)) {
            edges[i].valid = false;
            edges[j].valid = false;
          }
        }
      }

      // Create new faces from horizon edges
      for (int i = 0; i < num_edges; i++) {
        if (!edges[i].valid) continue;

        // Find next available face slot
        int new_face_idx = -1;
        #pragma unroll
        for (int j = 0; j < MAX_EPA_FACES; j++) {
          if (!poly.faces[j].valid) {
            new_face_idx = j;
            break;
          }
        }

        if (new_face_idx < 0 || new_face_idx >= MAX_EPA_FACES) break;

        // Create new face: edge horizon vertices + new vertex
        poly.faces[new_face_idx].v[0] = edges[i].v1;
        poly.faces[new_face_idx].v[1] = edges[i].v2;
        poly.faces[new_face_idx].v[2] = new_vertex_id;

        poly.faces[new_face_idx].v_idx[0][0] = edges[i].v_idx1[0];
        poly.faces[new_face_idx].v_idx[0][1] = edges[i].v_idx1[1];
        poly.faces[new_face_idx].v_idx[1][0] = edges[i].v_idx2[0];
        poly.faces[new_face_idx].v_idx[1][1] = edges[i].v_idx2[1];
        poly.faces[new_face_idx].v_idx[2][0] = new_vertex_idx[0];
        poly.faces[new_face_idx].v_idx[2][1] = new_vertex_idx[1];

        poly.faces[new_face_idx].valid = true;

        // Check winding and fix if necessary
        gkFloat* fv0 = poly.vertices[poly.faces[new_face_idx].v[0]];
        gkFloat* fv1 = poly.vertices[poly.faces[new_face_idx].v[1]];
        gkFloat* fv2 = poly.vertices[poly.faces[new_face_idx].v[2]];

        gkFloat fe0[3], fe1[3], fnormal[3];
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          fe0[c] = fv1[c] - fv0[c];
          fe1[c] = fv2[c] - fv0[c];
        }
        crossProduct(fe0, fe1, fnormal);

        // Vector from face to centroid
        gkFloat to_cent[3];
        #pragma unroll
        for (int c = 0; c < 3; c++) {
          to_cent[c] = centroid[c] - fv0[c];
        }

        // If normal points toward centroid flip winding
        if (dotProduct(fnormal, to_cent) > 0) {
          // Swap v[1] and v[2]
          int tmp_v = poly.faces[new_face_idx].v[1];
          poly.faces[new_face_idx].v[1] = poly.faces[new_face_idx].v[2];
          poly.faces[new_face_idx].v[2] = tmp_v;

          int tmp_idx0 = poly.faces[new_face_idx].v_idx[1][0];
          int tmp_idx1 = poly.faces[new_face_idx].v_idx[1][1];
          poly.faces[new_face_idx].v_idx[1][0] = poly.faces[new_face_idx].v_idx[2][0];
          poly.faces[new_face_idx].v_idx[1][1] = poly.faces[new_face_idx].v_idx[2][1];
          poly.faces[new_face_idx].v_idx[2][0] = tmp_idx0;
          poly.faces[new_face_idx].v_idx[2][1] = tmp_idx1;
        }

        // Update max face index
        if (new_face_idx >= poly.max_face_index) {
          poly.max_face_index = new_face_idx + 1;
        }
      }
    }

    __syncwarp(warp_mask);

    // Broadcast updated polytope state
    poly.max_face_index = __shfl_sync(warp_mask, poly.max_face_index, 0);

    for (int i = 0; i < poly.max_face_index; i++) {
      poly.faces[i].valid = __shfl_sync(warp_mask, poly.faces[i].valid ? 1 : 0, 0) != 0;
      if (poly.faces[i].valid) {
        #pragma unroll
        for (int v = 0; v < 3; v++) {
          poly.faces[i].v[v] = __shfl_sync(warp_mask, poly.faces[i].v[v], 0);
          poly.faces[i].v_idx[v][0] = __shfl_sync(warp_mask, poly.faces[i].v_idx[v][0], 0);
          poly.faces[i].v_idx[v][1] = __shfl_sync(warp_mask, poly.faces[i].v_idx[v][1], 0);
        }
      }
    }
  }

  // If we exited due to max iterations, recompute closest face and use it
  if (iteration >= max_iterations && warp_lane_idx == 0) {
    // Find closest face and compute result
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      compute_face_normal_distance(&poly, i, centroid);
    }

    int closest_face = -1;
    gkFloat closest_distance = 1e10f;
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      if (poly.faces[i].distance >= 0.0f && poly.faces[i].distance < closest_distance) {
        closest_distance = poly.faces[i].distance;
        closest_face = i;
      }
    }

    if (closest_face >= 0 && poly.faces[closest_face].valid) {
      EPAFace* closest = &poly.faces[closest_face];

      gkFloat* v0 = poly.vertices[closest->v[0]];
      gkFloat* v1 = poly.vertices[closest->v[1]];
      gkFloat* v2 = poly.vertices[closest->v[2]];

      gkFloat a0, a1, a2;
      compute_barycentric_origin(v0, v1, v2, &a0, &a1, &a2);

      int idx0[2] = { closest->v_idx[0][0], closest->v_idx[0][1] };
      int idx1[2] = { closest->v_idx[1][0], closest->v_idx[1][1] };
      int idx2[2] = { closest->v_idx[2][0], closest->v_idx[2][1] };

      #pragma unroll
      for (int i = 0; i < 3; i++) {
        witness1[warp_idx * 3 + i] =
          getCoord(bd1, idx0[0], i) * a0 +
          getCoord(bd1, idx1[0], i) * a1 +
          getCoord(bd1, idx2[0], i) * a2;
        witness2[warp_idx * 3 + i] =
          getCoord(bd2, idx0[1], i) * a0 +
          getCoord(bd2, idx1[1], i) * a1 +
          getCoord(bd2, idx2[1], i) * a2;
      }

      distances[warp_idx] = -closest_distance;
      
      // Store contact normal (points from polytope1 to polytope2)
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        contact_normals[warp_idx * 3 + i] = closest->normal[i];
      }
    }
  }
}

// ============================================================================
// HIGH-LEVEL API IMPLEMENTATIONS: Automatic memory management
// ============================================================================

void compute_minimum_distance(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances) {

    if (n <= 0) return;

    // Allocate device memory
    gkPolytope* d_bd1 = nullptr;
    gkPolytope* d_bd2 = nullptr;
    gkFloat* d_coord1 = nullptr;
    gkFloat* d_coord2 = nullptr;
    gkSimplex* d_simplices = nullptr;
    gkFloat* d_distances = nullptr;

    allocate_and_copy_device_arrays(n, bd1, bd2, &d_bd1, &d_bd2,
                                     &d_coord1, &d_coord2, &d_simplices, &d_distances);

    // Launch kernel
    compute_minimum_distance_device(n, d_bd1, d_bd2, d_simplices, d_distances);

    // Copy results back
    copy_results_from_device(n, d_simplices, d_distances, simplices, distances);

    // Free device memory
    free_device_arrays(d_bd1, d_bd2, d_coord1, d_coord2, d_simplices, d_distances);
}

void compute_epa(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances,
    gkFloat* witness1,
    gkFloat* witness2,
    gkFloat* contact_normals) {

    if (n <= 0) return;

    // Allocate device memory for GJK inputs
    gkPolytope* d_bd1 = nullptr;
    gkPolytope* d_bd2 = nullptr;
    gkFloat* d_coord1 = nullptr;
    gkFloat* d_coord2 = nullptr;
    gkSimplex* d_simplices = nullptr;
    gkFloat* d_distances = nullptr;

    allocate_and_copy_device_arrays(n, bd1, bd2, &d_bd1, &d_bd2,
                                     &d_coord1, &d_coord2, &d_simplices, &d_distances);

    // Allocate device memory for EPA outputs
    gkFloat* d_witness1 = nullptr;
    gkFloat* d_witness2 = nullptr;
    gkFloat* d_contact_normals = nullptr;

    allocate_epa_device_arrays(n, &d_witness1, &d_witness2,
                               contact_normals ? &d_contact_normals : nullptr);

    // Copy simplices to device
    cudaMemcpy(d_simplices, simplices, n * sizeof(gkSimplex), cudaMemcpyHostToDevice);

    // Launch EPA kernel
    compute_epa_device(n, d_bd1, d_bd2, d_simplices, d_distances,
                      d_witness1, d_witness2, d_contact_normals);

    // Copy results back
    copy_results_from_device(n, d_simplices, d_distances, simplices, distances);
    copy_epa_results_from_device(n, d_witness1, d_witness2, d_contact_normals,
                                 witness1, witness2, contact_normals);

    // Free device memory
    free_device_arrays(d_bd1, d_bd2, d_coord1, d_coord2, d_simplices, d_distances);
    free_epa_device_arrays(d_witness1, d_witness2, d_contact_normals);
}

void compute_gjk_epa(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances,
    gkFloat* witness1,
    gkFloat* witness2) {

    if (n <= 0) return;

    // Allocate device memory for GJK inputs
    gkPolytope* d_bd1 = nullptr;
    gkPolytope* d_bd2 = nullptr;
    gkFloat* d_coord1 = nullptr;
    gkFloat* d_coord2 = nullptr;
    gkSimplex* d_simplices = nullptr;
    gkFloat* d_distances = nullptr;

    allocate_and_copy_device_arrays(n, bd1, bd2, &d_bd1, &d_bd2,
                                     &d_coord1, &d_coord2, &d_simplices, &d_distances);

    // Allocate device memory for EPA outputs
    gkFloat* d_witness1 = nullptr;
    gkFloat* d_witness2 = nullptr;
    gkFloat* d_contact_normals = nullptr;  // Always allocated but not copied back

    allocate_epa_device_arrays(n, &d_witness1, &d_witness2, &d_contact_normals);

    // Launch GJK kernel
    compute_minimum_distance_device(n, d_bd1, d_bd2, d_simplices, d_distances);

    // Launch EPA kernel
    compute_epa_device(n, d_bd1, d_bd2, d_simplices, d_distances,
                      d_witness1, d_witness2, d_contact_normals);

    // Copy results back (contact_normals not copied, pass nullptr for host)
    copy_results_from_device(n, d_simplices, d_distances, simplices, distances);
    copy_epa_results_from_device(n, d_witness1, d_witness2, d_contact_normals,
                                 witness1, witness2, nullptr);

    // Free device memory
    free_device_arrays(d_bd1, d_bd2, d_coord1, d_coord2, d_simplices, d_distances);
    free_epa_device_arrays(d_witness1, d_witness2, d_contact_normals);
}

// ============================================================================
// MID-LEVEL API IMPLEMENTATIONS: Explicit memory management
// ============================================================================

void allocate_and_copy_device_arrays(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkPolytope** d_bd1,
    gkPolytope** d_bd2,
    gkFloat** d_coord1,
    gkFloat** d_coord2,
    gkSimplex** d_simplices,
    gkFloat** d_distances) {

    // Allocate device memory for polytope structures
    cudaMalloc((void**)d_bd1, n * sizeof(gkPolytope));
    cudaMalloc((void**)d_bd2, n * sizeof(gkPolytope));
    cudaMalloc((void**)d_simplices, n * sizeof(gkSimplex));
    cudaMalloc((void**)d_distances, n * sizeof(gkFloat));

    // Initialize simplices to zero (kernel expects clean state)
    cudaMemset(*d_simplices, 0, n * sizeof(gkSimplex));

    // Calculate total coordinate size needed
    int total_coords1 = 0;
    int total_coords2 = 0;
    for (int i = 0; i < n; i++) {
        total_coords1 += bd1[i].numpoints * 3;
        total_coords2 += bd2[i].numpoints * 3;
    }

    // Allocate concatenated coordinate arrays
    cudaMalloc((void**)d_coord1, total_coords1 * sizeof(gkFloat));
    cudaMalloc((void**)d_coord2, total_coords2 * sizeof(gkFloat));

    // Create temporary host arrays with updated pointers
    gkPolytope* temp_bd1 = new gkPolytope[n];
    gkPolytope* temp_bd2 = new gkPolytope[n];

    int offset1 = 0;
    int offset2 = 0;
    for (int i = 0; i < n; i++) {
        // Copy polytope metadata
        temp_bd1[i] = bd1[i];
        temp_bd2[i] = bd2[i];

        // Copy coordinate data to device
        int coord_size1 = bd1[i].numpoints * 3 * sizeof(gkFloat);
        int coord_size2 = bd2[i].numpoints * 3 * sizeof(gkFloat);

        cudaMemcpy(*d_coord1 + offset1, bd1[i].coord, coord_size1, cudaMemcpyHostToDevice);
        cudaMemcpy(*d_coord2 + offset2, bd2[i].coord, coord_size2, cudaMemcpyHostToDevice);

        // Update pointers to device memory locations
        temp_bd1[i].coord = *d_coord1 + offset1;
        temp_bd2[i].coord = *d_coord2 + offset2;

        offset1 += bd1[i].numpoints * 3;
        offset2 += bd2[i].numpoints * 3;
    }

    // Copy polytope structures to device
    cudaMemcpy(*d_bd1, temp_bd1, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_bd2, temp_bd2, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);

    // Cleanup temporary host arrays
    delete[] temp_bd1;
    delete[] temp_bd2;
}

void allocate_epa_device_arrays(
    const int n,
    gkFloat** d_witness1,
    gkFloat** d_witness2,
    gkFloat** d_contact_normals) {

    cudaMalloc((void**)d_witness1, n * 3 * sizeof(gkFloat));
    cudaMalloc((void**)d_witness2, n * 3 * sizeof(gkFloat));

    // ALWAYS allocate contact_normals (EPA kernel writes to it unconditionally)
    // Caller must provide a valid pointer, we'll only copy back if host pointer is non-null
    cudaMalloc((void**)d_contact_normals, n * 3 * sizeof(gkFloat));
}

void compute_minimum_distance_device(
    const int n,
    const gkPolytope* d_bd1,
    const gkPolytope* d_bd2,
    gkSimplex* d_simplices,
    gkFloat* d_distances) {

    // Each collision uses 16 threads (half-warp)
    int blockSize = 256;  // 256 threads = 16 collisions per block
    int collisionsPerBlock = blockSize / THREADS_PER_COMPUTATION;
    int numBlocks = (n + collisionsPerBlock - 1) / collisionsPerBlock;

    compute_minimum_distance_kernel<<<numBlocks, blockSize>>>(
        d_bd1, d_bd2, d_simplices, d_distances, n);

    cudaDeviceSynchronize();
}

void compute_epa_device(
    const int n,
    const gkPolytope* d_bd1,
    const gkPolytope* d_bd2,
    gkSimplex* d_simplices,
    gkFloat* d_distances,
    gkFloat* d_witness1,
    gkFloat* d_witness2,
    gkFloat* d_contact_normals) {

    // Each collision uses 32 threads (one full warp)
    int blockSize = 256;  // 256 threads = 8 collisions per block
    int collisionsPerBlock = blockSize / THREADS_PER_EPA;
    int numBlocks = (n + collisionsPerBlock - 1) / collisionsPerBlock;

    compute_epa_kernel<<<numBlocks, blockSize>>>(
        d_bd1, d_bd2, d_simplices, d_distances,
        d_witness1, d_witness2, d_contact_normals, n);

    cudaDeviceSynchronize();
}

void copy_results_from_device(
    const int n,
    const gkSimplex* d_simplices,
    const gkFloat* d_distances,
    gkSimplex* simplices,
    gkFloat* distances) {

    cudaMemcpy(distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
}

void copy_epa_results_from_device(
    const int n,
    const gkFloat* d_witness1,
    const gkFloat* d_witness2,
    const gkFloat* d_contact_normals,
    gkFloat* witness1,
    gkFloat* witness2,
    gkFloat* contact_normals) {

    cudaMemcpy(witness1, d_witness1, n * 3 * sizeof(gkFloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(witness2, d_witness2, n * 3 * sizeof(gkFloat), cudaMemcpyDeviceToHost);

    if (contact_normals != nullptr && d_contact_normals != nullptr) {
        cudaMemcpy(contact_normals, d_contact_normals, n * 3 * sizeof(gkFloat), cudaMemcpyDeviceToHost);
    }
}

void free_device_arrays(
    gkPolytope* d_bd1,
    gkPolytope* d_bd2,
    gkFloat* d_coord1,
    gkFloat* d_coord2,
    gkSimplex* d_simplices,
    gkFloat* d_distances) {

    if (d_bd1) cudaFree(d_bd1);
    if (d_bd2) cudaFree(d_bd2);
    if (d_coord1) cudaFree(d_coord1);
    if (d_coord2) cudaFree(d_coord2);
    if (d_simplices) cudaFree(d_simplices);
    if (d_distances) cudaFree(d_distances);
}

void free_epa_device_arrays(
    gkFloat* d_witness1,
    gkFloat* d_witness2,
    gkFloat* d_contact_normals) {

    if (d_witness1) cudaFree(d_witness1);
    if (d_witness2) cudaFree(d_witness2);
    if (d_contact_normals) cudaFree(d_contact_normals);
}

// ============================================================================
// INDEXED API IMPLEMENTATION
// ============================================================================

void compute_minimum_distance_indexed(
    const int num_polytopes,
    const int num_pairs,
    const gkPolytope* polytopes,
    const gkCollisionPair* pairs,
    gkSimplex* simplices,
    gkFloat* distances) {

    if (num_pairs <= 0 || num_polytopes <= 0) return;

    // Allocate device memory
    gkPolytope* d_polytopes = nullptr;
    gkCollisionPair* d_pairs = nullptr;
    gkSimplex* d_simplices = nullptr;
    gkFloat* d_distances = nullptr;
    gkFloat* d_coords = nullptr;

    // Allocate polytope and pair arrays
    cudaMalloc(&d_polytopes, num_polytopes * sizeof(gkPolytope));
    cudaMalloc(&d_pairs, num_pairs * sizeof(gkCollisionPair));
    cudaMalloc(&d_simplices, num_pairs * sizeof(gkSimplex));
    cudaMalloc(&d_distances, num_pairs * sizeof(gkFloat));

    // Calculate total coordinates size
    int total_verts = 0;
    for (int i = 0; i < num_polytopes; i++) {
        total_verts += polytopes[i].numpoints;
    }
    cudaMalloc(&d_coords, total_verts * 3 * sizeof(gkFloat));

    // Copy coordinates and update polytope pointers
    gkPolytope* temp_polytopes = (gkPolytope*)malloc(num_polytopes * sizeof(gkPolytope));
    int coord_offset = 0;
    for (int i = 0; i < num_polytopes; i++) {
        temp_polytopes[i] = polytopes[i];
        temp_polytopes[i].coord = d_coords + coord_offset;
        cudaMemcpy(temp_polytopes[i].coord, polytopes[i].coord,
                   polytopes[i].numpoints * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);
        coord_offset += polytopes[i].numpoints * 3;
    }

    // Copy polytope descriptors and pairs
    cudaMemcpy(d_polytopes, temp_polytopes, num_polytopes * sizeof(gkPolytope), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs, pairs, num_pairs * sizeof(gkCollisionPair), cudaMemcpyHostToDevice);
    cudaMemset(d_simplices, 0, num_pairs * sizeof(gkSimplex));

    // Launch kernel - same configuration as regular GJK
    int blockSize = 256;
    int collisionsPerBlock = blockSize / THREADS_PER_COMPUTATION;
    int numBlocks = (num_pairs + collisionsPerBlock - 1) / collisionsPerBlock;

    compute_minimum_distance_indexed_kernel<<<numBlocks, blockSize>>>(
        d_polytopes, d_pairs, d_simplices, d_distances, num_pairs);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(simplices, d_simplices, num_pairs * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, num_pairs * sizeof(gkFloat), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_polytopes);
    cudaFree(d_pairs);
    cudaFree(d_simplices);
    cudaFree(d_distances);
    cudaFree(d_coords);
    free(temp_polytopes);
}
