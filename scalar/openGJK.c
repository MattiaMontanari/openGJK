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

/**
 * @file openGJK.c
 * @author Mattia Montanari, Vismay Churiwala
 * @date 1 Jan 2022
 * @brief Source of OpenGJK and its fast sub-algorithm, 
 * an implementation of the Explanding Polytope Algorithm (EPA) in C++.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

#include "openGJK/openGJK.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math.h"

/** If instricuted, compile a mex function for Matlab.  */
#ifdef MATLAB_MEX_BUILD
#include "mex.h"
#else
#define mexPrintf printf
#endif

/** Maximum number of GJK iterations before termination */
#define GJK_MAX_ITERATIONS 25

/** Relative tolerance multiplier for convergence check */
#define GJK_EPSILON_REL_MULT 1e4

/** Absolute tolerance multiplier for convergence check */
#define GJK_EPSILON_ABS_MULT 1e2

/** Relative tolerance for convergence (scaled machine epsilon) */
#define GJK_EPSILON_REL ((gkFloat)(gkEpsilon * GJK_EPSILON_REL_MULT))

/** Absolute tolerance for convergence (scaled machine epsilon) */
#define GJK_EPSILON_ABS ((gkFloat)(gkEpsilon * GJK_EPSILON_ABS_MULT))

#define norm2(a) (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
#define dotProduct(a, b) (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

#define getCoord(body, index, component) body.coord[(index)][(component)]

#define S3Dregion1234() \
  v[0] = 0;             \
  v[1] = 0;             \
  v[2] = 0;             \
  s->nvrtx = 4;

#define select_1ik()                                             \
  s->nvrtx = 3;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = si[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = si_idx[t];         \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

#define select_1ij()                                             \
  s->nvrtx = 3;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = si[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = si_idx[t];         \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = sj[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sj_idx[t];

#define select_1jk()                                             \
  s->nvrtx = 3;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[2][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[2][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = sj[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = sj_idx[t];         \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

#define select_1i()                                              \
  s->nvrtx = 2;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = si[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = si_idx[t];

#define select_1j()                                              \
  s->nvrtx = 2;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = sj[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sj_idx[t];

#define select_1k()                                              \
  s->nvrtx = 2;                                                  \
  for (t = 0; t < 3; t++) s->vrtx[1][t] = s->vrtx[3][t];         \
  for (t = 0; t < 2; t++) s->vrtx_idx[1][t] = s->vrtx_idx[3][t]; \
  for (t = 0; t < 3; t++) s->vrtx[0][t] = sk[t];                 \
  for (t = 0; t < 2; t++) s->vrtx_idx[0][t] = sk_idx[t];

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

inline static gkFloat determinant(const gkFloat* restrict p,
                                  const gkFloat* restrict q,
                                  const gkFloat* restrict r) {
  return p[0] * ((q[1] * r[2]) - (r[1] * q[2])) -
         p[1] * (q[0] * r[2] - r[0] * q[2]) +
         p[2] * (q[0] * r[1] - r[0] * q[1]);
}

inline static void crossProduct(const gkFloat* restrict a,
                                const gkFloat* restrict b,
                                gkFloat* restrict c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

inline static void projectOnLine(const gkFloat* restrict p,
                                 const gkFloat* restrict q,
                                 gkFloat* restrict v) {
  gkFloat pq[3];
  pq[0] = p[0] - q[0];
  pq[1] = p[1] - q[1];
  pq[2] = p[2] - q[2];

  const gkFloat tmp = dotProduct(p, pq) / dotProduct(pq, pq);

  for (int i = 0; i < 3; i++) {
    v[i] = p[i] - pq[i] * tmp;
  }
}

inline static void projectOnPlane(const gkFloat* restrict p,
                                  const gkFloat* restrict q,
                                  const gkFloat* restrict r,
                                  gkFloat* restrict v) {
  gkFloat n[3], pq[3], pr[3];

  for (int i = 0; i < 3; i++) {
    pq[i] = p[i] - q[i];
  }
  for (int i = 0; i < 3; i++) {
    pr[i] = p[i] - r[i];
  }

  crossProduct(pq, pr, n);
  const gkFloat tmp = dotProduct(n, p) / dotProduct(n, n);

  for (int i = 0; i < 3; i++) {
    v[i] = n[i] * tmp;
  }
}

inline static int hff1(const gkFloat* restrict p, const gkFloat* restrict q) {
  gkFloat tmp = 0;

  for (int i = 0; i < 3; i++) {
    tmp += (p[i] * p[i] - p[i] * q[i]);
  }

  if (tmp > 0) {
    return 1;
  }
  return 0;
}

inline static int hff2(const gkFloat* restrict p, const gkFloat* restrict q,
                       const gkFloat* restrict r) {
  gkFloat ntmp[3];
  gkFloat n[3], pq[3], pr[3];

  for (int i = 0; i < 3; i++) {
    pq[i] = q[i] - p[i];
  }
  for (int i = 0; i < 3; i++) {
    pr[i] = r[i] - p[i];
  }

  crossProduct(pq, pr, ntmp);
  crossProduct(pq, ntmp, n);

  return dotProduct(p, n) < 0;
}

inline static int hff3(const gkFloat* restrict p, const gkFloat* restrict q,
                       const gkFloat* restrict r) {
  gkFloat n[3];
  crossProduct(q, r, n);
  return dotProduct(p, n) <= 0;  // discard s if true
}

inline static void S1D(gkSimplex* s, gkFloat* v) {
  const gkFloat* restrict s1p = s->vrtx[1];
  const gkFloat* restrict s2p = s->vrtx[0];

  if (hff1(s1p, s2p)) {
    projectOnLine(s1p, s2p, v);
    return;
  } else {
    S1Dregion1();
    return;
  }
}

inline static void S2D(gkSimplex* s, gkFloat* v) {
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
          projectOnPlane(s1p, s2p, s3p, v);
          return;
        } else {
          projectOnLine(s1p, s3p, v);
          S2Dregion13();
          return;
        }
      } else {
        projectOnPlane(s1p, s2p, s3p, v);
        return;
      }
    } else {
      projectOnLine(s1p, s2p, v);
      S2Dregion12();
      return;
    }
  } else if (hff1f_s13) {
    const int hff2f_32 = !hff2(s1p, s3p, s2p);
    if (hff2f_32) {
      projectOnPlane(s1p, s2p, s3p, v);
      return;
    } else {
      projectOnLine(s1p, s3p, v);
      S2Dregion13();
      return;
    }
  } else {
    S2Dregion1();
    return;
  }
}

inline static void S3D(gkSimplex* s, gkFloat* v) {
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

  int hff1_tests[3];
  hff1_tests[2] = hff1(s1, s2);
  hff1_tests[1] = hff1(s1, s3);
  hff1_tests[0] = hff1(s1, s4);
  testLineThree = hff1(s1, s3);
  testLineFour = hff1(s1, s4);

  dotTotal = hff1(s1, s2) + testLineThree + testLineFour;
  if (dotTotal == 0) { /* case 0.0 -------------------------------------- */
    S3Dregion1();
    return;
  }

  const gkFloat det134 = determinant(s1s3, s1s4, s1s2);
  const int sss = (det134 <= 0);

  testPlaneTwo = hff3(s1, s3, s4) - sss;
  testPlaneTwo = testPlaneTwo * testPlaneTwo;
  testPlaneThree = hff3(s1, s4, s2) - sss;
  testPlaneThree = testPlaneThree * testPlaneThree;
  testPlaneFour = hff3(s1, s2, s3) - sss;
  testPlaneFour = testPlaneFour * testPlaneFour;

  switch (testPlaneTwo + testPlaneThree + testPlaneFour) {
    case 3:
      S3Dregion1234();
      break;

    case 2:
      s->nvrtx = 3;
      if (!testPlaneTwo) {
        for (i = 0; i < 3; i++) {
          s->vrtx[2][i] = s->vrtx[3][i];
        }
        for (i = 0; i < 2; i++) {
          s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
        }
      } else if (!testPlaneThree) {
        for (i = 0; i < 3; i++) {
          s->vrtx[1][i] = s2[i];
          s->vrtx[2][i] = s->vrtx[3][i];
        }
        for (i = 0; i < 2; i++) {
          s->vrtx_idx[1][i] = s2_idx[i];
          s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
        }
      } else if (!testPlaneFour) {
        for (i = 0; i < 3; i++) {
          s->vrtx[0][i] = s3[i];
          s->vrtx[1][i] = s2[i];
          s->vrtx[2][i] = s->vrtx[3][i];
        }
        for (i = 0; i < 2; i++) {
          s->vrtx_idx[0][i] = s3_idx[i];
          s->vrtx_idx[1][i] = s2_idx[i];
          s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
        }
      }
      S2D(s, v);
      break;
    case 1:
      s->nvrtx = 3;
      if (testPlaneTwo) {
        k = 2;
        i = 1;
        j = 0;
      } else if (testPlaneThree) {
        k = 1;
        i = 0;
        j = 2;
      } else {
        k = 0;
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
          } else if (!hff2(s1, sk, sj)) {
            select_1jk();
            projectOnPlane(s1, sj, sk, v);
          } else {
            select_1k();
            projectOnLine(s1, sk, v);
          }
        } else if (hff1_tests[i]) {
          if (!hff2(s1, si, sk)) {
            select_1ik();
            projectOnPlane(s1, si, sk, v);
          } else {
            select_1i();
            projectOnLine(s1, si, v);
          }
        } else {
          if (!hff2(s1, sj, sk)) {
            select_1jk();
            projectOnPlane(s1, sj, sk, v);
          } else {
            select_1j();
            projectOnLine(s1, sj, v);
          }
        }
      } else if (dotTotal == 2) {
        if (hff1_tests[i]) {
          if (!hff2(s1, sk, si)) {
            if (!hff2(s1, si, sk)) {
              select_1ik();
              projectOnPlane(s1, si, sk, v);
            } else {
              select_1k();
              projectOnLine(s1, sk, v);
            }
          } else {
            if (!hff2(s1, sk, sj)) {
              select_1jk();
              projectOnPlane(s1, sj, sk, v);
            } else {
              select_1k();
              projectOnLine(s1, sk, v);
            }
          }
        } else if (hff1_tests[j]) {
          if (!hff2(s1, sk, sj)) {
            if (!hff2(s1, sj, sk)) {
              select_1jk();
              projectOnPlane(s1, sj, sk, v);
            } else {
              select_1j();
              projectOnLine(s1, sj, v);
            }
          } else {
            if (!hff2(s1, sk, si)) {
              select_1ik();
              projectOnPlane(s1, si, sk, v);
            } else {
              select_1k();
              projectOnLine(s1, sk, v);
            }
          }
        }

      } else if (dotTotal == 3) {
        int hff2_ik = hff2(s1, si, sk);
        int hff2_jk = hff2(s1, sj, sk);
        int hff2_ki = hff2(s1, sk, si);
        int hff2_kj = hff2(s1, sk, sj);

        if (hff2_ki == 0 && hff2_kj == 0) {
          mexPrintf("\n\n UNEXPECTED VALUES!!! \n\n");
        }
        if (hff2_ki == 1 && hff2_kj == 1) {
          select_1k();
          projectOnLine(s1, sk, v);
        } else if (hff2_ki) {
          if (hff2_jk) {
            select_1j();
            projectOnLine(s1, sj, v);
          } else {
            select_1jk();
            projectOnPlane(s1, sk, sj, v);
          }
        } else {
          if (hff2_ik) {
            select_1i();
            projectOnLine(s1, si, v);
          } else {
            select_1ik();
            projectOnPlane(s1, sk, si, v);
          }
        }
      }
      break;

    case 0:
      if (dotTotal == 1) {
        if (testLineThree) {
          k = 2;
          i = 1;
          j = 0;
        } else if (testLineFour) {
          k = 1;
          i = 0;
          j = 2;
        } else {
          k = 0;
          i = 2;
          j = 1;
        }
        getvrtxidx(si, si_idx, i);
        getvrtxidx(sj, sj_idx, j);
        getvrtxidx(sk, sk_idx, k);

        if (!hff2(s1, si, sj)) {
          select_1ij();
          projectOnPlane(s1, si, sj, v);
        } else if (!hff2(s1, si, sk)) {
          select_1ik();
          projectOnPlane(s1, si, sk, v);
        } else {
          select_1i();
          projectOnLine(s1, si, v);
        }
      } else if (dotTotal == 2) {
        s->nvrtx = 3;
        if (!testLineThree) {
          k = 2;
          i = 1;
          j = 0;
        } else if (!testLineFour) {
          k = 1;
          i = 0;
          j = 2;
        } else {
          k = 0;
          i = 2;
          j = 1;
        }
        getvrtxidx(si, si_idx, i);
        getvrtxidx(sj, sj_idx, j);
        getvrtxidx(sk, sk_idx, k);

        if (!hff2(s1, sj, sk)) {
          if (!hff2(s1, sk, sj)) {
            select_1jk();
            projectOnPlane(s1, sj, sk, v);
          } else if (!hff2(s1, sk, si)) {
            select_1ik();
            projectOnPlane(s1, sk, si, v);
          } else {
            select_1k();
            projectOnLine(s1, sk, v);
          }
        } else if (!hff2(s1, sj, si)) {
          select_1ij();
          projectOnPlane(s1, si, sj, v);
        } else {
          select_1j();
          projectOnLine(s1, sj, v);
        }
      }
      break;
    default:
      mexPrintf("\nERROR:\tunhandled");
  }
}

inline static void support(gkPolytope* restrict body,
                           const gkFloat* restrict v) {
  gkFloat s, maxs;
  gkFloat* vrt;
  int better = -1;

  maxs = dotProduct(body->s, v);

  for (int i = 0; i < body->numpoints; ++i) {
    vrt = body->coord[i];
    s = dotProduct(vrt, v);
    if (s > maxs) {
      maxs = s;
      better = i;
    }
  }

  if (better != -1) {
    body->s[0] = body->coord[better][0];
    body->s[1] = body->coord[better][1];
    body->s[2] = body->coord[better][2];
    body->s_idx = better;
  }
}

inline static void subalgorithm(gkSimplex* s, gkFloat* v) {
  switch (s->nvrtx) {
    case 4:
      S3D(s, v);
      break;
    case 3:
      S2D(s, v);
      break;
    case 2:
      S1D(s, v);
      break;
    default:
      mexPrintf("\nERROR:\t invalid simplex\n");
  }
}

inline static void W0D(const gkPolytope* bd1, const gkPolytope* bd2,
                       gkSimplex* smp) {
  const gkFloat* w00 = bd1->coord[smp->vrtx_idx[0][0]];
  const gkFloat* w01 = bd2->coord[smp->vrtx_idx[0][1]];
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t];
    smp->witnesses[1][t] = w01[t];
  }
}

inline static void W1D(const gkPolytope* bd1, const gkPolytope* bd2,
                       gkSimplex* smp) {
  gkFloat pq[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];

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
  const gkFloat a0 = (gkFloat)1.0 - a1;

  // Compute witness points
  const gkFloat* w00 = bd1->coord[smp->vrtx_idx[0][0]];
  const gkFloat* w01 = bd2->coord[smp->vrtx_idx[0][1]];
  const gkFloat* w10 = bd1->coord[smp->vrtx_idx[1][0]];
  const gkFloat* w11 = bd2->coord[smp->vrtx_idx[1][1]];
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t] * a0 + w10[t] * a1;
    smp->witnesses[1][t] = w01[t] * a0 + w11[t] * a1;
  }
}

inline static void W2D(const gkPolytope* bd1, const gkPolytope* bd2,
                       gkSimplex* smp) {
  gkFloat pq[3], pr[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];
  const gkFloat* r = smp->vrtx[2];

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
  const gkFloat a0 = (gkFloat)1.0 - a1 - a2;

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
  } else if (a1 < gkEpsilon) {
    smp->nvrtx = 2;
    smp->vrtx[1][0] = smp->vrtx[2][0];
    smp->vrtx[1][1] = smp->vrtx[2][1];
    smp->vrtx[1][2] = smp->vrtx[2][2];
    smp->vrtx_idx[1][0] = smp->vrtx_idx[2][0];
    smp->vrtx_idx[1][1] = smp->vrtx_idx[2][1];
    W1D(bd1, bd2, smp);
  } else if (a2 < gkEpsilon) {
    smp->nvrtx = 2;
    W1D(bd1, bd2, smp);
  }

  // Compute witness points
  // This is done by blending the source points using
  // the barycentric coordinates
  const gkFloat* w00 = bd1->coord[smp->vrtx_idx[0][0]];
  const gkFloat* w01 = bd2->coord[smp->vrtx_idx[0][1]];
  const gkFloat* w10 = bd1->coord[smp->vrtx_idx[1][0]];
  const gkFloat* w11 = bd2->coord[smp->vrtx_idx[1][1]];
  const gkFloat* w20 = bd1->coord[smp->vrtx_idx[2][0]];
  const gkFloat* w21 = bd2->coord[smp->vrtx_idx[2][1]];
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] = w00[t] * a0 + w10[t] * a1 + w20[t] * a2;
    smp->witnesses[1][t] = w01[t] * a0 + w11[t] * a1 + w21[t] * a2;
  }
}

inline static void W3D(const gkPolytope* bd1, const gkPolytope* bd2,
                       gkSimplex* smp) {
  gkFloat pq[3], pr[3], ps[3], po[3];

  const gkFloat* p = smp->vrtx[0];
  const gkFloat* q = smp->vrtx[1];
  const gkFloat* r = smp->vrtx[2];
  const gkFloat* s = smp->vrtx[3];

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
  const gkFloat a0 = (gkFloat)1.0 - a1 - a2 - a3;

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
  } else if (a1 < gkEpsilon) {
    smp->nvrtx = 3;
    smp->vrtx[1][0] = smp->vrtx[3][0];
    smp->vrtx[1][1] = smp->vrtx[3][1];
    smp->vrtx[1][2] = smp->vrtx[3][2];
    smp->vrtx_idx[1][0] = smp->vrtx_idx[3][0];
    smp->vrtx_idx[1][1] = smp->vrtx_idx[3][1];
    W2D(bd1, bd2, smp);
  } else if (a2 < gkEpsilon) {
    smp->nvrtx = 3;
    smp->vrtx[2][0] = smp->vrtx[3][0];
    smp->vrtx[2][1] = smp->vrtx[3][1];
    smp->vrtx[2][2] = smp->vrtx[3][2];
    smp->vrtx_idx[2][0] = smp->vrtx_idx[3][0];
    smp->vrtx_idx[2][1] = smp->vrtx_idx[3][1];
    W2D(bd1, bd2, smp);
  } else if (a3 < gkEpsilon) {
    smp->nvrtx = 3;
    W2D(bd1, bd2, smp);
  }

  // Compute witness points
  // This is done by blending the original points using
  // the barycentric coordinates
  const gkFloat* w00 = bd1->coord[smp->vrtx_idx[0][0]];
  const gkFloat* w01 = bd2->coord[smp->vrtx_idx[0][1]];
  const gkFloat* w10 = bd1->coord[smp->vrtx_idx[1][0]];
  const gkFloat* w11 = bd2->coord[smp->vrtx_idx[1][1]];
  const gkFloat* w20 = bd1->coord[smp->vrtx_idx[2][0]];
  const gkFloat* w21 = bd2->coord[smp->vrtx_idx[2][1]];
  const gkFloat* w30 = bd1->coord[smp->vrtx_idx[3][0]];
  const gkFloat* w31 = bd2->coord[smp->vrtx_idx[3][1]];
  for (int t = 0; t < 3; t++) {
    smp->witnesses[0][t] =
        w00[t] * a0 + w10[t] * a1 + w20[t] * a2 + w30[t] * a3;
    smp->witnesses[1][t] =
        w01[t] * a0 + w11[t] * a1 + w21[t] * a2 + w31[t] * a3;
  }
}

inline static void compute_witnesses(const gkPolytope* bd1,
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
      mexPrintf("\nERROR:\t invalid simplex\n");
  }
}

gkFloat compute_minimum_distance(gkPolytope bd1, gkPolytope bd2,
                                 gkSimplex* restrict s) {
  unsigned int k = 0;
  const int mk = GJK_MAX_ITERATIONS;
  const gkFloat eps_rel = GJK_EPSILON_REL;
  const gkFloat eps_tot = GJK_EPSILON_ABS;

  const gkFloat eps_rel2 = eps_rel * eps_rel;
  unsigned int i;
  gkFloat w[3];
  int w_idx[2];
  gkFloat v[3];
  gkFloat vminus[3];
  gkFloat norm2Wmax = 0;

  /* Initialise search direction */
  v[0] = bd1.coord[0][0] - bd2.coord[0][0];
  v[1] = bd1.coord[0][1] - bd2.coord[0][1];
  v[2] = bd1.coord[0][2] - bd2.coord[0][2];

  /* Initialise simplex */
  s->nvrtx = 1;
  for (int t = 0; t < 3; ++t) {
    s->vrtx[0][t] = v[t];
  }

  s->vrtx_idx[0][0] = 0;
  s->vrtx_idx[0][1] = 0;

  for (int t = 0; t < 3; ++t) {
    bd1.s[t] = bd1.coord[0][t];
  }

  bd1.s_idx = 0;

  for (int t = 0; t < 3; ++t) {
    bd2.s[t] = bd2.coord[0][t];
  }

  bd2.s_idx = 0;

  /* Begin GJK iteration */
  do {
    k++;

    /* Update negative search direction */
    for (int t = 0; t < 3; ++t) {
      vminus[t] = -v[t];
    }

    /* Support function */
    support(&bd1, vminus);
    support(&bd2, v);
    for (int t = 0; t < 3; ++t) {
      w[t] = bd1.s[t] - bd2.s[t];
    }
    w_idx[0] = bd1.s_idx;
    w_idx[1] = bd2.s_idx;

    /* Test first exit condition (new point already in simplex/can't move
     * further) */
    gkFloat exeedtol_rel = (norm2(v) - dotProduct(v, w));
    if (exeedtol_rel <= (eps_rel * norm2(v)) ||
        exeedtol_rel < GJK_EPSILON_ABS) {
      break;
    }

    if (norm2(v) < eps_rel2) {
      break;
    }

    /* Add new vertex to simplex */
    i = s->nvrtx;
    for (int t = 0; t < 3; ++t) {
      s->vrtx[i][t] = w[t];
    }
    s->vrtx_idx[i][0] = w_idx[0];
    s->vrtx_idx[i][1] = w_idx[1];
    s->nvrtx++;

    /* Invoke distance sub-algorithm */
    subalgorithm(s, v);

    /* Test */
    for (int jj = 0; jj < s->nvrtx; jj++) {
      gkFloat tesnorm = norm2(s->vrtx[jj]);
      if (tesnorm > norm2Wmax) {
        norm2Wmax = tesnorm;
      }
    }

    if ((norm2(v) <= (eps_tot * eps_tot * norm2Wmax))) {
      break;
    }

  } while ((s->nvrtx != 4) && (k != mk));

  if (k == mk) {
    mexPrintf(
        "\n * * * * * * * * * * * * MAXIMUM ITERATION NUMBER REACHED!!!  "
        " * * * * * * * * * * * * * * \n");
  }

  compute_witnesses(&bd1, &bd2, s);
  return gkSqrt(norm2(v));
}

//*******************************************************************************************
// EPA Algorithm
//*******************************************************************************************

// Compute face normal and distance of face from origin.
// Winding is already fixed at face creation time, so the cross product
// direction is trusted directly — no centroid-based orientation check needed.
inline static void compute_face_normal_distance(EPAPolytope* poly, int face_idx) {
  EPAFace* face = &poly->faces[face_idx];

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat* v1 = poly->vertices[face->v[1]];
  gkFloat* v2 = poly->vertices[face->v[2]];

  gkFloat e0[3], e1[3];
  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
  }

  crossProduct(e0, e1, face->normal);
  gkFloat norm_sq = norm2(face->normal);

  if (norm_sq > gkEpsilon * gkEpsilon) {
    gkFloat norm = gkSqrt(norm_sq);
    for (int i = 0; i < 3; i++) {
      face->normal[i] /= norm;
    }

    face->distance = dotProduct(face->normal, v0);

    // Safety: origin should be inside polytope so distance must be positive.
    // If negative, the winding was wrong — flip to recover.
    if (face->distance < 0) {
      for (int i = 0; i < 3; i++) {
        face->normal[i] = -face->normal[i];
      }
      face->distance = -face->distance;
    }
  }
  else {
    face->valid = false;
    face->distance = (gkFloat)1e10;
  }
}

// Check if a face is visible from a point (point is on positive side of face) Needed to determine which faces to restructure when vertex is added
inline static bool is_face_visible(EPAPolytope* poly, int face_idx, const gkFloat* point) {
  EPAFace* face = &poly->faces[face_idx];
  if (!face->valid) return false;

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat diff[3];
  for (int i = 0; i < 3; i++) diff[i] = point[i] - v0[i];
  return dotProduct(face->normal, diff) > gkEpsilon;
}


// Initialize EPA polytope from GJK simplex (should be a tetrahedron)
inline static void init_epa_polytope(EPAPolytope* poly, const gkSimplex* simplex, gkFloat* centroid) {
  memset(poly->faces, 0, sizeof(poly->faces));

  // Copy vertices from simplex
  poly->num_vertices = 4;
  for (int i = 0; i < 4; i++) {
    
    for (int j = 0; j < 3; j++) {
      poly->vertices[i][j] = simplex->vrtx[i][j];
    }
    poly->vertex_indices[i][0] = simplex->vrtx_idx[i][0];
    poly->vertex_indices[i][1] = simplex->vrtx_idx[i][1];
  }

  // Compute centroid of the tetrahedron
  centroid[0] = centroid[1] = centroid[2] = 0.0f;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      centroid[j] += poly->vertices[i][j] * 0.25f;
    }
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
  
  for (int f = 0; f < 4; f++) {
    for (int v = 0; v < 3; v++) {
      int vi = poly->faces[f].v[v];
      poly->faces[f].v_idx[v][0] = simplex->vrtx_idx[vi][0];
      poly->faces[f].v_idx[v][1] = simplex->vrtx_idx[vi][1];
    }
  }

  // Compute normals and fix winding
  for (int f = 0; f < 4; f++) {
    gkFloat* v0 = poly->vertices[poly->faces[f].v[0]];
    gkFloat* v1 = poly->vertices[poly->faces[f].v[1]];
    gkFloat* v2 = poly->vertices[poly->faces[f].v[2]];

    gkFloat e0[3], e1[3], normal[3];
    
    for (int i = 0; i < 3; i++) {
      e0[i] = v1[i] - v0[i];
      e1[i] = v2[i] - v0[i];
    }
    crossProduct(e0, e1, normal);

    // If normal points toward centroid need to flip the winding
    gkFloat to_centroid[3];
    for (int i = 0; i < 3; i++) to_centroid[i] = centroid[i] - v0[i];
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
inline static void compute_barycentric_origin(
  const gkFloat* v0, const gkFloat* v1, const gkFloat* v2,
  gkFloat* a0, gkFloat* a1, gkFloat* a2) {

  // Compute vectors
  gkFloat e0[3], e1[3];

  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
  }

  // Compute dot products for barycentric coords
  gkFloat d00 = dotProduct(e0, e0);
  gkFloat d01 = dotProduct(e0, e1);
  gkFloat d11 = dotProduct(e1, e1);
  gkFloat d20 = -dotProduct(v0, e0);
  gkFloat d21 = -dotProduct(v0, e1);

  gkFloat denom = d00 * d11 - d01 * d01;

  if (gkFabs(denom) < gkEpsilon) {
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
    gkFloat e12[3];
    for (int i = 0; i < 3; i++) e12[i] = v2[i] - v1[i];
    gkFloat t = -dotProduct(v1, e12) / dotProduct(e12, e12);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
    *a0 = 0;
    *a1 = (gkFloat)1.0 - t;
    *a2 = t;
  }
  else if (u < 0) {
    // Origin projects outside edge v0-v2
    gkFloat t = -dotProduct(v0, e1) / dotProduct(e1, e1);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
    *a0 = (gkFloat)1.0 - t;
    *a1 = 0;
    *a2 = t;
  }
  else if (v < 0) {
    // Origin projects outside edge v0-v1
    gkFloat t = -dotProduct(v0, e0) / dotProduct(e0, e0);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
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

// Support function for EPA basicallly GJK one but only care about minkowski difference point
inline static void support_epa(gkPolytope body1, gkPolytope body2,
  const gkFloat* direction, gkFloat* result, int* result_idx) {

  gkFloat local_max1 = -1e10f;
  gkFloat local_max2 = -1e10f;
  int local_best1 = -1;
  int local_best2 = -1;

  // Search body1
  for (int i = 0; i < body1.numpoints; i++) {
    gkFloat s = getCoord(body1, i, 0) * direction[0]
              + getCoord(body1, i, 1) * direction[1]
              + getCoord(body1, i, 2) * direction[2];
    if (s > local_max1) {
      local_max1 = s;
      local_best1 = i;
    }
  }

  // Search body2 in opposite direction
  for (int i = 0; i < body2.numpoints; i++) {
    gkFloat s = getCoord(body2, i, 0) * direction[0]
              + getCoord(body2, i, 1) * direction[1]
              + getCoord(body2, i, 2) * direction[2];
    if (-s > local_max2) {
      local_max2 = -s;
      local_best2 = i;
    }
  }
  // Compute Minkowski difference point
  if (local_best1 >= 0 && local_best2 >= 0) {
    for (int i = 0; i < 3; i++) {
      result[i] = getCoord(body1, local_best1, i) - getCoord(body2, local_best2, i);
    }
    result_idx[0] = local_best1;
    result_idx[1] = local_best2;
  }
}

static void set_contact_normal(const gkFloat* w1, const gkFloat* w2, gkFloat* contact_normal) {
  gkFloat d[3] = { w2[0] - w1[0], w2[1] - w1[1], w2[2] - w1[2] };
  gkFloat n = gkSqrt(norm2(d));
  if (n > gkEpsilon) {
    contact_normal[0] = d[0] / n;
    contact_normal[1] = d[1] / n;
    contact_normal[2] = d[2] / n;
  } else {
    contact_normal[0] = 1.0f; contact_normal[1] = 0.0f; contact_normal[2] = 0.0f;
  }
}


//*******************************************************************************************
// Entry point to the EPA Implementation
//*******************************************************************************************

  void computeCollisionInformation(
  gkPolytope bd1,
  gkPolytope bd2,
  gkSimplex* simplex,
  gkFloat* distance,
  gkFloat contact_normal[3]) {

  // if distance isn't 0 didn't detect collision - skip EPA
  if (*distance > gkEpsilon) {
    set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
    return;
  }

  // If GJK returned a degenerate simplex, rebuild it properly for EPA
  if (simplex->nvrtx != 4) {
    // Need to get it up to 4 vertices
    if (simplex->nvrtx == 1) {
      // Grow simplex from a single point: fire a support in some direction.
      // We use current simplex point for new direction for the
      // support; if this does not produce a new point, treat penetration as 0.
        gkFloat new_vertex[3];
        int new_vertex_idx[2];
        const gkFloat eps_sq = gkEpsilon * gkEpsilon;

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, simplex->vrtx[0], new_vertex, new_vertex_idx);

        // Check if this is a new point relative to the existing simplex vertex.
        bool is_new = true;
        gkFloat dx = new_vertex[0] - simplex->vrtx[0][0];
        gkFloat dy = new_vertex[1] - simplex->vrtx[0][1];
        gkFloat dz = new_vertex[2] - simplex->vrtx[0][2];
        gkFloat d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < eps_sq) {
          is_new = false;
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 2;
        }
        else {
          // No new support point means penetration depth effectively zero.
          *distance = 0.0f;
          for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
            simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
          }
          set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
          return;
        }
    }
    if (simplex->nvrtx == 2) {
      // Grow simplex from an edge: fire a support in a direction perpendicular
      // to the edge. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;

    gkFloat edge[3];

    for (int c = 0; c < 3; ++c) {
        edge[c] = simplex->vrtx[1][c] - simplex->vrtx[0][c];
    }

    // Build a perpindicular
    gkFloat axis[3] = { 1.0f, 0.0f, 0.0f };
    gkFloat edge_norm = gkSqrt(norm2(edge));
    if (edge_norm > gkEpsilon && gkFabs(edge[0]) > 0.9f * edge_norm) {
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

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
        // Check if this is a new point relative to both existing simplex vertices.
        bool is_new = true;
        for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 3;
        }
        else {
          // No new support point means penetration depth effectively zero.
          *distance = 0.0f;
          for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
            simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
          }
          set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
          return;
        }
    }
    if (simplex->nvrtx == 3) {
      // Grow simplex from a triangle: fire a support in the direction of the
      // triangle normal. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;

        gkFloat e0[3], e1[3];

        for (int c = 0; c < 3; ++c) {
            e0[c] = simplex->vrtx[1][c] - simplex->vrtx[0][c];
            e1[c] = simplex->vrtx[2][c] - simplex->vrtx[0][c];
        }
        // dir = e0 x e1 (normal to the triangle)
        crossProduct(e0, e1, dir);

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
        // Check if this is a new point relative to all three existing simplex vertices.
        bool is_new = true;
        for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 4;
        }
        else {
          // Try opposite direction
          dir[0] = -dir[0];
          dir[1] = -dir[1];
          dir[2] = -dir[2];
        }

      // If first direction didn't work, try opposite
      if (simplex->nvrtx == 3) {
        support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
          bool is_new = true;
          for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
            gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
            gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
            gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
            gkFloat d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < eps_sq) {
              is_new = false;
              break;
            }
          }

          if (is_new) {
            int idx = simplex->nvrtx;
            
            for (int c = 0; c < 3; ++c) {
              simplex->vrtx[idx][c] = new_vertex[c];
            }
            simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
            simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
            simplex->nvrtx = 4;
          }
          else {
            *distance = 0.0f;
            for (int c = 0; c < 3; ++c) {
              simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
              simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
            }
            set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
            return;
          }
      }
    }

    // If we still don't have 4 vertices, abort
    if (simplex->nvrtx != 4) {
        *distance = 0.0f;
        // Set witness points from best available simplex vertex
        int best = simplex->nvrtx > 0 ? simplex->nvrtx - 1 : 0;
        for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, simplex->vrtx_idx[best][0], c);
            simplex->witnesses[1][c] = getCoord(bd2, simplex->vrtx_idx[best][1], c);
        }
        set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
        return;
    }
  }

  // On to actual EPA alg with a valid tetrahedron simplex
  // Initialize EPA polytope from simplex
  EPAPolytope poly;
  gkFloat centroid[3];
    init_epa_polytope(&poly, simplex, centroid);

  // EPA iteration parameters
  const int max_iterations = 64;
  const gkFloat tolerance = GJK_EPSILON_ABS;
  int iteration = 0;

  // Main EPA loop
  while (iteration < max_iterations && poly.num_vertices < MAX_EPA_VERTICES - 1) {
    iteration++;
    // Recompute normals & distances for assigned faces
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (poly.faces[i].valid) {
        compute_face_normal_distance(&poly, i);
      }
    }

    // parallel reduction to find closest face
    // finds the closest face in the range
    int closest_face = -1;
    gkFloat closest_distance = 1e10f;

    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      if (poly.faces[i].distance >= 0.0f && poly.faces[i].distance < closest_distance) {
        closest_distance = poly.faces[i].distance;
        closest_face = i;
      }
    }


    if (closest_face < 0) {
      break;
    }

    EPAFace* closest = &poly.faces[closest_face];

    // Get support point in direction of closest face normal
    gkFloat new_vertex[3];
    int new_vertex_idx[2];
    support_epa(bd1, bd2, poly.faces[closest_face].normal, new_vertex, new_vertex_idx);

    // Check termination condition: if distance to new vertex along normal is not more than tolerance further than closest face
    gkFloat dist_to_new = dotProduct(poly.faces[closest_face].normal, new_vertex);
    gkFloat improvement = dist_to_new - closest_distance;

    if (improvement < tolerance) {
      // Converged, compute witness points with bary coords
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
      break;
    }

    /// Check if new vertex is duplicate
    bool is_duplicate = false;
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

    if (is_duplicate) {
      // Can't make progress, use current best
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
      break;
    }

    // Add new vertex to polytope
    int new_vertex_id = poly.num_vertices;
    for (int i = 0; i < 3; i++) {
      poly.vertices[new_vertex_id][i] = new_vertex[i];
    }
    poly.vertex_indices[new_vertex_id][0] = new_vertex_idx[0];
    poly.vertex_indices[new_vertex_id][1] = new_vertex_idx[1];
    poly.num_vertices++;

    // Update centroid incrementally (running mean)
    gkFloat inv_n = (gkFloat)1.0 / (gkFloat)poly.num_vertices;
    for (int i = 0; i < 3; i++) {
      centroid[i] += (new_vertex[i] - centroid[i]) * inv_n;
    }
    // Find horizon edges: collect edges from faces being removed this iteration
    // only, then mark them invalid. Collecting from ALL invalid faces (including
    // ones from previous iterations) would pull in stale interior edges.
      EPAEdge edges[MAX_EPA_FACES * 3];
      int num_edges = 0;

      for (int f = 0; f < poly.max_face_index; f++) {
        if (!poly.faces[f].valid) continue;
        if (!is_face_visible(&poly, f, new_vertex)) continue;

        // Collect edges before invalidating the face
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

        poly.faces[f].valid = false;
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

        for (int c = 0; c < 3; c++) {
          fe0[c] = fv1[c] - fv0[c];
          fe1[c] = fv2[c] - fv0[c];
        }
        crossProduct(fe0, fe1, fnormal);

        // If normal points toward centroid flip winding
        gkFloat to_cent[3];
        for (int c = 0; c < 3; c++) to_cent[c] = centroid[c] - fv0[c];
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

  // If we exited due to max iterations, recompute closest face and use it
  if (iteration >= max_iterations) {
    // Find closest face and compute result
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      compute_face_normal_distance(&poly, i);
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

    if (closest_face >= 0) {
      EPAFace* closest = &poly.faces[closest_face];
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
    }
  }
}


#ifdef MATLAB_MEX_BUILD
/**
 * @brief Mex function for Matlab.
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  gkFloat* inCoordsA;
  gkFloat* inCoordsB;
  size_t nCoordsA;
  size_t nCoordsB;
  int i;
  gkFloat* distance;
  int c = 3;
  int count = 0;
  gkFloat** arr1;
  gkFloat** arr2;

  /**************** PARSE INPUTS AND OUTPUTS **********************/
  /*----------------------------------------------------------------*/
  /* Examine input (right-hand-side) arguments. */
  if (nrhs != 2) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:nrhs", "Two inputs required.");
  }
  /* Examine output (left-hand-side) arguments. */
  if (nlhs != 1) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:nlhs", "One output required.");
  }

  /* make sure the two input arguments are any numerical type */
  /* .. first input */
  if (!mxIsNumeric(prhs[0])) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notNumeric",
                      "Input matrix must be type numeric.");
  }
  /* .. second input */
  if (!mxIsNumeric(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notNumeric",
                      "Input matrix must be type numeric.");
  }

  /* make sure the two input arguments have 3 columns */
  /* .. first input */
  if (mxGetM(prhs[0]) != 3) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notColumnVector",
                      "First input must have 3 columns.");
  }
  /* .. second input */
  if (mxGetM(prhs[1]) != 3) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notColumnVector",
                      "Second input must have 3 columns.");
  }

  /*----------------------------------------------------------------*/
  /* CREATE DATA COMPATIBLE WITH MATALB  */

  /* create a pointer to the real data in the input matrix  */
  inCoordsA = mxGetPr(prhs[0]);
  inCoordsB = mxGetPr(prhs[1]);

  /* get the length of each input vector */
  nCoordsA = mxGetN(prhs[0]);
  nCoordsB = mxGetN(prhs[1]);

  /* Create output */
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

  /* get a pointer to the real data in the output matrix */
  distance = mxGetPr(plhs[0]);

  /* Copy data from Matlab's vectors into two new arrays */
  arr1 = (gkFloat**)mxMalloc(sizeof(gkFloat*) * (int)nCoordsA);
  arr2 = (gkFloat**)mxMalloc(sizeof(gkFloat*) * (int)nCoordsB);

  for (i = 0; i < nCoordsA; i++) {
    arr1[i] = &inCoordsA[i * 3];
  }

  for (i = 0; i < nCoordsB; i++) {
    arr2[i] = &inCoordsB[i * 3];
  }

  /*----------------------------------------------------------------*/
  /* POPULATE BODIES' STRUCTURES  */

  gkPolytope bd1; /* Structure of body A */
  gkPolytope bd2; /* Structure of body B */

  /* Assign number of vertices to each body */
  bd1.numpoints = (int)nCoordsA;
  bd2.numpoints = (int)nCoordsB;

  bd1.coord = arr1;
  bd2.coord = arr2;

  /*----------------------------------------------------------------*/
  /*CALL COMPUTATIONAL ROUTINE  */

  gkSimplex s;
  s.nvrtx = 0;

  /* Compute squared distance using GJK algorithm */
  distance[0] = compute_minimum_distance(bd1, bd2, &s);

  mxFree(arr1);
  mxFree(arr2);
}
#endif
#ifdef CS_MONO_BUILD
/**
 * @brief Invoke this function from C# applications
 */
OPENGJK_EXPORT gkFloat csFunction(int nCoordsA, gkFloat* inCoordsA,
                                  int nCoordsB, gkFloat* inCoordsB) {
  gkFloat distance = 0;
  int i, j;

  /*----------------------------------------------------------------*/
  /* POPULATE BODIES' STRUCTURES  */

  gkPolytope bd1; /* Structure of body A */
  gkPolytope bd2; /* Structure of body B */

  /* Assign number of vertices to each body */
  bd1.numpoints = (int)nCoordsA;
  bd2.numpoints = (int)nCoordsB;

  gkFloat** pinCoordsA = (gkFloat**)malloc(bd1.numpoints * sizeof(gkFloat*));
  for (i = 0; i < bd1.numpoints; i++) {
    pinCoordsA[i] = (gkFloat*)malloc(3 * sizeof(gkFloat));
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < bd1.numpoints; j++) {
      pinCoordsA[j][i] = inCoordsA[i * bd1.numpoints + j];
    }
  }

  gkFloat** pinCoordsB = (gkFloat**)malloc(bd2.numpoints * sizeof(gkFloat*));
  for (i = 0; i < bd2.numpoints; i++) {
    pinCoordsB[i] = (gkFloat*)malloc(3 * sizeof(gkFloat));
  }

  for (i = 0; i < 3; i++) {
    for (j = 0; j < bd2.numpoints; j++) {
      pinCoordsB[j][i] = inCoordsB[i * bd2.numpoints + j];
    }
  }

  bd1.coord = pinCoordsA;
  bd2.coord = pinCoordsB;

  /*----------------------------------------------------------------*/
  /*CALL COMPUTATIONAL ROUTINE  */
  gkSimplex s;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

  /* Compute squared distance using GJK algorithm */
  distance = compute_minimum_distance(bd1, bd2, &s);

  for (i = 0; i < bd1.numpoints; i++) {
    free(pinCoordsA[i]);
  }
  free(pinCoordsA);

  for (i = 0; i < bd2.numpoints; i++) {
    free(pinCoordsB[i]);
  }
  free(pinCoordsB);

  return distance;
}
#endif  // CS_MONO_BUILD

/* ========================================================================== */
/* Testing wrappers - expose internal functions for cross-validation         */
/* ========================================================================== */

/**
 * @brief Wrapper to expose S1D for testing.
 *
 * @param[in,out] s  Simplex with nvrtx=2. vrtx[1] is newest, vrtx[0] is oldest.
 * @param[out]    v  Closest point to origin.
 */
void opengjk_test_S1D(gkSimplex* s, gkFloat* v) { S1D(s, v); }

/**
 * @brief Wrapper to expose S2D for testing.
 *
 * @param[in,out] s  Simplex with nvrtx=3. vrtx[2] is newest, vrtx[0] is oldest.
 * @param[out]    v  Closest point to origin.
 */
void opengjk_test_S2D(gkSimplex* s, gkFloat* v) { S2D(s, v); }

/**
 * @brief Wrapper to expose S3D for testing.
 *
 * @param[in,out] s  Simplex with nvrtx=4. vrtx[3] is newest, vrtx[0] is oldest.
 * @param[out]    v  Closest point to origin.
 */
void opengjk_test_S3D(gkSimplex* s, gkFloat* v) { S3D(s, v); }
