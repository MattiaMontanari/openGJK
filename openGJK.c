//                           _____      _ _  __ //
//                          / ____|    | | |/ / //
//    ___  _ __   ___ _ __ | |  __     | | ' / //
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  < //
//  | (_) | |_) |  __/ | | | |__| | |__| | . \ //
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\ //
//        | | //
//        |_| //
//                                                                                //
// Copyright 2022 Mattia Montanari, University of Oxford //
//                                                                                //
// This program is free software: you can redistribute it and/or modify it under
// // the terms of the GNU General Public License as published by the Free
// Software  // Foundation, either version 3 of the License. You should have
// received a copy   // of the GNU General Public License along with this
// program. If not, visit       //
//                                                                                //
//     https://www.gnu.org/licenses/ //
//                                                                                //
// This program is distributed in the hope that it will be useful, but WITHOUT
// // ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS  // FOR A PARTICULAR PURPOSE. See GNU General Public License for
// details.          //

/**
 * @file openGJK.c
 * @author Mattia Montanari
 * @date 1 Jan 2022
 * @brief Source of OpenGJK and its fast sub-algorithm.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

#include "openGJK/openGJK.h"

#include <stdio.h>
#include <stdlib.h>

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
    return 1;  // keep q
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

  return dotProduct(p, n) < 0;  // Discard r if true
}

inline static int hff3(const gkFloat* restrict p, const gkFloat* restrict q,
                       const gkFloat* restrict r) {
  gkFloat n[3], pq[3], pr[3];

  for (int i = 0; i < 3; i++) {
    pq[i] = q[i] - p[i];
  }
  for (int i = 0; i < 3; i++) {
    pr[i] = r[i] - p[i];
  }

  crossProduct(pq, pr, n);
  return dotProduct(p, n) <= 0;  // discard s if true
}

inline static void S1D(gkSimplex* s, gkFloat* v) {
  const gkFloat* restrict s1p = s->vrtx[1];
  const gkFloat* restrict s2p = s->vrtx[0];

  if (hff1(s1p, s2p)) {
    projectOnLine(s1p, s2p, v);  // Update v, no need to update s
    return;                      // Return V{1,2}
  } else {
    S1Dregion1();  // Update v and s
    return;        // Return V{1}
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
          projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
          return;                            // Return V{1,2,3}
        } else {
          projectOnLine(s1p, s3p, v);  // Update v
          S2Dregion13();               // Update s
          return;                      // Return V{1,3}
        }
      } else {
        projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update c
        return;                            // Return V{1,2,3}
      }
    } else {
      projectOnLine(s1p, s2p, v);  // Update v
      S2Dregion12();               // Update s
      return;                      // Return V{1,2}
    }
  } else if (hff1f_s13) {
    const int hff2f_32 = !hff2(s1p, s3p, s2p);
    if (hff2f_32) {
      projectOnPlane(s1p, s2p, s3p, v);  // Update s, no need to update v
      return;                            // Return V{1,2,3}
    } else {
      projectOnLine(s1p, s3p, v);  // Update v
      S2Dregion13();               // Update s
      return;                      // Return V{1,3}
    }
  } else {
    S2Dregion1();  // Update s and v
    return;        // Return V{1}
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
      // Only one facing the oring
      // 1,i,j, are the indices of the points on the triangle and remove k from
      // simplex
      s->nvrtx = 3;
      if (!testPlaneTwo) {  // k = 2;   removes s2
        for (i = 0; i < 3; i++) {
          s->vrtx[2][i] = s->vrtx[3][i];
        }
        for (i = 0; i < 2; i++) {
          s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
        }
      } else if (!testPlaneThree) {  // k = 1; // removes s3
        for (i = 0; i < 3; i++) {
          s->vrtx[1][i] = s2[i];
          s->vrtx[2][i] = s->vrtx[3][i];
        }
        for (i = 0; i < 2; i++) {
          s->vrtx_idx[1][i] = s2_idx[i];
          s->vrtx_idx[2][i] = s->vrtx_idx[3][i];
        }
      } else if (!testPlaneFour) {  // k = 0; // removes s4  and no need to
                                    // reorder
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
      } else if (testPlaneThree) {
        k = 1;  // s3
        i = 0;
        j = 2;
      } else {
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
          } else if (!hff2(s1, sk, sj)) {
            select_1jk();
            projectOnPlane(s1, sj, sk, v);
          } else {
            select_1k();  // select region 1i
            projectOnLine(s1, sk, v);
          }
        } else if (hff1_tests[i]) {
          if (!hff2(s1, si, sk)) {
            select_1ik();
            projectOnPlane(s1, si, sk, v);
          } else {
            select_1i();  // select region 1i
            projectOnLine(s1, si, v);
          }
        } else {
          if (!hff2(s1, sj, sk)) {
            select_1jk();
            projectOnPlane(s1, sj, sk, v);
          } else {
            select_1j();  // select region 1i
            projectOnLine(s1, sj, v);
          }
        }
      } else if (dotTotal == 2) {
        // Two edges have positive hff1, meaning that for two edges the origin's
        // project fall on the segement.
        //  Certainly the edge 1,k supports the the point of minimum norm, and
        //  so hff1_1k is positive

        if (hff1_tests[i]) {
          if (!hff2(s1, sk, si)) {
            if (!hff2(s1, si, sk)) {
              select_1ik();  // select region 1ik
              projectOnPlane(s1, si, sk, v);
            } else {
              select_1k();  // select region 1k
              projectOnLine(s1, sk, v);
            }
          } else {
            if (!hff2(s1, sk, sj)) {
              select_1jk();  // select region 1jk
              projectOnPlane(s1, sj, sk, v);
            } else {
              select_1k();  // select region 1k
              projectOnLine(s1, sk, v);
            }
          }
        } else if (hff1_tests[j]) {  //  there is no other choice
          if (!hff2(s1, sk, sj)) {
            if (!hff2(s1, sj, sk)) {
              select_1jk();  // select region 1jk
              projectOnPlane(s1, sj, sk, v);
            } else {
              select_1j();  // select region 1j
              projectOnLine(s1, sj, v);
            }
          } else {
            if (!hff2(s1, sk, si)) {
              select_1ik();  // select region 1ik
              projectOnPlane(s1, si, sk, v);
            } else {
              select_1k();  // select region 1k
              projectOnLine(s1, sk, v);
            }
          }
        } else {
          // ERROR;
        }

      } else if (dotTotal == 3) {
        // MM : ALL THIS HYPHOTESIS IS FALSE
        // sk is s.t. hff3 for sk < 0. So, sk must support the origin because
        // there are 2 triangles facing the origin.

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
          // discard i
          if (hff2_jk) {
            // discard k
            select_1j();
            projectOnLine(s1, sj, v);
          } else {
            select_1jk();
            projectOnPlane(s1, sk, sj, v);
          }
        } else {
          // discard j
          if (hff2_ik) {
            // discard k
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
      // The origin is outside all 3 triangles
      if (dotTotal == 1) {
        // Here si is set such that hff(s1,si) > 0
        if (testLineThree) {
          k = 2;
          i = 1;  // s3
          j = 0;
        } else if (testLineFour) {
          k = 1;  // s3
          i = 0;
          j = 2;
        } else {
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
        } else if (!hff2(s1, si, sk)) {
          select_1ik();
          projectOnPlane(s1, si, sk, v);
        } else {
          select_1i();
          projectOnLine(s1, si, v);
        }
      } else if (dotTotal == 2) {
        // Here si is set such that hff(s1,si) < 0
        s->nvrtx = 3;
        if (!testLineThree) {
          k = 2;
          i = 1;  // s3
          j = 0;
        } else if (!testLineFour) {
          k = 1;
          i = 0;  // s4
          j = 2;
        } else {
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
  const gkFloat a0 = 1.0 - a1;

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
  unsigned int k = 0;                /**< Iteration counter                 */
  const int mk = GJK_MAX_ITERATIONS; /**< Maximum number of GJK iterations  */
  const gkFloat eps_rel = GJK_EPSILON_REL; /**< Tolerance on relative */
  const gkFloat eps_tot =
      GJK_EPSILON_ABS; /**< Tolerance on absolute distance    */

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

    if (norm2(v) < eps_rel2) {  // it a null V
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
  return sqrt(norm2(v));
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
extern gkFloat csFunction(int nCoordsA, gkFloat* inCoordsA, int nCoordsB,
                          gkFloat* inCoordsB) {
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
