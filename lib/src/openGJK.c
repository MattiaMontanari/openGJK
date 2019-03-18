/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
*                                    #####        # #    #                *
*        ####  #####  ###### #    # #     #       # #   #                 *
*       #    # #    # #      ##   # #             # #  #                  *
*       #    # #    # #####  # #  # #  ####       # ###                   *
*       #    # #####  #      #  # # #     # #     # #  #                  *
*       #    # #      #      #   ## #     # #     # #   #                 *
*        ####  #      ###### #    #  #####   #####  #    #                *
*                                                                         *
*           Mattia Montanari    |   University of Oxford 2018             *
* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - *
*                                                                         *
* This file implements the GJK algorithm and the Signed Volumes method as *
* presented in:                                                           *
*   M. Montanari, N. Petrinic, E. Barbieri, "Improving the GJK Algorithm  *
*   for Faster and More Reliable Distance Queries Between Convex Objects" *
*   ACM Transactions on Graphics, vol. 36, no. 3, Jun. 2017.              *
*                                                                         *
* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

/**
 * @file openGJK.c
 * @author Mattia Montanari
 * @date April 2018
 * @brief File containing entire implementation of the GJK algorithm.
 *
 */

#define _CRT_HAS_CXX17 0
#include <stdio.h>

#include "openGJK/openGJK.h"

/* If defined, uses an advanced method to compute the determinant of matrices. */
#ifdef ADAPTIVEFP
#include "predicates.h"
#endif

/* Used for creating a mex function for Matlab.  */
#ifdef MATLABDOESMEXSTUFF
  #include "mex.h" 
#else
  #define mexPrintf  printf 
#endif

#ifdef unix
  #define _isnan(x) isnan(x)
#endif


/**
 * @brief Computes the norm of a vector.
 */
double norm2(double *v) {
  double n2;
  n2 = 0;
  for (int i = 0; i < 3; ++i) {
    n2 += v[i] * v[i];
  }
  return n2;
}

/**
 * @brief Computes the dot product between two vectors.
 */
inline double dotprod(double *a, double *b) {
  double dot;
  dot = 0;
  for (int i = 0; i < 3; ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

/**
* @brief Function invoked by the Signed Volumes sub-algorithm for 1-simplex.
*/
void S1D( struct simplex * s) {

  double    a[3],
    b[3],
    t[3],
    nu_test[3],
    nu_fabs[3];

  double    inv_detM = 0,
    det_ap = 0,
    det_pb = 0,
    pt = 0;

  int     i = 0,
    indexI = 1,
    FacetsTest[ 2 ];

  for ( i=0 ; i<3; ++i ) {
    b[i] = s->vrtx[0][i];
    a[i] = s->vrtx[1][i];
    t[i] = b[i] - a[i];
  }

  for ( i = 0; i < 3; ++i) {
    nu_test[i] = a[i] - b[i];
  }
  for ( i = 0; i < 3; ++i) {
    nu_fabs[i] = fabs( nu_test[i] );
  }

  if (nu_fabs[0] > nu_fabs[1]){
    if (nu_fabs[0] > nu_fabs[2]){
      indexI = 0;
    }
    else {
      indexI = 2;
    }
  }
  else if ( nu_fabs[0] < nu_fabs[1] ){
    if (nu_fabs[1] > nu_fabs[2]){
      indexI = 1;
    }
    else {
      indexI = 2;
    }
  }
  else if ( nu_fabs[0] < nu_fabs[2] ){
    indexI = 2;
  }
  else if ( nu_fabs[1] < nu_fabs[2] ){
    indexI = 2;
  }

  /* Compute inverse of detM */
  inv_detM = 1 / nu_test [ indexI ];

  /* Project origin onto the 1D simplex - line */
  pt = dotprod (b,t) / (dotprod(t,t)) * ( a[indexI] - b[indexI] ) + b[indexI];

  /* Compute signed determinants */
  det_ap = a[indexI] - pt;
  det_pb = pt - b[indexI];

  /* Test if sign of ABCD is equal to the signes of the auxiliary simplices */
  FacetsTest[ 0 ] = SAMESIGN( nu_test [ indexI ] , det_ap );
  FacetsTest[ 1 ] = SAMESIGN( nu_test [ indexI ] , det_pb );

  if ( FacetsTest[ 0 ] + FacetsTest[ 1 ] == 2){
    s->lambdas[0] = det_ap * inv_detM;
    s->lambdas[1] = 1 - s->lambdas[0];
    s->wids[0] = 0;
    s->wids[1] = 1;
    s->nvrtx = 2;
  }
  else if ( FacetsTest[ 0 ] == 0 ) {
    s->lambdas[0] = 1;
    s->wids[0] = 0;
    s->nvrtx = 1;

    for ( i = 0; i < 3; ++i) {
      s->vrtx[0][i] = s->vrtx[1][i];
    }
  }
  else {
    s->lambdas[0] = 1;
    s->wids[0] = 1;
    s->nvrtx = 1;
  }
}



/**
* @brief Function invoked by the Signed Volumes sub-algorithm for 2-simplex.
*/
static void S2D( struct simplex * s) {

  double    a[3],
    b[3],
    c[3],
    s21[3],
    s31[3],
    nu_test[3],
    nu_fabs[3],
    B[ 3 ],
    n[3],
    v[3],
    vtmp[3];


  double pp[3-1],
    sa[3-1],
    sb[3-1],
    sc[3-1];

  double nu_max = 0,
    inv_detM = 0,
    nnorm_sqrd = 0,
    nnnorm = 0,
    dotNA;

  int i = 0,
    FacetsTest[3],
    k,
    l,
    j;

  int indexI = 1,
    indexJ[2] = {0, 2},
    stemp[3];

  for ( i=0 ; i<3; ++i ) {
    c[i] = s->vrtx[0][i];
    b[i] = s->vrtx[1][i];
    a[i] = s->vrtx[2][i];
    s21[i] = b[i] - a[i];
    s31[i] = c[i] - a[i];
  }

  k = 1; l = 2;
  for (i = 0; i < 3 ; ++i) {
    nu_test[i] = pow(-1.0,i) * (b[k]*c[l] + a[k]*b[l] + c[k]*a[l] - b[k]*a[l] - c[k]*b[l] - a[k]*c[l]);
    k = l; l = i;
  }

  for ( i = 0; i < 3; ++i) {
    nu_fabs[i] = fabs( nu_test[i]);
  }

  if (nu_fabs[0] > nu_fabs[1]){
    if (nu_fabs[0] > nu_fabs[2]){
      indexI = 0;
      indexJ[0] = 1;
      indexJ[1] = 2;
    }
    else {
      indexJ[0] = 0;
      indexJ[1] = 1;
      indexI = 2;
    }
  }
  else if ( nu_fabs[0] < nu_fabs[1] ){
    if (nu_fabs[1] > nu_fabs[2]){
      indexJ[0] = 0;
      indexI = 1;
      indexJ[1] = 2;
    }
    else {
      indexJ[0] = 0;
      indexJ[1] = 1;
      indexI = 2;
    }
  }
  else if ( nu_fabs[0] < nu_fabs[2] ){
    indexJ[0] = 0;
    indexJ[1] = 1;
    indexI = 2;
  }

  nu_max = nu_test[ indexI ];

  k = 1;      l = 2;
  for ( i = 0; i < 3; ++i ) {
    n[i] = s21[k] * s31[l] - s21[l] * s31[k];
    nnorm_sqrd += n[i] * n[i];
    k = l;     l = i;
  }
  nnnorm = 1 / sqrt( nnorm_sqrd );
  for ( i = 0; i < 3; ++i) {
    n[i] = n[i] * nnnorm;
  }

  dotNA = dotprod(n,a);
  pp[0] = dotNA * n[ indexJ[0] ];
  pp[1] = dotNA * n[ indexJ[1] ];

  /* Compute signed determinants */
#ifndef ADAPTIVEFP
  double    ss[3][3-1];

  ss[0][0] = sa[0] = a[ indexJ[0] ];
  ss[0][1] = sa[1] = a[ indexJ[1] ];
  ss[1][0] = sb[0] = b[ indexJ[0] ];
  ss[1][1] = sb[1] = b[ indexJ[1] ];
  ss[2][0] = sc[0] = c[ indexJ[0] ];
  ss[2][1] = sc[1] = c[ indexJ[1] ];

  k = 1;   l = 2;
  for ( i = 0; i < 3 ; ++i ) {
    B[i] = pp[0] * ss[k][1] + pp[1] * ss[l][0] + ss[k][0] * ss[l][1] - pp[0] * ss[l][1] - pp[1] * ss[k][0] -
           ss[l][0] * ss[k][1];
    k = l;
    l = i;
  }
#else

    sa[0] = a[ indexJ[0] ];
    sa[1] = a[ indexJ[1] ];
    sb[0] = b[ indexJ[0] ];
    sb[1] = b[ indexJ[1] ];
    sc[0] = c[ indexJ[0] ];
    sc[1] = c[ indexJ[1] ];

    B[0] = orient2d( sa, pp, sc );
    B[1] = orient2d( sa, pp, sc );
    B[2] = orient2d( sa, sb, pp );
#endif

  /* Test if sign of ABC is equal to the signes of the auxiliary simplices */
  for ( int m = 0; m < 3 ; ++m) {
    FacetsTest[ m ] = SAMESIGN( nu_max , B[ m ] );
  }

  if ( FacetsTest[0] +FacetsTest[1] + FacetsTest[2] == 0  || isnan(n[0]) ){

    struct simplex  sTmp;

    sTmp.nvrtx = 2;
    s->nvrtx = 2;
    for ( i = 0; i < 3; ++i) {

      sTmp.vrtx[0][i] = s->vrtx[1][i];
      sTmp.vrtx[1][i] = s->vrtx[2][i];

      s->vrtx[0][i] = s->vrtx[0][i];
      s->vrtx[1][i] = s->vrtx[2][i];
    }

    S1D(&sTmp);
    S1D(s);

    for (j = 0; j < 3; ++j) {
      vtmp[j] = 0;
      v[j] = 0;
      for (i = 0; i < sTmp.nvrtx ; ++i) {
        vtmp[j] +=  (sTmp.lambdas[i] * sTmp.vrtx[i][j] );
        v[j] += (s->lambdas[i] * s->vrtx[i][j] );
      }
    }

    if( dotprod(v,v) < dotprod(vtmp,vtmp) ){
      for (i = 1; i < s->nvrtx ; ++i) {
        s->wids[i] = s->wids[i] + 1;
      }
    }
    else {
      s->nvrtx = sTmp.nvrtx;
      for (j = 0; j < 3; ++j) {
        for (i = 0; i < s->nvrtx ; ++i) {
          s->vrtx[i][j] = s->vrtx[i][j];
          s->lambdas[i] = sTmp.lambdas[i];
          /* No need to convert sID here since sTmp deal with the vertices A and B. ;*/
          s->wids[i] = sTmp.wids[i];
        }
      }
    }
  }
  else if ( (FacetsTest[0] + FacetsTest[1] + FacetsTest[2] ) == 3 ) {
    /* The origin projections lays onto the triangle */
    inv_detM = 1 / nu_max;
    s->lambdas[0] = B[2] * inv_detM;
    s->lambdas[1] = B[1] * inv_detM;
    s->lambdas[2] = 1 - s->lambdas[0] - s->lambdas[1];
    s->wids[0] = 0;
    s->wids[1] = 1;
    s->wids[2] = 2;
    s->nvrtx = 3;
  }
  else if ( FacetsTest[2] == 0 ){
    /* The origin projection P faces the segment AB */
    s->nvrtx = 2;
    for ( i = 0; i < 3; ++i) {
      s->vrtx[0][i] = s->vrtx[1][i];
      s->vrtx[1][i] = s->vrtx[2][i];
    }
    S1D(s);
  }
  else if ( FacetsTest[1] == 0 ){
    /* The origin projection P faces the segment AC */
    stemp[0] = 0;   stemp[1] = 2;
    s->nvrtx = 2;
    for ( i = 0; i < 3; ++i) {
      s->vrtx[0][i] = s->vrtx[ stemp[0] ][i];
      s->vrtx[1][i] = s->vrtx[ stemp[1] ][i];
    }
    S1D(s);
    for (i = 1; i < s->nvrtx ; ++i) {
      s->wids[i] = s->wids[i] + 1;
    }
  }
  else {
    s->nvrtx = 2;
    S1D(s);
  }
}


/**
* @brief Function invoked by the Signed Volumes sub-algorithm for 3-simplex.
*
*/
static void S3D( struct simplex * s) {

  double  a[3];
  double  b[3];
  double  c[3];
  double  d[3];
#ifdef ADAPTIVEFP
  double  o[3] = {0}; 
#endif
  double  vtmp[3];
  double  v[3];
  double  B[ 4 ];
  double  sqdist_tmp = 0 ;
  double  detM;
  double  inv_detM;
  double  tmplamda[4] = {0, 0, 0, 0};
  double  tmpscoo1[4][3] = {0};
  int     i, j, k, l,
          Flag_sAuxused = 0,
          firstaux = 0,
          secondaux = 0;
  int     FacetsTest[ 4 ] = { 1,1,1,1 },
          sID[4] = {0, 0, 0, 0},
          nclosestS = 0;
  int     TrianglesToTest[9] = { 3, 3, 3, 1, 2, 2, 0, 0, 1 };
  int     ndoubts = 0,
          vtxid;

  for ( i=0 ; i<3; ++i ) {
    d[i] = s->vrtx[0][i];
    c[i] = s->vrtx[1][i];
    b[i] = s->vrtx[2][i];
    a[i] = s->vrtx[3][i];
  }

#ifndef ADAPTIVEFP
  B[ 0 ] = -1 * ( b[0]*c[1]*d[2] + b[1]*c[2]*d[0] + b[2]*c[0]*d[1] - b[2]*c[1]*d[0] - b[1]*c[0]*d[2] - b[0]*c[2]*d[1]);
  B[ 1 ] = +1 * ( a[0]*c[1]*d[2] + a[1]*c[2]*d[0] + a[2]*c[0]*d[1] - a[2]*c[1]*d[0] - a[1]*c[0]*d[2] - a[0]*c[2]*d[1]);
  B[ 2 ] = -1 * ( a[0]*b[1]*d[2] + a[1]*b[2]*d[0] + a[2]*b[0]*d[1] - a[2]*b[1]*d[0] - a[1]*b[0]*d[2] - a[0]*b[2]*d[1]);
  B[ 3 ] = +1 * ( a[0]*b[1]*c[2] + a[1]*b[2]*c[0] + a[2]*b[0]*c[1] - a[2]*b[1]*c[0] - a[1]*b[0]*c[2] - a[0]*b[2]*c[1]);
  detM =  B[ 0 ] + B[ 1 ] + B[ 2 ] + B[ 3 ] ;
#else
  B[ 0 ]  = orient3d(o,b,c,d);
  B[ 1 ]  = orient3d(a,o,c,d);
  B[ 2 ]  = orient3d(a,b,o,d);
  B[ 3 ]  = orient3d(a,b,c,o);
  detM    = orient3d(a,b,c,d);
#endif

  /* Test if sign of ABCD is equal to the signes of the auxiliary simplices */
  double eps = 1e-13;
  if ( fabs(detM) < eps)
  {
    if (fabs(B[2]) < eps && fabs(B[3]) < eps)
    {
      FacetsTest[1] = 0; /* A = B. Test only ACD */
    }
    else if (  fabs(B[1]) < eps && fabs(B[3]) < eps)
    {
      FacetsTest[2] = 0; /* A = C. Test only ABD */
    }
    else if ( fabs(B[1]) < eps && fabs(B[2]) < eps)
    {
      FacetsTest[3] = 0; /* A = D. Test only ABC */
    }
    else if ( fabs(B[0]) < eps && fabs(B[3]) < eps)
    {
      FacetsTest[1] = 0; /* B = C. Test only ACD */
    }
    else if ( fabs(B[0]) < eps && fabs(B[2]) < eps)
    {
      FacetsTest[1] = 0; /* B = D. Test only ABD */
    }
    else if ( fabs(B[0]) < eps && fabs(B[1]) < eps)
    {
      FacetsTest[2] = 0; /* C = D. Test only ABC */
    }
    else
    {
      for (i = 0; i < 4; i++)
      {
        FacetsTest[i] = 0; 
      }
    }

  }
  else
  {
    for (i = 0; i < 4 ; ++i) {
      FacetsTest[i] = SAMESIGN( detM , B[i] );
    }
  }

  /* Compare signed volumes and compute barycentric coordinates */
  if ( FacetsTest[0] + FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 4 ){
    /* All signs are equal, therefore the origin is inside the simplex */
    inv_detM = 1 / detM;
    s->lambdas[3] = B[0] * inv_detM;
    s->lambdas[2] = B[1] * inv_detM;
    s->lambdas[1] = B[2] * inv_detM;
    s->lambdas[0] = 1 - s->lambdas[1] - s->lambdas[2] - s->lambdas[3];
    s->wids[0] = 0;
    s->wids[1] = 1;
    s->wids[2] = 2;
    s->wids[3] = 3;
    s->nvrtx   = 4;
    return;
  }
  else if (  FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 0 ){
    /* There are three facets facing the origin  */
    ndoubts = 3;

    struct simplex  sTmp;

    for (i = 0; i < ndoubts; ++i) {
      sTmp.nvrtx = 3;
      /* Assign coordinates to simplex */

      for ( k = 0; k < sTmp.nvrtx; ++k) {
        vtxid = TrianglesToTest[ i+(k*3) ];
        for ( j = 0; j < 3; ++j) {
          sTmp.vrtx[2-k][j] = s->vrtx[ vtxid ][j];
        }
      }
      S2D(&sTmp);
      for (j = 0; j < 3; ++j) {
        vtmp[j] = 0;
        for (l = 0; l < sTmp.nvrtx ; ++l) {
          vtmp[j] += sTmp.lambdas[l] * ( sTmp.vrtx[l][j] );
        }
      }

      if (i == 0){
        sqdist_tmp = dotprod(vtmp,vtmp);
        nclosestS = sTmp.nvrtx;
        for (l = 0; l < nclosestS ; ++l) {
          sID[l] = TrianglesToTest[ i+( sTmp.wids[l] *3) ];
          tmplamda[l] = sTmp.lambdas[l];
        }
      }
      else if (dotprod(vtmp,vtmp) < sqdist_tmp){
        sqdist_tmp = dotprod(vtmp,vtmp);
        nclosestS = sTmp.nvrtx;
        for (l = 0; l < nclosestS ; ++l) {
          sID[l] = TrianglesToTest[ i+( sTmp.wids[l] *3) ];
          tmplamda[l] = sTmp.lambdas[l];
        }
      }
    }

    for (i = 0; i < 4; ++i) {
      for (j = 0; j < 3; ++j) {
        tmpscoo1[i][j] = s->vrtx[i][j];
      }
    }

    /* Store closest simplex */
    s->nvrtx = nclosestS;
    for (i = 0; i < s->nvrtx ; ++i) {
      for (j = 0; j < 3; ++j) {
        s->vrtx[ nclosestS-1 - i][j] = tmpscoo1[sID[i]][j];
      }
      s->lambdas[i] = tmplamda[i];
      s->wids[ nclosestS-1 - i ] = sID[i];
    }

    return;
  }
  else if ( FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 1 ){
    /* There are two   facets facing the origin, need to find the closest.   */
    struct simplex  sTmp;
    sTmp.nvrtx = 3;

    if ( FacetsTest[1] == 0 ){
      /* ... Test facet ACD  */
      for ( i = 0; i < 3; ++i) {
        sTmp.vrtx[0][i] = s->vrtx[ 0 ][i];
        sTmp.vrtx[1][i] = s->vrtx[ 1 ][i];
        sTmp.vrtx[2][i] = s->vrtx[ 3 ][i];
      }
      S2D(&sTmp);
      for (j = 0; j < 3; ++j) {
        vtmp[j] = 0;
        for (i = 0; i < sTmp.nvrtx ; ++i) {
          vtmp[j] += sTmp.lambdas[i] * (sTmp.vrtx[i][j] );
        }
      }
      sqdist_tmp = dotprod(vtmp,vtmp);
      Flag_sAuxused = 1;
      firstaux = 0;
    }
    if ( FacetsTest[2] == 0 ){
      if ( Flag_sAuxused == 0 ){

        for ( i = 0; i < 3; ++i) {
          sTmp.vrtx[0][i] = s->vrtx[ 0 ][i];
          sTmp.vrtx[1][i] = s->vrtx[ 2 ][i];
          sTmp.vrtx[2][i] = s->vrtx[ 3 ][i];
        }
        S2D(&sTmp);
        for (j = 0; j < 3; ++j) {
          vtmp[j] = 0;
          for (i = 0; i < sTmp.nvrtx ; ++i) {
            vtmp[j] += sTmp.lambdas[i] * (sTmp.vrtx[i][j] );
          }
        }
        sqdist_tmp = dotprod(vtmp,vtmp);
        firstaux = 1;
      }
      else {
        s->nvrtx = 3;
        for ( i = 0; i < 3; ++i) {
          s->vrtx[0][i] = s->vrtx[ 0 ][i];
          s->vrtx[1][i] = s->vrtx[ 2 ][i];
          s->vrtx[2][i] = s->vrtx[ 3 ][i];
        }
        /* ... and call S2D itself */
        S2D(s);
        secondaux = 1;
      }
    }

    if  ( FacetsTest[3] == 0 ){
      /* ... Test facet ABC  */
      s->nvrtx = 3;
      for ( i = 0; i < 3; ++i) {
        s->vrtx[0][i] = s->vrtx[ 1 ][i];
        s->vrtx[1][i] = s->vrtx[ 2 ][i];
        s->vrtx[2][i] = s->vrtx[ 3 ][i];
      }
      /* ... and call S2D itself */
      S2D(s);
      secondaux = 2;
    }
    /* Do a loop and compare current outcomes */
    for (j = 0; j < 3; ++j) {
      v[j] = 0;
      for (i = 0; i < s->nvrtx ; ++i) {
        v[j] += s->lambdas[i] * (s->vrtx[i][j] );
      }
    }
    if( dotprod(v,v) < sqdist_tmp){
      /* Keep simplex. Need to update sID only*/
      for (i = 0; i < s->nvrtx ; ++i) {
        s->wids[ s->nvrtx - 1 - i] = TrianglesToTest[ secondaux +( s->wids[i] *3) ] ;
      }
    }
    else {

      s->nvrtx = sTmp.nvrtx;
      for (i = 0; i < s->nvrtx ; ++i) {
        for (j = 0; j < 3; ++j) {
          s->vrtx[i][j] = sTmp.vrtx[i][j];
        }
        s->lambdas[i] = sTmp.lambdas[i];
        s->wids[ sTmp.nvrtx - 1 - i] = TrianglesToTest[ firstaux +( sTmp.wids[i] *3) ] ;
      }
    }

    return;
  }
  else if ( FacetsTest[1] + FacetsTest[2] + FacetsTest[3] == 2 ){
    /* Only one facet is facing the origin */
    if  ( FacetsTest[1] == 0 ){
      s->nvrtx = 3;
      for ( i = 0; i < 3; ++i) {
        s->vrtx[0][i] = s->vrtx[ 0 ][i];
        s->vrtx[1][i] = s->vrtx[ 1 ][i];
        s->vrtx[2][i] = s->vrtx[ 3 ][i];
      }
      S2D(s);
      for (i = 0; i < s->nvrtx ; ++i) {
        s->wids[i] = s->wids[i];
      }
      return;
    }
    else if  ( FacetsTest[2] == 0 ){
      /* The origin projection P faces the facet ABD */
      /* ... thus, remove the vertex C from the simplex. */
      s->nvrtx = 3;
      for ( i = 0; i < 3; ++i) {
        s->vrtx[0][i] = s->vrtx[ 0 ][i];
        s->vrtx[1][i] = s->vrtx[ 2 ][i];
        s->vrtx[2][i] = s->vrtx[ 3 ][i];
      }
      S2D(s);
      /* Keep simplex. Need to update sID only*/
      for (i = 2; i < s->nvrtx ; ++i) {
        /* Assume that vertex a is always included in sID. */
        s->wids[i] = s->wids[i] + 1;
      }
      return;
    }
    else if  ( FacetsTest[3] == 0 ){
      /* The origin projection P faces the facet ABC */
      s->nvrtx = 3;
      for ( i = 0; i < 3; ++i) {
        s->vrtx[0][i] = s->vrtx[ 1 ][i];
        s->vrtx[1][i] = s->vrtx[ 2 ][i];
        s->vrtx[2][i] = s->vrtx[ 3 ][i];
      }
      S2D(s);
      return;
    }
  }
  else {
    s->nvrtx = 3;
    for ( i = 0; i < 3; ++i) {
      s->vrtx[0][i] = s->vrtx[ 0 ][i];
      s->vrtx[1][i] = s->vrtx[ 1 ][i];
      s->vrtx[2][i] = s->vrtx[ 2 ][i];
    }
    S2D(s);
    for (i = 0; i < s->nvrtx ; ++i) {
      /* Assume that vertex a is always included in sID. */
      s->wids[i] = s->wids[i] + 1;
    }
    return;
  }
}


/**
* @brief Evaluate support function for polytopes.
*/
void support ( struct bd *body, double *v ) {

  int i;
  double s;
  double maxs;
  double *vrt;

  /* Initialise variables */
  maxs = dotprod (body->coord[0],v);
  body->s[0] = body->coord[0][0];
  body->s[1] = body->coord[0][1];
  body->s[2] = body->coord[0][2];

  for (i = 1; i < body->numpoints; ++i) {
    vrt = body->coord[i];
    /* Evaluate function */
    s = dotprod (vrt,v);
    if ( s > maxs ){
      maxs = s;
      body->s[0] = vrt[0];
      body->s[1] = vrt[1];
      body->s[2] = vrt[2];
    }
  }
}


/**
* @brief Signed Volumes method.
*/
void subalgorithm  ( struct simplex *s ) {

  switch ( s->nvrtx ){
    case 4: 
      S3D( s );
      break;
    case 3: 
      S2D( s );
      break;
    case 2: 
      S1D( s );
      break;
    default : 
      s->lambdas[0] = 1.0;
      break;
  }

}


/**
* @brief ~The GJK algorithm
*
*/
double gjk ( struct bd bd1, struct bd bd2, struct simplex *s) {

  int k = 0;                /**< Iteration counter            */
  int i;                    /**< General purpose counter      */
  int mk = 100;             /**< Maximum number of iterations of the GJK algorithm */
  double v[3];              /**< Search direction             */
  double vminus[3];         /**< Search direction             */
  double w[3];              /**< Vertex on CSO frontier       */
  double eps_rel = 1e-10;   /**< Tolerance on relative        */
  double eps_tot = 1e-13;
  double dd = -1;           /**< Squared distance             */
  int maxitreached = 0;     /**< Flag for maximum iterations  */
  int origininsimplex = 0;  /**< Flag for origin in simples   */
  int exeedtol_rel = 0;     /**< Flag for 1st exit condition  */
  int exeedtol_tot = 0;     /**< Flag for 2nd exit condition  */ 
  int nullV = 0;            /**< Flag for 3rd exit condition  */ 

#ifdef ADAPTIVEFP
  exactinit();
#endif

  /* Initialise search direction */
  for ( i = 0; i < 3; ++i) {
    v[i] = bd1.coord[0][i] - bd2.coord[0][i];
    vminus[i] = -v[i];
  }

  /* Begin GJK iteration */
  do {
    /* Increment iteration counter */
    k++;

    /* Support function on polytope A */
    support( &bd1 , vminus );
    /* Support function on polytope B */
    support( &bd2 , v );
    /* Minkowski difference */
    for ( i = 0; i < 3; ++i) {
      w[i] = bd1.s[i] - bd2.s[i];
    }

    /* Test first exit condition (can't move further) */
    exeedtol_rel = (norm2(v) - dotprod (v,w) ) <= eps_rel * norm2(v);
    if ( exeedtol_rel ){
      break;
    }
    nullV = norm2(v) < eps_rel;
    if (nullV) {
      break;
    }

    /* Add support vertex to simplex at the position nvrtx+1 */
    s->nvrtx++;
    for ( i = 0; i < 3; ++i) {
      s->vrtx[  s->nvrtx - 1  ][i] = w[i];
    }
    /* Invoke sub-distance algorithm */
    subalgorithm ( s );

    /* Termination tests */
    maxitreached = k == mk;
    origininsimplex = s->nvrtx == 4;
    exeedtol_tot = norm2 (v) <= eps_tot ;

    /* Update search direction */
    for ( i = 0; i < 3; ++i) {
      v[i] = 0;
    }

    for ( i = 0; i < s->nvrtx; ++i) {
      v[0] += (s->lambdas[i] * s->vrtx[i][0]);
      v[1] += (s->lambdas[i] * s->vrtx[i][1]);
      v[2] += (s->lambdas[i] * s->vrtx[i][2]);
    }
    for ( i = 0; i < 3; ++i) {
      vminus[i] = -v[i];
    }

  }while( !maxitreached && !origininsimplex && !exeedtol_tot && !exeedtol_rel );

  /* Compute distance */
  dd = sqrt ( norm2 (v));

  return dd;
}

#ifdef MATLABDOESMEXSTUFF 
/**
 * @brief Mex function for Matlab.
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

  double  multiplier;           
  double *inCoordsA; 
  double *inCoordsB;  
  size_t  nCoordsA;   
  size_t  nCoordsB;  
  size_t  ncols;    
  double *outMatrix;  
  int     i,j,idx;    
  double *distance;  
  int     r = 30;
  int     c = 3;
  int     count;

  /**************** PARSE INPUTS AND OUTPUTS **********************/
  /*----------------------------------------------------------------*/
  /* Examine input (right-hand-side) arguments. */
  if(nrhs!=2) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:nrhs","Two inputs required.");
  }
  /* Examine output (left-hand-side) arguments. */
  if(nlhs!=1) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:nlhs","One output required.");
  }

  /* make sure the two input arguments are any numerical type */
  /* .. first input */
  if( !mxIsNumeric(prhs[0])) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notNumeric","Input matrix must be type numeric.");
  }
  /* .. second input */
  if( !mxIsNumeric(prhs[1])) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notNumeric","Input matrix must be type numeric.");
  }

  /* make sure the two input arguments have 3 columns */
  /* .. first input */
  if(mxGetM(prhs[0])!=3) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notColumnVector","First input must have 3 columns.");
  }
  /* .. second input */
  if(mxGetM(prhs[1])!=3) {
    mexErrMsgIdAndTxt("MyToolbox:gjk:notColumnVector","Second input must have 3 columns.");
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
  plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);

  /* get a pointer to the real data in the output matrix */
  distance = mxGetPr(plhs[0]);


  /*----------------------------------------------------------------*/
  /* POPULATE BODIES' STRUCTURES  */

  struct bd       bd1; /* Structure of body A */
  struct bd       bd2; /* Structure of body B */
  
  /* Assign number of vertices to each body */
  bd1.numpoints = (int) nCoordsA;
  bd2.numpoints = (int) nCoordsB;

  double **pinCoordsA = (double **)malloc(bd1.numpoints * sizeof(double *));
  for (i=0; i< bd1.numpoints ; i++)
     pinCoordsA[i] = (double *)malloc(3 * sizeof(double));

  for (i = 0; i <  3; i++)
    for (j = 0; j < bd1.numpoints; j++)
      pinCoordsA[j][i] = inCoordsA[ i + j*3] ;

  double **pinCoordsB = (double **)malloc(bd2.numpoints * sizeof(double *));
  for (i=0; i< bd2.numpoints ; i++)
    pinCoordsB[i] = (double *)malloc(3 * sizeof(double));

  for (i = 0; i <  3; i++)
    for (j = 0; j < bd2.numpoints; j++)
      pinCoordsB[j][i] = inCoordsB[ i + j*3] ;

  bd1.coord = pinCoordsA;
  bd2.coord = pinCoordsB;

  /*----------------------------------------------------------------*/
  /*CALL COMPUTATIONAL ROUTINE  */
  
  struct simplex s;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

  /* Compute squared distance using GJK algorithm */
  distance[0] = gjk (bd1, bd2, &s);

  for (i=0; i< bd1.numpoints ; i++)
    free(pinCoordsA[i]);
  free( pinCoordsA );

  for (i=0; i< bd2.numpoints ; i++)
    free(pinCoordsB[i]);
  free( pinCoordsB );

}
#endif

/**
 * @brief Invoke this function from C# applications
 */
double csFunction( int nCoordsA, double *inCoordsA, int nCoordsB, double *inCoordsB )
{ 
  double distance = 0;
  int i, j;

  /*----------------------------------------------------------------*/
  /* POPULATE BODIES' STRUCTURES  */

  struct bd       bd1; /* Structure of body A */
  struct bd       bd2; /* Structure of body B */

  /* Assign number of vertices to each body */
  bd1.numpoints = (int) nCoordsA;
  bd2.numpoints = (int) nCoordsB;

  double **pinCoordsA = (double **)malloc(bd1.numpoints * sizeof(double *));
  for (i=0; i< bd1.numpoints ; i++)
     pinCoordsA[i] = (double *)malloc(3 * sizeof(double));

  for (i = 0; i <  3; i++)
    for (j = 0; j < bd1.numpoints; j++)
      pinCoordsA[j][i] = inCoordsA[ i*bd1.numpoints + j] ;

  double **pinCoordsB = (double **)malloc(bd2.numpoints * sizeof(double *));
  for (i=0; i< bd2.numpoints ; i++)
    pinCoordsB[i] = (double *)malloc(3 * sizeof(double));

  for (i = 0; i <  3; i++)
    for (j = 0; j < bd2.numpoints; j++)
      pinCoordsB[j][i] = inCoordsB[ i*bd2.numpoints + j] ;

  bd1.coord = pinCoordsA;
  bd2.coord = pinCoordsB;


  /*----------------------------------------------------------------*/
  /*CALL COMPUTATIONAL ROUTINE  */
  struct simplex s;

  /* Initialise simplex as empty */
  s.nvrtx = 0;

  /* Compute squared distance using GJK algorithm */
  distance = gjk (bd1, bd2, &s);

  for (i=0; i< bd1.numpoints ; i++)
    free(pinCoordsA[i]);
  free( pinCoordsA );

  for (i=0; i< bd2.numpoints ; i++)
    free(pinCoordsB[i]);
  free( pinCoordsB );

  return distance;
}
