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

// Unit tests for full GJK algorithm

#include <cmath>
#include <cstdlib>

#include <gtest/gtest.h>

#include "opengjk_simd.hh"

namespace opengjk {
namespace simd {
namespace {

// ============================================================================
// Helper Functions
// ============================================================================

void create_cube(Polytope* bd, gjkFloat cx, gjkFloat cy, gjkFloat cz, gjkFloat half_size) {
  bd->numpoints = 8;

  // Allocate coordinates
  static gjkFloat coords[8][3];
  static gjkFloat* coord_ptrs[8];

  gjkFloat offsets[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                         {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    coords[i][0] = cx + offsets[i][0] * half_size;
    coords[i][1] = cy + offsets[i][1] * half_size;
    coords[i][2] = cz + offsets[i][2] * half_size;
    coord_ptrs[i] = coords[i];
  }

  bd->coord = coord_ptrs;
}

void create_tetrahedron(Polytope* bd, gjkFloat cx, gjkFloat cy, gjkFloat cz,
                        gjkFloat scale) {
  bd->numpoints = 4;

  static gjkFloat coords[4][3];
  static gjkFloat* coord_ptrs[4];

  // Regular tetrahedron vertices
  coords[0][0] = cx + scale * 1.0;
  coords[0][1] = cy + scale * 1.0;
  coords[0][2] = cz + scale * 1.0;

  coords[1][0] = cx + scale * 1.0;
  coords[1][1] = cy + scale * (-1.0);
  coords[1][2] = cz + scale * (-1.0);

  coords[2][0] = cx + scale * (-1.0);
  coords[2][1] = cy + scale * 1.0;
  coords[2][2] = cz + scale * (-1.0);

  coords[3][0] = cx + scale * (-1.0);
  coords[3][1] = cy + scale * (-1.0);
  coords[3][2] = cz + scale * 1.0;

  for (int i = 0; i < 4; ++i) {
    coord_ptrs[i] = coords[i];
  }

  bd->coord = coord_ptrs;
}

// ============================================================================
// GJK Tests
// ============================================================================

TEST(GJKTest, IdenticalCubes) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    c2[i][0] = offs[i][0];
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Identical cubes should have distance 0
  EXPECT_NEAR(dist, 0.0, 1e-6);
}

TEST(GJKTest, SeparatedCubes) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    // Cube 1 centered at origin
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    // Cube 2 centered at (5, 0, 0)
    c2[i][0] = offs[i][0] + 5.0;
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Distance should be 5 - 1 - 1 = 3
  EXPECT_NEAR(dist, 3.0, 1e-6);
}

TEST(GJKTest, TouchingCubes) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    // Cube 2 touching cube 1 at x = 1
    c2[i][0] = offs[i][0] + 2.0;
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Touching cubes should have distance 0
  EXPECT_NEAR(dist, 0.0, 1e-6);
}

TEST(GJKTest, OverlappingCubes) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    // Cube 2 overlapping by 0.5
    c2[i][0] = offs[i][0] + 1.5;
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Overlapping cubes should have distance 0
  EXPECT_NEAR(dist, 0.0, 1e-6);
}

TEST(GJKTest, DiagonalSeparation) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    // Cube 2 separated diagonally
    c2[i][0] = offs[i][0] + 5.0;
    c2[i][1] = offs[i][1] + 5.0;
    c2[i][2] = offs[i][2] + 5.0;
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Corners (1,1,1) and (4,4,4) are closest
  // Distance = sqrt((4-1)^2 * 3) = sqrt(27) = 5.196...
  gjkFloat expected = std::sqrt(27.0);
  EXPECT_NEAR(dist, expected, 1e-4);
}

TEST(GJKTest, TetrahedronVsCube) {
  Polytope bd1, bd2;
  static gjkFloat c1[4][3], c2[8][3];
  static gjkFloat* p1[4];
  static gjkFloat* p2[8];

  // Tetrahedron centered at (5, 0, 0)
  gjkFloat tet[4][3] = {
      {5 + 1, 1, 1}, {5 + 1, -1, -1}, {5 - 1, 1, -1}, {5 - 1, -1, 1}};

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 4; ++i) {
    c1[i][0] = tet[i][0];
    c1[i][1] = tet[i][1];
    c1[i][2] = tet[i][2];
    p1[i] = c1[i];
  }

  for (int i = 0; i < 8; ++i) {
    c2[i][0] = offs[i][0];
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 4;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Distance should be positive (separated)
  EXPECT_GT(dist, 0.0);
  EXPECT_LT(dist, 10.0);
}

TEST(GJKTest, WitnessPoints) {
  Polytope bd1, bd2;
  static gjkFloat c1[8][3], c2[8][3];
  static gjkFloat* p1[8];
  static gjkFloat* p2[8];

  gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                      {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    c2[i][0] = offs[i][0] + 4.0;  // Separated by 2 units
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    p1[i] = c1[i];
    p2[i] = c2[i];
  }

  bd1.coord = p1;
  bd1.numpoints = 8;
  bd2.coord = p2;
  bd2.numpoints = 8;

  Simplex s;
  gjkFloat dist = compute_minimum_distance(bd1, bd2, &s);

  // Witness point 1 should be on face x = 1
  EXPECT_NEAR(s.witnesses[0][0], 1.0, 0.5);

  // Witness point 2 should be on face x = 3
  EXPECT_NEAR(s.witnesses[1][0], 3.0, 0.5);

  // Distance between witnesses should match computed distance
  gjkFloat w_dist = std::sqrt(std::pow(s.witnesses[1][0] - s.witnesses[0][0], 2) +
                              std::pow(s.witnesses[1][1] - s.witnesses[0][1], 2) +
                              std::pow(s.witnesses[1][2] - s.witnesses[0][2], 2));
  EXPECT_NEAR(dist, w_dist, 1e-4);
}

}  // namespace
}  // namespace simd
}  // namespace opengjk

