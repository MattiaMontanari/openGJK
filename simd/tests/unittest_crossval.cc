// Cross-validation tests: SIMD vs Scalar OpenGJK
//
// IMPORTANT: These tests compare the DISTANCE output (v) between SIMD and scalar
// implementations. The support vertices in the simplex may differ between
// implementations due to:
//   - Different floating-point rounding across SIMD architectures
//   - Different vertex selection order in degenerate cases
//   - Tie-breaking differences when multiple vertices are equidistant
//
// This is acceptable because the GJK algorithm guarantees the same minimum
// distance regardless of which valid support vertices are selected.

#include <cmath>
#include <cstdlib>
#include <random>

#include <gtest/gtest.h>

#include "opengjk_simd.hh"

// Include scalar version
extern "C" {
#include "openGJK/openGJK.h"
}

namespace opengjk {
namespace {

// ============================================================================
// Helper Functions
// ============================================================================

// Random number generator
class RandomGenerator {
public:
  RandomGenerator(unsigned seed = 42) : gen_(seed), dist_(-10.0, 10.0) {}

  simd::gjkFloat
  random() {
    return dist_(gen_);
  }

  void
  random_point(simd::gjkFloat* p) {
    p[0] = random();
    p[1] = random();
    p[2] = random();
  }

private:
  std::mt19937 gen_;
  std::uniform_real_distribution<simd::gjkFloat> dist_;
};

void
create_random_convex_hull(simd::Polytope* simd_poly, gkPolytope* scalar_poly, int num_points, RandomGenerator& rng) {
  static simd::gjkFloat coords[100][3];
  static simd::gjkFloat* coord_ptrs[100];
  static gkFloat scalar_coords[100][3];
  static gkFloat* scalar_ptrs[100];

  for (int i = 0; i < num_points && i < 100; ++i) {
    rng.random_point(coords[i]);
    scalar_coords[i][0] = static_cast<gkFloat>(coords[i][0]);
    scalar_coords[i][1] = static_cast<gkFloat>(coords[i][1]);
    scalar_coords[i][2] = static_cast<gkFloat>(coords[i][2]);
    coord_ptrs[i] = coords[i];
    scalar_ptrs[i] = scalar_coords[i];
  }

  simd_poly->coord = coord_ptrs;
  simd_poly->numpoints = num_points;

  scalar_poly->coord = scalar_ptrs;
  scalar_poly->numpoints = num_points;
}

void
create_random_cube(simd::Polytope* simd_poly, gkPolytope* scalar_poly, simd::gjkFloat cx, simd::gjkFloat cy,
                   simd::gjkFloat cz, simd::gjkFloat half_size) {
  static simd::gjkFloat coords[8][3];
  static simd::gjkFloat* coord_ptrs[8];
  static gkFloat scalar_coords[8][3];
  static gkFloat* scalar_ptrs[8];

  simd::gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                               {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    coords[i][0] = cx + offs[i][0] * half_size;
    coords[i][1] = cy + offs[i][1] * half_size;
    coords[i][2] = cz + offs[i][2] * half_size;
    scalar_coords[i][0] = static_cast<gkFloat>(coords[i][0]);
    scalar_coords[i][1] = static_cast<gkFloat>(coords[i][1]);
    scalar_coords[i][2] = static_cast<gkFloat>(coords[i][2]);
    coord_ptrs[i] = coords[i];
    scalar_ptrs[i] = scalar_coords[i];
  }

  simd_poly->coord = coord_ptrs;
  simd_poly->numpoints = 8;

  scalar_poly->coord = scalar_ptrs;
  scalar_poly->numpoints = 8;
}

// ============================================================================
// Cross-validation Tests
// ============================================================================

TEST(CrossValidationTest, IdenticalCubes) {
  simd::Polytope simd_bd1, simd_bd2;
  gkPolytope scalar_bd1, scalar_bd2;

  static simd::gjkFloat c1[8][3], c2[8][3];
  static simd::gjkFloat* p1[8];
  static simd::gjkFloat* p2[8];
  static gkFloat sc1[8][3], sc2[8][3];
  static gkFloat* sp1[8];
  static gkFloat* sp2[8];

  simd::gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                               {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    c2[i][0] = offs[i][0];
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    sc1[i][0] = static_cast<gkFloat>(offs[i][0]);
    sc1[i][1] = static_cast<gkFloat>(offs[i][1]);
    sc1[i][2] = static_cast<gkFloat>(offs[i][2]);
    sc2[i][0] = static_cast<gkFloat>(offs[i][0]);
    sc2[i][1] = static_cast<gkFloat>(offs[i][1]);
    sc2[i][2] = static_cast<gkFloat>(offs[i][2]);
    p1[i] = c1[i];
    p2[i] = c2[i];
    sp1[i] = sc1[i];
    sp2[i] = sc2[i];
  }

  simd_bd1.coord = p1;
  simd_bd1.numpoints = 8;
  simd_bd2.coord = p2;
  simd_bd2.numpoints = 8;

  scalar_bd1.coord = sp1;
  scalar_bd1.numpoints = 8;
  scalar_bd2.coord = sp2;
  scalar_bd2.numpoints = 8;

  simd::Simplex simd_s;
  gkSimplex scalar_s;

  simd::gjkFloat simd_dist = simd::compute_minimum_distance(simd_bd1, simd_bd2, &simd_s);
  gkFloat scalar_dist = compute_minimum_distance(scalar_bd1, scalar_bd2, &scalar_s);

  EXPECT_NEAR(simd_dist, static_cast<simd::gjkFloat>(scalar_dist), 1e-5);
}

TEST(CrossValidationTest, SeparatedCubes) {
  simd::Polytope simd_bd1, simd_bd2;
  gkPolytope scalar_bd1, scalar_bd2;

  static simd::gjkFloat c1[8][3], c2[8][3];
  static simd::gjkFloat* p1[8];
  static simd::gjkFloat* p2[8];
  static gkFloat sc1[8][3], sc2[8][3];
  static gkFloat* sp1[8];
  static gkFloat* sp2[8];

  simd::gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                               {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    c2[i][0] = offs[i][0] + 5.0;
    c2[i][1] = offs[i][1];
    c2[i][2] = offs[i][2];
    sc1[i][0] = static_cast<gkFloat>(c1[i][0]);
    sc1[i][1] = static_cast<gkFloat>(c1[i][1]);
    sc1[i][2] = static_cast<gkFloat>(c1[i][2]);
    sc2[i][0] = static_cast<gkFloat>(c2[i][0]);
    sc2[i][1] = static_cast<gkFloat>(c2[i][1]);
    sc2[i][2] = static_cast<gkFloat>(c2[i][2]);
    p1[i] = c1[i];
    p2[i] = c2[i];
    sp1[i] = sc1[i];
    sp2[i] = sc2[i];
  }

  simd_bd1.coord = p1;
  simd_bd1.numpoints = 8;
  simd_bd2.coord = p2;
  simd_bd2.numpoints = 8;

  scalar_bd1.coord = sp1;
  scalar_bd1.numpoints = 8;
  scalar_bd2.coord = sp2;
  scalar_bd2.numpoints = 8;

  simd::Simplex simd_s;
  gkSimplex scalar_s;

  simd::gjkFloat simd_dist = simd::compute_minimum_distance(simd_bd1, simd_bd2, &simd_s);
  gkFloat scalar_dist = compute_minimum_distance(scalar_bd1, scalar_bd2, &scalar_s);

  EXPECT_NEAR(simd_dist, static_cast<simd::gjkFloat>(scalar_dist), 1e-5);
}

TEST(CrossValidationTest, RandomCubes) {
  RandomGenerator rng(12345);

  for (int test = 0; test < 10; ++test) {
    simd::Polytope simd_bd1, simd_bd2;
    gkPolytope scalar_bd1, scalar_bd2;

    static simd::gjkFloat c1[8][3], c2[8][3];
    static simd::gjkFloat* p1[8];
    static simd::gjkFloat* p2[8];
    static gkFloat sc1[8][3], sc2[8][3];
    static gkFloat* sp1[8];
    static gkFloat* sp2[8];

    simd::gjkFloat cx1 = rng.random();
    simd::gjkFloat cy1 = rng.random();
    simd::gjkFloat cz1 = rng.random();
    simd::gjkFloat cx2 = rng.random();
    simd::gjkFloat cy2 = rng.random();
    simd::gjkFloat cz2 = rng.random();

    simd::gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                                 {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

    for (int i = 0; i < 8; ++i) {
      c1[i][0] = cx1 + offs[i][0];
      c1[i][1] = cy1 + offs[i][1];
      c1[i][2] = cz1 + offs[i][2];
      c2[i][0] = cx2 + offs[i][0];
      c2[i][1] = cy2 + offs[i][1];
      c2[i][2] = cz2 + offs[i][2];
      sc1[i][0] = static_cast<gkFloat>(c1[i][0]);
      sc1[i][1] = static_cast<gkFloat>(c1[i][1]);
      sc1[i][2] = static_cast<gkFloat>(c1[i][2]);
      sc2[i][0] = static_cast<gkFloat>(c2[i][0]);
      sc2[i][1] = static_cast<gkFloat>(c2[i][1]);
      sc2[i][2] = static_cast<gkFloat>(c2[i][2]);
      p1[i] = c1[i];
      p2[i] = c2[i];
      sp1[i] = sc1[i];
      sp2[i] = sc2[i];
    }

    simd_bd1.coord = p1;
    simd_bd1.numpoints = 8;
    simd_bd2.coord = p2;
    simd_bd2.numpoints = 8;

    scalar_bd1.coord = sp1;
    scalar_bd1.numpoints = 8;
    scalar_bd2.coord = sp2;
    scalar_bd2.numpoints = 8;

    simd::Simplex simd_s;
    gkSimplex scalar_s;

    simd::gjkFloat simd_dist = simd::compute_minimum_distance(simd_bd1, simd_bd2, &simd_s);
    gkFloat scalar_dist = compute_minimum_distance(scalar_bd1, scalar_bd2, &scalar_s);

    EXPECT_NEAR(simd_dist, static_cast<simd::gjkFloat>(scalar_dist), 1e-4) << "Test " << test << " failed";
  }
}

TEST(CrossValidationTest, DiagonalSeparation) {
  simd::Polytope simd_bd1, simd_bd2;
  gkPolytope scalar_bd1, scalar_bd2;

  static simd::gjkFloat c1[8][3], c2[8][3];
  static simd::gjkFloat* p1[8];
  static simd::gjkFloat* p2[8];
  static gkFloat sc1[8][3], sc2[8][3];
  static gkFloat* sp1[8];
  static gkFloat* sp2[8];

  simd::gjkFloat offs[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                               {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};

  for (int i = 0; i < 8; ++i) {
    c1[i][0] = offs[i][0];
    c1[i][1] = offs[i][1];
    c1[i][2] = offs[i][2];
    c2[i][0] = offs[i][0] + 5.0;
    c2[i][1] = offs[i][1] + 5.0;
    c2[i][2] = offs[i][2] + 5.0;
    sc1[i][0] = static_cast<gkFloat>(c1[i][0]);
    sc1[i][1] = static_cast<gkFloat>(c1[i][1]);
    sc1[i][2] = static_cast<gkFloat>(c1[i][2]);
    sc2[i][0] = static_cast<gkFloat>(c2[i][0]);
    sc2[i][1] = static_cast<gkFloat>(c2[i][1]);
    sc2[i][2] = static_cast<gkFloat>(c2[i][2]);
    p1[i] = c1[i];
    p2[i] = c2[i];
    sp1[i] = sc1[i];
    sp2[i] = sc2[i];
  }

  simd_bd1.coord = p1;
  simd_bd1.numpoints = 8;
  simd_bd2.coord = p2;
  simd_bd2.numpoints = 8;

  scalar_bd1.coord = sp1;
  scalar_bd1.numpoints = 8;
  scalar_bd2.coord = sp2;
  scalar_bd2.numpoints = 8;

  simd::Simplex simd_s;
  gkSimplex scalar_s;

  simd::gjkFloat simd_dist = simd::compute_minimum_distance(simd_bd1, simd_bd2, &simd_s);
  gkFloat scalar_dist = compute_minimum_distance(scalar_bd1, scalar_bd2, &scalar_s);

  EXPECT_NEAR(simd_dist, static_cast<simd::gjkFloat>(scalar_dist), 1e-4);
}

} // namespace
} // namespace opengjk
