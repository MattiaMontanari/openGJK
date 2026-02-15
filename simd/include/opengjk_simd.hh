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
 * @file opengjk_simd.hh
 * @author Mattia Montanari
 * @date Jan 2026
 * @brief SIMD-accelerated GJK implementation using Google Highway.
 *
 * This header provides the main API for the SIMD GJK implementation.
 * The implementation uses dynamic dispatch to select the best SIMD
 * instruction set at runtime.
 *
 * @see https://www.mattiamontanari.com/opengjk/
 */

#ifndef OPENGJK_SIMD_HH__
#define OPENGJK_SIMD_HH__

#include <cstddef>
#include <cstdint>

// Highway base for HWY_ALIGN_MAX
#include "hwy/base.h"

namespace opengjk {
namespace simd {

/** Maximum number of GJK iterations before termination */
constexpr unsigned int kMaxIterations = 25;

/** Relative tolerance multiplier for convergence check */
constexpr double kEpsilonRelMult = 1e4;

/** Absolute tolerance multiplier for convergence check */
constexpr double kEpsilonAbsMult = 1e2;

/**
 * @brief Precision selector for GJK computation.
 *
 * The SIMD implementation supports both single and double precision.
 * Use float for better performance, double for higher accuracy.
 */
#ifdef OPENGJK_SIMD_USE_FLOAT
using gjkFloat = float;
constexpr gjkFloat kEpsilon = 1.192092896e-07f; // FLT_EPSILON
#else
using gjkFloat = double;
constexpr gjkFloat kEpsilon = 2.2204460492503131e-16; // DBL_EPSILON
#endif

constexpr gjkFloat kEpsilonRel = kEpsilon * static_cast<gjkFloat>(kEpsilonRelMult);
constexpr gjkFloat kEpsilonAbs = kEpsilon * static_cast<gjkFloat>(kEpsilonAbsMult);

/**
 * @brief Data structure for convex polytopes.
 *
 * Polytopes are three-dimensional convex shapes. The GJK algorithm works
 * directly on their vertices without explicitly computing the convex-hull.
 */
struct Polytope {
  int numpoints;            ///< Number of vertices defining the polytope.
  gjkFloat s[4];            ///< Furthest point from support function (padded to 4).
  int s_idx;                ///< Index of the furthest point.
  const gjkFloat* coords_x; ///< X coordinates of all vertices (SoA layout).
  const gjkFloat* coords_y; ///< Y coordinates of all vertices (SoA layout).
  const gjkFloat* coords_z; ///< Z coordinates of all vertices (SoA layout).

  // Alternative: Array of Structures layout (for compatibility with scalar)
  gjkFloat** coord; ///< Coordinates in AoS format [numpoints][3].
};

/**
 * @brief Data structure for simplex.
 *
 * The simplex is updated at each GJK iteration and contains up to 4 vertices.
 * Uses aligned storage for SIMD operations. HWY_ALIGN_MAX ensures proper
 * alignment for all SIMD targets (typically 64 bytes for AVX-512).
 */
struct HWY_ALIGN_MAX Simplex {
  int nvrtx;                              ///< Number of vertices in the simplex (1-4).
  HWY_ALIGN_MAX gjkFloat vrtx[4][4];      ///< Vertices of the simplex (padded to 4 components).
  int vrtx_idx[4][2];                     ///< Indices [vertex][body] for witness computation.
  HWY_ALIGN_MAX gjkFloat witnesses[2][4]; ///< Closest points on each body (padded).
};

/**
 * @brief Compute minimum distance between two convex polytopes using GJK.
 *
 * This is the main entry point for the SIMD GJK implementation. It uses
 * dynamic dispatch to select the optimal SIMD instruction set at runtime.
 *
 * @param[in]     bd1  First polytope.
 * @param[in]     bd2  Second polytope.
 * @param[in,out] s    Simplex structure. Must have nvrtx = 0 before first call.
 *                     After return, contains the final simplex and witness points.
 * @return The minimum Euclidean distance between the two polytopes.
 *         Returns 0 if the polytopes are intersecting or touching.
 */
gjkFloat compute_minimum_distance(const Polytope& bd1, const Polytope& bd2, Simplex* s);

/**
 * @brief Compute minimum distance (templated version for specific precision).
 *
 * This overload allows explicit precision selection at compile time.
 *
 * @tparam T  Floating-point type (float or double).
 */
template <typename T>
T compute_minimum_distance_t(const Polytope& bd1, const Polytope& bd2, Simplex* s);

} // namespace simd
} // namespace opengjk

#endif // OPENGJK_SIMD_HH__
