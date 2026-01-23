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
 * @file opengjk_gpu.h
 * @author Mattia Montanari, Vismay Churiwala, Marcus Hedlund
 * @date 22 Jan 2026
 * @brief GPU (CUDA) implementation of OpenGJK - Public API
 *
 * GPU acceleration implementation with warp-level parallelism for
 * high-performance collision detection on NVIDIA GPUs.
 *
 * @see https://github.com/vismaychuriwala/OpenGJK-GPU
 * @see https://www.mattiamontanari.com/opengjk/
 */

#ifndef OPENGJK_GPU_H
#define OPENGJK_GPU_H

#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! @brief Precision of floating-point numbers.
 *
 * Controlled by USE_32BITS compile flag (set in root CMakeLists.txt). */
#ifdef USE_32BITS
#define gkFloat   float
#define gkEpsilon FLT_EPSILON
#else
#define gkFloat   double
#define gkEpsilon DBL_EPSILON
#endif

/*! @brief Data structure for convex polytopes (GPU version).
 *
 * GPU implementation uses flattened coordinate array for efficient
 * memory access and coalescing. */
typedef struct gkPolytope_ {
  int numpoints;   /*!< Number of points defining the polytope. */
  gkFloat s[3];    /*!< Furthest point returned by support function. */
  int s_idx;       /*!< Index of the furthest point. */
  gkFloat* coord;  /*!< Flattened coordinates [x0,y0,z0,x1,y1,z1,...]. */
} gkPolytope;

/*! @brief Data structure for simplex. */
typedef struct gkSimplex_ {
  int nvrtx;               /*!< Number of vertices in simplex (1-4). */
  gkFloat vrtx[4][3];      /*!< Simplex vertex coordinates. */
  int vrtx_idx[4][2];      /*!< Vertex indices [vertex][body]. */
  gkFloat witnesses[2][3]; /*!< Witness points (closest points on each body). */
} gkSimplex;

/**
 * @brief Computes minimum distance between polytopes using GJK algorithm on GPU.
 *
 * GPU implementation with warp-level parallelism (16 threads per collision pair)
 * for high-performance batch collision detection. Handles all GPU memory allocation
 * and transfers internally.
 *
 * @param n         Number of polytope pairs to process
 * @param bd1       Array of first polytopes (host memory)
 * @param bd2       Array of second polytopes (host memory)
 * @param simplices Array to store resulting simplices (host memory)
 * @param distances Array to store distances (host memory, 0.0 indicates collision)
 *
 * @note Polytope coordinates (bd.coord) should use flattened array format:
 *       [x0, y0, z0, x1, y1, z1, ...] for efficient GPU memory access.
 * @note Precision (float/double) controlled by USE_32BITS compile flag.
 */
void compute_minimum_distance(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances
);

#ifdef __cplusplus
}
#endif

#endif  // OPENGJK_GPU_H
