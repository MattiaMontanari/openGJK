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
#define gkFloat float
#define gkEpsilon FLT_EPSILON
#define gkSqrt sqrtf
#else
#define gkFloat double
#define gkEpsilon DBL_EPSILON
#define gkSqrt sqrt
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

/**
 * @brief Computes minimum distance using GPU pointers (device memory).
 *
 * Low-level API that assumes all data is already on the GPU. Does not perform
 * any memory allocation or transfers - only launches the kernel. Use this when
 * you manage GPU memory externally for optimal performance.
 *
 * @param n         Number of polytope pairs to process
 * @param d_bd1     Array of first polytopes (device memory)
 * @param d_bd2     Array of second polytopes (device memory)
 * @param d_simplices Array to store resulting simplices (device memory)
 * @param d_distances Array to store distances (device memory)
 *
 * @note All pointers must point to device (GPU) memory.
 * @note Polytope coord pointers within gkPolytope structs must also point to device memory.
 */
void compute_minimum_distance_device(
    const int n,
    const gkPolytope* d_bd1,
    const gkPolytope* d_bd2,
    gkSimplex* d_simplices,
    gkFloat* d_distances
);

/**
 * @brief Allocate device memory and copy polytope data to GPU.
 *
 * Mid-level API for managing GPU memory explicitly. Useful for static objects
 * that persist across multiple frames. Call once to allocate, then use
 * compute_minimum_distance_device repeatedly, and free_device_arrays when done.
 *
 * @param n           Number of polytope pairs to allocate for
 * @param bd1         Array of first polytopes (host memory)
 * @param bd2         Array of second polytopes (host memory)
 * @param d_bd1       Output: device pointer to first polytope array
 * @param d_bd2       Output: device pointer to second polytope array
 * @param d_coord1    Output: host array of device coordinate pointers for bd1
 * @param d_coord2    Output: host array of device coordinate pointers for bd2
 * @param d_simplices Output: device pointer to simplex array
 * @param d_distances Output: device pointer to distance array
 */
void allocate_and_copy_device_arrays(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkPolytope** d_bd1,
    gkPolytope** d_bd2,
    gkFloat*** d_coord1,
    gkFloat*** d_coord2,
    gkSimplex** d_simplices,
    gkFloat** d_distances
);

/**
 * @brief Free device memory allocated by allocate_and_copy_device_arrays.
 *
 * @param n           Number of polytope pairs (same as allocate call)
 * @param d_bd1       Device pointer to first polytope array
 * @param d_bd2       Device pointer to second polytope array
 * @param d_coord1    Host array of device coordinate pointers for bd1
 * @param d_coord2    Host array of device coordinate pointers for bd2
 * @param d_simplices Device pointer to simplex array
 * @param d_distances Device pointer to distance array
 */
void free_device_arrays(
    const int n,
    gkPolytope* d_bd1,
    gkPolytope* d_bd2,
    gkFloat** d_coord1,
    gkFloat** d_coord2,
    gkSimplex* d_simplices,
    gkFloat* d_distances
);

#ifdef __cplusplus
}
#endif

#endif  // OPENGJK_GPU_H
