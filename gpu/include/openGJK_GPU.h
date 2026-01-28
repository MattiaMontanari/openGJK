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

#include <cuda.h>
#include <cuda_runtime.h>
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
#define gkFmax fmaxf
#else
#define gkFloat double
#define gkEpsilon DBL_EPSILON
#define gkSqrt sqrt
#define gkFmax fmax
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
 * @brief Index pair specifying which polytopes to check for collision.
 */
struct gkCollisionPair {
    int idx1;  /*!< Index of first polytope. */
    int idx2;  /*!< Index of second polytope. */
};

// ============================================================================
// LOW-LEVEL API: Direct kernel access
// ============================================================================

/*! @brief Invoke the warp-parallel GJK algorithm to compute the minimum distance between two
 * polytopes using 16 threads per collision. */
__global__ void compute_minimum_distance_kernel(const gkPolytope* polytopes1, const gkPolytope* polytopes2,
  gkSimplex* simplices, gkFloat* distances, const int n);

/*! @brief Invoke the warp-parallel GJK algorithm using indexed polytope pairs.
 * Uses 16 threads per collision. Thread i uses pairs[i] to look up polytopes. */
__global__ void compute_minimum_distance_indexed_kernel(
    const gkPolytope* polytopes,
    const gkCollisionPair* pairs,
    gkSimplex* simplices,
    gkFloat* distances,
    const int n
);

/*! @brief Invoke the warp-parallel EPA algorithm to compute penetration depth and witness points
 * for colliding polytopes using 32 threads (one warp) per collision.
 *
 * This should be called after GJK when a collision is detected (simplex has 4 vertices).
 * The function expands the GJK simplex into a full polytope to find the closest points
 * on the surfaces of the two polytopes.
 *
 * @param polytopes1 First set of polytopes
 * @param polytopes2 Second set of polytopes
 * @param simplices Simplex results from GJK (should have 4 vertices for collisions)
 * @param distances Output array for penetration depths (or distances if no collision)
 * @param witness1 Output array for witness points on first polytope (3 floats per collision)
 * @param witness2 Output array for witness points on second polytope (3 floats per collision)
 * @param contact_normals Output array for contact normals (3 floats per collision, points from polytope1 to polytope2)
 * @param n Number of polytope pairs to process
 */
__global__ void compute_epa_kernel(const gkPolytope* polytopes1, const gkPolytope* polytopes2,
  gkSimplex* simplices, gkFloat* distances, gkFloat* witness1, gkFloat* witness2, gkFloat* contact_normals, const int n);

// ============================================================================
// HIGH-LEVEL API: Automatic memory management
// ============================================================================

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
 * @brief Computes collision information using EPA algorithm on GPU.
 *
 * GPU implementation with warp-level parallelism (32 threads per collision pair).
 * Handles all GPU memory allocation and transfers internally.
 *
 * @param n               Number of polytope pairs to process
 * @param bd1             Array of first polytopes (host memory)
 * @param bd2             Array of second polytopes (host memory)
 * @param simplices       Array of simplices (host memory, input/output)
 * @param distances       Array of distances (host memory, input/output)
 * @param witness1        Array to store witness points on first polytope (n*3 floats)
 * @param witness2        Array to store witness points on second polytope (n*3 floats)
 * @param contact_normals Optional array to store contact normals (n*3 floats, can be nullptr)
 */
void compute_epa(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances,
    gkFloat* witness1,
    gkFloat* witness2,
    gkFloat* contact_normals = nullptr
);

/**
 * @brief Computes GJK and EPA combined on GPU.
 *
 * First runs GJK to detect collisions, then runs EPA for colliding pairs.
 * Handles all GPU memory allocation and transfers internally.
 *
 * @param n         Number of polytope pairs to process
 * @param bd1       Array of first polytopes (host memory)
 * @param bd2       Array of second polytopes (host memory)
 * @param simplices Array to store resulting simplices (host memory)
 * @param distances Array to store distances/penetration depths (host memory)
 * @param witness1  Array to store witness points on first polytope (n*3 floats)
 * @param witness2  Array to store witness points on second polytope (n*3 floats)
 */
void compute_gjk_epa(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances,
    gkFloat* witness1,
    gkFloat* witness2
);

/**
 * @brief Computes minimum distance using indexed polytope pairs (high-level API).
 *
 * Uses a single polytope array with index pairs for collision checks.
 * More efficient when polytopes are reused in multiple collision tests.
 * Handles all GPU memory allocation and transfers internally.
 *
 * @param num_polytopes Total number of unique polytopes
 * @param num_pairs     Number of collision pairs to check
 * @param polytopes     Array of all polytopes (host memory)
 * @param pairs         Array of index pairs specifying which polytopes to check (host memory)
 * @param simplices     Array to store resulting simplices (host memory)
 * @param distances     Array to store distances (host memory)
 */
void compute_minimum_distance_indexed(
    const int num_polytopes,
    const int num_pairs,
    const gkPolytope* polytopes,
    const gkCollisionPair* pairs,
    gkSimplex* simplices,
    gkFloat* distances
);

// ============================================================================
// MID-LEVEL API: Explicit memory management
// ============================================================================

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
 * @brief Computes EPA using GPU pointers (device memory).
 *
 * @param n               Number of polytope pairs
 * @param d_bd1           Device pointer to first polytopes
 * @param d_bd2           Device pointer to second polytopes
 * @param d_simplices     Device pointer to simplices
 * @param d_distances     Device pointer to distances
 * @param d_witness1      Device pointer to witness points for bd1
 * @param d_witness2      Device pointer to witness points for bd2
 * @param d_contact_normals Device pointer to contact normals (can be nullptr)
 */
void compute_epa_device(
    const int n,
    const gkPolytope* d_bd1,
    const gkPolytope* d_bd2,
    gkSimplex* d_simplices,
    gkFloat* d_distances,
    gkFloat* d_witness1,
    gkFloat* d_witness2,
    gkFloat* d_contact_normals = nullptr
);

/**
 * @brief Allocate device memory and copy polytope data to GPU (GJK only).
 *
 * Mid-level API for managing GPU memory explicitly. Use this for static objects
 * that persist across multiple frames. Only allocates memory needed for GJK.
 *
 * @param n           Number of polytope pairs to allocate for
 * @param bd1         Array of first polytopes (host memory)
 * @param bd2         Array of second polytopes (host memory)
 * @param d_bd1       Output: device pointer to first polytope array
 * @param d_bd2       Output: device pointer to second polytope array
 * @param d_coord1    Output: device pointer to concatenated coordinates for bd1
 * @param d_coord2    Output: device pointer to concatenated coordinates for bd2
 * @param d_simplices Output: device pointer to simplex array
 * @param d_distances Output: device pointer to distance array
 */
void allocate_and_copy_device_arrays(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkPolytope** d_bd1,
    gkPolytope** d_bd2,
    gkFloat** d_coord1,
    gkFloat** d_coord2,
    gkSimplex** d_simplices,
    gkFloat** d_distances
);

/**
 * @brief Allocate device memory for EPA outputs only.
 *
 * Call this separately if you need EPA. More efficient than allocating
 * witness points when only running GJK.
 *
 * @param n               Number of polytope pairs
 * @param d_witness1      Output: device pointer to witness points for bd1
 * @param d_witness2      Output: device pointer to witness points for bd2
 * @param d_contact_normals Output: device pointer to contact normals (nullptr to skip)
 */
void allocate_epa_device_arrays(
    const int n,
    gkFloat** d_witness1,
    gkFloat** d_witness2,
    gkFloat** d_contact_normals = nullptr
);

/**
 * @brief Copy GJK results from device to host memory.
 *
 * @param n           Number of polytope pairs
 * @param d_simplices Device pointer to simplex array (source)
 * @param d_distances Device pointer to distance array (source)
 * @param simplices   Host array to store simplices (destination)
 * @param distances   Host array to store distances (destination)
 */
void copy_results_from_device(
    const int n,
    const gkSimplex* d_simplices,
    const gkFloat* d_distances,
    gkSimplex* simplices,
    gkFloat* distances
);

/**
 * @brief Copy EPA results from device to host memory.
 *
 * @param n               Number of polytope pairs
 * @param d_witness1      Device pointer to witness points for bd1
 * @param d_witness2      Device pointer to witness points for bd2
 * @param d_contact_normals Device pointer to contact normals (nullptr to skip)
 * @param witness1        Host array for witness points (destination)
 * @param witness2        Host array for witness points (destination)
 * @param contact_normals Host array for contact normals (destination, nullptr to skip)
 */
void copy_epa_results_from_device(
    const int n,
    const gkFloat* d_witness1,
    const gkFloat* d_witness2,
    const gkFloat* d_contact_normals,
    gkFloat* witness1,
    gkFloat* witness2,
    gkFloat* contact_normals = nullptr
);

/**
 * @brief Free device memory allocated by allocate_and_copy_device_arrays.
 *
 * @param d_bd1       Device pointer to first polytope array
 * @param d_bd2       Device pointer to second polytope array
 * @param d_coord1    Device pointer to concatenated coordinates for bd1
 * @param d_coord2    Device pointer to concatenated coordinates for bd2
 * @param d_simplices Device pointer to simplex array
 * @param d_distances Device pointer to distance array
 */
void free_device_arrays(
    gkPolytope* d_bd1,
    gkPolytope* d_bd2,
    gkFloat* d_coord1,
    gkFloat* d_coord2,
    gkSimplex* d_simplices,
    gkFloat* d_distances
);

/**
 * @brief Free EPA device arrays allocated by allocate_epa_device_arrays.
 *
 * @param d_witness1      Device pointer to witness points for bd1
 * @param d_witness2      Device pointer to witness points for bd2
 * @param d_contact_normals Device pointer to contact normals (nullptr to skip)
 */
void free_epa_device_arrays(
    gkFloat* d_witness1,
    gkFloat* d_witness2,
    gkFloat* d_contact_normals = nullptr
);

#ifdef __cplusplus
}
#endif

#endif  // OPENGJK_GPU_H
