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
 * @file opengjk_gpu.cu
 * @author Mattia Montanari, Vismay Churiwala, Marcus Hedlund
 * @date 22 Jan 2026
 * @brief GPU (CUDA) implementation of OpenGJK - Public API implementation
 *
 * Public API functions that handle memory management and kernel launches
 * for the GPU-accelerated GJK algorithm.
 *
 * @see https://github.com/vismaychuriwala/OpenGJK-GPU
 * @see https://www.mattiamontanari.com/opengjk/
 */

#include "opengjk_gpu.h"

#include <stdio.h>
#include <stdlib.h>

// Declare kernel function (defined in gjk_kernel.cu)
extern __global__ void compute_minimum_distance_kernel(
    const gkPolytope* polytypes1,
    const gkPolytope* polytypes2,
    gkSimplex* simplices,
    gkFloat* distances,
    const int n
);

// Threads per computation (must match gjk_kernel.cu)
#define THREADS_PER_COMPUTATION 16
#define THREADS_PER_BLOCK 256

/**
 * @brief Host wrapper function for GPU distance computation
 *
 * Handles all GPU memory allocation, data transfers, kernel launch,
 * and cleanup for batch collision detection.
 *
 * @param n         Number of polytope pairs to process
 * @param bd1       Array of first polytopes (host memory)
 * @param bd2       Array of second polytopes (host memory)
 * @param simplices Array to store resulting simplices (host memory)
 * @param distances Array to store distances (host memory)
 */
void compute_minimum_distance(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances
) {
    gkPolytope *d_bd1, *d_bd2;
    gkFloat **d_coord1, **d_coord2;
    gkSimplex *d_simplices;
    gkFloat *d_distances;

    // Allocate and copy polytope data to device
    allocate_and_copy_device_arrays(n, bd1, bd2, &d_bd1, &d_bd2, &d_coord1, &d_coord2,
                           &d_simplices, &d_distances);

    // Compute distances (kernel initializes simplices internally)
    compute_minimum_distance_device(n, d_bd1, d_bd2, d_simplices, d_distances);

    // Copy results back to host
    cudaMemcpy(simplices, d_simplices, n * sizeof(gkSimplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, n * sizeof(gkFloat), cudaMemcpyDeviceToHost);

    // Free device memory
    free_device_arrays(n, d_bd1, d_bd2, d_coord1, d_coord2, d_simplices, d_distances);
}

/**
 * @brief Low-level GPU distance computation with device pointers
 *
 * Assumes all data is already on the GPU. Only launches the kernel without
 * performing any memory allocation or transfers. Use this when you manage
 * GPU memory externally for optimal performance.
 *
 * @param n           Number of polytope pairs to process
 * @param d_bd1       Array of first polytopes (device memory)
 * @param d_bd2       Array of second polytopes (device memory)
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
) {
    // Launch kernel (16 threads per collision pair)
    const int blocks = (n * THREADS_PER_COMPUTATION + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_minimum_distance_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_bd1, d_bd2, d_simplices, d_distances, n
    );

    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

/**
 * @brief Allocate device memory and copy polytope data to GPU
 *
 * Allocates GPU memory for polytopes, coordinate arrays, simplices, and distances.
 * Copies polytope data from host to device. Use with compute_minimum_distance_device
 * for repeated computations on static geometry.
 *
 * @param n           Number of polytope pairs
 * @param bd1         Array of first polytopes (host memory)
 * @param bd2         Array of second polytopes (host memory)
 * @param d_bd1       Output: device pointer to first polytope array
 * @param d_bd2       Output: device pointer to second polytope array
 * @param d_coord1    Output: host array of device coordinate pointers
 * @param d_coord2    Output: host array of device coordinate pointers
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
) {
    // Allocate device memory for polytopes, simplices, and distances
    cudaMalloc(d_bd1, n * sizeof(gkPolytope));
    cudaMalloc(d_bd2, n * sizeof(gkPolytope));
    cudaMalloc(d_simplices, n * sizeof(gkSimplex));
    cudaMalloc(d_distances, n * sizeof(gkFloat));

    // Allocate host arrays to hold device coordinate pointers
    *d_coord1 = (gkFloat**)malloc(n * sizeof(gkFloat*));
    *d_coord2 = (gkFloat**)malloc(n * sizeof(gkFloat*));

    // Allocate device memory for coordinate arrays and copy them
    for (int i = 0; i < n; i++) {
        cudaMalloc(&(*d_coord1)[i], bd1[i].numpoints * 3 * sizeof(gkFloat));
        cudaMalloc(&(*d_coord2)[i], bd2[i].numpoints * 3 * sizeof(gkFloat));

        cudaMemcpy((*d_coord1)[i], bd1[i].coord, bd1[i].numpoints * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);
        cudaMemcpy((*d_coord2)[i], bd2[i].coord, bd2[i].numpoints * 3 * sizeof(gkFloat), cudaMemcpyHostToDevice);
    }

    // Create temporary host arrays with updated coord pointers
    gkPolytope *h_bd1_temp = (gkPolytope*)malloc(n * sizeof(gkPolytope));
    gkPolytope *h_bd2_temp = (gkPolytope*)malloc(n * sizeof(gkPolytope));

    for (int i = 0; i < n; i++) {
        h_bd1_temp[i] = bd1[i];
        h_bd2_temp[i] = bd2[i];
        h_bd1_temp[i].coord = (*d_coord1)[i];  // Update to device pointer
        h_bd2_temp[i].coord = (*d_coord2)[i];  // Update to device pointer
    }

    // Copy polytope structures to device (with device coord pointers)
    cudaMemcpy(*d_bd1, h_bd1_temp, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_bd2, h_bd2_temp, n * sizeof(gkPolytope), cudaMemcpyHostToDevice);

    // Free temporary host arrays
    free(h_bd1_temp);
    free(h_bd2_temp);
}

/**
 * @brief Free device memory allocated by allocate_and_copy_device_arrays
 *
 * Frees all GPU memory and host tracking arrays allocated by allocate_and_copy_device_arrays.
 *
 * @param n           Number of polytope pairs (same as allocate call)
 * @param d_bd1       Device pointer to first polytope array
 * @param d_bd2       Device pointer to second polytope array
 * @param d_coord1    Host array of device coordinate pointers
 * @param d_coord2    Host array of device coordinate pointers
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
) {
    // Free device coordinate arrays
    for (int i = 0; i < n; i++) {
        cudaFree(d_coord1[i]);
        cudaFree(d_coord2[i]);
    }

    // Free device polytope, simplex, and distance arrays
    cudaFree(d_bd1);
    cudaFree(d_bd2);
    cudaFree(d_simplices);
    cudaFree(d_distances);

    // Free host arrays that tracked device coordinate pointers
    free(d_coord1);
    free(d_coord2);
}
