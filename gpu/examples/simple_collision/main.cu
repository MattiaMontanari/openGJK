/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \\___|_| |_|\\_____|\\____/|_|\\_\\
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
 * @file example.cu
 * @author Vismay Churiwala, Marcus Hedlund
 * @date 22 Jan 2026
 * @brief Simple example demonstrating GPU-accelerated GJK collision detection
 *
 * This example reads polytopes from data files (same as scalar example)
 * and computes their minimum distance using the GPU implementation.
 */

#include "openGJK_GPU.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define fscanf_s fscanf

/**
 * @brief Read polytope coordinates from input file
 *
 * Reads vertex data and stores it in flattened format for GPU.
 * Format: [x0, y0, z0, x1, y1, z1, ...]
 */
int readinput(const char* inputfile, gkFloat** coords, int* numpoints) {
    int npoints = 0;
    FILE* fp;

    // Open file
#ifdef WIN32
    errno_t err;
    if ((err = fopen_s(&fp, inputfile, "r")) != 0) {
#else
    if ((fp = fopen(inputfile, "r")) == NULL) {
#endif
        fprintf(stdout, "ERROR: input file %s not found!\n", inputfile);
        fprintf(stdout, "  -> The file must be in the folder from which this "
                        "program is launched\n\n");
        return 1;
    }

    // Read number of vertices
    if (fscanf_s(fp, "%d", &npoints) != 1) {
        fclose(fp);
        return 1;
    }

    // Allocate flattened array for coordinates [x0,y0,z0,x1,y1,z1,...]
    gkFloat* arr = (gkFloat*)malloc(npoints * 3 * sizeof(gkFloat));

    // Read vertices into flattened array
    for (int i = 0; i < npoints; i++) {
#ifdef USE_32BITS
        if (fscanf_s(fp, "%f %f %f\n",
                     &arr[i*3 + 0], &arr[i*3 + 1], &arr[i*3 + 2]) != 3) {
#else
        if (fscanf_s(fp, "%lf %lf %lf\n",
                     &arr[i*3 + 0], &arr[i*3 + 1], &arr[i*3 + 2]) != 3) {
#endif
            free(arr);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    *coords = arr;
    *numpoints = npoints;

    return 0;
}

int main() {
    // Number of collision pairs (1 pair for this example)
    const int n = 1;

    // Allocate arrays for input polytopes and output results
    gkPolytope* bd1 = (gkPolytope*)malloc(n * sizeof(gkPolytope));
    gkPolytope* bd2 = (gkPolytope*)malloc(n * sizeof(gkPolytope));
    gkSimplex* simplices = (gkSimplex*)malloc(n * sizeof(gkSimplex));
    gkFloat* distances = (gkFloat*)malloc(n * sizeof(gkFloat));

    // Input file names
    char inputfileA[40] = "userP.dat";
    char inputfileB[40] = "userQ.dat";

    gkFloat* coords1 = NULL;
    gkFloat* coords2 = NULL;
    int nvrtx1, nvrtx2;

    // ========================================================================
    // Read polytope data from files
    // ========================================================================

    // Import coordinates of object 1
    if (readinput(inputfileA, &coords1, &nvrtx1)) {
        fprintf(stderr, "Failed to read %s\n", inputfileA);
        free(bd1);
        free(bd2);
        free(simplices);
        free(distances);
        return 1;
    }
    bd1[0].numpoints = nvrtx1;
    bd1[0].coord = coords1;

    // Import coordinates of object 2
    if (readinput(inputfileB, &coords2, &nvrtx2)) {
        fprintf(stderr, "Failed to read %s\n", inputfileB);
        free(coords1);
        free(bd1);
        free(bd2);
        free(simplices);
        free(distances);
        return 1;
    }
    bd2[0].numpoints = nvrtx2;
    bd2[0].coord = coords2;

    // ========================================================================
    // Compute minimum distance using GPU
    // ========================================================================

    printf("Computing minimum distance on GPU...\n");

    compute_minimum_distance(n, bd1, bd2, simplices, distances);

    // ========================================================================
    // Print results
    // ========================================================================

    printf("Distance between bodies %f\n", distances[0]);
    printf("Witnesses: (%f, %f, %f) and (%f, %f, %f)\n",
           simplices[0].witnesses[0][0], simplices[0].witnesses[0][1], simplices[0].witnesses[0][2],
           simplices[0].witnesses[1][0], simplices[0].witnesses[1][1], simplices[0].witnesses[1][2]);

    // Verify distance by computing Euclidean distance between witnesses
    gkFloat dx = simplices[0].witnesses[0][0] - simplices[0].witnesses[1][0];
    gkFloat dy = simplices[0].witnesses[0][1] - simplices[0].witnesses[1][1];
    gkFloat dz = simplices[0].witnesses[0][2] - simplices[0].witnesses[1][2];
    gkFloat computed_dist = gkSqrt(dx*dx + dy*dy + dz*dz);
    printf("Verification (witness distance): %f\n", computed_dist);

    // ========================================================================
    // Cleanup
    // ========================================================================

    free(coords1);
    free(coords2);
    free(bd1);
    free(bd2);
    free(simplices);
    free(distances);

    return 0;
}
