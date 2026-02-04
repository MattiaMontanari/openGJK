"""
Example usage of openGJK GPU Python wrapper.

This demonstrates:
1. Single collision pair detection
2. Batch processing multiple pairs
3. EPA for penetration depth
4. Indexed API for efficient polytope reuse
"""

import numpy as np
from pyopengjk_gpu import (
    compute_minimum_distance,
    compute_epa,
    compute_gjk_epa,
    compute_minimum_distance_indexed
)


def example_single_pair():
    """Example 1: Single collision pair (same as C example)."""
    print("=" * 70)
    print("Example 1: Single Collision Pair")
    print("=" * 70)

    # Define two polytopes (same as userP.dat and userQ.dat from C example)
    vertices1 = np.array([
        [0.0, 5.5, 0.0],
        [2.3, 1.0, -2.0],
        [8.1, 4.0, 2.4],
        [4.3, 5.0, 2.2],
        [2.5, 1.0, 2.3],
        [7.1, 1.0, 2.4],
        [1.0, 1.5, 0.3],
        [3.3, 0.5, 0.3],
        [6.0, 1.4, 0.2]
    ], dtype=np.float32)

    vertices2 = np.array([
        [0.0, -5.5, 0.0],
        [-2.3, -1.0, 2.0],
        [-8.1, -4.0, -2.4],
        [-4.3, -5.0, -2.2],
        [-2.5, -1.0, -2.3],
        [-7.1, -1.0, -2.4],
        [-1.0, -1.5, -0.3],
        [-3.3, -0.5, -0.3],
        [-6.0, -1.4, -0.2]
    ], dtype=np.float32)

    # Compute distance
    result = compute_minimum_distance(vertices1, vertices2)

    distance = result['distances'][0]
    w1 = result['witnesses1'][0]
    w2 = result['witnesses2'][0]

    print(f"Distance: {distance:.6f}")
    print(f"Collision: {result['is_collision'][0]}")
    print(f"Witness points:")
    print(f"  Body 1: ({w1[0]:.6f}, {w1[1]:.6f}, {w1[2]:.6f})")
    print(f"  Body 2: ({w2[0]:.6f}, {w2[1]:.6f}, {w2[2]:.6f})")

    # Verify distance
    computed_dist = np.linalg.norm(w1 - w2)
    print(f"Verification (witness distance): {computed_dist:.6f}")
    print()


def example_batch_processing():
    """Example 2: Batch processing multiple collision pairs."""
    print("=" * 70)
    print("Example 2: Batch Processing (1000 collision pairs)")
    print("=" * 70)

    # Generate 1000 random polytope pairs
    num_pairs = 1000
    np.random.seed(42)

    vertices1_list = []
    vertices2_list = []

    for i in range(num_pairs):
        # Random polytopes with 5-10 vertices
        n1 = np.random.randint(5, 11)
        n2 = np.random.randint(5, 11)

        # First polytope centered around (10, 0, 0)
        v1 = np.random.randn(n1, 3).astype(np.float32) * 2 + np.array([10.0, 0.0, 0.0], dtype=np.float32)

        # Second polytope centered around (-10, 0, 0) - separated
        v2 = np.random.randn(n2, 3).astype(np.float32) * 2 + np.array([-10.0, 0.0, 0.0], dtype=np.float32)

        vertices1_list.append(v1)
        vertices2_list.append(v2)

    # Process all pairs in one GPU call
    import time
    start = time.time()
    result = compute_minimum_distance(vertices1_list, vertices2_list)
    elapsed = time.time() - start

    # Analyze results using vectorized operations
    distances = result['distances']
    is_collision = result['is_collision']
    num_collisions = is_collision.sum()

    print(f"Processed {num_pairs} pairs in {elapsed*1000:.2f} ms")
    print(f"Throughput: {num_pairs/elapsed:.0f} pairs/second")
    print(f"Collisions detected: {num_collisions}")
    print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"Average distance: {distances.mean():.3f}")
    print()


def example_collision_and_epa():
    """Example 3: Collision detection with EPA for penetration depth."""
    print("=" * 70)
    print("Example 3: Collision Detection with EPA")
    print("=" * 70)

    # Create two overlapping cubes
    cube1 = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    # Second cube slightly overlapping (shifted by 1.5 instead of 2)
    cube2 = np.array([
        [0.5, -1, -1], [2.5, -1, -1], [0.5, 1, -1], [2.5, 1, -1],
        [0.5, -1, 1], [2.5, -1, 1], [0.5, 1, 1], [2.5, 1, 1]
    ], dtype=np.float32)

    # Run GJK first
    gjk_result = compute_minimum_distance(cube1, cube2)
    print(f"GJK Distance: {gjk_result['distances'][0]:.6f}")
    print(f"Collision: {gjk_result['is_collision'][0]}")

    if gjk_result['is_collision'][0]:
        # Run EPA for penetration info
        epa_result = compute_epa(cube1, cube2, return_normals=True)
        print(f"\nEPA Results:")
        print(f"Penetration depth: {epa_result['penetration_depths'][0]:.6f}")
        print(f"Contact point 1: {epa_result['witnesses1'][0]}")
        print(f"Contact point 2: {epa_result['witnesses2'][0]}")
        print(f"Contact normal: {epa_result['contact_normals'][0]}")

    # Combined GJK+EPA (more efficient)
    print(f"\nUsing combined GJK+EPA:")
    combined_result = compute_gjk_epa(cube1, cube2)
    print(f"Collision: {combined_result['is_collision'][0]}")
    if combined_result['is_collision'][0]:
        print(f"Penetration: {combined_result['distances'][0]:.6f}")
        print(f"Contact points: {combined_result['witnesses1'][0]} <-> {combined_result['witnesses2'][0]}")
    print()


def example_indexed_api():
    """Example 4: Indexed API for efficient polytope reuse."""
    print("=" * 70)
    print("Example 4: Indexed API (Polytope Reuse)")
    print("=" * 70)

    # Define 4 unique polytopes
    cube = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    sphere_approx = np.random.randn(20, 3).astype(np.float32)
    sphere_approx = sphere_approx / np.linalg.norm(sphere_approx, axis=1, keepdims=True) * 2

    tetra1 = np.array([
        [0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 2]
    ], dtype=np.float32) + np.array([10, 0, 0], dtype=np.float32)

    tetra2 = np.array([
        [0, 0, 0], [2, 0, 0], [1, 2, 0], [1, 1, 2]
    ], dtype=np.float32) + np.array([-10, 0, 0], dtype=np.float32)

    polytopes = [cube, sphere_approx, tetra1, tetra2]

    # Check all unique pairs using NumPy array
    pairs = np.array([
        [0, 1],  # cube vs sphere
        [0, 2],  # cube vs tetra1
        [0, 3],  # cube vs tetra2
        [1, 2],  # sphere vs tetra1
        [1, 3],  # sphere vs tetra2
        [2, 3],  # tetra1 vs tetra2
    ], dtype=np.int32)

    print(f"Checking {len(polytopes)} unique polytopes in {len(pairs)} pairs")
    print(f"Pairs: {pairs.tolist()}")

    # Run indexed collision detection
    result = compute_minimum_distance_indexed(polytopes, pairs)

    print(f"\nResults:")
    for i, (idx1, idx2) in enumerate(pairs):
        print(f"  Pair ({idx1},{idx2}): distance = {result['distances'][i]:.3f}, "
              f"collision = {result['is_collision'][i]}")
    print()


def example_vectorized_analysis():
    """Example 5: Vectorized analysis of batch results."""
    print("=" * 70)
    print("Example 5: Vectorized Analysis")
    print("=" * 70)

    # Generate random polytopes
    num_pairs = 100
    np.random.seed(123)

    # Mix of colliding and non-colliding pairs
    vertices1_list = []
    vertices2_list = []

    for i in range(num_pairs):
        n1, n2 = 8, 8
        v1 = np.random.randn(n1, 3).astype(np.float32) * 2

        # 50% colliding, 50% separated
        if i < num_pairs // 2:
            # Colliding: small offset
            v2 = v1 + np.random.randn(n2, 3).astype(np.float32) * 0.5
        else:
            # Separated: large offset
            v2 = v1 + np.array([20, 0, 0], dtype=np.float32) + np.random.randn(n2, 3).astype(np.float32) * 2

        vertices1_list.append(v1)
        vertices2_list.append(v2)

    # Batch process
    result = compute_minimum_distance(vertices1_list, vertices2_list)

    # Vectorized analysis
    distances = result['distances']
    is_collision = result['is_collision']
    witnesses1 = result['witnesses1']
    witnesses2 = result['witnesses2']

    # Statistics using NumPy vectorized operations
    print(f"Total pairs: {num_pairs}")
    print(f"Collisions: {is_collision.sum()}")
    print(f"Non-collisions: {(~is_collision).sum()}")
    print(f"\nDistance statistics:")
    print(f"  Min: {distances.min():.6f}")
    print(f"  Max: {distances.max():.6f}")
    print(f"  Mean: {distances.mean():.6f}")
    print(f"  Std: {distances.std():.6f}")

    # Analyze only non-colliding pairs using boolean indexing
    non_colliding_dists = distances[~is_collision]
    if len(non_colliding_dists) > 0:
        print(f"\nNon-colliding pairs statistics:")
        print(f"  Mean distance: {non_colliding_dists.mean():.3f}")
        print(f"  Median distance: {np.median(non_colliding_dists):.3f}")

    # Verify witness distances match reported distances (vectorized)
    witness_dists = np.linalg.norm(witnesses1 - witnesses2, axis=1)
    dist_error = np.abs(distances - witness_dists)
    print(f"\nWitness verification:")
    print(f"  Max error: {dist_error.max():.9f}")
    print(f"  All within tolerance: {np.all(dist_error < 1e-5)}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" OpenGJK GPU - Python Examples (NumPy-based)")
    print("=" * 70 + "\n")

    try:
        example_single_pair()
        example_batch_processing()
        example_collision_and_epa()
        example_indexed_api()
        example_vectorized_analysis()

        print("=" * 70)
        print(" All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
