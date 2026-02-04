"""
Simple collision example using openGJK GPU Python wrapper.

This demonstrates basic usage with a single collision pair
(same polytopes as userP.dat and userQ.dat from the C example).
"""

import numpy as np
from pyopengjk_gpu import compute_minimum_distance


def main():
    """Simple collision pair example."""
    print("=" * 70)
    print("OpenGJK GPU - Simple Collision Example")
    print("=" * 70)

    # Define two polytopes (same as userP.dat and userQ.dat from C example)
    polytope1 = np.array([
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

    polytope2 = np.array([
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

    print(f"Polytope 1: {len(polytope1)} vertices")
    print(f"Polytope 2: {len(polytope2)} vertices")
    print()

    # Compute minimum distance using GPU
    result = compute_minimum_distance(polytope1, polytope2)

    distance = result['distances'][0]
    witness1 = result['witnesses1'][0]
    witness2 = result['witnesses2'][0]

    print(f"Results:")
    print(f"  Distance: {distance:.6f}")
    print(f"  Witness point 1: ({witness1[0]:.6f}, {witness1[1]:.6f}, {witness1[2]:.6f})")
    print(f"  Witness point 2: ({witness2[0]:.6f}, {witness2[1]:.6f}, {witness2[2]:.6f})")
    print()

    # Verify distance by computing Euclidean distance between witness points
    computed_dist = np.linalg.norm(witness1 - witness2)
    print(f"Verification:")
    print(f"  Distance from witnesses: {computed_dist:.6f}")
    print(f"  Match: {abs(distance - computed_dist) < 1e-5}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
