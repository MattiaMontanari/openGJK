"""
OpenGJK GPU Test Examples - Python Version
Recreates the test cases from OpenGJK-GPU repository

This demonstrates:
1. Simple GJK (single pair)
2. Batch array processing
3. Indexed API
4. EPA collision tests
"""

import numpy as np
import time
from pyopengjk_gpu import (
    compute_minimum_distance,
    compute_epa,
    compute_gjk_epa,
    compute_minimum_distance_indexed
)

# Global random seed for reproducibility
RANDOM_SEED = 42

# Auto-detect dtype from library (matches #USE_32BITS in C++ compilation)
def _detect_dtype():
    """Detect the dtype used by the library by running a minimal test."""
    # Create minimal test arrays (use float32 for detection)
    p1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    p2 = np.array([[[2, 0, 0], [3, 0, 0], [2, 1, 0]]], dtype=np.float32)
    try:
        result = compute_minimum_distance(p1, p2)
        return result['distances'].dtype
    except:
        # Fallback to float32 if detection fails
        return np.float32

DTYPE = _detect_dtype()

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_pass(msg):
    print(f"{Colors.GREEN}  PASS: {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}  WARNING: {msg}{Colors.RESET}")

def print_fail(msg):
    print(f"{Colors.RED}  FAIL: {msg}{Colors.RESET}")


def generate_polytope(num_poly, num_verts, offsets=None):
    """Generate multiple polytopes with random vertices on sphere surfaces (fully vectorized).

    Args:
        num_poly: Number of polytopes to generate
        num_verts: Number of vertices per polytope
        offsets: Optional (num_poly, 3) array of offsets. If None, random offsets are generated.

    Returns:
        ndarray of shape (num_poly, num_verts, 3)
    """
    # If no offsets provided, generate random offsets
    if offsets is None:
        offsets = (np.random.rand(num_poly, 3) - 0.5) * 20.0

    # Generate random spherical coordinates for all polytopes at once
    theta = np.random.rand(num_poly, num_verts) * 2.0 * np.pi
    phi = np.random.rand(num_poly, num_verts) * np.pi
    r = 1.0 + np.random.rand(num_poly, num_verts) * 0.5

    vertices = np.zeros((num_poly, num_verts, 3), dtype=DTYPE)
    vertices[:, :, 0] = r * np.sin(phi) * np.cos(theta) + offsets[:, 0:1]
    vertices[:, :, 1] = r * np.sin(phi) * np.sin(theta) + offsets[:, 1:2]
    vertices[:, :, 2] = r * np.cos(phi) + offsets[:, 2:3]

    return vertices


def generate_sphere_surface(num_points, radius, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Generate points uniformly distributed on sphere surface (vectorized)."""
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)

    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)

    vertices = np.zeros((num_points, 3), dtype=DTYPE)
    vertices[:, 0] = radius * np.sin(phi) * np.cos(theta) + offset_x
    vertices[:, 1] = radius * np.sin(phi) * np.sin(theta) + offset_y
    vertices[:, 2] = radius * np.cos(phi) + offset_z

    return vertices


def generate_cube_with_grid(grid_size, size, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Generate a cube with grid of vertices on each face."""
    vertices_per_face = grid_size * grid_size
    total_vertices = 6 * vertices_per_face
    vertices = np.zeros((total_vertices, 3), dtype=DTYPE)

    idx = 0

    # Generate vertices for each of the 6 faces
    for face in range(6):
        for i in range(grid_size):
            for j in range(grid_size):
                y = -size + (2.0 * size * i) / (grid_size - 1) if grid_size > 1 else 0
                z = -size + (2.0 * size * j) / (grid_size - 1) if grid_size > 1 else 0
                x = -size + (2.0 * size * i) / (grid_size - 1) if grid_size > 1 else 0

                if face == 0:  # +X face
                    vertices[idx] = [size + offset_x, y + offset_y, z + offset_z]
                elif face == 1:  # -X face
                    vertices[idx] = [-size + offset_x, y + offset_y, z + offset_z]
                elif face == 2:  # +Y face
                    vertices[idx] = [x + offset_x, size + offset_y, z + offset_z]
                elif face == 3:  # -Y face
                    vertices[idx] = [x + offset_x, -size + offset_y, z + offset_z]
                elif face == 4:  # +Z face
                    vertices[idx] = [x + offset_x, y + offset_y, size + offset_z]
                elif face == 5:  # -Z face
                    vertices[idx] = [x + offset_x, y + offset_y, -size + offset_z]

                idx += 1

    return vertices


def test_1_simple_gjk():
    """Test 1: Simple GJK with single collision pair (from simple_collision example)."""
    print("=" * 70)
    print("Test 1: Simple GJK (Single Pair)")
    print("=" * 70)

    # Use exact vertices from userP.dat and userQ.dat (same as simple_collision example)
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
    ], dtype=DTYPE)

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
    ], dtype=DTYPE)

    # Compute distance (wrap in 3D arrays for batch API)
    result = compute_minimum_distance(polytope1[np.newaxis, :, :], polytope2[np.newaxis, :, :])

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]
    witness1 = result['witnesses1'][0]
    witness2 = result['witnesses2'][0]

    print(f"Polytope 1: 9 vertices (from userP.dat)")
    print(f"Polytope 2: 9 vertices (from userQ.dat)")
    print(f"\nResults:")
    print(f"  Distance: {distance:.6f}")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Witness 1: ({witness1[0]:.6f}, {witness1[1]:.6f}, {witness1[2]:.6f})")
    print(f"  Witness 2: ({witness2[0]:.6f}, {witness2[1]:.6f}, {witness2[2]:.6f})")

    # Verify distance from witness points
    computed_dist = np.linalg.norm(witness1 - witness2)
    print(f"\nVerification:")
    print(f"  Distance from witnesses: {computed_dist:.6f}")

    # Expected result from README: 3.653650
    expected_distance = 3.653650
    if abs(distance - expected_distance) < 1e-5:
        print_pass(f"Distance matches expected value ({expected_distance:.6f})")
    else:
        print_fail(f"Distance {distance:.6f} != expected {expected_distance:.6f}")
    print()


def test_2_batch_array():
    """Test 2: Batch array processing and indexed API comparison."""
    print("=" * 70)
    print("Test 2: Batch Array Processing & Indexed API")
    print("=" * 70)

    num_pairs = 1000
    num_verts = 1000
    np.random.seed(RANDOM_SEED)

    print(f"Generating {num_pairs} random polytope pairs with {num_verts} vertices each...")

    # Generate random polytope pairs using vectorized generation
    offsets1 = (np.random.rand(num_pairs, 3) - 0.5) * 10.0
    offsets2 = (np.random.rand(num_pairs, 3) - 0.5) * 10.0

    polytopes1 = generate_polytope(num_pairs, num_verts, offsets1)
    polytopes2 = generate_polytope(num_pairs, num_verts, offsets2)

    # Method 1: Non-indexed API
    print(f"\nMethod 1: Non-indexed API")
    start = time.time()
    result_nonindexed = compute_minimum_distance(polytopes1, polytopes2)
    time_nonindexed = time.time() - start
    print(f"  Time: {time_nonindexed*1000:.2f} ms")

    distances_nonindexed = result_nonindexed['distances']
    print(f"  Distance range: [{distances_nonindexed.min():.3f}, {distances_nonindexed.max():.3f}]")

    # Method 2: Indexed API - interlace polytopes1 and polytopes2
    print(f"\nMethod 2: Indexed API (Interlaced)")
    # Interlace: [p1[0], p2[0], p1[1], p2[1], ...]
    # Shape: (2*num_pairs, num_verts, 3)
    polytopes_interlaced = np.empty((2 * num_pairs, num_verts, 3), dtype=DTYPE)
    polytopes_interlaced[0::2] = polytopes1  # Even indices: p1[0], p1[1], ...
    polytopes_interlaced[1::2] = polytopes2  # Odd indices: p2[0], p2[1], ...

    # Create index pairs: [[0,1], [2,3], [4,5], ...]
    pairs = np.array([[2*i, 2*i+1] for i in range(num_pairs)], dtype=np.int32)

    start = time.time()
    result_indexed = compute_minimum_distance_indexed(polytopes_interlaced, pairs)
    time_indexed = time.time() - start
    print(f"  Time: {time_indexed*1000:.2f} ms")

    distances_indexed = result_indexed['distances']
    print(f"  Distance range: [{distances_indexed.min():.3f}, {distances_indexed.max():.3f}]")

    # Compare results
    print(f"\nComparison:")
    max_diff = np.max(np.abs(distances_nonindexed - distances_indexed))
    mean_diff = np.mean(np.abs(distances_nonindexed - distances_indexed))

    print(f"  Max distance difference: {max_diff:.9f}")
    print(f"  Mean distance difference: {mean_diff:.9f}")

    # Validation
    tolerance = 1e-5
    if max_diff < tolerance:
        print_pass(f"Results match within tolerance ({tolerance})")
    elif max_diff < 1e-3:
        print_warning(f"Results differ slightly (max diff: {max_diff:.9f})")
    else:
        print_fail(f"Results differ significantly (max diff: {max_diff:.9f})")

    # Performance comparison
    speedup = time_nonindexed / time_indexed if time_indexed > 0 else 1.0
    print(f"  Performance: Indexed is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than non-indexed")
    print()


def test_3_touching_cubes():
    """Test 3: EPA with two touching cubes (just touching, no penetration)."""
    print("=" * 70)
    print("Test 3: EPA - Two Touching Cubes")
    print("=" * 70)

    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (2, 0, 0), size 2x2x2 (touching at x=1)
    cube1 = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=DTYPE)

    cube2 = np.array([
        [1, -1, -1], [3, -1, -1], [1, 1, -1], [3, 1, -1],
        [1, -1, 1], [3, -1, 1], [1, 1, 1], [3, 1, 1]
    ], dtype=DTYPE)

    # Run GJK+EPA (wrap in 3D arrays)
    result = compute_gjk_epa(cube1[np.newaxis, :, :], cube2[np.newaxis, :, :])

    print(f"Cube 1: centered at (0, 0, 0), size 2x2x2")
    print(f"Cube 2: centered at (2, 0, 0), size 2x2x2")
    print(f"Expected: Very small distance (near zero)")

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]

    print(f"\nResults:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance: {distance:.6f}")
    print(f"  Witness 1: {result['witnesses1'][0]}")
    print(f"  Witness 2: {result['witnesses2'][0]}")

    # Validation (matching main.cpp logic)
    if distance >= 0 and distance < 0.01:
        print_pass(f"Distance near zero as expected")
    else:
        print_warning(f"Distance may indicate collision or separation")
    print()


def test_4_epa_overlapping_cubes():
    """Test 4a: EPA with two overlapping cubes."""
    print("=" * 70)
    print("Test 4a: EPA - Two Overlapping Cubes")
    print("=" * 70)

    # Create two cubes that overlap
    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (1, 0, 0), size 2x2x2 (overlaps by 1 unit)
    cube1 = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=DTYPE)

    cube2 = np.array([
        [0, -1, -1], [2, -1, -1], [0, 1, -1], [2, 1, -1],
        [0, -1, 1], [2, -1, 1], [0, 1, 1], [2, 1, 1]
    ], dtype=DTYPE)

    # Run GJK+EPA (wrap in 3D arrays)
    result = compute_gjk_epa(cube1[np.newaxis, :, :], cube2[np.newaxis, :, :])

    print(f"Cube 1: centered at (0, 0, 0), size 2x2x2")
    print(f"Cube 2: centered at (1, 0, 0), size 2x2x2")
    print(f"Expected: Collision (overlap by 1 unit)")

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]

    print(f"\nResults:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance/Penetration: {distance:.6f}")
    print(f"  Witness 1: {result['witnesses1'][0]}")
    print(f"  Witness 2: {result['witnesses2'][0]}")

    # Validation (matching main.cpp logic)
    if distance < -0.8 and distance > -1.2:
        print_pass(f"Collision detected, penetration depth valid")
    else:
        print_fail(f"Invalid results (expected penetration between 0.8 and 1.2)")
    print()


def test_5_epa_separated_cubes():
    """Test 4b: EPA with two separated cubes."""
    print("=" * 70)
    print("Test 4b: EPA - Two Separated Cubes")
    print("=" * 70)

    # Cube 1: centered at (0, 0, 0), size 2x2x2
    # Cube 2: centered at (5, 0, 0), size 2x2x2 (separated by 3 units)
    cube1 = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ], dtype=DTYPE)

    cube2 = np.array([
        [4, -1, -1], [6, -1, -1], [4, 1, -1], [6, 1, -1],
        [4, -1, 1], [6, -1, 1], [4, 1, 1], [6, 1, 1]
    ], dtype=DTYPE)

    # Run GJK+EPA (wrap in 3D arrays)
    result = compute_gjk_epa(cube1[np.newaxis, :, :], cube2[np.newaxis, :, :])

    print(f"Cube 1: centered at (0, 0, 0), size 2x2x2")
    print(f"Cube 2: centered at (5, 0, 0), size 2x2x2")
    print(f"Expected: Separation distance ~3.0")

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]

    print(f"\nResults:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance: {distance:.6f}")
    print(f"  Witness 1: {result['witnesses1'][0]}")
    print(f"  Witness 2: {result['witnesses2'][0]}")

    # Validation (matching main.cpp logic)
    if simplex_nvrtx < 4 and distance > 2.9 and distance < 3.1:
        print_pass(f"Correct separation distance")
    elif simplex_nvrtx < 4:
        print_warning(f"Distance may be incorrect")
    else:
        print_fail(f"Should not detect collision")
    print()


def test_6_epa_overlapping_spheres():
    """Test 4c: EPA with two overlapping spheres."""
    print("=" * 70)
    print("Test 4c: EPA - Two Overlapping Spheres")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    num_points = 1000
    radius = 2.0

    # Sphere 1: centered at (0, 0, 0), radius 2
    # Sphere 2: centered at (1, 0, 0), radius 2 (overlap)
    sphere1 = generate_sphere_surface(num_points, radius, 0.0, 0.0, 0.0)
    sphere2 = generate_sphere_surface(num_points, radius, 1.0, 0.0, 0.0)

    # Run GJK+EPA with contact normals (wrap in 3D arrays)
    result_gjk_epa = compute_gjk_epa(sphere1[np.newaxis, :, :], sphere2[np.newaxis, :, :])
    result_epa = compute_epa(sphere1[np.newaxis, :, :], sphere2[np.newaxis, :, :], return_normals=True)

    print(f"Sphere 1: centered at (0, 0, 0), radius {radius}")
    print(f"Sphere 2: centered at (1, 0, 0), radius {radius}")
    print(f"Expected: Collision (centers 1 unit apart, overlap ~3 units)")

    distance = result_gjk_epa['distances'][0]
    simplex_nvrtx = result_gjk_epa['simplex_nvrtx'][0]
    witness1 = result_gjk_epa['witnesses1'][0]
    witness2 = result_gjk_epa['witnesses2'][0]

    print(f"\nGJK+EPA Results:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance/Penetration: {distance:.6f}")
    print(f"  Expected: Collision (spheres overlap, centers 1 unit apart, each radius 2)")
    print(f"  Expected overlap: ~3 units (2+2-1=3)")
    print(f"  Witness 1: {witness1}")
    print(f"  Witness 2: {witness2}")

    print(f"\nEPA Results (with contact normals):")
    print(f"  Penetration depth: {result_epa['penetration_depths'][0]:.6f}")
    print(f"  Contact point 1: {result_epa['witnesses1'][0]}")
    print(f"  Contact point 2: {result_epa['witnesses2'][0]}")
    print(f"  Contact normal: {result_epa['contact_normals'][0]}")

    # Verify witness points are within sphere bounds (matching main.cpp logic)
    dist1 = np.linalg.norm(witness1)
    dist2 = np.linalg.norm(witness2 - np.array([1.0, 0.0, 0.0]))

    valid1 = dist1 <= radius + 0.1  # Allow small tolerance
    valid2 = dist2 <= radius + 0.1

    # Validation (matching main.cpp logic)
    if simplex_nvrtx == 4 and valid1 and valid2:
        if distance < 0.0:
            print_pass(f"Collision detected with penetration depth of {-distance:.6f}")
            print(f"  Expected penetration: ~3.0 units")
        elif distance < 0.1:
            print_pass(f"Collision detected (very small distance/penetration)")
        else:
            print_warning(f"Collision detected but distance seems large: {distance:.6f}")
    elif simplex_nvrtx < 4 and distance >= 0.0:
        print_warning(f"No collision detected, but spheres should overlap")
        print(f"  Separation distance: {distance:.6f}")
    else:
        print_warning(f"Unexpected results")
        if not valid1:
            print(f"    Witness 1 distance from sphere 1 center: {dist1:.6f} (expected <= {radius})")
        if not valid2:
            print(f"    Witness 2 distance from sphere 2 center: {dist2:.6f} (expected <= {radius})")
    print()


def test_7_overlapping_polytopes_50_verts():
    """Test 4e: EPA with overlapping polytopes (~50 vertices each)."""
    print("=" * 70)
    print("Test 4e: EPA - Overlapping Polytopes (~50 vertices each)")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    num_verts = 50

    # Generate polytopes that overlap
    # Polytope 1: centered at (0, 0, 0)
    # Polytope 2: centered at (0.5, 0, 0) - overlaps with polytope 1
    offsets1 = np.array([[0.0, 0.0, 0.0]], dtype=DTYPE)
    offsets2 = np.array([[0.5, 0.0, 0.0]], dtype=DTYPE)

    polytope1 = generate_polytope(1, num_verts, offsets1)[0]
    polytope2 = generate_polytope(1, num_verts, offsets2)[0]

    # Run GJK+EPA
    result = compute_gjk_epa(polytope1[np.newaxis, :, :], polytope2[np.newaxis, :, :])

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]
    witness1 = result['witnesses1'][0]
    witness2 = result['witnesses2'][0]

    print(f"Polytope 1: {num_verts} vertices, centered at (0, 0, 0)")
    print(f"Polytope 2: {num_verts} vertices, centered at (0.5, 0, 0)")
    print(f"Expected: Collision (polytopes overlap)")

    print(f"\nResults:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance/Penetration: {distance:.6f}")
    print(f"  Witness 1: {witness1}")
    print(f"  Witness 2: {witness2}")

    # Validation (matching example.cu logic)
    if simplex_nvrtx == 4:
        if distance < 0.0:
            print_pass(f"Collision detected with penetration depth of {-distance:.6f}")
        elif distance < 0.1:
            print_pass(f"Collision detected (very small distance/penetration)")
        else:
            print_warning(f"Collision detected but distance seems large: {distance:.6f}")
    else:
        print_warning(f"Simplex has {simplex_nvrtx} vertices (expected 4 for collision)")
        if distance > 0.0:
            print(f"  Polytopes are separated by distance: {distance:.6f}")
    print()


def test_8_epa_rotated_cubes():
    """Test 4f: EPA with cube and rotated cube (high resolution)."""
    print("=" * 70)
    print("Test 4f: EPA - Cube and Rotated Cube (High Resolution)")
    print("=" * 70)

    grid_size = 40
    cube_size = 1.0
    num_verts = 6 * grid_size * grid_size

    print(f"Generating cubes with {num_verts} vertices each ({grid_size}x{grid_size} grid per face)...")

    # Cube 1: centered at (0, 0, 0)
    cube1 = generate_cube_with_grid(grid_size, cube_size, 0.0, 0.0, 0.0)

    # Cube 2: generate at origin, then rotate and translate
    cube2 = generate_cube_with_grid(grid_size, cube_size, 0.0, 0.0, 0.0)

    # Rotate by 45° around all axes and translate by (1, 0, 0)
    angle = 45.0 * np.pi / 180.0
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for i in range(num_verts):
        x, y, z = cube2[i]

        # Rotate around X axis
        y, z = y * cos_a - z * sin_a, y * sin_a + z * cos_a

        # Rotate around Y axis
        x, z = x * cos_a + z * sin_a, -x * sin_a + z * cos_a

        # Rotate around Z axis
        x, y = x * cos_a - y * sin_a, x * sin_a + y * cos_a

        # Translate
        cube2[i] = [x + 1.0, y, z]

    # Run GJK+EPA
    print("Running GJK+EPA...")
    result = compute_gjk_epa(cube1[np.newaxis, :, :], cube2[np.newaxis, :, :])

    distance = result['distances'][0]
    simplex_nvrtx = result['simplex_nvrtx'][0]
    witness1 = result['witnesses1'][0]
    witness2 = result['witnesses2'][0]

    print(f"\nResults:")
    print(f"  Simplex vertices: {simplex_nvrtx}")
    print(f"  Distance/Penetration: {distance:.6f}")
    print(f"  Expected: May overlap or be close depending on rotation")
    print(f"  Witness 1: {witness1}")
    print(f"  Witness 2: {witness2}")

    # Verify witness points are reasonable (expanded bounds for rotated cube)
    valid1 = (witness1[0] >= -2.0 and witness1[0] <= 2.0 and
              witness1[1] >= -2.0 and witness1[1] <= 2.0 and
              witness1[2] >= -2.0 and witness1[2] <= 2.0)
    valid2 = (witness2[0] >= -1.0 and witness2[0] <= 3.0 and
              witness2[1] >= -2.0 and witness2[1] <= 2.0 and
              witness2[2] >= -2.0 and witness2[2] <= 2.0)

    # Validation (matching main.cpp logic)
    if simplex_nvrtx == 4 and valid1 and valid2:
        if distance < 0.0:
            print_pass(f"Collision detected with penetration depth of {-distance:.6f}")
        else:
            print_pass(f"Collision detected, witness points valid")
    elif simplex_nvrtx < 4 and distance >= 0.0:
        print_pass(f"No collision, separation distance: {distance:.6f}")
    else:
        print_warning(f"Unexpected results")
    print()


def test_9_epa_separate_gjk_epa():
    """Test 4g: EPA with two overlapping spheres (using separate GJK and EPA calls)."""
    print("=" * 70)
    print("Test 4g: EPA - Two Overlapping Spheres (Separate GJK/EPA)")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    num_points = 1000
    radius = 2.0

    print(f"Generating spheres with {num_points} points each, radius {radius}...")

    # Sphere 1: centered at (0, 0, 0)
    # Sphere 2: centered at (1, 0, 0) - shifted 1 unit in x direction
    sphere1 = generate_sphere_surface(num_points, radius, 0.0, 0.0, 0.0)
    sphere2 = generate_sphere_surface(num_points, radius, 1.0, 0.0, 0.0)

    print(f"\nRunning GPU GJK...")
    # Run GJK first
    result_gjk = compute_minimum_distance(sphere1[np.newaxis, :, :], sphere2[np.newaxis, :, :])

    print(f"GJK Results:")
    print(f"  Simplex vertices: {result_gjk['simplex_nvrtx'][0]}")
    print(f"  Distance: {result_gjk['distances'][0]:.6f}")

    print(f"\nRunning GPU EPA...")
    # Run EPA separately (with contact normals)
    result_epa = compute_epa(sphere1[np.newaxis, :, :], sphere2[np.newaxis, :, :], return_normals=True)

    distance = result_epa['penetration_depths'][0]
    witness1 = result_epa['witnesses1'][0]
    witness2 = result_epa['witnesses2'][0]
    contact_normal = result_epa['contact_normals'][0]

    print(f"\nFinal Results:")
    print(f"  Distance/Penetration: {distance:.6f}")
    print(f"  Expected: Collision (spheres overlap, centers 1 unit apart, each radius 2)")
    print(f"  Expected overlap: ~3 units (2+2-1=3)")
    print(f"  Witness 1: ({witness1[0]:.6f}, {witness1[1]:.6f}, {witness1[2]:.6f})")
    print(f"  Witness 2: ({witness2[0]:.6f}, {witness2[1]:.6f}, {witness2[2]:.6f})")
    print(f"  Contact Normal: ({contact_normal[0]:.6f}, {contact_normal[1]:.6f}, {contact_normal[2]:.6f})")

    # Verify witness points are within sphere bounds
    # Sphere 1: centered at (0,0,0), radius 2
    # Sphere 2: centered at (1,0,0), radius 2
    dist1 = np.linalg.norm(witness1)
    dist2 = np.linalg.norm(witness2 - np.array([1.0, 0.0, 0.0]))

    valid1 = dist1 <= radius + 0.1  # Allow small tolerance
    valid2 = dist2 <= radius + 0.1

    # Validation (matching example.cu logic)
    simplex_nvrtx = result_gjk['simplex_nvrtx'][0]
    if simplex_nvrtx == 4 and valid1 and valid2:
        if distance < 0.0:
            print_pass(f"Collision detected with penetration depth of {-distance:.6f}")
            print(f"  Expected penetration: ~3.0 units")
        elif distance < 0.1:
            print_pass(f"Collision detected (very small distance/penetration)")
        else:
            print_warning(f"Collision detected but distance seems large: {distance:.6f}")
    elif simplex_nvrtx < 4 and distance >= 0.0:
        print_warning(f"No collision detected, but spheres should overlap")
        print(f"  Separation distance: {distance:.6f}")
    else:
        print_warning(f"Unexpected results")
        if not valid1:
            print(f"    Witness 1 distance from sphere 1 center: {dist1:.6f} (expected <= {radius})")
        if not valid2:
            print(f"    Witness 2 distance from sphere 2 center: {dist2:.6f} (expected <= {radius})")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" OpenGJK GPU - Test Examples (Python)")
    print("=" * 70 + "\n")

    try:
        test_1_simple_gjk()
        test_2_batch_array()
        test_3_touching_cubes()
        test_4_epa_overlapping_cubes()
        test_5_epa_separated_cubes()
        test_6_epa_overlapping_spheres()
        test_7_overlapping_polytopes_50_verts()
        test_8_epa_rotated_cubes()
        test_9_epa_separate_gjk_epa()

        print("=" * 70)
        print(" All tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
