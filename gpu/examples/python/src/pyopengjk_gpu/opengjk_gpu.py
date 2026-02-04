"""
openGJK GPU - High-performance Python wrapper using NumPy

GPU-accelerated GJK/EPA collision detection library for batch processing.
Optimized for performance with NumPy arrays and vectorized operations.

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund
SPDX-License-Identifier: GPL-3.0-only
"""

import ctypes
import os
import sys
import numpy as np
from typing import List, Tuple, Dict


# ============================================================================
# Determine precision and load library
# ============================================================================

# Check if library was built with 32-bit precision
# This should match the USE_32BITS flag used during compilation
USE_32BITS = True  # Default for GPU build

if USE_32BITS:
    gkFloat = ctypes.c_float
    DTYPE = np.float32
else:
    gkFloat = ctypes.c_double
    DTYPE = np.float64


# ============================================================================
# C Structure Definitions
# ============================================================================

class gkPolytope(ctypes.Structure):
    """
    GPU polytope structure.

    Note: coord must be a flattened array [x0,y0,z0, x1,y1,z1, ...]
    for efficient GPU memory access (coalescing).
    """
    _fields_ = [
        ("numpoints", ctypes.c_int),
        ("s", gkFloat * 3),
        ("s_idx", ctypes.c_int),
        ("coord", ctypes.POINTER(gkFloat))
    ]


class gkSimplex(ctypes.Structure):
    """GJK simplex structure containing closest points and witness information."""
    _fields_ = [
        ("nvrtx", ctypes.c_int),
        ("vrtx", (gkFloat * 3) * 4),
        ("vrtx_idx", (ctypes.c_int * 2) * 4),
        ("witnesses", (gkFloat * 3) * 2)
    ]


class gkCollisionPair(ctypes.Structure):
    """Index pair for indexed collision detection."""
    _fields_ = [
        ("idx1", ctypes.c_int),
        ("idx2", ctypes.c_int)
    ]


# ============================================================================
# Load Shared Library
# ============================================================================

def _find_library():
    """Find the openGJK_GPU shared library."""
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Search paths (in order of preference)
    # File is at: gpu/examples/python/src/pyopengjk_gpu/opengjk_gpu.py
    # Build output: build/gpu/Release/openGJK_GPU.dll
    search_paths = [
        module_dir,
        os.path.join(module_dir, "..", "..", "..", "..", "..", "build", "gpu", "Release"),
        os.path.join(module_dir, "..", "..", "..", "..", "..", "build", "gpu"),
        os.path.join(module_dir, "..", "..", "..", "..", "Release"),
        os.path.join(module_dir, "..", "..", "..", ".."),
    ]

    # Library names to try
    if sys.platform == "win32":
        lib_names = ["openGJK_GPU.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libopenGJK_GPU.dylib", "libopenGJK_GPU.so"]
    else:
        lib_names = ["libopenGJK_GPU.so"]

    # Search for library
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(search_path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

    raise RuntimeError(
        f"Could not find openGJK_GPU shared library. Searched:\n" +
        "\n".join(f"  {p}/{n}" for p in search_paths for n in lib_names) +
        f"\n\nMake sure to build the library with SHARED option enabled."
    )


# Load library
_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path)


# ============================================================================
# Function Signatures
# ============================================================================

# High-level API
_lib.compute_minimum_distance.argtypes = [
    ctypes.c_int,                   # n
    ctypes.POINTER(gkPolytope),     # bd1
    ctypes.POINTER(gkPolytope),     # bd2
    ctypes.POINTER(gkSimplex),      # simplices
    ctypes.POINTER(gkFloat)         # distances
]
_lib.compute_minimum_distance.restype = None

_lib.compute_epa.argtypes = [
    ctypes.c_int,                   # n
    ctypes.POINTER(gkPolytope),     # bd1
    ctypes.POINTER(gkPolytope),     # bd2
    ctypes.POINTER(gkSimplex),      # simplices
    ctypes.POINTER(gkFloat),        # distances
    ctypes.POINTER(gkFloat),        # witness1
    ctypes.POINTER(gkFloat),        # witness2
    ctypes.POINTER(gkFloat)         # contact_normals (nullable)
]
_lib.compute_epa.restype = None

_lib.compute_gjk_epa.argtypes = [
    ctypes.c_int,                   # n
    ctypes.POINTER(gkPolytope),     # bd1
    ctypes.POINTER(gkPolytope),     # bd2
    ctypes.POINTER(gkSimplex),      # simplices
    ctypes.POINTER(gkFloat),        # distances
    ctypes.POINTER(gkFloat),        # witness1
    ctypes.POINTER(gkFloat)         # witness2
]
_lib.compute_gjk_epa.restype = None

_lib.compute_minimum_distance_indexed.argtypes = [
    ctypes.c_int,                   # num_polytopes
    ctypes.c_int,                   # num_pairs
    ctypes.POINTER(gkPolytope),     # polytopes
    ctypes.POINTER(gkCollisionPair), # pairs
    ctypes.POINTER(gkSimplex),      # simplices
    ctypes.POINTER(gkFloat)         # distances
]
_lib.compute_minimum_distance_indexed.restype = None


# ============================================================================
# Helper Functions
# ============================================================================

def _prepare_polytope_array(vertices: np.ndarray) -> Tuple[gkPolytope, np.ndarray]:
    """
    Convert numpy array to GPU polytope structure.

    Args:
        vertices: NumPy array of shape (n, 3)

    Returns:
        (gkPolytope, coord_array) where coord_array must be kept alive
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Vertices must have shape (n, 3), got {vertices.shape}")

    # Ensure correct dtype and contiguous memory
    coords_flat = np.ascontiguousarray(vertices.flatten(), dtype=DTYPE)

    polytope = gkPolytope()
    polytope.numpoints = vertices.shape[0]
    polytope.s = (gkFloat * 3)(0, 0, 0)
    polytope.s_idx = 0
    polytope.coord = coords_flat.ctypes.data_as(ctypes.POINTER(gkFloat))

    return polytope, coords_flat


def _prepare_polytope_batch(vertices_list: List[np.ndarray]) -> Tuple[ctypes.Array, List[np.ndarray]]:
    """
    Prepare a batch of polytopes for GPU processing.

    Args:
        vertices_list: List of NumPy arrays, each shape (n, 3)

    Returns:
        (ctypes array of polytopes, list of coord arrays to keep alive)
    """
    n = len(vertices_list)
    bd_array = (gkPolytope * n)()
    coords_keep_alive = []

    for i, vertices in enumerate(vertices_list):
        polytope, coords = _prepare_polytope_array(vertices)
        bd_array[i] = polytope
        coords_keep_alive.append(coords)

    return bd_array, coords_keep_alive


# ============================================================================
# High-Level Python API (NumPy-based)
# ============================================================================

def compute_minimum_distance(
    vertices1: np.ndarray,
    vertices2: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute minimum distance between polytope pairs using GPU-accelerated GJK.

    Args:
        vertices1: NumPy array of shape (n, 3) for single pair, or list of arrays for batch
        vertices2: NumPy array of shape (n, 3) for single pair, or list of arrays for batch

    Returns:
        Dictionary with NumPy arrays:
            'distances': (n,) distances between polytopes (0.0 = collision)
            'witnesses1': (n, 3) closest points on first polytopes
            'witnesses2': (n, 3) closest points on second polytopes
            'is_collision': (n,) boolean array indicating collisions
            'simplex_nvrtx': (n,) number of vertices in each simplex

    Example:
        Single pair:
        >>> v1 = np.array([[0, 5.5, 0], [2.3, 1.0, -2.0], [8.1, 4.0, 2.4]], dtype=np.float32)
        >>> v2 = np.array([[0, -5.5, 0], [-2.3, -1.0, 2.0], [-8.1, -4.0, -2.4]], dtype=np.float32)
        >>> result = compute_minimum_distance(v1, v2)
        >>> print(f"Distance: {result['distances'][0]}")

        Batch:
        >>> v1_list = [np.random.randn(10, 3) for _ in range(100)]
        >>> v2_list = [np.random.randn(10, 3) for _ in range(100)]
        >>> result = compute_minimum_distance(v1_list, v2_list)
        >>> print(f"Collisions: {result['is_collision'].sum()}")
    """
    # Check if batch or single
    is_batch = isinstance(vertices1, list)

    if not is_batch:
        # Single pair - wrap in list for uniform processing
        vertices1 = [vertices1]
        vertices2 = [vertices2]

    n = len(vertices1)
    if len(vertices2) != n:
        raise ValueError(f"Mismatch: {len(vertices1)} polytopes in bd1, {len(vertices2)} in bd2")

    # Prepare polytopes (coords_alive arrays must stay in scope to prevent garbage collection)
    bd1_array, coords1_alive = _prepare_polytope_batch(vertices1)
    bd2_array, coords2_alive = _prepare_polytope_batch(vertices2)

    # Allocate outputs
    simplices = (gkSimplex * n)()
    distances = np.zeros(n, dtype=DTYPE)

    # Call GPU function
    _lib.compute_minimum_distance(
        n, bd1_array, bd2_array, simplices,
        distances.ctypes.data_as(ctypes.POINTER(gkFloat))
    )

    # Extract results as NumPy arrays (vectorized)
    witnesses1 = np.zeros((n, 3), dtype=DTYPE)
    witnesses2 = np.zeros((n, 3), dtype=DTYPE)
    simplex_nvrtx = np.zeros(n, dtype=np.int32)

    for i in range(n):
        witnesses1[i] = simplices[i].witnesses[0]
        witnesses2[i] = simplices[i].witnesses[1]
        simplex_nvrtx[i] = simplices[i].nvrtx

    # Vectorized collision detection
    is_collision = np.abs(distances) < 1e-6

    return {
        'distances': distances,
        'witnesses1': witnesses1,
        'witnesses2': witnesses2,
        'is_collision': is_collision,
        'simplex_nvrtx': simplex_nvrtx
    }


def compute_epa(
    vertices1: np.ndarray,
    vertices2: np.ndarray,
    return_normals: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute penetration depth and witness points using GPU-accelerated EPA.

    EPA (Expanding Polytope Algorithm) is used when polytopes are colliding
    to find penetration depth and contact points.

    Args:
        vertices1: NumPy array (n, 3) or list of arrays for batch
        vertices2: NumPy array (n, 3) or list of arrays for batch
        return_normals: If True, compute and return contact normals

    Returns:
        Dictionary with NumPy arrays:
            'penetration_depths': (n,) penetration distances
            'witnesses1': (n, 3) contact points on first polytopes
            'witnesses2': (n, 3) contact points on second polytopes
            'contact_normals': (n, 3) contact normals (if return_normals=True)
    """
    # Check if batch or single
    is_batch = isinstance(vertices1, list)

    if not is_batch:
        vertices1 = [vertices1]
        vertices2 = [vertices2]

    n = len(vertices1)

    # Prepare polytopes (coords_alive arrays must stay in scope to prevent garbage collection)
    bd1_array, coords1_alive = _prepare_polytope_batch(vertices1)
    bd2_array, coords2_alive = _prepare_polytope_batch(vertices2)

    # Allocate outputs
    simplices = (gkSimplex * n)()
    distances = np.zeros(n, dtype=DTYPE)
    witnesses1 = np.zeros(n * 3, dtype=DTYPE)
    witnesses2 = np.zeros(n * 3, dtype=DTYPE)
    contact_normals = np.zeros(n * 3, dtype=DTYPE) if return_normals else np.empty(0, dtype=DTYPE)

    # Call GPU function
    _lib.compute_epa(
        n, bd1_array, bd2_array, simplices,
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        witnesses1.ctypes.data_as(ctypes.POINTER(gkFloat)),
        witnesses2.ctypes.data_as(ctypes.POINTER(gkFloat)),
        contact_normals.ctypes.data_as(ctypes.POINTER(gkFloat)) if return_normals else None
    )

    # Reshape witness points and build result
    if return_normals:
        return {
            'penetration_depths': distances,
            'witnesses1': witnesses1.reshape(n, 3),
            'witnesses2': witnesses2.reshape(n, 3),
            'contact_normals': contact_normals.reshape(n, 3)
        }
    else:
        return {
            'penetration_depths': distances,
            'witnesses1': witnesses1.reshape(n, 3),
            'witnesses2': witnesses2.reshape(n, 3)
        }


def compute_gjk_epa(
    vertices1: np.ndarray,
    vertices2: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Combined GJK+EPA pipeline: runs GJK first, then EPA for colliding pairs.

    This is more efficient than calling both separately as it reuses
    the GJK simplex for EPA initialization.

    Args:
        vertices1: NumPy array (n, 3) or list of arrays for batch
        vertices2: NumPy array (n, 3) or list of arrays for batch

    Returns:
        Dictionary with NumPy arrays:
            'distances': (n,) distances (0.0 for collisions)
            'is_collision': (n,) boolean collision flags
            'witnesses1': (n, 3) witness/contact points on first polytopes
            'witnesses2': (n, 3) witness/contact points on second polytopes
            'simplex_nvrtx': (n,) number of simplex vertices

        For colliding pairs, witnesses are EPA contact points.
        For separated pairs, witnesses are GJK closest points.
    """
    # Check if batch or single
    is_batch = isinstance(vertices1, list)

    if not is_batch:
        vertices1 = [vertices1]
        vertices2 = [vertices2]

    n = len(vertices1)

    # Prepare polytopes (coords_alive arrays must stay in scope to prevent garbage collection)
    bd1_array, coords1_alive = _prepare_polytope_batch(vertices1)
    bd2_array, coords2_alive = _prepare_polytope_batch(vertices2)

    # Allocate outputs
    simplices = (gkSimplex * n)()
    distances = np.zeros(n, dtype=DTYPE)
    witnesses1 = np.zeros(n * 3, dtype=DTYPE)
    witnesses2 = np.zeros(n * 3, dtype=DTYPE)

    # Call GPU function
    _lib.compute_gjk_epa(
        n, bd1_array, bd2_array, simplices,
        distances.ctypes.data_as(ctypes.POINTER(gkFloat)),
        witnesses1.ctypes.data_as(ctypes.POINTER(gkFloat)),
        witnesses2.ctypes.data_as(ctypes.POINTER(gkFloat))
    )

    # Extract simplex info
    simplex_nvrtx = np.array([simplices[i].nvrtx for i in range(n)], dtype=np.int32)

    # Vectorized collision detection
    is_collision = np.abs(distances) < 1e-6

    return {
        'distances': distances,
        'is_collision': is_collision,
        'witnesses1': witnesses1.reshape(n, 3),
        'witnesses2': witnesses2.reshape(n, 3),
        'simplex_nvrtx': simplex_nvrtx
    }


def compute_minimum_distance_indexed(
    polytopes: List[np.ndarray],
    pairs: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute distances for indexed polytope pairs (efficient for reuse).

    When polytopes are reused across multiple collision checks, this API
    is more efficient as it avoids redundant memory transfers.

    Args:
        polytopes: List of NumPy arrays, each shape (n, 3)
        pairs: NumPy array of shape (m, 2) with integer indices

    Returns:
        Dictionary with NumPy arrays (same as compute_minimum_distance)

    Example:
        >>> polytopes = [cube1, cube2, sphere, capsule]
        >>> pairs = np.array([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int32)
        >>> result = compute_minimum_distance_indexed(polytopes, pairs)
        >>> print(f"Collision mask: {result['is_collision']}")
    """
    num_polytopes = len(polytopes)

    # Validate and prepare pairs
    if isinstance(pairs, list):
        pairs = np.array(pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"Pairs must have shape (m, 2), got {pairs.shape}")

    num_pairs = pairs.shape[0]

    # Prepare polytopes
    bd_array, coords_alive = _prepare_polytope_batch(polytopes)

    # Prepare pairs
    pairs_array = (gkCollisionPair * num_pairs)()
    for i in range(num_pairs):
        pairs_array[i].idx1 = int(pairs[i, 0])
        pairs_array[i].idx2 = int(pairs[i, 1])

    # Allocate outputs
    simplices = (gkSimplex * num_pairs)()
    distances = np.zeros(num_pairs, dtype=DTYPE)

    # Call GPU function
    _lib.compute_minimum_distance_indexed(
        num_polytopes, num_pairs, bd_array, pairs_array, simplices,
        distances.ctypes.data_as(ctypes.POINTER(gkFloat))
    )

    # Extract results (vectorized)
    witnesses1 = np.zeros((num_pairs, 3), dtype=DTYPE)
    witnesses2 = np.zeros((num_pairs, 3), dtype=DTYPE)
    simplex_nvrtx = np.zeros(num_pairs, dtype=np.int32)

    for i in range(num_pairs):
        witnesses1[i] = simplices[i].witnesses[0]
        witnesses2[i] = simplices[i].witnesses[1]
        simplex_nvrtx[i] = simplices[i].nvrtx

    # Vectorized collision detection
    is_collision = np.abs(distances) < 1e-6

    return {
        'distances': distances,
        'witnesses1': witnesses1,
        'witnesses2': witnesses2,
        'is_collision': is_collision,
        'simplex_nvrtx': simplex_nvrtx
    }


# ============================================================================
# Module Info
# ============================================================================

__version__ = "3.0.0"
__all__ = [
    "compute_minimum_distance",
    "compute_epa",
    "compute_gjk_epa",
    "compute_minimum_distance_indexed",
]
