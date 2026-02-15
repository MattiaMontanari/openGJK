"""
openGJK GPU - Python wrapper for GPU-accelerated collision detection

High-performance GJK/EPA algorithms running on NVIDIA GPUs with CUDA.
"""

from .opengjk_gpu import (
    compute_minimum_distance,
    compute_epa,
    compute_gjk_epa,
    compute_minimum_distance_indexed,
)

__version__ = "3.0.0"
__all__ = [
    "compute_minimum_distance",
    "compute_epa",
    "compute_gjk_epa",
    "compute_minimum_distance_indexed",
]
