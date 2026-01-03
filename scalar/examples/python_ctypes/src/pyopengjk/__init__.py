"""Python wrapper for openGJK.

openGJK is a fast and robust C implementation of the
Gilbert-Johnson-Keerthi (GJK) algorithm.
"""

from .opengjk import compute_minimum_distance, Point3, Simplex, DistanceResult

__all__ = ["compute_minimum_distance", "Point3", "Simplex", "DistanceResult"]
