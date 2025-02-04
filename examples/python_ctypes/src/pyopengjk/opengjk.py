import ctypes
from ctypes import Structure, POINTER, c_int, c_double
import os
from typing import NamedTuple, Tuple, List


class POLYTOPE(Structure):
    _fields_ = [
        ("numpoints", c_int),
        ("s", c_double * 3),
        ("s_idx", c_int),
        ("coord", POINTER(POINTER(c_double)))
    ]


class SIMPLEX(Structure):
    _fields_ = [
        ("nvrtx", c_int),
        ("vrtx", (c_double * 3) * 4),
        ("vrtx_idx", (c_int * 2) * 4),
        ("witnesses", (c_double * 3) * 2)
    ]


module_dir = os.path.dirname(__file__)
if os.path.exists(os.path.join(module_dir, "opengjk_ce.dll")):
    path = os.path.join(module_dir, "opengjk_ce.dll")
elif os.path.exists(os.path.join(module_dir, "libopengjk_ce.so")):
    path = os.path.join(module_dir, "libopengjk_ce.so")
elif os.path.exists(os.path.join(module_dir, "libopengjk_ce.dylib")):
    path = os.path.join(module_dir, "libopengjk_ce.dylib")
else:
    raise RuntimeError("Could not find rego_shared library")


opengjk = ctypes.cdll.LoadLibrary(path)

opengjk.compute_minimum_distance.restype = ctypes.c_double
opengjk.compute_minimum_distance.argtypes = [POLYTOPE,
                                             POLYTOPE,
                                             POINTER(SIMPLEX)]


Point3 = NamedTuple("Point3", [("x", float), ("y", float), ("z", float)])

Simplex = NamedTuple("Simplex", [("vertices", List[Point3]),
                                 ("indices", List[Tuple[int, int]]),
                                 ("witnesses", Tuple[Point3, Point3])])

DistanceResult = NamedTuple("Result", [("distance", float),
                                       ("simplex", Simplex)])


def compute_minimum_distance(vertices0: List[Point3],
                             vertices1: List[Point3]) -> DistanceResult:
    """Compute the minimum distance between two polytopes.

    Args:
        vertices0: Vertices of the first polytope.
        vertices1: Vertices of the second polytope.

    Returns:
        DistanceResult: A named tuple containing the distance and the final
                        simplex, which contains the points on the polytopes
                        that are closest to each other.
    """
    polytope0 = POLYTOPE()
    polytope0.numpoints = len(vertices0)
    polytope0.s = (c_double * 3)(0, 0, 0)
    polytope0.coord = (POINTER(c_double) * len(vertices0))()
    for i, vertex in enumerate(vertices0):
        polytope0.coord[i] = (c_double * 3)(*vertex)

    polytope1 = POLYTOPE()
    polytope1.numpoints = len(vertices1)
    polytope1.s = (c_double * 3)(0, 0, 0)
    polytope1.coord = (POINTER(c_double) * len(vertices1))()
    for i, vertex in enumerate(vertices1):
        polytope1.coord[i] = (c_double * 3)(*vertex)

    simplex = SIMPLEX()
    simplex.nvrtx = 0
    distance = opengjk.compute_minimum_distance(polytope0, polytope1, simplex)
    vertices = [Point3(*simplex.vrtx[i]) for i in range(simplex.nvrtx)]
    indices = [(simplex.vrtx_idx[i][0], simplex.vrtx_idx[i][1])
               for i in range(simplex.nvrtx)]
    witnesses = (Point3(*simplex.witnesses[0]), Point3(*simplex.witnesses[1]))
    return DistanceResult(distance, Simplex(vertices, indices, witnesses))
