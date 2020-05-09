import opengjkc as opengjk
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest
#from IPython import embed

def settol():
    return 1e-12

def distance_point_to_line_3D(P1, P2, point):
    """
    distance from point to line
    """
    return np.linalg.norm(np.cross(P2-P1, P1-point))/np.linalg.norm(P2-P1)


def distance_point_to_plane_3D(P1, P2, P3, point):
    """
    Distance from point to plane
    """
    return np.abs(np.dot(np.cross(P2-P1, P3-P1) /
                         np.linalg.norm(np.cross(P2-P1, P3-P1)), point-P2))


@pytest.mark.parametrize("delta", [0.1, 1e-12, 0, -2])
def test_line_point_distance(delta):
    line = np.array([[0.1, 0.2, 0.3], [0.5, 0.8, 0.7]], dtype=np.float64)
    point_on_line = line[0] + 0.27*(line[1]-line[0])
    normal = np.cross(line[0], line[1])
    point = point_on_line + delta * normal
    distance = opengjk.gjk(line, point)
    actual_distance = distance_point_to_line_3D(
        line[0], line[1], point)
    print(distance, actual_distance)
    assert(np.isclose(distance, actual_distance, atol=settol() ))


@pytest.mark.parametrize("delta", [0.1, 1e-12, 0])
def test_line_line_distance(delta):
    line = np.array([[-0.5, -0.7, -0.3], [1, 2, 3]], dtype=np.float64)
    point_on_line = line[0] + 0.38*(line[1]-line[0])
    normal = np.cross(line[0], line[1])
    point = point_on_line + delta * normal
    line_2 = np.array([point, [2, 5, 6]], dtype=np.float64)
    distance = opengjk.gjk(line, line_2)
    actual_distance = distance_point_to_line_3D(
        line[0], line[1], line_2[0])
    print(distance, actual_distance)
    assert(np.isclose(distance, actual_distance, atol=settol() ))


@pytest.mark.parametrize("delta", [0.1**(3*i) for i in range(6)])
def test_tri_distance(delta):
    tri_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tri_2 = np.array([[1, delta, 0], [3, 1.2, 0], [
        1, 1, 0]], dtype=np.float64)
    P1 = tri_1[2]
    P2 = tri_1[1]
    point = tri_2[0]
    actual_distance = distance_point_to_line_3D(P1, P2, point)
    distance = opengjk.gjk(tri_1, tri_2) 
    print("Computed distance ", distance, "Actual distance ", actual_distance)

    #embed()
    assert(np.isclose(distance, actual_distance, atol=settol() ))


@pytest.mark.parametrize("delta", [0.1*0.1**(3*i) for i in range(6)])
def test_quad_distance2d(delta):
    quad_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                       [1, 1, 0]], dtype=np.float64)
    quad_2 = np.array([[0, 1+delta, 0], [2, 2, 0],
                       [2, 4, 0], [4, 4, 0]], dtype=np.float64)
    P1 = quad_1[2]
    P2 = quad_1[3]
    point = quad_2[0]
    actual_distance = distance_point_to_line_3D(P1, P2, point)
    distance = opengjk.gjk(quad_1, quad_2)
    print("Computed distance ", distance, "Actual distance ", actual_distance)

    assert(np.isclose(distance, actual_distance, atol=settol() ))


@pytest.mark.parametrize("delta", [1*0.5**(3*i) for i in range(7)])
def test_tetra_distance_3d(delta):
    tetra_1 = np.array([[0, 0, 0.2], [1, 0, 0.1], [0, 1, 0.3],
                        [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3],
                        [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1],
                                                 tetra_1[2], tetra_2[3])
    distance = opengjk.gjk(tetra_1, tetra_2)
    print("Computed distance ", distance, "Actual distance ", actual_distance)

    assert(np.isclose(distance, actual_distance, atol=settol() ))


@pytest.mark.parametrize("delta", [(-1)**i*np.sqrt(2)*0.1**(3*i)
                                   for i in range(6)])
def test_tetra_collision_3d(delta):
    tetra_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                        [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3],
                        [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1],
                                                 tetra_1[2], tetra_2[3])
    distance = opengjk.gjk(tetra_1, tetra_2)

    if delta < 0:
        assert(np.isclose(distance, 0, atol=settol()))
    else:
        print("Computed distance ", distance,
              "Actual distance ", actual_distance)
        assert(np.isclose(distance, actual_distance, atol=settol()))


@pytest.mark.parametrize("delta", [0, -0.1, -0.49, -0.51])
def test_hex_collision_3d(delta):
    hex_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                      [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                     dtype=np.float64)
    P0 = np.array([1.5+delta, 1.5+delta, 0.5], dtype=np.float64)
    P1 = np.array([2, 2, 1], dtype=np.float64)
    P2 = np.array([2, 1.25, 0.25], dtype=np.float64)
    P3 = P1 + P2 - P0
    quad_1 = np.array([P0, P1, P2, P3], dtype=np.float64)
    n = (np.cross(quad_1[1]-quad_1[0], quad_1[2]-quad_1[0]) /
         np.linalg.norm(
        np.cross(quad_1[1]-quad_1[0],
                 quad_1[2]-quad_1[0])))
    quad_2 = quad_1 + n
    hex_2 = np.zeros((8, 3), dtype=np.float64)
    hex_2[:4, :] = quad_1
    hex_2[4:, :] = quad_2
    actual_distance = np.linalg.norm(
        np.array([1, 1, P0[2]], dtype=np.float64)-hex_2[0])
    distance = opengjk.gjk(hex_1, hex_2)

    if P0[0] < 1:
        assert(np.isclose(distance, 0, atol=settol()))
    else:
        print("Computed distance ", distance,
              "Actual distance ", actual_distance)
        assert(np.isclose(distance, actual_distance, atol=settol()))


@pytest.mark.parametrize("c0", [0, 1, 2, 3])
@pytest.mark.parametrize("c1", [0, 1, 2, 3])
def test_cube_distance(c0, c1):
    cubes = [np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                      dtype=np.float64)]

    r = R.from_euler('z', 45, degrees=True)
    cubes.append(r.apply(cubes[0]))
    r = R.from_euler('y', np.arctan2(1.0, np.sqrt(2)))
    cubes.append(r.apply(cubes[1]))
    r = R.from_euler('y', 45, degrees=True)
    cubes.append(r.apply(cubes[0]))

    dx = cubes[c0][:,0].max() - cubes[c1][:,0].min()
    cube0 = cubes[c0]

    for delta in [1e8, 1.0, 1e-4, 1e-8, 1e-12]:
        cube1 = cubes[c1] + np.array([dx + delta, 0, 0])
        distance = opengjk.gjk(cube0, cube1)
        print(distance, delta)
        assert(np.isclose(distance, delta))

def test_random_objects():
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1000):    
                arr1 = np.random.rand(i, 3)
                arr2 = np.random.rand(j, 3)
                opengjk.gjk(arr1, arr2) 


def test_large_random_objects():
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1000):    
                arr1 = 10000.0*np.random.rand(i, 3)
                arr2 = 10000.0*np.random.rand(j, 3)
                opengjk.gjk(arr1, arr2) 
