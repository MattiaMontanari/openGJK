# Python `ctypes` wrapper

This wrapper uses `ctypes` to wrap the full C API for use from Python.

## Getting Started

To start, build the project with `BUILD_CTYPES` enabled. From the
project root:

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CTYPES=1
    cmake --build .
    cd ../examples/python_ctypes
    ls src/pyopengjk/

At this point you should see either one of `libopengjk_ce.so`,
`opengjk_ce.dll`, or `libopengjk_ce.dylib` depending on your OS.
If not, then something has gone wrong with the build process. Make
certain `BUILD_CTYPES` is enabled, as that should copy the shared
library. If present, then proceed with creating a virtual environment
(posix version shown below):

    python -m venv .env
    . .env/bin/activate
    (.env) pip install --upgrade pip
    (.env) pip install -e .[test]
    (.env) pytest test/
    (.env) pip wheel .

The tests should run (and pass), and then a wheel will be built that
you can install and use in other Python environments.

## Usage

The API exposes the `compute_minimum_distance` function, taking as an
argument two lists of vertices. A `Vertex` type is provided for
convenience, for example:

```python
vertices0 = [
    Point3(0.0, 5.5, 0.0),
    Point3(2.3, 1.0, -2.0),
    Point3(8.1, 4.0, 2.4),
    Point3(4.3, 5.0, 2.2),
    Point3(2.5, 1.0, 2.3),
    Point3(7.1, 1.0, 2.4),
    Point3(1.0, 1.5, 0.3),
    Point3(3.3, 0.5, 0.3),
    Point3(6.0, 1.4, 0.2)
]

vertices1 = [
    Point3(0.0, -5.5, 0.0),
    Point3(-2.3, -1.0, 2.0),
    Point3(-8.1, -4.0, -2.4),
    Point3(-4.3, -5.0, -2.2),
    Point3(-2.5, -1.0, -2.3),
    Point3(-7.1, -1.0, -2.4),
    Point3(-1.0, -1.5, -0.3),
    Point3(-3.3, -0.5, -0.3),
    Point3(-6.0, -1.4, -0.2)
]

distance, simplex = compute_minimum_distance(vertices0, vertices1)
print(f"Minimum distance: {distance}")
print("Witness points:")
print(simplex.witnesses[0])
print(simplex.witnesses[1])
```

will produce the output:

    Minimum distance: 3.653649722294501
    Witness points:
    Point3(x=1.0251728907330566, y=1.4903181189488242, z=0.2554633471645919)
    Point3(x=-1.0251728907330566, y=-1.4903181189488242, z=-0.2554633471645919)

However, any "list of lists" will suffice, *i.e.* a (N, 3) `numpy` array
will work just as well. The simplex that is returned contains
the final simplex vertices and their source indices in addition to
the witness points.
