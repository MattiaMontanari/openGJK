# OpenGJK GPU - Python Wrapper

GPU-accelerated GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) for high-performance collision detection in Python.

## Features

- **Batch Processing**: Process thousands of collision pairs in a single GPU call
- **Warp-Level Parallelism**: 8 threads per GJK by default for efficient GPU utilization
- **NumPy Integration**: Native support for NumPy arrays with vectorized operations
- **Multiple Modes**:
  - GJK for distance computation
  - EPA for penetration depth and contact normals
  - Combined GJK+EPA pipeline
  - Indexed API for polytope reuse

## Requirements

- **NVIDIA GPU** with CUDA support (compute capability 6.0+)
- **CUDA Toolkit** (11.0 or higher)
- **Python** 3.7+
- **NumPy** (required)
- **CMake** 3.18+
- **C/C++ compiler** with CUDA support

## Installation

### 1. Build the Shared Library

From the repository root:

```bash
# Configure with shared library enabled
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF

# Build
cmake --build build --config Release

# The library will be at:
# Windows: build/gpu/Release/openGJK_GPU.dll
# Linux:   build/gpu/libopenGJK_GPU.so
# macOS:   build/gpu/libopenGJK_GPU.dylib
```

### 2. Set Up Python

```bash
cd gpu/examples/python

# Install NumPy if not already installed
pip install numpy

# Run example
python example.py

# Run comprehensive test suite
python test_examples.py
```

The Python wrapper will automatically search for the shared library in common locations.

## Usage

See [example.py](example.py) for a simple collision detection example and [test_examples.py](test_examples.py) for comprehensive test cases demonstrating all API features.

## API Reference

### `compute_minimum_distance(polytopes1, polytopes2)`

Compute minimum distance using GJK algorithm.

**Parameters:**
- `polytopes1`: NumPy array of shape `(n_pairs, n_vertices, 3)`
- `polytopes2`: NumPy array of shape `(n_pairs, n_vertices, 3)`

**Returns:**
Dictionary with NumPy arrays:
- `'distances'`: shape `(n_pairs,)` - minimum distances between polytope pairs
- `'witnesses1'`: shape `(n_pairs, 3)` - closest points on first polytopes
- `'witnesses2'`: shape `(n_pairs, 3)` - closest points on second polytopes
- `'simplex_nvrtx'`: shape `(n_pairs,)` - number of vertices in final simplex

---

### `compute_epa(polytopes1, polytopes2, return_normals=False)`

Compute penetration depth and contact information using EPA.

**Parameters:**
- `polytopes1`: NumPy array of shape `(n_pairs, n_vertices, 3)`
- `polytopes2`: NumPy array of shape `(n_pairs, n_vertices, 3)`
- `return_normals`: bool - whether to compute contact normals (default: False)

**Returns:**
Dictionary with NumPy arrays:
- `'penetration_depths'`: shape `(n_pairs,)` - penetration distances
- `'witnesses1'`: shape `(n_pairs, 3)` - contact points on first polytopes
- `'witnesses2'`: shape `(n_pairs, 3)` - contact points on second polytopes
- `'contact_normals'`: shape `(n_pairs, 3)` - contact normals (only if `return_normals=True`)

---

### `compute_gjk_epa(polytopes1, polytopes2)`

Combined GJK+EPA pipeline. More efficient than separate calls.

**Parameters:**
- `polytopes1`: NumPy array of shape `(n_pairs, n_vertices, 3)`
- `polytopes2`: NumPy array of shape `(n_pairs, n_vertices, 3)`

**Returns:**
Dictionary with NumPy arrays:
- `'distances'`: shape `(n_pairs,)` - distances or penetration depths
- `'witnesses1'`: shape `(n_pairs, 3)` - witness or contact points on first polytopes
- `'witnesses2'`: shape `(n_pairs, 3)` - witness or contact points on second polytopes
- `'simplex_nvrtx'`: shape `(n_pairs,)` - number of vertices in final simplex

---

### `compute_minimum_distance_indexed(polytopes, pairs)`

Indexed collision detection for efficient polytope reuse.

**Parameters:**
- `polytopes`: NumPy array of shape `(n_polytopes, n_vertices, 3)`
- `pairs`: NumPy array of shape `(n_pairs, 2)` with dtype `int32` - indices into `polytopes` array

**Returns:**
Dictionary with NumPy arrays (same structure as `compute_minimum_distance`):
- `'distances'`: shape `(n_pairs,)`
- `'witnesses1'`: shape `(n_pairs, 3)`
- `'witnesses2'`: shape `(n_pairs, 3)`
- `'simplex_nvrtx'`: shape `(n_pairs,)`

---

**Supported dtypes:** `np.float32` (32-bit), `np.float64` (64-bit)

## License

GPL-3.0-only (same as OpenGJK)

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund

## See Also

- [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) - GPU implementation
- [OpenGJK](https://www.mattiamontanari.com/opengjk/) - Main project
- [GPU API Documentation](../../README.md) - C/CUDA API reference
