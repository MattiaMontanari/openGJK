# OpenGJK GPU - Python Wrapper

GPU-accelerated GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm) for high-performance collision detection in Python.

## Features

- **Batch Processing**: Process thousands of collision pairs in a single GPU call
- **High Performance**: Warp-level parallelism (16 threads per GJK, 32 per EPA)
- **NumPy-First API**: Optimized for NumPy arrays with vectorized operations
- **Zero-Copy Results**: Returns NumPy arrays directly from GPU memory
- **Multiple Modes**:
  - GJK for distance computation
  - EPA for penetration depth and witness points
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

# Install NumPy (optional but recommended)
pip install numpy

# Run example
python example.py
```

The Python wrapper will automatically search for the shared library in common locations.

## Quick Start

### Single Collision Pair

```python
import numpy as np
from opengjk_gpu import compute_minimum_distance

# Define two polytopes as NumPy arrays
vertices1 = np.array([
    [0.0, 5.5, 0.0],
    [2.3, 1.0, -2.0],
    [8.1, 4.0, 2.4],
    # ... more vertices
], dtype=np.float32)

vertices2 = np.array([
    [0.0, -5.5, 0.0],
    [-2.3, -1.0, 2.0],
    [-8.1, -4.0, -2.4],
    # ... more vertices
], dtype=np.float32)

# Compute distance - returns NumPy arrays
result = compute_minimum_distance(vertices1, vertices2)

print(f"Distance: {result['distances'][0]}")
print(f"Collision: {result['is_collision'][0]}")
print(f"Witness 1: {result['witnesses1'][0]}")
print(f"Witness 2: {result['witnesses2'][0]}")
```

### Batch Processing (1000 pairs)

```python
import numpy as np
from opengjk_gpu import compute_minimum_distance

# Generate 1000 random polytope pairs as lists of NumPy arrays
vertices1_list = [np.random.randn(10, 3).astype(np.float32) + [10, 0, 0]
                  for _ in range(1000)]
vertices2_list = [np.random.randn(10, 3).astype(np.float32) - [10, 0, 0]
                  for _ in range(1000)]

# Process all pairs in ONE GPU call - returns NumPy arrays
result = compute_minimum_distance(vertices1_list, vertices2_list)

# Vectorized analysis
distances = result['distances']  # NumPy array (1000,)
is_collision = result['is_collision']  # NumPy boolean array (1000,)

print(f"Processed 1000 pairs")
print(f"Collisions: {is_collision.sum()}")  # Vectorized count
print(f"Average distance: {distances.mean():.3f}")  # Vectorized mean
print(f"Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
```

### Collision Detection with EPA

```python
import numpy as np
from opengjk_gpu import compute_gjk_epa

# Two overlapping cubes as NumPy arrays
cube1 = np.array([
    [-1,-1,-1], [1,-1,-1], [-1,1,-1], [1,1,-1],
    [-1,-1,1], [1,-1,1], [-1,1,1], [1,1,1]
], dtype=np.float32)

cube2 = np.array([
    [0.5,-1,-1], [2.5,-1,-1], [0.5,1,-1], [2.5,1,-1],
    [0.5,-1,1], [2.5,-1,1], [0.5,1,1], [2.5,1,1]
], dtype=np.float32)

# Combined GJK+EPA - returns NumPy arrays
result = compute_gjk_epa(cube1, cube2)

if result['is_collision'][0]:
    print(f"Penetration depth: {result['distances'][0]}")
    print(f"Contact point 1: {result['witnesses1'][0]}")
    print(f"Contact point 2: {result['witnesses2'][0]}")
```

### Indexed API (Polytope Reuse)

```python
import numpy as np
from opengjk_gpu import compute_minimum_distance_indexed

# Define unique polytopes as NumPy arrays
polytopes = [
    cube_vertices,      # np.array (8, 3)
    sphere_vertices,    # np.array (20, 3)
    capsule_vertices,   # np.array (12, 3)
    tetrahedron_vertices  # np.array (4, 3)
]

# Specify which pairs to check as NumPy array
pairs = np.array([
    [0, 1],  # cube vs sphere
    [0, 2],  # cube vs capsule
    [1, 3],  # sphere vs tetrahedron
    # ... more pairs
], dtype=np.int32)

# Efficient: polytopes transferred once, reused for all pairs
result = compute_minimum_distance_indexed(polytopes, pairs)

# Vectorized access to results
print(f"Collision mask: {result['is_collision']}")
print(f"Distances: {result['distances']}")
```

## API Reference

### `compute_minimum_distance(vertices1, vertices2)`

Compute minimum distance using GJK algorithm.

**Args:**
- `vertices1`: NumPy array (n,3) for single pair, or list of arrays for batch
- `vertices2`: NumPy array (n,3) for single pair, or list of arrays for batch

**Returns:** Dictionary with NumPy arrays:
- `'distances'`: (n,) array of minimum distances (0.0 = collision)
- `'witnesses1'`: (n,3) array of closest points on first polytopes
- `'witnesses2'`: (n,3) array of closest points on second polytopes
- `'is_collision'`: (n,) boolean array indicating collisions
- `'simplex_nvrtx'`: (n,) array of simplex vertex counts

**Batch mode:** Pass lists of NumPy arrays to process multiple pairs efficiently.

### `compute_epa(vertices1, vertices2, return_normals=False)`

Compute penetration depth and witness points using EPA.

**Args:**
- `vertices1`, `vertices2`: NumPy array (n,3) or list of arrays
- `return_normals`: If True, compute contact normals

**Returns:** Dictionary with NumPy arrays:
- `'penetration_depths'`: (n,) array of penetration distances
- `'witnesses1'`: (n,3) array of contact points on first polytopes
- `'witnesses2'`: (n,3) array of contact points on second polytopes
- `'contact_normals'`: (n,3) array of contact normals (if `return_normals=True`)

### `compute_gjk_epa(vertices1, vertices2)`

Combined GJK+EPA pipeline (more efficient than separate calls).

**Returns:** Dictionary with NumPy arrays (combines GJK and EPA results):
- `'distances'`: (n,) array of distances/penetration depths
- `'is_collision'`: (n,) boolean array
- `'witnesses1'`, `'witnesses2'`: (n,3) arrays of witness/contact points
- `'simplex_nvrtx'`: (n,) array of simplex vertex counts

### `compute_minimum_distance_indexed(polytopes, pairs)`

Indexed collision detection for polytope reuse.

**Args:**
- `polytopes`: List of NumPy arrays (each shape (n,3))
- `pairs`: NumPy array of shape (m,2) with integer indices

**Returns:** Dictionary with NumPy arrays (same structure as `compute_minimum_distance`)

## Performance

Typical performance on NVIDIA RTX 3080:

| Pairs | Time | Throughput |
|-------|------|------------|
| 1 | ~0.5 ms | 2K pairs/sec |
| 100 | ~1 ms | 100K pairs/sec |
| 1000 | ~5 ms | 200K pairs/sec |
| 10000 | ~40 ms | 250K pairs/sec |

Performance scales with:
- GPU compute capability
- Polytope complexity (vertex count)
- Memory transfer overhead (batch larger for better throughput)

## Troubleshooting

### Library Not Found

If you get `RuntimeError: Could not find openGJK_GPU shared library`:

1. Verify the library was built:
   ```bash
   ls build/gpu/Release/openGJK_GPU.dll  # Windows
   ls build/gpu/libopenGJK_GPU.so        # Linux
   ```

2. Copy the library to the Python directory:
   ```bash
   cp build/gpu/Release/openGJK_GPU.dll gpu/examples/python/
   ```

3. Or set the library path explicitly:
   ```python
   import os
   os.add_dll_directory(r"d:\gpu_programming\openGJK\build\gpu\Release")
   from opengjk_gpu import compute_minimum_distance
   ```

### Precision Mismatch

The wrapper auto-detects precision. If you built with 64-bit:

```bash
cmake -B build -DUSE_32BITS=OFF ...
```

Update `opengjk_gpu.py` line 20:
```python
USE_32BITS = False  # Match your build configuration
```

### CUDA Errors

If you get CUDA runtime errors:

1. Check GPU compatibility: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Ensure compute capability ≥ 6.0

## Examples

Run the included examples:

```bash
python example.py
```

This demonstrates:
1. Single collision pair (same as C example)
2. Batch processing (1000 pairs)
3. Collision detection with EPA
4. Indexed API usage
5. NumPy array input

## Integration with Existing Code

The Python wrapper is self-contained - just copy `opengjk_gpu.py` to your project:

```python
# In your project
from opengjk_gpu import compute_minimum_distance

# Your collision detection code
result = compute_minimum_distance(obj1_vertices, obj2_vertices)
if result.is_collision:
    handle_collision(result)
```

## License

GPL-3.0-only (same as OpenGJK)

Copyright 2022-2026 Mattia Montanari, University of Oxford
Copyright 2025-2026 Vismay Churiwala, Marcus Hedlund

## See Also

- [OpenGJK](https://www.mattiamontanari.com/opengjk/) - Main project
- [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) - Original GPU implementation
- [GPU API Documentation](../../README.md) - C/CUDA API reference
