# OpenGJK GPU Implementation

CUDA-accelerated implementation of the Gilbert-Johnson-Keerthi (GJK) algorithm and Expanding Polytope Algorithm (EPA) for batch collision detection on NVIDIA GPUs.

## Features

- **GJK Algorithm**: Compute minimum distances between convex polytopes
- **EPA Algorithm**: Compute penetration depth and witness points for colliding objects
- **Warp-level parallelism**: 16 threads per collision for GJK, 32 threads per collision for EPA
- **Batch processing**: Process thousands of polytope pairs simultaneously
- **Flexible APIs**: Three API levels for different use cases
- **Indexed API**: Efficient handling of shared polytopes across multiple collision queries

## Build Instructions

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON -DBUILD_SCALAR=OFF -DBUILD_SIMD=OFF
cmake --build build --config Release
```

Run the example:

```bash
cd build/gpu/examples/simple_collision/Release
./example_lib_opengjk_gpu.exe
```

### Build Options

- `BUILD_GPU` - Enable GPU build (default: OFF)
- `USE_32BITS` - Use 32-bit float precision (default: ON, recommended for GPU)

## API Variants

OpenGJK GPU provides three API levels:

### 1. High-level API

Handles all GPU memory management automatically. Best for simple integration or one-off computations.

```c
gkFloat compute_minimum_distance(
    const int n,
    const gkPolytope* bd1,
    const gkPolytope* bd2,
    gkSimplex* simplices,
    gkFloat* distances
);
```

### 2. Mid-level API

Separate allocation, computation, and deallocation phases. Ideal for persistent objects across multiple frames.

```c
// Allocate once
allocate_and_copy_device_arrays(n, bd1, bd2, &d_bd1, &d_bd2, &d_coord1, &d_coord2, &d_simplices, &d_distances);

// Compute many times
compute_minimum_distance_device(n, d_bd1, d_bd2, d_simplices, d_distances);
copy_results_from_device(n, d_simplices, d_distances, simplices, distances);

// Free when done
free_device_arrays(d_bd1, d_bd2, d_coord1, d_coord2, d_simplices, d_distances);
```

### 3. Low-level API (Kernel Access)

Direct kernel invocation for users managing their own GPU memory.

```c
compute_minimum_distance_kernel<<<grid, block>>>(d_polytopes1, d_polytopes2, d_simplices, d_distances, n);
```

### Indexed API

Efficiently handle scenarios where polytopes are shared across multiple collision queries:

```c
compute_minimum_distance_indexed(
    const int n_pairs,
    const gkPolytope* polytopes,    // Single array of unique polytopes
    const gkCollisionPair* pairs,   // Array of index pairs
    gkSimplex* simplices,
    gkFloat* distances
);
```

### EPA API

Compute penetration depth and witness points for colliding objects:

```c
// Combined GJK + EPA
compute_gjk_epa(n, bd1, bd2, simplices, distances, witness1, witness2, contact_normals);

// EPA only (after GJK)
compute_epa(n, bd1, bd2, simplices, distances, witness1, witness2, contact_normals);
```

## Files

- `openGJK_GPU.cu` - Complete CUDA implementation (kernels + host API)
- `include/openGJK_GPU.h` - Public API header
- `examples/simple_collision/` - Basic usage example

## Performance

For detailed performance benchmarks and analysis, see the [OpenGJK-GPU repository](https://github.com/vismaychuriwala/OpenGJK-GPU).

## Data Format

GPU implementation uses **flattened coordinate arrays** for efficient memory access:

```c
// GPU format: flattened array [x0,y0,z0, x1,y1,z1, ...]
gkPolytope.coord = gkFloat*

// Scalar format: array of pointers [[x0,y0,z0], [x1,y1,z1], ...]
gkPolytope.coord = gkFloat**
```

## Credits

Based on [OpenGJK-GPU](https://github.com/vismaychuriwala/OpenGJK-GPU) by Vismay Churiwala and Marcus Hedlund.

Original GJK implementation by Mattia Montanari.
