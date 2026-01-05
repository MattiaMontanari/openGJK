<!--                        _____      _ _  __                                      >
<                          / ____|    | | |/ /                                      >
<    ___  _ __   ___ _ __ | |  __     | | ' /                                       >
<   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <                                        >
<  | (_) | |_) |  __/ | | | |__| | |__| | . \                                       >
<   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\                                      >
<        | |                                                                        >
<        |_|                                                                        >
<                                                                                   >
< Copyright 2022-2026 Mattia Montanari, University of Oxford                        >
<                                                                                   >
< This program is free software: you can redistribute it and/or modify it under     >
< the terms of the GNU General Public License as published by the Free Software     >
< Foundation, either version 3 of the License. You should have received a copy      >
< of the GNU General Public License along with this program. If not, visit          >
<                                                                                   >
<     https://www.gnu.org/licenses/                                                 >
<                                                                                   >
< This program is distributed in the hope that it will be useful, but WITHOUT       >
< ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS     >
< FOR A PARTICULAR PURPOSE. See GNU General Public License for details.           -->

[![Language Bindings](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-examples.yml/badge.svg)](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-examples.yml)
[![Tests](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/MattiaMontanari/openGJK/actions/workflows/ci-tests.yml)

# OpenGJK

A fast and robust implementation of the Gilbert-Johnson-Keerthi (GJK) algorithm for computing minimum distances between convex polytopes. Available in two flavors:

- **Scalar** (`scalar/`): Portable C implementation with interfaces for C#, Go, Matlab, Python, and Zig
- **SIMD** (`simd/`): High-performance C++ implementation using [Google Highway](https://github.com/google/highway) for automatic SIMD acceleration (SSE4, AVX2, AVX-512, NEON)

A Unity Plug-in [is also available in another repository](https://github.com/MattiaMontanari/urban-couscous).

Useful links: [API references](https://www.mattiamontanari.com/opengjk/docsapi/), [documentation](https://www.mattiamontanari.com/opengjk/docs/) and [automated benchmarks](https://www.mattiamontanari.com/opengjk/docs/benchmarks/).

## Getting started

On Linux, Mac or Windows, install a basic C/C++ toolchain - for example: git, compiler and cmake.

### Prerequisites

**Required:**

- Git
- C/C++ compiler (GCC, Clang, or MSVC)
- CMake (version 3.5 or higher)

**Recommended for faster builds:**

- Ninja build system (provides ~60% faster compilation)

```bash
# Install Ninja (if not already installed)
# Ubuntu/Debian: sudo apt install ninja-build
# macOS: brew install ninja
# Windows: choco install ninja
```

Next, clone this repo:

``` bash
git clone https://github.com/MattiaMontanari/openGJK
```

Then use these commands to build and run an example:

``` bash
cmake -E make_directory build
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. 
cmake --build build 
cmake -E chdir build/examples/c ./example_lib_opengjk_ce
```

The successful output should be:

>
> `Distance between bodies 3.653650`
>

However, if you do get an error - any error - please file a bug. Support requests are welcome.

## Build Options

OpenGJK supports several CMake options to customize the build:

- `BUILD_EXAMPLE` (default: ON) - Build the C demo example
- `BUILD_MONO` (default: OFF) - Build C# example (requires Mono)
- `BUILD_CTYPES` (default: OFF) - Expose symbols for Python ctypes
- `FORCE_CXX_COMPILER` (default: OFF) - Force C++ compiler for C files (useful for cross-compilation)
- `SINGLE_PRECISION` (default: OFF) - Use 32-bit floating point instead of 64-bit

### SIMD Build (simd/)

The SIMD implementation requires the [Google Highway](https://github.com/google/highway) library. It's fetched automatically via CMake FetchContent:

```bash
cd simd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

SIMD-specific options:

- `OPENGJK_SIMD_USE_FLOAT` - Use 32-bit float (default: 64-bit double)
- `OPENGJK_SIMD_MINIMAL_WIDTH` - Prefer smallest viable SIMD width

Example with custom options:

```bash
cmake -E chdir build cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_MONO=ON -DFORCE_CXX_COMPILER=ON -G Ninja ..
```

## Use OpenGJK in your project

The best source to learn how to use OpenGJK are the examples. They are listed [here](https://www.mattiamontanari.com/opengjk/docs/examples/) for C, C#, Go, Matlab, Zig and Python. I aim to publish a few more for Julia.

Take a look at the `examples` folder in this repo and have fun. File a request if you wish to see more!

## Contribute

You are very welcome to:

- Create pull requests of any kind
- Let me know if you are using this library and find it useful
- Open issues with request for support because they will help you and many others
- Cite this repository ([a sweet GitHub feature](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files#about-citation-files)) or my paper: Montanari, M. et at, *Improving the GJK Algorithm for Faster and More Reliable Distance Queries Between Convex Objects* (2017). ACM Trans. Graph.
