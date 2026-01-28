# Agent Guidelines for OpenGJK

This document defines coding standards and GPL3 compliance requirements for all source files in this repository.

## License Compliance (GPL-3.0)

**Every new or modified source file MUST include the standardized license header.**

### Standardized License Header

Use this exact header at the top of all `.c`, `.h`, `.cc`, `.hh`, and `.hpp` files:

```c
/*
 *                          _____      _ _  __
 *                         / ____|    | | |/ /
 *   ___  _ __   ___ _ __ | |  __     | | ' /
 *  / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  <
 * | (_) | |_) |  __/ | | | |__| | |__| | . \
 *  \___/| .__/ \___|_| |_|\_____|\____/|_|\_\
 *       | |
 *       |_|
 *
 * Copyright YYYY Mattia Montanari, University of Oxford
 *
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3. See https://www.gnu.org/licenses/
 */
```

Replace `YYYY` with the appropriate year (use range `YYYY-YYYY` if modified across years).

---

## C Code Standards (scalar/)

**All C source (`.c`) and header (`.h`) files MUST follow the [CMU C Coding Standard](https://users.ece.cmu.edu/~eno/coding/CCodingStandard.html).**

This is the authoritative reference for C code style in this project. The key rules are summarized below, but when in doubt, consult the full standard.

### Naming Conventions

> "Names are the heart of programming... A name is the result of a long deep thought process about the ecology it lives in."
> — CMU C Coding Standard

| Element | Convention | Example |
|---------|------------|---------|
| Functions | Verbs with underscores (action names) | `check_for_errors()`, `dump_data_to_file()` |
| Variables | Lowercase with `_` separators | `retry_cnt`, `timeout_msecs` |
| Global constants/macros | ALL_CAPS with `_` | `GJK_MAX_ITERATIONS`, `PIN_OFF` |
| Global variables | Prefix with `g_` | `g_iteration_count` |
| Pointers | `*` near variable name | `char *name`, `gkFloat *v` |
| Struct members | Optional meaningful prefix | `sc_` for `struct softc` |

### Braces and Formatting

```c
if (condition) {
    /* Always use braces, even for single statements */
    do_something();
}
```

- **Lines ≤ 78 characters** — Hard limit
- **One statement per line**
- **Space after keywords** (`if`, `while`, `for`), no space after function names
- **Brace placement**: Opening brace on same line as control statement

### Headers

- **Include guards**: `#ifndef HEADER_H__` / `#define HEADER_H__`
- **No data definitions** in headers — use `extern` declarations
- **`extern "C"` wrappers** for C++ compatibility:

```c
#ifdef __cplusplus
extern "C" {
#endif
/* ... declarations ... */
#ifdef __cplusplus
}
#endif
```

### Critical Rules (from CMU Standard)

1. **No magic numbers** — Use `#define`, `const`, or `enum` with meaningful names
2. **Check return values** of system calls (malloc, fopen, etc.)
3. **Use `#if MACRO` not `#ifdef MACRO`** — Avoids silent misconfiguration
4. **Use `#if 0` to comment out code blocks** — Not `/* */` which can't nest
5. **Be const correct** — Use `const` for parameters that shouldn't change
6. **Document decisions** — Comments should explain *why*, not *what*

### Commenting Out Code

Use `#if 0` with a descriptive macro name:

```c
#if NOT_YET_IMPLEMENTED
/* ... code ... */
#endif

#if OBSOLETE
/* ... code ... */
#endif
```

See full standard: <https://users.ece.cmu.edu/~eno/coding/CCodingStandard.html>

---

## C++ Code Standards (simd/)

The SIMD implementation uses modern C++ with Google Highway.

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Functions | `snake_case` or `PascalCase` | `compute_witnesses()`, `S3D_vector()` |
| Variables | `snake_case` | `num_items`, `max_idx` |
| Constants | `kPascalCase` | `kMaxIterations`, `kEpsilonRel` |
| Template parameters | `PascalCase` | `Policy`, `SearchType` |
| Namespaces | `lowercase` | `opengjk::simd` |

### Modern C++ Practices

- Use `constexpr` for compile-time constants
- Use `[[nodiscard]]` for functions with important return values
- Prefer `const` and `restrict` where applicable
- Use `HWY_INLINE` for SIMD hot paths

### Highway-Specific

- Always include `opengjk_simd_compile_config.h` before Highway headers
- Use `HWY_ALIGN` for SIMD-aligned arrays
- Follow Highway's foreach_target pattern for dynamic dispatch

---

## Comments Policy

### Required Comments

1. **License header** — Every file (see above)
2. **API documentation** — Doxygen `@brief`, `@param`, `@return` for public functions
3. **File-level documentation** — `@file`, `@author`, `@date` for main headers

### Discouraged Comments

- **Inline developer notes** — Code should be self-documenting with clear variable names
- **Redundant comments** — Don't restate what the code clearly does
- **Outdated comments** — Remove rather than risk misleading readers

### Example of Good vs Bad

```c
/* BAD: Redundant inline comment */
i++;  // increment i

/* GOOD: Self-documenting variable name, no comment needed */
iteration_count++;

/* BAD: Developer note that may become stale */
if (det == 0.0) {
    return;  // degenerate case, fall back
}

/* GOOD: Clear code structure, no comment needed */
if (determinant == 0.0) {
    handle_degenerate_simplex();
    return;
}
```

---

## File Structure

```
src/
├── scalar/           # C implementation
│   ├── openGJK.c     # Main algorithm
│   ├── include/      # Public API headers
│   ├── tests/        # Unit tests (cmocka)
│   └── examples/     # Usage examples
├── simd/             # C++ SIMD implementation
│   ├── opengjk.cc    # Dynamic dispatch entry
│   ├── opengjk-inl.h # SIMD kernels
│   ├── include/      # Public API headers
│   └── tests/        # Unit tests (gtest)
└── AGENTS.md         # This file
```

---

## Checklist for Contributors

Before committing changes:

- [ ] License header present and correct
- [ ] Code follows naming conventions
- [ ] No inline developer comments (code is self-documenting)
- [ ] API functions have Doxygen documentation
- [ ] Tests pass (`ctest` or platform-specific test runner)
