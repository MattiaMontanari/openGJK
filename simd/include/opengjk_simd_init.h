// OpenGJK SIMD Runtime Target Initialization
//
// This header provides runtime control over which SIMD targets Highway uses.
// Call InitSIMDTargets() once at program startup before any SIMD code runs.
//
// This approach is future-proof because it filters targets based on properties
// (vector width) rather than hardcoded target lists.

#ifndef OPENGJK_SIMD_INIT_H_
#define OPENGJK_SIMD_INIT_H_

#include <hwy/highway.h>
#include <hwy/targets.h>

namespace opengjk {
namespace simd {

// Target width classification
// These masks identify targets by their vector width capability

// 128-bit or smaller targets (not suitable for double precision 3D vectors)
// Includes: SSE2, SSSE3, SSE4, NEON (128-bit), WASM, EMU128, SCALAR
inline constexpr int64_t kTargets128BitOrLess = HWY_SCALAR | HWY_EMU128 | HWY_SSE2 | HWY_SSSE3 | HWY_SSE4 | HWY_NEON
                                                | HWY_NEON_WITHOUT_AES | HWY_WASM | HWY_WASM_EMU256;

// 256-bit targets (minimum for double precision, optional for float)
// Includes: AVX2, SVE_256, SVE2_128 (can be 256-bit in some configs)
inline constexpr int64_t kTargets256Bit = HWY_AVX2;

// 512-bit+ targets (AVX-512 variants, large SVE)
// These may be disabled when preferring minimal width
inline constexpr int64_t kTargets512BitPlus =
    HWY_AVX3 | HWY_AVX3_DL | HWY_AVX3_ZEN4 | HWY_AVX3_SPR | HWY_SVE | HWY_SVE2 | HWY_SVE_256 | HWY_SVE2_128;

// SVE targets (ARM Scalable Vector Extension)
// SVE uses "sizeless types" which cannot be stored in arrays. Our simplex code
// uses V (&S)[4] arrays, making SVE incompatible. SVE is also NOT supported on
// Apple Silicon (M1/M2/M3/M4) - only NEON. SVE is primarily for ARM servers
// (AWS Graviton3, Fujitsu A64FX, etc.).
// Note: SVE is disabled at compile-time in opengjk_simd_compile_config.h for Apple.
inline constexpr int64_t kTargetsSVE = HWY_SVE | HWY_SVE2 | HWY_SVE_256 | HWY_SVE2_128;

// NEON targets (ARM Advanced SIMD)
// NEON is the primary SIMD instruction set for Apple Silicon and most ARM64.
// Always 128-bit vectors. This is what Apple Silicon uses.
inline constexpr int64_t kTargetsNEON = HWY_NEON | HWY_NEON_WITHOUT_AES | HWY_NEON_BF16;

// Returns true if target provides at least 256-bit vectors (4 doubles)
inline bool
IsAtLeast256Bit(int64_t target) {
  // A target is 256-bit+ if it's NOT in the 128-bit-or-less category
  return (target & kTargets128BitOrLess) == 0;
}

// Returns true if target provides at least 128-bit vectors (4 floats)
inline bool
IsAtLeast128Bit(int64_t target) {
  return target != HWY_SCALAR;
}

// Returns true if target is 512-bit or larger
inline bool
Is512BitOrMore(int64_t target) {
  return (target & kTargets512BitPlus) != 0;
}

// Returns true if target is an SVE variant (incompatible with array storage)
inline bool
IsSVE(int64_t target) {
  return (target & kTargetsSVE) != 0;
}

// Filter supported targets based on requirements
// Returns a mask of targets that meet the specified criteria
inline int64_t
FilterTargets(int64_t supported, bool require_256bit, bool prefer_minimal_width) {
  int64_t filtered = 0;

  // Iterate through each supported target
  for (int64_t remaining = supported; remaining != 0;) {
    int64_t target = remaining & -remaining; // Extract lowest set bit
    remaining &= remaining - 1;              // Clear lowest bit

    bool keep = true;

    // Always filter out SVE targets - our code uses V (&S)[4] arrays which
    // are incompatible with SVE's sizeless types. NEON is used instead on ARM.
    if (IsSVE(target)) {
      keep = false;
    }

    // Filter out targets that don't meet minimum width requirement
    if (keep && require_256bit && !IsAtLeast256Bit(target)) {
      keep = false;
    }

    // For minimal width mode, filter out targets wider than necessary
    if (keep && prefer_minimal_width) {
      if (require_256bit) {
        // For double: keep 256-bit, disable 512-bit+
        if (Is512BitOrMore(target)) {
          keep = false;
        }
      } else {
        // For float: keep 128-bit, disable 256-bit+
        if (IsAtLeast256Bit(target)) {
          keep = false;
        }
      }
    }

    if (keep) {
      filtered |= target;
    }
  }

  return filtered;
}

// Initialize SIMD target selection
//
// Parameters:
//   require_256bit: If true, only 256-bit+ targets are allowed (for double precision)
//   prefer_minimal_width: If true, prefer smallest viable width over widest
//
// Call this once at program startup before any HWY_DYNAMIC_DISPATCH calls.
// This affects all subsequent dynamic dispatch decisions.
inline void
InitSIMDTargets(bool require_256bit, bool prefer_minimal_width) {
  int64_t supported = hwy::SupportedTargets();
  int64_t filtered = FilterTargets(supported, require_256bit, prefer_minimal_width);

  // Ensure we have at least one viable target
  if (filtered == 0) {
    // Fall back to whatever is supported - better than crashing
    filtered = supported;
  }

  // Apply the filtered targets
  // This affects future HWY_DYNAMIC_DISPATCH calls
  hwy::SetSupportedTargetsForTest(filtered);
}

// Initialize with compile-time configuration
// Uses OPENGJK_SIMD_USE_FLOAT and OPENGJK_SIMD_MINIMAL_WIDTH defines
inline void
InitSIMDTargetsFromConfig() {
#ifdef OPENGJK_SIMD_USE_FLOAT
  constexpr bool kRequire256Bit = false; // float works with 128-bit
#else
  constexpr bool kRequire256Bit = true; // double needs 256-bit
#endif

#ifdef OPENGJK_SIMD_MINIMAL_WIDTH
  constexpr bool kMinimalWidth = true;
#else
  constexpr bool kMinimalWidth = false;
#endif

  InitSIMDTargets(kRequire256Bit, kMinimalWidth);
}

// Reset to auto-detect all CPU-supported targets
// Useful for testing or when you want to restore default behavior
inline void
ResetSIMDTargets() {
  hwy::SetSupportedTargetsForTest(0); // 0 means auto-detect
}

// Get human-readable description of current SIMD configuration
inline const char*
GetSIMDConfigDescription() {
#ifdef OPENGJK_SIMD_USE_FLOAT
#ifdef OPENGJK_SIMD_MINIMAL_WIDTH
  return "float (128-bit SSE4/NEON preferred)";
#else
  return "float (widest available, NEON on ARM)";
#endif
#else
#ifdef OPENGJK_SIMD_MINIMAL_WIDTH
  return "double (256-bit AVX2 preferred, NEON on ARM)";
#else
  return "double (widest available, NEON on ARM)";
#endif
#endif
}

} // namespace simd
} // namespace opengjk

#endif // OPENGJK_SIMD_INIT_H_
