// OpenGJK SIMD Configuration Header
//
// This header is included before Highway headers for configuration.
// Target selection is now done at runtime via opengjk_simd_init.h
// using Highway's SetSupportedTargetsForTest() mechanism.
//
// This approach is future-proof and works on x86, ARM, and future architectures.

#ifndef OPENGJK_SIMD_CONFIG_H_
#define OPENGJK_SIMD_CONFIG_H_

#include "hwy/detect_compiler_arch.h"

// Disable SVE targets on Apple Silicon (M1/M2/M3/M4).
// Apple Silicon only supports NEON, not SVE. Additionally, SVE uses "sizeless
// types" which cannot be used in arrays (e.g., V (&S)[4] in our simplex code).
// Highway normally marks SVE as broken on Apple, but we explicitly disable it
// here to ensure compile-time safety regardless of Highway version.
#if HWY_ARCH_ARM_A64 && HWY_OS_APPLE
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (HWY_SVE | HWY_SVE2 | HWY_SVE_256 | HWY_SVE2_128)
#endif
#endif

#endif // OPENGJK_SIMD_CONFIG_H_
