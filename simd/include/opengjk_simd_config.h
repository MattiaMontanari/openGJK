// OpenGJK SIMD Configuration Header
//
// This header is included before Highway headers for configuration.
// Target selection is now done at runtime via opengjk_simd_init.h
// using Highway's SetSupportedTargetsForTest() mechanism.
//
// This approach is future-proof and works on x86, ARM, and future architectures.

#ifndef OPENGJK_SIMD_CONFIG_H_
#define OPENGJK_SIMD_CONFIG_H_

// No compile-time target restrictions.
// Runtime target filtering is handled by InitSIMDTargets() in opengjk_simd_init.h

#endif  // OPENGJK_SIMD_CONFIG_H_
