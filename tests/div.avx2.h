#ifndef DIV_AVX2_H
#define DIV_AVX2_H

#include <immintrin.h>
#include <stdint.h>

// Compute magic multiplier for division by constant d
// Returns multiplier m and shift s such that: floor(n/d) = (n*m) >> (64+s)
void
compute_magic_div(uint32_t d, uint64_t* m, int* s);

// Scalar magic division using 128-bit arithmetic
// Returns n / d using formula: n/d = (high64(n*m)) >> s
uint64_t
magic_div(uint64_t n, uint64_t m, int s);

// AVX2: 64-bit multiply for 4x 64-bit values
// Computes a * b for each lane
__m256i
mm256_mullo_epi64(__m256i a, __m256i b);

// AVX2: divide 4x 64-bit values by constant using magic multiplier
// Computes high 64 bits of n*m, then shifts right by s
// Formula: n/d = (high64(n*m)) >> s
__m256i
mm256_div_epu64_const(__m256i n, uint64_t m, int s);

#endif // DIV_AVX2_H
