#include "div.avx2.h"

void
compute_magic_div(uint32_t d, uint64_t* m, int* s)
{
  // We want to find m and s such that:
  // floor(n/d) = floor((n * m) / 2^(64+s)) for all n in [0, 2^64)

  // Start with p = 0 and increase until we find a good one
  for (int shift = 0; shift < 32; shift++) {
    // 2^(64+shift) = 2^64 * 2^shift
    // We want to compute ceil(2^(64+shift) / d)

    // Use 128-bit arithmetic
    // 2^(64+shift) / d = (2^shift * 2^64) / d
    __uint128_t numerator = (__uint128_t)1 << (64 + shift);
    __uint128_t quotient = numerator / d;
    __uint128_t remainder = numerator % d;

    if (remainder != 0) {
      quotient++; // Round up
    }

    // Check if this quotient fits in 64 bits and works
    if (quotient <= ((__uint128_t)1 << 64)) {
      *m = (uint64_t)quotient;
      *s = shift;
      return;
    }
  }

  // Fallback - should not reach here for reasonable divisors
  *m = (((__uint128_t)1 << 64) / d) + 1;
  *s = 0;
}

uint64_t
magic_div(uint64_t n, uint64_t m, int s)
{
  // Compute full 128-bit product: n * m
  __uint128_t product = (__uint128_t)n * (__uint128_t)m;

  // Get high 64 bits and shift by s
  uint64_t hi = (uint64_t)(product >> 64);

  return hi >> s;
}

__m256i
mm256_mullo_epi64(__m256i a, __m256i b)
{
  // Use _mm256_mul_epu32 which multiplies 32-bit unsigned integers
  // and produces 64-bit results for even-indexed elements

  // Get low 32 bits
  __m256i a_lo = a;
  __m256i b_lo = b;

  // Get high 32 bits (shift right by 32)
  __m256i a_hi = _mm256_srli_epi64(a, 32);
  __m256i b_hi = _mm256_srli_epi64(b, 32);

  // Multiply: (a_hi*b_lo + a_lo*b_hi) << 32 + a_lo*b_lo
  __m256i lo_lo = _mm256_mul_epu32(a_lo, b_lo);
  __m256i hi_lo = _mm256_mul_epu32(a_hi, b_lo);
  __m256i lo_hi = _mm256_mul_epu32(a_lo, b_hi);

  // Add cross terms and shift
  __m256i cross = _mm256_add_epi64(hi_lo, lo_hi);
  cross = _mm256_slli_epi64(cross, 32);

  return _mm256_add_epi64(lo_lo, cross);
}

__m256i
mm256_div_epu64_const(__m256i n, uint64_t m, int s)
{
  __m256i vm = _mm256_set1_epi64x(m);

  // Compute high 64 bits of 64x64->128 multiply for each lane
  // Split into 32-bit parts: n = n_hi:n_lo, m = m_hi:m_lo
  // Product = n*m = n_hi*m_hi*2^64 + (n_hi*m_lo + n_lo*m_hi)*2^32 + n_lo*m_lo
  //
  // High 64 bits = n_hi*m_hi + high32(n_hi*m_lo + n_lo*m_hi + high32(n_lo*m_lo))

  __m256i n_lo = n;                         // Low 32 bits of n (in low part of 64-bit)
  __m256i n_hi = _mm256_srli_epi64(n, 32);  // High 32 bits of n
  __m256i m_lo = vm;
  __m256i m_hi = _mm256_srli_epi64(vm, 32);

  // Compute partial products (each is 32x32->64)
  __m256i p_lo_lo = _mm256_mul_epu32(n_lo, m_lo);  // n_lo * m_lo -> 64-bit
  __m256i p_hi_lo = _mm256_mul_epu32(n_hi, m_lo);  // n_hi * m_lo -> 64-bit
  __m256i p_lo_hi = _mm256_mul_epu32(n_lo, m_hi);  // n_lo * m_hi -> 64-bit
  __m256i p_hi_hi = _mm256_mul_epu32(n_hi, m_hi);  // n_hi * m_hi -> 64-bit

  // Combine to get high 64 bits
  // high64 = p_hi_hi + high32(p_hi_lo) + high32(p_lo_hi) + high32(p_lo_lo + (p_hi_lo<<32) + (p_lo_hi<<32))
  //
  // Simplify: Start with high32(p_lo_lo), add low32(p_hi_lo) and low32(p_lo_hi)
  // Then add carries to p_hi_hi

  __m256i mid_sum = _mm256_add_epi64(
    _mm256_srli_epi64(p_lo_lo, 32),   // high32(p_lo_lo)
    _mm256_add_epi64(
      _mm256_and_si256(p_hi_lo, _mm256_set1_epi64x(0xFFFFFFFFULL)),  // low32(p_hi_lo)
      _mm256_and_si256(p_lo_hi, _mm256_set1_epi64x(0xFFFFFFFFULL))   // low32(p_lo_hi)
    )
  );

  __m256i high64 = _mm256_add_epi64(
    _mm256_add_epi64(p_hi_hi, _mm256_srli_epi64(p_hi_lo, 32)),
    _mm256_add_epi64(_mm256_srli_epi64(p_lo_hi, 32), _mm256_srli_epi64(mid_sum, 32))
  );

  // Shift to get final quotient
  return _mm256_srli_epi64(high64, s);
}
