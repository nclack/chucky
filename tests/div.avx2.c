#include "div.avx2.h"

void
compute_magic_div(uint32_t d, uint64_t* m, int* s)
{
  // We want to find m and s such that:
  // floor(n/d) = floor((n * m) / 2^(64+s)) for all n in [0, 2^64)
  //
  // Compute ceil(2^(64+shift) / d) using iterative 32-bit long division.
  // Since d is uint32_t, each step is a 64÷32 operation in standard C.

  for (int shift = 0; shift < 32; shift++) {
    // Divide 2^(64+shift) by d.
    // Represent the numerator as (shift+64+1) bits: a 1 followed by
    // (64+shift) zero bits.  We process 32 bits at a time, high to low.
    //
    // Number of 32-bit "digits" covering bits [0, 64+shift):
    //   ndigits = ceil((64+shift) / 32) + 1   (the leading 1-bit is the
    //   most-significant digit).
    // Rather than dynamically sizing, we note 64+shift <= 95, so at most
    // 4 digits (indices 3..0, big-endian).

    // Build the numerator digits (big-endian, base 2^32).
    // 2^(64+shift) has bit (64+shift) set and nothing else.
    uint32_t digits[4] = { 0, 0, 0, 0 };
    {
      int bit = 64 + shift;          // which bit is set
      int digit_idx = 3 - bit / 32;  // big-endian index (digit 0 is MSB)
      digits[digit_idx] = (uint32_t)1 << (bit % 32);
    }

    // Long division: divide digits[] (base 2^32) by d, collect quotient
    // digits into a 128-bit result (only low 65 bits can be nonzero).
    uint64_t q_hi = 0; // bits [64..127] of quotient
    uint64_t q_lo = 0; // bits [0..63]  of quotient
    uint64_t rem = 0;
    for (int i = 0; i < 4; ++i) {
      uint64_t cur = (rem << 32) | digits[i];
      uint64_t qi  = cur / d;
      rem          = cur % d;
      // Accumulate qi into (q_hi:q_lo) at position (3-i)*32.
      int pos = (3 - i) * 32;
      if (pos >= 64)
        q_hi |= qi << (pos - 64);
      else if (pos == 0)
        q_lo |= qi;
      else
        q_lo |= qi << pos;
    }

    // Round up if remainder != 0  (we need ceil)
    if (rem != 0) {
      q_lo++;
      if (q_lo == 0)
        q_hi++;
    }

    // Check if quotient fits in 64 bits (q_hi must be 0 or exactly 1
    // with q_lo == 0, i.e. quotient <= 2^64).
    if (q_hi == 0 || (q_hi == 1 && q_lo == 0)) {
      *m = q_lo;
      *s = shift;
      return;
    }
  }

  // Fallback — should not reach here for reasonable divisors.
  // Compute ceil(2^64 / d) via: 2^64 / d = (2^64 - d) / d + 1.
  *m = (UINT64_MAX / d) + 1;
  *s = 0;
}

uint64_t
magic_div(uint64_t n, uint64_t m, int s)
{
  // Compute high 64 bits of (n * m) using the same 32-bit decomposition
  // that mm256_div_epu64_const uses for its AVX2 lanes.
  //
  //   n = n_hi:n_lo  (each 32 bits)
  //   m = m_hi:m_lo
  //   n*m = n_hi*m_hi*2^64
  //       + (n_hi*m_lo + n_lo*m_hi)*2^32
  //       + n_lo*m_lo
  //
  //   high64 = n_hi*m_hi
  //          + high32(n_hi*m_lo + n_lo*m_hi + high32(n_lo*m_lo))

  uint64_t n_lo = (uint32_t)n;
  uint64_t n_hi = n >> 32;
  uint64_t m_lo = (uint32_t)m;
  uint64_t m_hi = m >> 32;

  uint64_t p_lo_lo = n_lo * m_lo;
  uint64_t p_hi_lo = n_hi * m_lo;
  uint64_t p_lo_hi = n_lo * m_hi;
  uint64_t p_hi_hi = n_hi * m_hi;

  // Sum the three terms that contribute to bits [32..95] and propagate carry
  uint64_t mid = (p_lo_lo >> 32)
               + (uint32_t)p_hi_lo
               + (uint32_t)p_lo_hi;

  uint64_t hi = p_hi_hi
              + (p_hi_lo >> 32)
              + (p_lo_hi >> 32)
              + (mid >> 32);

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
