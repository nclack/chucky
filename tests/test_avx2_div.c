#include "div.avx2.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int
main(void)
{
  uint32_t divisor = 10;
  uint64_t m;
  int s;
  compute_magic_div(divisor, &m, &s);

  printf("Divisor %u: m=%llu, s=%d\n", divisor, (unsigned long long)m, s);

  // Test with values: 40, 41, 42, 43
  __m256i test_values = _mm256_set_epi64x(43, 42, 41, 40);
  __m256i result = mm256_div_epu64_const(test_values, m, s);

  uint64_t results[4];
  _mm256_storeu_si256((__m256i*)results, result);

  printf("Input: [40, 41, 42, 43]\n");
  printf("Output: [%llu, %llu, %llu, %llu]\n", (unsigned long long)results[0], (unsigned long long)results[1], (unsigned long long)results[2], (unsigned long long)results[3]);
  printf("Expected: [4, 4, 4, 4]\n");

  int errors = 0;
  uint64_t expected[] = {4, 4, 4, 4};
  for (int i = 0; i < 4; ++i) {
    if (results[i] != expected[i]) {
      printf("ERROR at index %d: expected %llu, got %llu\n", i, (unsigned long long)expected[i], (unsigned long long)results[i]);
      errors++;
    }
  }

  return errors > 0 ? 1 : 0;
}
