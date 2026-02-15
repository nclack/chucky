#include "div.avx2.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
  // Test various divisors and dividends
  uint32_t divisors[] = { 3, 10, 64, 100, 300, 1920, 19200 };
  int num_divisors = sizeof(divisors) / sizeof(divisors[0]);

  int errors = 0;

  for (int d_idx = 0; d_idx < num_divisors; ++d_idx) {
    uint32_t d = divisors[d_idx];
    uint64_t m;
    int s;
    compute_magic_div(d, &m, &s);

    printf("Divisor %u: m=%llu, s=%d\n", d, (unsigned long long)m, s);

    // Test a range of values
    for (uint64_t n = 0; n < 100000; n += 17) {
      uint64_t expected = n / d;
      uint64_t actual = magic_div(n, m, s);

      if (expected != actual) {
        printf("  ERROR: %llu / %u: expected %llu, got %llu\n", (unsigned long long)n, d, (unsigned long long)expected, (unsigned long long)actual);
        errors++;
        if (errors > 10) {
          printf("Too many errors, stopping\n");
          return 1;
        }
      }
    }

    printf("  OK\n");
  }

  if (errors == 0) {
    printf("\nAll tests passed!\n");
  } else {
    printf("\n%d errors found\n", errors);
  }

  return errors > 0 ? 1 : 0;
}
