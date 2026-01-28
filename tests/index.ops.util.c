#include "index.ops.util.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Returns elapsed time in seconds since last toc()
float
toc(struct clock* clock)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  float elapsed = (now.tv_sec - clock->last.tv_sec) +
                  (now.tv_nsec - clock->last.tv_nsec) / 1e9f;

  clock->last = now;
  return elapsed;
}

uint64_t
add(int rank,
    const int* __restrict__ shape,
    const int* __restrict__ strides,
    uint64_t a,
    uint64_t b)
{
  uint64_t o = 0;
  {
    uint64_t rest = a + b;
    for (int d = rank - 1; d >= 0; --d) {
      const int r = rest % shape[d];
      o += r * strides[d];
      rest /= shape[d];
    }
  }

  return o;
}

void
print_vi32(int n, const int* v)
{
  putc('[', stdout);
  if (n)
    printf("%d", v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %d", v[i]);
  putc(']', stdout);
}

void
println_vi32(int n, const int* v)
{
  print_vi32(n, v);
  putc('\n', stdout);
}

void
print_vu64(int n, const uint64_t* v)
{
  putc('[', stdout);
  if (n)
    printf("%lu", v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %lu", v[i]);
  putc(']', stdout);
}

void
println_vu64(int n, const uint64_t* v)
{
  print_vu64(n, v);
  putc('\n', stdout);
}

void
compute_strides(int rank, const int* shape, int* strides)
{
  strides[rank - 1] = 1;
  for (int d = rank - 1; d > 0; --d) {
    strides[d - 1] = shape[d] * strides[d];
  }
}

void
permute_i32(int n,
            const int* __restrict__ p,
            const int* __restrict__ in,
            int* __restrict__ out)
{
  for (int i = 0; i < n; ++i) {
    out[i] = in[p[i]];
  }
}

void
inverse_permutation_i32(int n, const int* __restrict__ p, int* __restrict__ inv)
{
  for (int i = 0; i < n; ++i) {
    inv[p[i]] = i;
  }
}

// Helper to create expected array using add()
uint64_t*
make_expected(int rank,
              const int* shape,
              const int* strides,
              uint64_t beg,
              uint64_t end)
{
  const size_t n = end - beg;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  for (uint64_t i = 0; i < n; ++i) {
    out[i] = add(rank, shape, strides, beg, i);
  }
  return out;
}

// Helper to create expected array using add() with step
uint64_t*
make_expected_step(int rank,
                   const int* shape,
                   const int* strides,
                   uint64_t beg,
                   uint64_t end,
                   uint64_t step)
{
  const size_t n = (end - beg + step - 1) / step;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  for (uint64_t i = 0; i < n; ++i) {
    out[i] = add(rank, shape, strides, beg, i * step);
  }
  return out;
}

// Compare arrays and report first mismatch
// Returns 0 if arrays match, 1 if they differ
int
expect_arrays_equal(const uint64_t* expected,
                    const uint64_t* actual,
                    size_t n,
                    const char* test_name)
{
  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      printf("%s: Expected %lu but got %lu at element %zu\n",
             test_name,
             expected[i],
             actual[i],
             i);
      return 1;
    }
  }
  return 0;
}

uint64_t*
random_vu64(int count, uint64_t max)
{
  uint64_t* cases = (uint64_t*)malloc(count * sizeof(uint64_t));
  if (!cases)
    return 0;
  for (int i = 0; i < count; ++i) {
    cases[i] = (uint64_t)rand() % max;
  }
  return cases;
}
