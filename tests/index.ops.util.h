#ifndef INDEX_OPS_UTIL_H
#define INDEX_OPS_UTIL_H

#include <stdint.h>
#include <time.h>

// Timing utilities
struct clock
{
  struct timespec last;
};

// Returns elapsed time in seconds since last toc()
float
toc(struct clock* clock);

// Scalar index computation
// For an array of dimension `rank` of size `shape`, a transpose is expressed
// via `strides`. Consider an "input" array that has a row-major layout. The
// "output" array also has a row-major layout but with dimensions in a different
// order.
//
// This function computes the output index corresponding to add input offsets
// a and b. That is if r(i) is coordinate of the i'th element of the input
// array, and T(r) is the transpose, and r'(j) is the j'th element of the
// output array, this function should return j such that r'(j)=T(r(a+b)).
uint64_t
add(int rank,
    const int* __restrict__ shape,
    const int* __restrict__ strides,
    uint64_t a,
    uint64_t b);

// Array printing utilities
void
print_vi32(int n, const int* v);

void
println_vi32(int n, const int* v);

void
print_vu64(int n, const uint64_t* v);

void
println_vu64(int n, const uint64_t* v);

// Array manipulation utilities
void
compute_strides(int rank, const int* shape, int* strides);

void
permute_i32(int n,
            const int* __restrict__ p,
            const int* __restrict__ in,
            int* __restrict__ out);

void
inverse_permutation_i32(int n, const int* __restrict__ p, int* __restrict__ inv);

// Test utilities
// Helper to create expected array using add()
uint64_t*
make_expected(int rank,
              const int* shape,
              const int* strides,
              uint64_t beg,
              uint64_t end);

// Helper to create expected array using add() with step
uint64_t*
make_expected_step(int rank,
                   const int* shape,
                   const int* strides,
                   uint64_t beg,
                   uint64_t end,
                   uint64_t step);

// Compare arrays and report first mismatch
// Returns 0 if arrays match, 1 if they differ
int
expect_arrays_equal(const uint64_t* expected,
                    const uint64_t* actual,
                    size_t n,
                    const char* test_name);

// Generate random test cases
uint64_t*
random_vu64(int count, uint64_t max);

#endif // INDEX_OPS_UTIL_H
