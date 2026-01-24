#include <bits/posix1_lim.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// # add
//
// For an array of dimension `rank` of size `shape`, a transpose is expressed
// via `strides`. Consider an "input" array that has a row-major layout. The
// "output" array also has a row-major layout but with dimensions in a different
// order.
//
// This function computes the output index corresponding to add input offsets
// a and b. That is if r(i) is coordinate of the i'th element of the input
// array, and T(r) is the transpose, and r'(j) is the j'th element of the
// output array, this function should return j such that r'(j)=T(r(a+b)).
static uint64_t
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

// # vadd
// Return the output offsets corresponding to input offsets beg..end
static uint64_t*
vadd(int rank,
     const int* __restrict__ shape,
     const int* __restrict__ strides,
     uint64_t beg,
     uint64_t end)
{
  // get the output offset corresponding to beg
  uint64_t o = add(rank, shape, strides, beg, 0);

  const size_t n = end - beg;
  uint64_t* out = (uint64_t*)malloc((end - beg) * sizeof(uint64_t));
  if (!out)
    return 0;

  // init out with deltas - later we'll do a scan
  for (int i = 0; i < n; ++i) {
    out[i] = strides[rank - 1];
  }

  // transposed carries
  {
    uint64_t rest = beg;
    int input_stride = 1; // Start with the stride for the last dimension
    int first_carry = (beg % shape[rank - 1]);
    first_carry = (shape[rank - 1] - first_carry);

    for (int d = rank - 1; d > 0; --d) {
      const int e = shape[d];
      const int r = rest % e;
      rest /= e;
      // Q: Over the next n elements, where will there be a carry from d to d-1?
      // A: Every input_stride*shape[d] elements starting at first_carry.
      const int next_input_stride = input_stride * shape[d];
      for (int i = first_carry; i < n; i += next_input_stride) {
        // correct for carry from d to d-1
        out[i] += strides[d - 1] - shape[d] * strides[d];
      }

      // Compute first_carry for the next dimension (d - 1)
      if (d > 1) {
        const int r_next = rest % shape[d - 1];
        first_carry =
          first_carry + next_input_stride * (shape[d - 1] - r_next - 1);
      }

      input_stride = next_input_stride;
    }
  }

  // prefix sum
  out[0] = o;
  for (int i = 1; i < n; ++i)
    out[i] += out[i - 1];
  return out;
}

static void
print_vi32(int n, int* v)
{
  putc('[', stdout);
  if (n)
    printf("%d", v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %d", v[i]);
  putc(']', stdout);
}

static void
println_vi32(int n, int* v)
{
  print_vi32(n, v);
  putc('\n', stdout);
}

static void
print_vu64(int n, const uint64_t* v)
{
  putc('[', stdout);
  if (n)
    printf("%d", (int)v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %d", (int)v[i]);
  putc(']', stdout);
}

static void
println_vu64(int n, const uint64_t* v)
{
  print_vu64(n, v);
  putc('\n', stdout);
}

static void
compute_strides(int rank, const int* shape, int* strides)
{
  strides[rank - 1] = 1;
  for (int d = rank - 1; d > 0; --d) {
    strides[d - 1] = shape[d] * strides[d];
  }
}

static void
permute_i32(int n,
            const int* __restrict__ p,
            const int* __restrict__ in,
            int* __restrict__ out)
{
  for (int i = 0; i < n; ++i) {
    out[i] = in[p[i]];
  }
}

// Helper to create expected array using add()
static uint64_t*
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

// Helper to compare arrays and report first mismatch
// Returns 0 if arrays match, 1 if they differ
static int
expect_arrays_equal(const uint64_t* expected,
                    const uint64_t* actual,
                    size_t n,
                    const char* test_name)
{
  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      printf("%s: Expected %d but got %d at element %zu\n",
             test_name,
             (int)expected[i],
             (int)actual[i],
             i);
      return 1;
    }
  }
  return 0;
}

static int
vadd_agrees_with_add()
{
  // Setup a tiling of a 640x480x3 array into tiles of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  int transposed_strides[6] = {};
  {
    int strides[6] = {};
    compute_strides(rank, shape, strides);
    println_vi32(rank, strides);
    {
      const int p[] = { 0, 2, 4, 1, 3, 5 };
      permute_i32(rank, p, strides, transposed_strides);
    }
    println_vi32(rank, transposed_strides);
  }

  int ecode = 0;
  const int n = transposed_strides[0] * shape[0];

  // Test case 1: beg = 0
  {
    const uint64_t* actual = vadd(rank, shape, transposed_strides, 0, n);
    const uint64_t* expected = make_expected(rank, shape, transposed_strides, 0, n);

    if (!actual || !expected) {
      free((void*)actual);
      free((void*)expected);
      return 1;
    }

    println_vu64(15, actual);
    ecode |= expect_arrays_equal(expected, actual, n, "beg=0");

    free((void*)actual);
    free((void*)expected);
  }

  // Test case 2: beg = 110
  {
    const uint64_t beg = 110;
    const uint64_t* actual = vadd(rank, shape, transposed_strides, beg, n);
    const uint64_t* expected = make_expected(rank, shape, transposed_strides, beg, n);

    if (!actual || !expected) {
      free((void*)actual);
      free((void*)expected);
      return 1;
    }

    println_vu64(15, actual);
    ecode |= expect_arrays_equal(expected, actual, n - beg, "beg=110");

    free((void*)actual);
    free((void*)expected);
  }

  return ecode;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;

  ecode |= vadd_agrees_with_add();

  return ecode;
}
