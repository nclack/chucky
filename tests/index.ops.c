#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

struct clock
{
  struct timespec last;
};

// Returns elapsed time in seconds since last toc()
static float
toc(struct clock* clock)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  float elapsed = (now.tv_sec - clock->last.tv_sec) +
                  (now.tv_nsec - clock->last.tv_nsec) / 1e9f;

  clock->last = now;
  return elapsed;
}

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
  for (size_t i = 0; i < n; ++i) {
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
      rest /= e;
      // Q: Over the next n elements, where will there be a carry from d to d-1?
      // A: Every input_stride*shape[d] elements starting at first_carry.
      const int next_input_stride = input_stride * shape[d];
      // correct for carry from d to d-1
      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];
      for (size_t i = first_carry; i < n; i += next_input_stride) {
        out[i] += correction;
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
  {
    out[0] = o;
    for (size_t i = 1; i < n; ++i)
      out[i] += out[i - 1];
  }
  return out;
}

// # vadd2
// Return the output offsets corresponding to input offsets beg..end
// stepping by step.
//
// The strides array (indexed by input dimensions) tells us the output offset
// change for a +1 move in each input dimension.
//
// e.g. For a transpose from 5x7x3->3x5x7, strides should be `{7,1,35}`.
static uint64_t*
vadd2(int rank,
      const int* __restrict__ shape,
      const int* __restrict__ strides,
      uint64_t beg,
      uint64_t end,
      uint64_t step)
{
  // get the output offset corresponding to beg
  uint64_t o = add(rank, shape, strides, beg, 0);

  const size_t n = (end - beg + step - 1) / step;
  uint64_t* out = (uint64_t*)malloc(n * sizeof(uint64_t));
  if (!out)
    return 0;

  {
    // delta: output offset corresponding to a `step` in the input index space
    const uint64_t delta = add(rank, shape, strides, 0, step);

    // init out with delta - later we'll apply carry corrections
    for (size_t i = 0; i < n; ++i) {
      out[i] = delta;
    }
  }

  // transposed carries
  {
    // Allocate carry stencil (tracks carries from previous dimension iteration)
    int* carry = (int*)calloc(n, sizeof(int));
    if (!carry) {
      free(carry);
      free(out);
      return 0;
    }

    // Initialize carry stencil for dimensions with extent 1
    // When shape[rank-1] == 1, we can't move within that dimension,
    // so every step produces a carry
    if (shape[rank - 1] == 1 && step > 0) {
      for (size_t i = 1; i < n; ++i) {
        carry[i] = 1;
      }
    }

    uint64_t rest_beg = beg;
    uint64_t rest_step = step;

    for (int d = rank - 1; d > 0; --d) {
      const uint64_t e = rest_beg % shape[d];
      const uint64_t s = rest_step % shape[d];
      rest_beg /= shape[d];
      rest_step /= shape[d];

      const uint64_t correction = strides[d - 1] - shape[d] * strides[d];

      for (size_t i = 1; i < n; ++i) {
        // Check for natural carry
        const uint64_t quot_prev = (e + (i - 1) * s) / shape[d];
        const uint64_t quot_curr = (e + i * s) / shape[d];
        const int is_natural_carry = (quot_prev < quot_curr);

        // Check if coordinate at i-1 is at boundary (would carry if
        // incremented)
        const uint64_t coord_prev = (e + (i - 1) * s) % shape[d];
        const int almost_carry = (coord_prev + 1 == (uint64_t)shape[d]);

        // Update carry stencil: natural carry OR (almost-carry AND carry from
        // d+1)
        const int has_carry = is_natural_carry || (almost_carry && carry[i]);
        carry[i] = has_carry;

        if (has_carry) {
          out[i] += correction;
        }

        if (i == 1920) {
#define XXX(e) printf("%-20s: %d\n", #e, (int)e)
          printf("--- %d\n", (int)i);
          XXX(d);
          XXX(e);
          XXX(s);
          XXX(is_natural_carry);
          XXX(almost_carry);
          XXX(carry[i]);
          XXX(has_carry);
          XXX(shape[d]);
          XXX(correction);
#undef XXX
        }
      }
    }

    free(carry);
  }

  // prefix sum
  {
    out[0] = o;
    for (size_t i = 1; i < n; ++i)
      out[i] += out[i - 1];
  }
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
println_vi32(int n, const int* v)
{
  print_vi32(n, v);
  putc('\n', stdout);
}

static void
print_vu64(int n, const uint64_t* v)
{
  putc('[', stdout);
  if (n)
    printf("%lu", v[0]);
  for (int i = 1; i < n; ++i)
    printf(", %lu", v[i]);
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

// Helper to create expected array using add() with step
static uint64_t*
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

static uint64_t*
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

static int
vadd_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into tiles of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  int transposed_strides[6] = { 0 };
  {
    int transposed_shape[6] = { 0 };
    {
      const int p[] = { 0, 2, 4, 1, 3, 5 };
      permute_i32(rank, p, shape, transposed_shape);
    }
    compute_strides(rank, transposed_shape, transposed_strides);
    printf("%30s", "Input shape: ");
    println_vi32(rank, shape);
    {
      printf("%30s", "Input strides (row-major): ");
      int strides[6] = { 0 };
      compute_strides(rank, shape, strides);
      println_vi32(rank, strides);
    }
    printf("%30s", "Transposed shape: ");
    println_vi32(rank, transposed_shape);
    printf("%30s", "Transposed strides: ");
    println_vi32(rank, transposed_strides);
  }

  const int n = transposed_strides[0] * shape[0];
  const int num_tests = 10000;

  const uint64_t* expected_all =
    make_expected(rank, shape, transposed_strides, 0, n);
  if (!expected_all)
    return 1;

  uint64_t* test_cases = random_vu64(num_tests, n);
  if (!test_cases) {
    free((void*)expected_all);
    return 1;
  }

  struct
  {
    _Atomic int completed;
    _Atomic int ecode;
  } state = { 0 };

#pragma omp parallel
  {
#pragma omp master
    {
      struct clock clk = { 0 };
      toc(&clk);
      int last_completed = 0;

      while (state.completed < num_tests) {
        float dt = toc(&clk);
        int delta = state.completed - last_completed;
        float velocity = dt > 0 ? delta / dt : 0;
        last_completed = state.completed;

        printf("\rProgress: %d/%d (%.1f%%) - %.0f/s",
               state.completed,
               num_tests,
               100.0 * state.completed / num_tests,
               velocity);
        fflush(stdout);
        usleep(500000);
      }
      printf("\n");
    }

#pragma omp for schedule(guided)
    for (int i = 0; i < num_tests; ++i) {
      if (state.ecode)
        continue;

      const uint64_t beg = test_cases[i];
      const uint64_t* actual = vadd(rank, shape, transposed_strides, beg, n);
      if (!actual) {
        state.ecode = 1;
        continue;
      }

      const uint64_t* expected = expected_all + beg;

      {
        char buf[64] = { 0 };
        snprintf(buf, sizeof(buf), "beg=%lu", beg);
        int err = expect_arrays_equal(expected, actual, n - beg, buf);
        if (err) {
          state.ecode |= err;
        }
      }

      free((void*)actual);

      atomic_fetch_add_explicit(&state.completed, 1, memory_order_relaxed);
    }
  }

  free(test_cases);
  free((void*)expected_all);
  return state.ecode;
}

static int
vadd2_agrees_with_add(void)
{
  // Setup a tiling of a 640x480x3 array into tiles of size 10x10x1
  const int rank = 6;
  const int shape[] = { 48, 10, 64, 10, 3, 1 };
  int transposed_strides[6] = { 0 };
  {
    int transposed_shape[6] = { 0 };
    {
      const int p[] = { 0, 2, 4, 1, 3, 5 };
      permute_i32(rank, p, shape, transposed_shape);
    }
    compute_strides(rank, transposed_shape, transposed_strides);
    printf("%30s", "Input shape: ");
    println_vi32(rank, shape);
    {
      printf("%30s", "Input strides (row-major): ");
      int strides[6] = { 0 };
      compute_strides(rank, shape, strides);
      println_vi32(rank, strides);
    }
    printf("%30s", "Transposed shape: ");
    println_vi32(rank, transposed_shape);
    printf("%30s", "Transposed strides: ");
    println_vi32(rank, transposed_strides);
  }

  const int n = transposed_strides[0] * shape[0];
  const int num_tests = 1;
  const uint64_t steps[] = { 1 }; //, 2, 3, 5, 7, 11, 32, 100, 1000 };
  const int num_steps = sizeof(steps) / sizeof(steps[0]);

  struct
  {
    _Atomic int completed;
    _Atomic int ecode;
  } state = { 0 };

  // #pragma omp parallel
  {
#if 0
// #pragma omp master
    {
      struct clock clk = { 0 };
      toc(&clk);
      int last_completed = 0;
      const int total_tests = num_tests * num_steps;

      while (state.completed < total_tests) {
        float dt = toc(&clk);
        int delta = state.completed - last_completed;
        float velocity = dt > 0 ? delta / dt : 0;
        last_completed = state.completed;

        printf("\rProgress: %d/%d (%.1f%%) - %.0f/s",
               state.completed,
               total_tests,
               100.0 * state.completed / total_tests,
               velocity);
        fflush(stdout);
        usleep(500000);
      }
      printf("\n");
    }
#endif

    // #pragma omp for schedule(guided) collapse(2)
    for (int step_idx = 0; step_idx < num_steps; ++step_idx) {
      for (int i = 0; i < num_tests; ++i) {
        if (state.ecode)
          continue;

        const uint64_t step = steps[step_idx];
        const uint64_t beg = 0; //(uint64_t)rand() % n;
        const uint64_t* expected =
          make_expected_step(rank, shape, transposed_strides, beg, n, step);
        if (!expected) {
          state.ecode = 1;
          continue;
        }

        const uint64_t* actual =
          vadd2(rank, shape, transposed_strides, beg, n, step);
        if (!actual) {
          free((void*)expected);
          state.ecode = 1;
          continue;
        }

        const size_t count = (n - beg + step - 1) / step;
        {
          char buf[64] = { 0 };
          snprintf(buf, sizeof(buf), "beg=%lu, step=%lu", beg, step);
          int err = expect_arrays_equal(expected, actual, count, buf);
          if (err) {
            printf("Expected (first 10): ");
            println_vu64(count < 10 ? count : 10, expected);
            printf("Actual (first 10): ");
            println_vu64(count < 10 ? count : 10, actual);
            state.ecode |= err;
          }
        }

        free((void*)expected);
        free((void*)actual);

        atomic_fetch_add_explicit(&state.completed, 1, memory_order_relaxed);
      }
    }
  }

  return state.ecode;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  srand(42);

  int ecode = 0;

  // ecode |= vadd_agrees_with_add();
  ecode |= vadd2_agrees_with_add();

  return ecode;
}
