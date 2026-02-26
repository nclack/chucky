#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM 8

static uint32_t
max_shape(int ndim, const uint32_t* shape)
{
  uint32_t m = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > m)
      m = shape[d];
  return m;
}

// Smallest p such that 2^p >= v. Returns 0 for v <= 1.
static int
ceil_log2(uint32_t v)
{
  int p = 0;
  while ((1u << p) < v)
    ++p;
  return p;
}

// Number of d-bit digits needed to represent k.
static int
morton_digits(int ndim, uint64_t k)
{
  int n = 0;
  while (k > 0) {
    k >>= ndim;
    ++n;
  }
  return n;
}

// Extract the d-bit digit at the given level from a Morton code.
// Level 0 is the most significant digit.
static int
morton_digit(int ndim, uint64_t k, int p, int level)
{
  return (int)((k >> (ndim * (p - 1 - level))) & ((1u << ndim) - 1));
}

// Clamped extent: how many valid coordinates in [lo, lo+scale) for a
// dimension of size `shape_d`.
static uint64_t
clamped_extent(uint32_t shape_d, uint32_t lo, uint32_t scale)
{
  if (lo >= shape_d)
    return 0;
  uint32_t e = shape_d - lo;
  return (e < scale) ? e : scale;
}

// Decode a Morton code back into coordinates.
static void
morton_decode(int ndim, uint64_t code, uint32_t* coords)
{
  memset(coords, 0, (size_t)ndim * sizeof(*coords));
  for (int bit = 0; bit < 32; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      coords[d] |= (uint32_t)(code & 1) << bit;
      code >>= 1;
    }
  }
}

// Count how many Morton codes in [0, k) decode to coordinates within the array
// bounds shape[0..ndim-1].
//
// Walks the Morton code top-down (MSB to LSB), one d-bit digit per level.
// At each level, sibling subtrees before k's digit contribute a known count.
// Instead of enumerating all v < digit (2^d siblings), we decompose the digit
// bit by bit across dimensions. For each dimension where the digit's bit is 1,
// we split: that dimension takes 0 (all lower dimensions free) or takes 1
// (continue matching). This is O(d) per level, or O(p*d) overall where p*d
// is the number of bits used in the morton code.
static uint64_t
morton_rank(int ndim, const uint32_t* shape, uint64_t k)
{
  int p = ceil_log2(max_shape(ndim, shape));
  int pk = morton_digits(ndim, k);
  if (pk > p)
    p = pk;

  uint64_t count = 0;
  uint32_t prefix[MAX_NDIM] = { 0 };

  for (int level = 0; level < p; ++level) {
    uint32_t scale = 1u << (p - 1 - level);
    int digit = morton_digit(ndim, k, p, level);

    // For each dimension, precompute clamped extents for bit=0 and bit=1.
    uint64_t ext[MAX_NDIM][2];
    for (int d = 0; d < ndim; ++d) {
      for (int b = 0; b < 2; ++b) {
        uint32_t lo = (prefix[d] * 2 + (uint32_t)b) * scale;
        ext[d][b] = clamped_extent(shape[d], lo, scale);
      }
    }

    // Prefix product of "free" extents (both bit values) for dims 0..d-1.
    uint64_t free_prefix[MAX_NDIM + 1];
    free_prefix[0] = 1;
    for (int d = 0; d < ndim; ++d)
      free_prefix[d + 1] = free_prefix[d] * (ext[d][0] + ext[d][1]);

    // Scan dimensions from highest to lowest. For each dimension where
    // the digit's bit is 1, split: this dim takes 0 (lower dims free),
    // or this dim takes 1 (continue matching tight prefix above).
    uint64_t tight = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      int bit = (digit >> d) & 1;
      if (bit == 1)
        count += tight * ext[d][0] * free_prefix[d];
      tight *= ext[d][bit];
    }

    // Descend: extend prefix with this digit's per-dimension bits
    for (int d = 0; d < ndim; ++d)
      prefix[d] = prefix[d] * 2 + ((digit >> d) & 1);
  }

  return count;
}

static uint64_t
morton_encode(int ndim, const uint32_t* coords)
{
  uint64_t code = 0;
  for (int bit = 31; bit >= 0; --bit) {
    for (int d = ndim - 1; d >= 0; --d) {
      code <<= 1;
      code |= (coords[d] >> bit) & 1;
    }
  }
  return code;
}

static int
is_all_ones(int n, const uint32_t* v)
{
  for (int d = 0; d < n; ++d)
    if (v[d] > 1)
      return 0;
  return 1;
}

// Row-major linear index to coordinates.
static void
linear_to_coords(int ndim,
                 const uint32_t* shape,
                 uint64_t idx,
                 uint32_t* coords)
{
  for (int d = 0; d < ndim; ++d) {
    coords[d] = (uint32_t)(idx % shape[d]);
    idx /= shape[d];
  }
}

// Scatter array (row-major) into compacted Morton order.
// buf must have prod(shape) elements.
static void
lod_scatter(int ndim, const uint32_t* shape, const float* src, float* dst)
{
  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  uint32_t coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(ndim, shape, i, coords);
    uint64_t m = morton_encode(ndim, coords);
    uint64_t pos = morton_rank(ndim, shape, m);
    dst[pos] = src[i];
  }
}

#define MAX_LOD 32

// Compute the shape pyramid and per-level element counts.
// shapes[k] = ceil(shape / 2^k). counts[k] = prod(shapes[k+1]).
// Returns number of LOD levels (including level 0).
static int
lod_plan(int ndim,
         const uint32_t* shape,
         uint32_t shapes[][MAX_NDIM],
         uint64_t* counts,
         int max_levels)
{
  memcpy(shapes[0], shape, (size_t)ndim * sizeof(uint32_t));
  int nlev = 1;
  while (nlev < max_levels && !is_all_ones(ndim, shapes[nlev - 1])) {
    for (int d = 0; d < ndim; ++d)
      shapes[nlev][d] = (shapes[nlev - 1][d] + 1) / 2;
    ++nlev;
  }
  for (int k = 0; k < nlev - 1; ++k) {
    uint64_t c = 1;
    for (int d = 0; d < ndim; ++d)
      c *= shapes[k + 1][d];
    counts[k] = c;
  }
  return nlev;
}

// Fill one level's segment-end array. Embarrassingly parallel: each iteration
// is independent (one iteration = one GPU thread).
//
// ends[pos] receives the exclusive end of parent pos's children in the
// compacted Morton buffer for child_shape.
static void
lod_fill_ends(int ndim,
              const uint32_t* child_shape,
              const uint32_t* parent_shape,
              uint64_t n_parents,
              uint64_t* ends)
{
  uint32_t coords[MAX_NDIM];
  for (uint64_t j = 0; j < n_parents; ++j) {
    linear_to_coords(ndim, parent_shape, j, coords);
    uint64_t m = morton_encode(ndim, coords);
    uint64_t pos = morton_rank(ndim, parent_shape, m);
    uint64_t val = morton_rank(ndim, child_shape, (m + 1) << ndim);
    ends[pos] = val;
  }
}

// Segmented mean reduction using precomputed segment ends.
// ends[i] is the exclusive end of segment i (start of segment 0 is 0).
// Embarrassingly parallel: each iteration is independent.
static void
lod_reduce(uint64_t n_next, const uint64_t* ends, const float* src, float* dst)
{
  for (uint64_t i = 0; i < n_next; ++i) {
    uint64_t start = i > 0 ? ends[i - 1] : 0;
    uint64_t end = ends[i];
    uint64_t len = end - start;
    float sum = 0;
    for (uint64_t j = start; j < end; ++j)
      sum += src[j];
    dst[i] = sum / (float)len;
  }
}

// Compute all LOD levels. Returns number of levels (including level 0).
// levels[0] = original data in compacted Morton order
// levels[l] = reduced data for shape ceil(original_shape / 2^l)
// Returns 0 on failure. Caller must free each levels[l].
static int
lod_compute(int ndim,
            const uint32_t* shape,
            const float* src,
            float** levels,
            int max_levels)
{
  int ok = 0;
  uint64_t* ends = NULL;

  uint32_t shapes[MAX_LOD][MAX_NDIM];
  uint64_t counts[MAX_LOD] = { 0 };
  int nlev = lod_plan(ndim, shape, shapes, counts, max_levels);

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  // Level 0: scatter into Morton order
  levels[0] = (float*)malloc(n * sizeof(float));
  CHECK(Error, levels[0]);
  lod_scatter(ndim, shape, src, levels[0]);

  // Reduce each level
  for (int l = 0; l < nlev - 1; ++l) {
    ends = (uint64_t*)malloc(counts[l] * sizeof(uint64_t));
    CHECK(Error, ends);
    lod_fill_ends(ndim, shapes[l], shapes[l + 1], counts[l], ends);

    levels[l + 1] = (float*)malloc(counts[l] * sizeof(float));
    CHECK(Error, levels[l + 1]);
    lod_reduce(counts[l], ends, levels[l], levels[l + 1]);

    free(ends);
    ends = NULL;
  }

  ok = 1;
Error:
  free(ends);
  return ok ? nlev : 0;
}

// Reference: brute-force downsample by averaging only valid children in each
// 2^d block. This matches lod_reduce's semantics: boundary parents have fewer
// than 2^d valid children and the mean is taken over the valid ones only.
// (This is equivalent to replicate-BC averaging because replicated values are
// copies of boundary elements, so the weighted and unweighted means coincide.)
static void
downsample_ref(int ndim,
               const uint32_t* cur_shape,
               const uint32_t* next_shape,
               const float* src,
               float* dst)
{
  uint64_t n_next = 1;
  for (int d = 0; d < ndim; ++d)
    n_next *= next_shape[d];

  uint32_t coords[MAX_NDIM];
  for (uint64_t j = 0; j < n_next; ++j) {
    linear_to_coords(ndim, next_shape, j, coords);

    float sum = 0;
    int count = 0;
    int n_children = 1 << ndim;
    for (int c = 0; c < n_children; ++c) {
      uint64_t lin = 0;
      uint64_t stride = 1;
      int valid = 1;
      for (int d = 0; d < ndim; ++d) {
        uint32_t child_coord = coords[d] * 2 + ((c >> d) & 1);
        if (child_coord >= cur_shape[d]) {
          valid = 0;
          break;
        }
        lin += child_coord * stride;
        stride *= cur_shape[d];
      }
      if (valid) {
        sum += src[lin];
        ++count;
      }
    }
    dst[j] = sum / (float)count;
  }
}

// Unshuffle: convert from compacted Morton order back to row-major.
static void
morton_unshuffle(int ndim,
                 const uint32_t* shape,
                 const float* morton_buf,
                 float* rowmajor)
{
  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  uint32_t coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(ndim, shape, i, coords);
    uint64_t m = morton_encode(ndim, coords);
    uint64_t pos = morton_rank(ndim, shape, m);
    rowmajor[i] = morton_buf[pos];
  }
}

// brute-force
static uint64_t
morton_rank_ref(int ndim, const uint32_t* shape, uint64_t k)
{
  uint64_t count = 0;
  uint32_t coords[MAX_NDIM];
  for (uint64_t m = 0; m < k; ++m) {
    morton_decode(ndim, m, coords);
    int valid = 1;
    for (int d = 0; d < ndim; ++d) {
      if (coords[d] >= shape[d]) {
        valid = 0;
        break;
      }
    }
    count += valid;
  }
  return count;
}

static int
test_3d(void)
{
  printf("--- test_3d ---\n");
  int ok = 0;
  const int ndim = 3;
  const uint32_t shape[] = { 3, 2, 5 };

  int p = ceil_log2(max_shape(ndim, shape));
  uint64_t box = 1ull << (ndim * p);
  for (uint64_t k = 0; k <= box; ++k) {
    uint64_t r = morton_rank(ndim, shape, k);
    uint64_t r_ref = morton_rank_ref(ndim, shape, k);
    if (r != r_ref) {
      printf("  FAIL at k=%llu: got %llu, expected %llu\n",
             (unsigned long long)k,
             (unsigned long long)r,
             (unsigned long long)r_ref);
      goto Fail;
    }
  }
  uint64_t total = morton_rank(ndim, shape, box);
  printf("  total valid in 3x2x5 = %llu\n", (unsigned long long)total);
  CHECK(Fail, total == 30);
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_1d(void)
{
  printf("--- test_1d ---\n");
  int ok = 0;
  const int ndim = 1;
  const uint32_t shape[] = { 7 };
  for (uint64_t k = 0; k <= 8; ++k) {
    uint64_t r = morton_rank(ndim, shape, k);
    uint64_t expected = k < 7 ? k : 7;
    CHECK(Fail, r == expected);
  }
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_lod(const char* label, int ndim, const uint32_t* shape)
{
  printf("--- %s ---\n", label);
  int ok = 0;
  const int max_lev = 16;
  float* src = NULL;
  float* levels[16] = { 0 };
  float* prev_rm = NULL;
  float* ref = NULL;
  float* cur_rm = NULL;

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  int nlev = lod_compute(ndim, shape, src, levels, max_lev);
  printf("  levels: %d\n", nlev);
  CHECK(Fail, nlev >= 2);

  // Verify level 0 scatter roundtrips
  prev_rm = (float*)malloc(n * sizeof(float));
  CHECK(Fail, prev_rm);
  morton_unshuffle(ndim, shape, levels[0], prev_rm);
  for (uint64_t i = 0; i < n; ++i) {
    if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
      printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
             (unsigned long long)i, prev_rm[i], src[i]);
      goto Fail;
    }
  }
  printf("  level 0 scatter: ok\n");

  // Verify each subsequent level against brute-force downsample
  {
    uint32_t prev_shape[MAX_NDIM];
    memcpy(prev_shape, shape, (size_t)ndim * sizeof(*shape));

    for (int l = 1; l < nlev; ++l) {
      uint32_t cur_shape[MAX_NDIM];
      for (int d = 0; d < ndim; ++d)
        cur_shape[d] = (prev_shape[d] + 1) / 2;

      uint64_t cur_n = 1;
      for (int d = 0; d < ndim; ++d)
        cur_n *= cur_shape[d];

      ref = (float*)malloc(cur_n * sizeof(float));
      CHECK(Fail, ref);
      downsample_ref(ndim, prev_shape, cur_shape, prev_rm, ref);

      cur_rm = (float*)malloc(cur_n * sizeof(float));
      CHECK(Fail, cur_rm);
      morton_unshuffle(ndim, cur_shape, levels[l], cur_rm);

      for (uint64_t i = 0; i < cur_n; ++i) {
        if (fabsf(cur_rm[i] - ref[i]) > 1e-5f) {
          uint32_t coords[MAX_NDIM];
          linear_to_coords(ndim, cur_shape, i, coords);
          printf("  FAIL level %d at (", l);
          for (int d = 0; d < ndim; ++d)
            printf("%s%u", d ? "," : "", coords[d]);
          printf("): got %f, expected %f\n", cur_rm[i], ref[i]);
          goto Fail;
        }
      }
      printf("  level %d: ok\n", l);

      free(ref);
      ref = NULL;
      free(prev_rm);
      prev_rm = cur_rm;
      cur_rm = NULL;
      memcpy(prev_shape, cur_shape, (size_t)ndim * sizeof(*cur_shape));
    }
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(prev_rm);
  free(ref);
  free(cur_rm);
  for (int l = 0; l < max_lev; ++l)
    free(levels[l]);
  return ok;
}

int
main(void)
{
  int nfail = 0;
  nfail += !test_3d();
  nfail += !test_1d();
  nfail += !test_lod("test_lod_2d", 2, (uint32_t[]){ 3, 5 });
  nfail += !test_lod("test_lod_3d", 3, (uint32_t[]){ 3, 2, 5 });
  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);
  return nfail ? 1 : 0;
}
