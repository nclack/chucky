// morton.util.c — shared CPU reference for LOD via compacted Morton codes.
// Intended to be #include'd (no main).

#include "lod_plan.h"
#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM LOD_MAX_NDIM
#define MAX_LOD  LOD_MAX_LEVELS

static uint64_t
max_shape(int ndim, const uint64_t* shape)
{
  uint64_t m = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > m)
      m = shape[d];
  return m;
}

static int
ceil_log2(uint64_t v)
{
  int p = 0;
  while ((1ull << p) < v)
    ++p;
  return p;
}

static uint64_t
clamped_extent(uint64_t shape_d, uint64_t lo, uint64_t scale)
{
  if (lo >= shape_d)
    return 0;
  uint64_t e = shape_d - lo;
  return (e < scale) ? e : scale;
}

static void
linear_to_coords(int ndim,
                 const uint64_t* shape,
                 uint64_t idx,
                 uint64_t* coords)
{
  for (int d = 0; d < ndim; ++d) {
    coords[d] = idx % shape[d];
    idx /= shape[d];
  }
}

static uint64_t
morton_rank(int ndim,
            const uint64_t* shape,
            const uint64_t* coords,
            int depth)
{
  int p = ceil_log2(max_shape(ndim, shape));

  for (int d = 0; d < ndim; ++d) {
    int pc = coords[d] > 0 ? ceil_log2(coords[d] + 1) : 0;
    if (pc > p)
      p = pc;
  }

  int total_levels = p + depth;

  uint64_t count = 0;
  uint64_t prefix[MAX_NDIM] = { 0 };

  for (int level = 0; level < total_levels; ++level) {
    uint64_t scale = 1ull << (total_levels - 1 - level);

    int digit = 0;
    if (level < p) {
      int bit_idx = p - 1 - level;
      for (int d = 0; d < ndim; ++d)
        digit |= (int)((coords[d] >> bit_idx) & 1) << d;
    }

    uint64_t ext[MAX_NDIM][2];
    for (int d = 0; d < ndim; ++d) {
      for (int b = 0; b < 2; ++b) {
        uint64_t lo = (prefix[d] * 2 + (uint64_t)b) * scale;
        ext[d][b] = clamped_extent(shape[d], lo, scale);
      }
    }

    uint64_t free_prefix[MAX_NDIM + 1];
    free_prefix[0] = 1;
    for (int d = 0; d < ndim; ++d)
      free_prefix[d + 1] = free_prefix[d] * (ext[d][0] + ext[d][1]);

    uint64_t tight = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      int bit = (digit >> d) & 1;
      if (bit == 1)
        count += tight * ext[d][0] * free_prefix[d];
      tight *= ext[d][bit];
    }

    for (int d = 0; d < ndim; ++d)
      prefix[d] = prefix[d] * 2 + ((digit >> d) & 1);
  }

  return count;
}

static void
coords_morton_next(int ndim, int p, uint64_t* coords)
{
  for (int bit = 0; bit < p; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      uint64_t mask = 1ull << bit;
      coords[d] ^= mask;
      if (coords[d] & mask)
        return;
    }
  }
  memset(coords, 0, (size_t)ndim * sizeof(uint64_t));
  coords[0] = 1ull << p;
}

static void
morton_decode(int ndim, uint64_t code, uint64_t* coords)
{
  memset(coords, 0, (size_t)ndim * sizeof(*coords));
  for (int bit = 0; bit < 64 / ndim; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      coords[d] |= (code & 1) << bit;
      code >>= 1;
    }
  }
}

static uint64_t
morton_rank_ref(int ndim, const uint64_t* shape, uint64_t k)
{
  uint64_t count = 0;
  uint64_t coords[MAX_NDIM];
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

// --- CPU reference functions using lod_plan ---

static uint64_t
plan_batch_index(const struct lod_plan* p, const uint64_t* full_coords)
{
  uint64_t idx = 0, stride = 1;
  for (int k = 0; k < p->batch_ndim; ++k) {
    idx += full_coords[p->batch_map[k]] * stride;
    stride *= p->batch_shape[k];
  }
  return idx;
}

static void
plan_extract_lod(const struct lod_plan* p,
                 const uint64_t* full_coords,
                 uint64_t* lod_coords)
{
  for (int k = 0; k < p->lod_ndim; ++k)
    lod_coords[k] = full_coords[p->lod_map[k]];
}

static void
lod_scatter_cpu(const struct lod_plan* p, const float* src, float* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = lod_span_len(lod_spans_at(&p->levels, 0));

  uint64_t full_coords[MAX_NDIM];
  uint64_t lod_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_lod(p, full_coords, lod_coords);
    uint64_t pos = morton_rank(p->lod_ndim, p->lod_shapes[0], lod_coords, 0);
    dst[b * p->lod_counts[0] + pos] = src[i];
  }
}

static void
lod_reduce_cpu(const struct lod_plan* p, float* values)
{
  for (int l = 0; l < p->nlev - 1; ++l) {
    struct lod_span seg = lod_segment(p, l);
    uint64_t src_ds = p->lod_counts[l];
    uint64_t dst_ds = p->lod_counts[l + 1];
    struct lod_span src_level = lod_spans_at(&p->levels, l);
    struct lod_span dst_level = lod_spans_at(&p->levels, l + 1);

    for (uint64_t b = 0; b < p->batch_count; ++b) {
      uint64_t src_base = src_level.beg + b * src_ds;
      uint64_t dst_base = dst_level.beg + b * dst_ds;

      for (uint64_t i = 0; i < dst_ds; ++i) {
        uint64_t start = (i > 0) ? p->ends[seg.beg + i - 1] : 0;
        uint64_t end = p->ends[seg.beg + i];
        uint64_t len = end - start;
        float sum = 0;
        for (uint64_t j = start; j < end; ++j)
          sum += values[src_base + j];
        values[dst_base + i] = sum / (float)len;
      }
    }
  }
}

static int
lod_compute(const struct lod_plan* p, const float* src, float** out_values)
{
  int ok = 0;
  *out_values = NULL;

  uint64_t total_vals = p->levels.ends[p->nlev - 1];
  float* values = (float*)malloc(total_vals * sizeof(float));
  CHECK(Error, values);
  *out_values = values;

  lod_scatter_cpu(p, src, values);
  lod_reduce_cpu(p, values);

  ok = 1;
Error:
  if (!ok) {
    free(*out_values);
    *out_values = NULL;
  }
  return ok;
}

static void
downsample_ref(int ndim,
               uint8_t ds_mask,
               const uint64_t* cur_shape,
               const uint64_t* next_shape,
               const float* src,
               float* dst)
{
  uint64_t n_next = 1;
  for (int d = 0; d < ndim; ++d)
    n_next *= next_shape[d];

  int n_ds = 0;
  for (int d = 0; d < ndim; ++d)
    if (ds_mask & (1 << d))
      n_ds++;
  int n_children = 1 << n_ds;

  uint64_t coords[MAX_NDIM];
  for (uint64_t j = 0; j < n_next; ++j) {
    linear_to_coords(ndim, next_shape, j, coords);

    float sum = 0;
    int count = 0;
    for (int c = 0; c < n_children; ++c) {
      uint64_t lin = 0;
      uint64_t stride = 1;
      int valid = 1;
      int ds_bit = 0;

      for (int d = 0; d < ndim; ++d) {
        uint64_t child_coord;
        if (ds_mask & (1 << d)) {
          child_coord = coords[d] * 2 + ((c >> ds_bit) & 1);
          ds_bit++;
        } else {
          child_coord = coords[d];
        }
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

static void
morton_unshuffle(const struct lod_plan* p,
                 int level,
                 const float* morton_buf,
                 float* rowmajor)
{
  const uint64_t* full_shape = p->shapes[level];
  uint64_t n = 1;
  for (int d = 0; d < p->ndim; ++d)
    n *= full_shape[d];

  uint64_t full_coords[MAX_NDIM];
  uint64_t lod_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_lod(p, full_coords, lod_coords);
    uint64_t pos =
      morton_rank(p->lod_ndim, p->lod_shapes[level], lod_coords, 0);
    rowmajor[i] = morton_buf[b * p->lod_counts[level] + pos];
  }
}
