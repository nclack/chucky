// morton.util.c — shared CPU reference for LOD via compacted Morton codes.
// Intended to be #include'd (no main).

#include "prelude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM 64
#define MAX_LOD 32

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

static int
is_all_ones(int n, const uint64_t* v)
{
  for (int d = 0; d < n; ++d)
    if (v[d] > 1)
      return 0;
  return 1;
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

// --- LOD plan ---

struct slice
{
  uint64_t beg, end;
};

static uint64_t
slice_len(struct slice s)
{
  return s.end - s.beg;
}

struct spans
{
  uint64_t* ends;
  uint64_t n;
};

static struct slice
spans_at(const struct spans* s, uint64_t i)
{
  return (struct slice){
    .beg = i > 0 ? s->ends[i - 1] : 0,
    .end = s->ends[i],
  };
}

struct lod_plan
{
  int ndim;
  int nlev;
  uint64_t shapes[MAX_LOD][MAX_NDIM];

  uint8_t ds_mask;
  int ds_ndim;
  int ds_map[MAX_NDIM];
  int batch_ndim;
  int batch_map[MAX_NDIM];
  uint64_t batch_shape[MAX_NDIM];
  uint64_t batch_count;

  uint64_t ds_shapes[MAX_LOD][MAX_NDIM];
  uint64_t ds_counts[MAX_LOD];

  struct spans levels;
  struct spans ds_levels;
  uint64_t* ends;
};

static struct slice
lod_segment(const struct lod_plan* p, int level)
{
  struct slice next = spans_at(&p->ds_levels, level + 1);
  uint64_t base = p->ds_levels.ends[0];
  return (struct slice){ .beg = next.beg - base, .end = next.end - base };
}

static void
lod_plan_free(struct lod_plan* p);

static void
lod_fill_ends(int ndim,
              const uint64_t* child_shape,
              const uint64_t* parent_shape,
              uint64_t n_parents,
              uint64_t* ends)
{
  int p = ceil_log2(max_shape(ndim, parent_shape));

  uint64_t coords[MAX_NDIM];
  uint64_t next[MAX_NDIM];
  for (uint64_t j = 0; j < n_parents; ++j) {
    linear_to_coords(ndim, parent_shape, j, coords);
    uint64_t pos = morton_rank(ndim, parent_shape, coords, 0);

    memcpy(next, coords, (size_t)ndim * sizeof(uint64_t));
    coords_morton_next(ndim, p, next);
    uint64_t val = morton_rank(ndim, child_shape, next, 1);

    ends[pos] = val;
  }
}

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
plan_extract_ds(const struct lod_plan* p,
                const uint64_t* full_coords,
                uint64_t* ds_coords)
{
  for (int k = 0; k < p->ds_ndim; ++k)
    ds_coords[k] = full_coords[p->ds_map[k]];
}

static int
lod_plan_init(struct lod_plan* p,
              int ndim,
              const uint64_t* shape,
              uint8_t ds_mask,
              int max_levels)
{
  memset(p, 0, sizeof(*p));
  p->ndim = ndim;
  p->ds_mask = ds_mask;

  for (int d = 0; d < ndim; ++d) {
    if (ds_mask & (1 << d)) {
      p->ds_map[p->ds_ndim++] = d;
    } else {
      p->batch_map[p->batch_ndim] = d;
      p->batch_shape[p->batch_ndim] = shape[d];
      p->batch_ndim++;
    }
  }
  p->batch_count = 1;
  for (int k = 0; k < p->batch_ndim; ++k)
    p->batch_count *= p->batch_shape[k];

  memcpy(p->shapes[0], shape, (size_t)ndim * sizeof(uint64_t));
  for (int k = 0; k < p->ds_ndim; ++k)
    p->ds_shapes[0][k] = shape[p->ds_map[k]];

  p->nlev = 1;
  while (p->nlev < max_levels &&
         !is_all_ones(p->ds_ndim, p->ds_shapes[p->nlev - 1])) {
    for (int k = 0; k < p->ds_ndim; ++k)
      p->ds_shapes[p->nlev][k] = (p->ds_shapes[p->nlev - 1][k] + 1) / 2;
    memcpy(p->shapes[p->nlev], p->shapes[p->nlev - 1],
           (size_t)ndim * sizeof(uint64_t));
    for (int k = 0; k < p->ds_ndim; ++k)
      p->shapes[p->nlev][p->ds_map[k]] = p->ds_shapes[p->nlev][k];
    ++p->nlev;
  }

  for (int k = 0; k < p->nlev; ++k) {
    p->ds_counts[k] = 1;
    for (int d = 0; d < p->ds_ndim; ++d)
      p->ds_counts[k] *= p->ds_shapes[k][d];
  }

  p->ds_levels.n = (uint64_t)p->nlev;
  p->ds_levels.ends = (uint64_t*)malloc(p->nlev * sizeof(uint64_t));
  if (!p->ds_levels.ends)
    goto Fail;
  p->ds_levels.ends[0] = p->ds_counts[0];
  for (int k = 1; k < p->nlev; ++k)
    p->ds_levels.ends[k] = p->ds_levels.ends[k - 1] + p->ds_counts[k];

  p->levels.n = (uint64_t)p->nlev;
  p->levels.ends = (uint64_t*)malloc(p->nlev * sizeof(uint64_t));
  if (!p->levels.ends)
    goto Fail;
  for (int k = 0; k < p->nlev; ++k)
    p->levels.ends[k] = p->batch_count * p->ds_levels.ends[k];

  {
    uint64_t total_ends =
      p->ds_levels.ends[p->nlev - 1] - p->ds_levels.ends[0];
    if (total_ends > 0) {
      p->ends = (uint64_t*)malloc(total_ends * sizeof(uint64_t));
      if (!p->ends)
        goto Fail;
      for (int l = 0; l < p->nlev - 1; ++l) {
        struct slice seg = lod_segment(p, l);
        lod_fill_ends(p->ds_ndim,
                      p->ds_shapes[l],
                      p->ds_shapes[l + 1],
                      slice_len(seg),
                      p->ends + seg.beg);
      }
    }
  }

  return 1;
Fail:
  lod_plan_free(p);
  return 0;
}

static void
lod_plan_free(struct lod_plan* p)
{
  if (!p)
    return;
  free(p->levels.ends);
  free(p->ds_levels.ends);
  free(p->ends);
  memset(p, 0, sizeof(*p));
}

static void
lod_scatter(const struct lod_plan* p, const float* src, float* dst)
{
  const uint64_t* full_shape = p->shapes[0];
  uint64_t n = slice_len(spans_at(&p->levels, 0));

  uint64_t full_coords[MAX_NDIM];
  uint64_t ds_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_ds(p, full_coords, ds_coords);
    uint64_t pos = morton_rank(p->ds_ndim, p->ds_shapes[0], ds_coords, 0);
    dst[b * p->ds_counts[0] + pos] = src[i];
  }
}

static void
lod_reduce(const struct lod_plan* p, float* values)
{
  for (int l = 0; l < p->nlev - 1; ++l) {
    struct slice seg = lod_segment(p, l);
    uint64_t src_ds = p->ds_counts[l];
    uint64_t dst_ds = p->ds_counts[l + 1];
    struct slice src_level = spans_at(&p->levels, l);
    struct slice dst_level = spans_at(&p->levels, l + 1);

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

  lod_scatter(p, src, values);
  lod_reduce(p, values);

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
  uint64_t ds_coords[MAX_NDIM];
  for (uint64_t i = 0; i < n; ++i) {
    linear_to_coords(p->ndim, full_shape, i, full_coords);
    uint64_t b = plan_batch_index(p, full_coords);
    plan_extract_ds(p, full_coords, ds_coords);
    uint64_t pos =
      morton_rank(p->ds_ndim, p->ds_shapes[level], ds_coords, 0);
    rowmajor[i] = morton_buf[b * p->ds_counts[level] + pos];
  }
}
