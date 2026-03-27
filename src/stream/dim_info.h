#pragma once

#include "dimension.h"
#include "util/ceildiv.h"
#include <assert.h>
#include <stdint.h>

// View into a contiguous range of dimensions (beg/end pointer pair).
struct dim_slice
{
  const struct dimension* beg;
  const struct dimension* end;
};

static inline uint8_t
dim_slice_len(struct dim_slice s)
{
  return (uint8_t)(s.end - s.beg);
}

// Resolved partition of dims into append and inner groups.
// Constructed once by dim_info_init(); immutable after.
//
// Append dims: always includes dim 0. When additional dims also have
// chunk_size == 1, they extend the prefix — truncated at the first dim
// with downsample set (only the rightmost append dim may be downsampled).
// Inner dims: everything after the append prefix.
// The slices point into the original dimension array (no copies).
//
// Lifetime: slices point into the dimension array passed to dim_info_init().
// That array must outlive this struct and remain unmodified.
struct dim_info
{
  struct dim_slice append; // dims[0 .. n_append)
  struct dim_slice inner;  // dims[n_append .. rank)

  int append_downsample;       // rightmost append dim has downsample
  uint32_t lod_mask;           // bitmask: inner dims with downsample=1
  uint64_t inner_append_count; // prod(chunk_count[d] for d=1..n_append-1)
};

static inline uint8_t
dim_info_n_append(const struct dim_info* info)
{
  return dim_slice_len(info->append);
}

static inline uint8_t
dim_info_rank(const struct dim_info* info)
{
  return dim_slice_len(info->append) + dim_slice_len(info->inner);
}

// Absolute index of a dimension pointer within the original array.
static inline int
dim_index(const struct dim_info* info, const struct dimension* d)
{
  return (int)(d - info->append.beg);
}

// Decompose a flat append-chunk count into per-dimension append sizes.
// Fills append_sizes[0..n_append-1].
static inline void
dim_info_decompose_append_sizes(const struct dim_info* info,
                                uint64_t total_append_chunks,
                                uint64_t* append_sizes)
{
  uint8_t na = dim_info_n_append(info);
  for (uint8_t i = 0; i < na; ++i)
    append_sizes[i] = 0;
  for (const struct dimension* d = info->append.beg + 1; d < info->append.end; ++d)
    append_sizes[dim_index(info, d)] = d->size;
  const uint64_t iac = info->inner_append_count;
  assert(iac > 0 && "inner_append_count must be > 0 after valid dim_info_init");
  append_sizes[0] =
    ceildiv(total_append_chunks, iac) * info->append.beg[0].chunk_size;
}

// Partition dims into append/inner, validate constraints, precompute
// derived values. Returns 0 on success.
//
// Validates via dims_validate(): chunk_size > 0, storage_position is a
// valid permutation with append dims pinned, only dim 0 may be unbounded.
// Note: downsample on a non-rightmost append dim is not rejected — it
// causes dims_n_append to truncate the prefix there (see dims_n_append).
int
dim_info_init(struct dim_info* info,
              const struct dimension* dims,
              uint8_t rank);
