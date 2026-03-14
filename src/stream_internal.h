#pragma once

#include "stream.h"
#include "stream_flush.h"
#include "stream_ingest.h"
#include "stream_lod.h"

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// Set writer vtable (append/flush).
// Called from tile_stream_gpu_create after zeroing *out.
void
tile_stream_gpu_init_writer(struct tile_stream_gpu* s);

// Per-level pre-computed layout information (CPU only, no GPU pointers).
struct level_layout_info
{
  struct aggregate_layout agg_layout; // host fields only, d_* = NULL
  uint32_t batch_active_count;
  uint64_t tiles_per_shard_0;
  uint64_t tiles_per_shard_inner;
  uint64_t tiles_per_shard_total;
  uint64_t shard_inner_count;
};

// All pre-computed layout data from CPU-only math.
// Produced by compute_stream_layouts, consumed by the create path
// and the memory estimate path.
struct computed_stream_layouts
{
  struct stream_layout l0; // host fields; d_* = NULL
  struct lod_plan plan;    // owned if enable_multiscale
  struct stream_layout
    lod_layouts[LOD_MAX_LEVELS]; // host; d_* = NULL; [0] unused
  struct level_geometry levels;
  uint32_t epochs_per_batch;
  size_t max_output_size; // codec-derived compressed tile bound
  struct level_layout_info per_level[LOD_MAX_LEVELS];
};

// Validate config and compute all CPU-only layout math. Returns 0 on success.
int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       struct computed_stream_layouts* out);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);
