#pragma once

#include "aggregate.h"
#include "lod_plan.h"
#include "metric.h"
#include "transpose.h"
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct slice
{
  const void* beg;
  const void* end;
};

struct writer_result
{
  int error;
  struct slice rest; // unconsumed input (empty on success for append)
};

struct writer
{
  struct writer_result (*append)(struct writer* self, struct slice data);
  struct writer_result (*flush)(struct writer* self);
};

struct shard_writer
{
  int (*write)(struct shard_writer* self,
               uint64_t offset, // byte offset within the shard
               const void* beg,
               const void* end);
  int (*finalize)(struct shard_writer* self); // shard complete, close/flush
};

struct shard_sink
{
  // Open/get a writer for the given flat shard index.
  struct shard_writer* (*open)(struct shard_sink* self,
                               uint8_t level,
                               uint64_t shard_index);
};

struct stream_metrics
{
  struct stream_metric memcpy;
  struct stream_metric h2d;
  struct stream_metric lod_scatter;
  struct stream_metric lod_reduce;
  struct stream_metric lod_m2t;
  struct stream_metric scatter;
  struct stream_metric compress;
  struct stream_metric aggregate;
  struct stream_metric d2h;
};

enum domain
{
  host,
  device
};

struct buffer
{
  void* data;
  CUevent ready;
  enum domain domain;
};

struct dimension
{
  uint64_t size;
  uint64_t tile_size;
  uint64_t tiles_per_shard; // 0 means all tiles along this dimension
  const char* name;         // optional label (e.g. "x"), may be NULL
};

struct tile_stream_configuration
{
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct writer* sink;           // downstream writer (uncompressed), not owned
  struct shard_sink* shard_sink; // downstream shard writer factory, not owned
  int compress;                  // enable nvcomp zstd compression
  int enable_multiscale;         // enable LOD pyramid generation
  uint32_t lod_mask;             // bitmask of dims to downsample (0 = dim0)
};

struct staging_slot
{
  struct buffer h_in;      // pinned host WC, size = buffer_capacity_bytes
  struct buffer d_in;      // device, size = buffer_capacity_bytes
  CUevent t_h2d_start;     // recorded before H2D memcpy
  CUevent t_scatter_start; // recorded before scatter kernel
  CUevent t_scatter_end;   // recorded after scatter kernel
};

struct staging_state
{
  struct staging_slot slot[2];
  int current; // 0 or 1: which buffer the host is filling
  size_t fill; // bytes written to current slot's h_in so far
};

struct stream_layout
{
  uint8_t lifted_rank;
  uint64_t lifted_shape[MAX_RANK];
  int64_t lifted_strides[MAX_RANK];

  uint64_t* d_lifted_shape;  // device copy (allocated once)
  int64_t* d_lifted_strides; // device copy (allocated once)

  uint64_t tile_elements;  // elements per tile
  uint64_t tile_stride;    // elements between tile starts (>= tile_elements)
  uint64_t slot_count;     // M = prod of tile_count[i] for i > 0
  uint64_t epoch_elements; // elements per epoch = M * tile_elements
  size_t tile_pool_bytes;  // slot_count * tile_stride * bpe
};

struct active_shard
{
  size_t data_cursor;
  uint64_t* index;             // 2 * tiles_per_shard_total entries
  struct shard_writer* writer; // from sink->open, NULL until first use
};

struct shard_state
{
  uint64_t epoch_in_shard;        // 0..tiles_per_shard[0]-1
  uint64_t shard_epoch;           // s_0 coordinate (0, 1, 2, ...)
  uint64_t shard_inner_count;     // S_inner = prod(shard_count[d] for d>0)
  uint64_t tiles_per_shard_inner; // prod(tps[d] for d>0)
  uint64_t tiles_per_shard_total; // prod(tps[d] for all d)
  uint64_t tiles_per_shard_0;     // tps[0]
  struct active_shard* shards;    // array[shard_inner_count]
};

// Per flush-slot: holds compressed output + pre-built pointer arrays.
// flush[0] is used for A-pool epochs, flush[1] for B-pool epochs.
struct flush_slot_gpu
{
  struct buffer d_compressed; // device: M0 * max_comp_chunk_bytes
  void** d_uncomp_ptrs;       // device [M0], pre-built at init
  void** d_comp_ptrs;         // device [M0], pre-built at init
  CUevent t_compress_start;
  CUevent t_d2h_start;
  CUevent ready; // signals all D2H for this slot is done
};

struct tile_stream_gpu;

// Dispatch function: H2D + scatter (+ optional d_linear copy).
typedef int (*dispatch_scatter_fn)(struct tile_stream_gpu*);

struct tile_stream_gpu
{
  struct writer writer;
  dispatch_scatter_fn dispatch;
  CUstream h2d, compute, compress, d2h;
  struct staging_state stage;
  struct stream_layout layout; // L0 layout
  struct tile_stream_configuration config;
  struct stream_metrics metrics;
  uint64_t cursor;

  // Tile pools (double-buffered)
  //   A, B:     M0 tiles each (device)
  //   A_host, B_host: M0 tiles each (host pinned, uncompressed path)
  struct buffer pool_A;      // device: M0 * tile_stride * bpe
  struct buffer pool_B;      // device: M0 * tile_stride * bpe
  struct buffer pool_A_host; // host pinned (uncompressed path)
  struct buffer pool_B_host; // host pinned (uncompressed path)
  int pool_current;          // 0=A, 1=B — which pool scatter writes to

  // Flush pipeline
  struct flush_slot_gpu flush[2]; // [0]=A epochs, [1]=B epochs
  int flush_current;              // 0 or 1
  int flush_pending;

  // Compress (shared across flushes)
  size_t* d_comp_sizes;   // device [M0]
  size_t* d_uncomp_sizes; // device [M0], all same
  void* d_comp_temp;
  size_t comp_temp_bytes;
  size_t max_comp_chunk_bytes;
  size_t comp_pool_bytes; // M0 * max_comp_chunk_bytes

  // Aggregate + shard
  struct aggregate_layout agg_layout;
  struct aggregate_slot agg[2]; // indexed by flush_current
  struct shard_state shard;

  // LOD (multiscale) state
  struct lod_plan lod;
  struct buffer d_linear;       // linear epoch buffer (device)
  struct buffer d_morton;       // morton-ordered LOD output (all levels packed)
  CUdeviceptr d_lod_full_shape; // device copy of shapes[0]
  CUdeviceptr d_lod_shape;      // device copy of lod_shapes[0]
  CUdeviceptr d_lod_ends;       // device copy of lod_plan.ends
  CUdeviceptr d_lod_child_shapes[LOD_MAX_LEVELS];  // per-level child shapes
  CUdeviceptr d_lod_parent_shapes[LOD_MAX_LEVELS]; // per-level parent shapes
  CUdeviceptr d_lod_level_ends[LOD_MAX_LEVELS];    // per-level ends
  CUevent t_lod_start;
  CUevent t_lod_scatter_end;
  CUevent t_lod_reduce_end;
  CUevent t_lod_end;

  // Per-level tile layout for morton-to-tile scatter
  struct stream_layout
    lod_layouts[LOD_MAX_LEVELS]; // [1..nlev-1], index 0 unused
  CUdeviceptr
    d_lod_lv_full_shapes[LOD_MAX_LEVELS]; // per-level device full shapes
  CUdeviceptr
    d_lod_lv_lod_shapes[LOD_MAX_LEVELS]; // per-level device lod shapes
  struct buffer d_lod_tiles; // unified tile buffer for all LOD levels
  uint64_t lod_tile_ends[LOD_MAX_LEVELS]; // exclusive end offsets in elements

  // Per-level compress + aggregate + shard delivery
  struct
  {
    struct buffer d_compressed; // compressed tiles for this level
    void** d_uncomp_ptrs;       // device [M_level]
    void** d_comp_ptrs;         // device [M_level]
    size_t* d_comp_sizes;       // device [M_level]
    size_t* d_uncomp_sizes;     // device [M_level]
    void* d_comp_temp;
    size_t comp_temp_bytes;
    size_t max_comp_chunk_bytes;
    size_t comp_pool_bytes;
    struct aggregate_layout agg_layout;
    struct aggregate_slot agg_slot;
    struct shard_state shard;
  } lod_levels[LOD_MAX_LEVELS]; // [1..nlev-1]
};

// Initialize a tile_stream_gpu. Returns 0 on success, non-zero on error.
// On failure, *out is zeroed and safe to pass to tile_stream_gpu_destroy.
int
tile_stream_gpu_create(const struct tile_stream_configuration* config,
                       struct tile_stream_gpu* out);

void
tile_stream_gpu_destroy(struct tile_stream_gpu* stream);

// Dispatch to the writer's append method.
struct writer_result
writer_append(struct writer* w, struct slice data);

// Dispatch to the writer's flush method.
struct writer_result
writer_flush(struct writer* w);

// Append data to a writer, retrying with exponential back-off on stall.
struct writer_result
writer_append_wait(struct writer* w, struct slice data);

// Return accumulated timing metrics.
struct stream_metrics
tile_stream_gpu_get_metrics(const struct tile_stream_gpu* s);
