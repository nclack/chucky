#pragma once

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

struct tile_writer
{
  int (*append)(struct tile_writer* self,
                const void* const* tiles, // array of pointers to tile data
                const size_t* sizes,      // array of byte counts per tile
                size_t count);            // number of tiles (= slot_count)
  int (*flush)(struct tile_writer* self);
};

struct stream_metrics
{
  float h2d_ms;
  float scatter_ms;
  float compress_ms;
  float d2h_ms;
  float scatter_best_ms;
  float compress_best_ms;
  float d2h_best_ms;
  int epoch_count;
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
};

struct transpose_stream_configuration
{
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct writer* sink;              // downstream writer (uncompressed), not owned
  struct tile_writer* compressed_sink; // downstream writer (compressed), not owned
  int compress;                     // enable nvcomp zstd compression
};

struct transpose_stream
{
  struct writer writer;

  CUstream h2d, compute, d2h;

  // Input staging (double-buffered)
  struct buffer h_in[2]; // pinned host WC, size = buffer_capacity_bytes
  struct buffer d_in[2]; // device, size = buffer_capacity_bytes
  int stage_idx;         // 0 or 1: which buffer the host is filling

  // Tile pool (double-buffered: scatter into one while flushing the other)
  struct buffer d_tiles[2]; // device: tile_pool_bytes
  struct buffer h_tiles[2]; // host:   tile_pool_bytes
  int tile_idx;             // which pool scatter writes to
  int flush_pending;        // async D2H in flight, not yet delivered to sink

  // Precomputed layout (lifted rank = 2 * config.rank)
  uint8_t lifted_rank;
  uint64_t lifted_shape[2 * MAX_RANK];
  int64_t lifted_strides[2 * MAX_RANK]; // strides[0] = 0

  // Device copies (allocated once, used by every kernel dispatch)
  uint64_t* d_lifted_shape;
  int64_t* d_lifted_strides;

  uint64_t tile_elements;  // elements per tile
  uint64_t tile_stride;    // elements between tile starts (>= tile_elements)
  uint64_t slot_count;     // M = prod of tile_count[i] for i > 0
  uint64_t epoch_elements; // elements per epoch = M * tile_elements
  size_t tile_pool_bytes;  // slot_count * tile_stride * bpe

  // Compression state (all zero when compress == 0)
  struct buffer d_compressed[2]; // device: comp_pool_bytes
  struct buffer h_compressed[2]; // host:   comp_pool_bytes

  void** d_uncomp_ptrs[2]; // device arrays of pointers into d_tiles[i]
  void** d_comp_ptrs[2];   // device arrays of pointers into d_compressed[i]
  size_t* d_uncomp_sizes;  // device: all = tile_stride * bpe
  size_t* d_comp_sizes[2]; // device: actual compressed sizes per tile
  size_t* h_comp_sizes[2]; // host (pinned): compressed sizes per tile
  void* d_comp_temp;       // device scratch workspace
  size_t comp_temp_bytes;
  size_t max_comp_chunk_bytes; // per-tile max compressed size
  size_t comp_pool_bytes;      // slot_count * max_comp_chunk_bytes

  // GPU timing instrumentation
  CUevent t_h2d_start[2];      // per-slot: recorded before H2D memcpy
  CUevent t_scatter_start[2];  // per-slot: recorded before scatter kernel
  CUevent t_compress_start[2]; // per-pool: recorded before compress
  CUevent t_d2h_start[2];      // per-pool: recorded before D2H memcpy
  struct stream_metrics metrics;

  // Runtime state
  uint64_t cursor;   // current element position in input stream
  size_t stage_fill; // bytes written to h_in so far

  struct transpose_stream_configuration config;
};

// Initialize a transpose_stream. Returns 0 on success, non-zero on error.
// On failure, *out is zeroed and safe to pass to transpose_stream_destroy.
int
transpose_stream_create(const struct transpose_stream_configuration* config,
                        struct transpose_stream* out);

void
transpose_stream_destroy(struct transpose_stream* stream);

// Dispatch to the writer's append method.
struct writer_result
writer_append(struct writer* w, struct slice data);

// Dispatch to the writer's flush method.
struct writer_result
writer_flush(struct writer* w);

// Append data to a writer, retrying with exponential back-off on stall.
struct writer_result
writer_append_wait(struct writer* w, struct slice data);

// Return accumulated GPU timing metrics.
struct stream_metrics
transpose_stream_get_metrics(const struct transpose_stream* s);
