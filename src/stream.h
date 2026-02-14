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

struct stream_metric
{
  const char* name;
  float ms;      // cumulative
  float best_ms; // best single measurement (1e30f = not yet measured)
  int count;
};

struct stream_metrics
{
  struct stream_metric h2d;
  struct stream_metric scatter;
  struct stream_metric compress;
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
};

struct transpose_stream_configuration
{
  size_t buffer_capacity_bytes;
  size_t bytes_per_element;
  uint8_t rank;
  const struct dimension* dimensions;
  struct writer* sink; // downstream writer (uncompressed), not owned
  struct tile_writer*
    compressed_sink; // downstream writer (compressed), not owned
  int compress;      // enable nvcomp zstd compression
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

struct compression_slot
{
  struct buffer d_compressed; // device: comp_pool_bytes
  struct buffer h_compressed; // host:   comp_pool_bytes
  void** d_uncomp_ptrs;       // device array of pointers into d_tiles
  void** d_comp_ptrs;         // device array of pointers into d_compressed
  size_t* d_comp_sizes;       // device: actual compressed sizes per tile
  size_t* h_comp_sizes;       // host (pinned): compressed sizes per tile
};

struct compression_shared
{
  size_t* d_uncomp_sizes; // device: all = tile_stride * bpe
  void* d_comp_temp;      // device scratch workspace
  size_t comp_temp_bytes;
  size_t max_comp_chunk_bytes; // per-tile max compressed size
  size_t comp_pool_bytes;      // slot_count * max_comp_chunk_bytes
};

struct tile_pool_slot
{
  struct buffer d_tiles; // device: tile_pool_bytes
  struct buffer h_tiles; // host:   tile_pool_bytes
  struct compression_slot comp;
  CUevent t_compress_start; // recorded before compress
  CUevent t_d2h_start;      // recorded before D2H memcpy
};

struct tile_pool_state
{
  struct tile_pool_slot slot[2];
  int current;       // which pool scatter writes to
  int flush_pending; // async D2H in flight, not yet delivered to sink
};

struct transpose_stream
{
  struct writer writer;
  CUstream h2d, compute, d2h;
  struct staging_state stage;
  struct tile_pool_state tiles;
  struct stream_layout layout;
  struct compression_shared comp;
  struct stream_metrics metrics;
  uint64_t cursor;
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
