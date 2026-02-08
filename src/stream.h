#pragma once

#include "transpose.h"
#include <cuda.h>
#include <stddef.h>

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
  struct writer* sink; // downstream writer, not owned
};

struct transpose_stream
{
  struct writer writer;

  CUstream h2d, compute, d2h;

  // Input staging (single-buffered)
  struct buffer h_in; // pinned host, size = buffer_capacity_bytes
  struct buffer d_in; // device, size = buffer_capacity_bytes

  // Tile pool (one epoch at a time)
  struct buffer d_tiles; // device: slot_count * tile_elements * bpe
  struct buffer h_tiles; // host:   slot_count * tile_elements * bpe

  // Precomputed layout (lifted rank = 2 * config.rank)
  uint8_t lifted_rank;
  uint64_t lifted_shape[2 * MAX_RANK];
  int64_t lifted_strides[2 * MAX_RANK]; // strides[0] = 0

  uint64_t tile_elements;  // elements per tile
  uint64_t slot_count;     // M = prod of tile_count[i] for i > 0
  uint64_t epoch_elements; // elements per epoch = M * tile_elements

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
