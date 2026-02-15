#pragma once

#include "stream.h"

#include <stdint.h>

struct io_queue;

struct zarr_dimension
{
  uint64_t size;
  uint64_t tile_size;
  uint64_t tiles_per_shard; // 0 = all tiles along this dimension
  const char* name;         // dimension name, may be NULL
};

enum zarr_dtype
{
  zarr_dtype_uint8,
  zarr_dtype_uint16,
  zarr_dtype_uint32,
  zarr_dtype_uint64,
  zarr_dtype_int8,
  zarr_dtype_int16,
  zarr_dtype_int32,
  zarr_dtype_int64,
  zarr_dtype_float32,
  zarr_dtype_float64,
};

struct zarr_config
{
  const char* store_path; // root directory
  const char* array_name; // e.g. "0"
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct zarr_dimension* dimensions;
};

struct zarr_sink;

// Create a zarr v3 store sink. Writes directory structure and metadata files.
// queue may be NULL for synchronous I/O.
// Returns NULL on error.
struct zarr_sink* zarr_sink_create(const struct zarr_config* cfg,
                                   struct io_queue* queue);

void zarr_sink_destroy(struct zarr_sink* s);

// Get the shard_sink interface for use with transpose_stream.
struct shard_sink* zarr_sink_as_shard_sink(struct zarr_sink* s);
