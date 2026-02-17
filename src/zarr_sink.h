#pragma once

#include "stream.h"

#include <stdint.h>

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
  const struct dimension* dimensions;
};

struct zarr_sink;

// Create a zarr v3 store sink. Writes directory structure and metadata files.
// Returns NULL on error.
struct zarr_sink* zarr_sink_create(const struct zarr_config* cfg);

void zarr_sink_destroy(struct zarr_sink* s);

// Block until all queued I/O has completed.
void zarr_sink_flush(struct zarr_sink* s);

// Get the shard_sink interface for use with transpose_stream.
struct shard_sink* zarr_sink_as_shard_sink(struct zarr_sink* s);

// --- Multiscale (LOD) zarr sink ---

struct zarr_multiscale_config
{
  const char* store_path;
  enum zarr_dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions; // level 0
  int num_levels;                     // total levels (including level 0)
};

struct zarr_multiscale_sink;

// Create a multiscale zarr v3 store sink with OME-NGFF metadata.
// Creates arrays "s0", "s1", ... "s{num_levels-1}".
// Returns NULL on error.
struct zarr_multiscale_sink*
zarr_multiscale_sink_create(const struct zarr_multiscale_config* cfg);

void zarr_multiscale_sink_destroy(struct zarr_multiscale_sink* s);

// Block until all queued I/O has completed across all levels.
void zarr_multiscale_sink_flush(struct zarr_multiscale_sink* s);

// Get the shard_sink interface for a specific level.
struct shard_sink*
zarr_multiscale_get_level_sink(struct zarr_multiscale_sink* s, uint8_t level);

// Get a combined shard_sink that routes open(level, shard_index) to the
// correct level's zarr_sink. Use this with transpose_stream when enable_lod=1.
struct shard_sink*
zarr_multiscale_as_shard_sink(struct zarr_multiscale_sink* s);
