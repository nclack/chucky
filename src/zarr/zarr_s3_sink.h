#pragma once

#include "dimension.h"
#include "types.codec.h"
#include "writer.h"
#include "dtype.h"

#include <stdint.h>

struct zarr_s3_config
{
  const char* bucket;     // S3 bucket name
  const char* prefix;     // key prefix (like store_path), no leading/trailing /
  const char* array_name; // e.g. "0"
  const char* region;     // NULL = auto (us-east-1)
  const char* endpoint;   // NULL = AWS default, or e.g. "http://localhost:9000"
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions;
  enum compression_codec codec; // CODEC_ZSTD, CODEC_LZ4, or CODEC_NONE
  size_t part_size;             // 0 = default (8 MiB)
  double throughput_gbps;       // 0 = default (10.0)
  size_t max_retries;           // 0 = CRT default (10)
  uint32_t backoff_scale_ms;    // 0 = CRT default (500)
  uint32_t max_backoff_secs;    // 0 = CRT default (20)
  uint64_t timeout_ns;          // 0 = no timeout (infinite)
};

struct zarr_s3_sink;

// Create a zarr v3 store sink backed by S3.
// Writes zarr.json metadata and shard objects to the bucket.
// Returns NULL on error.
struct zarr_s3_sink*
zarr_s3_sink_create(const struct zarr_s3_config* cfg);

// Returns non-zero if any upload has failed.
int
zarr_s3_sink_has_error(const struct zarr_s3_sink* s);

// Returns non-zero if any upload had failed.
int
zarr_s3_sink_destroy(struct zarr_s3_sink* s);

// Block until all in-flight uploads have completed.
// Returns non-zero if any upload failed.
int
zarr_s3_sink_flush(struct zarr_s3_sink* s);

// Get the shard_sink interface for use with the chunk stream.
struct shard_sink*
zarr_s3_sink_as_shard_sink(struct zarr_s3_sink* s);

// --- Multiscale (OME-NGFF) ---

struct zarr_s3_multiscale_config
{
  const char* bucket;
  const char* prefix;
  const char* array_name; // group name; NULL = write at prefix
  const char* region;
  const char* endpoint;
  enum dtype data_type;
  double fill_value;
  uint8_t rank;
  const struct dimension* dimensions; // L0 dimensions
  int nlod;                           // number of levels (0 = auto)
  enum compression_codec codec;
  size_t part_size;          // 0 = default (8 MiB)
  double throughput_gbps;    // 0 = default (10.0)
  size_t max_retries;        // 0 = CRT default (10)
  uint32_t backoff_scale_ms; // 0 = CRT default (500)
  uint32_t max_backoff_secs; // 0 = CRT default (20)
  uint64_t timeout_ns;       // 0 = no timeout (infinite)
};

struct zarr_s3_multiscale_sink;

struct zarr_s3_multiscale_sink*
zarr_s3_multiscale_sink_create(const struct zarr_s3_multiscale_config* cfg);

// Returns non-zero if any upload had failed.
int
zarr_s3_multiscale_sink_destroy(struct zarr_s3_multiscale_sink* s);

// Block until all in-flight uploads have completed.
// Returns non-zero if any upload failed.
int
zarr_s3_multiscale_sink_flush(struct zarr_s3_multiscale_sink* s);

struct shard_sink*
zarr_s3_multiscale_sink_as_shard_sink(struct zarr_s3_multiscale_sink* s);
