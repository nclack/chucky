#include "zarr_metadata.h"
#include "defs.limits.h"
#include "json_writer.h"
#include "prelude.h"

#include <stdio.h>

int
zarr_root_json(char* buf, size_t cap)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}

int
zarr_array_json(char* buf,
                size_t cap,
                uint8_t rank,
                const struct dimension* dimensions,
                enum dtype data_type,
                double fill_value,
                const uint64_t* chunks_per_shard,
                enum compression_codec codec)
{
  struct json_writer jw;
  jw_init(&jw, buf, cap);

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "array");

  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].size);
  jw_array_end(&jw);

  jw_key(&jw, "data_type");
  jw_string(&jw, dtype_zarr_string(data_type));

  // chunk_grid: shard = chunk_size * chunks_per_shard
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size * chunks_per_shard[d]);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  jw_key(&jw, "chunk_key_encoding");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "default");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "separator");
  jw_string(&jw, "/");
  jw_object_end(&jw);
  jw_object_end(&jw);

  // codecs: sharding_indexed
  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "sharding_indexed");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);

  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < rank; ++d)
    jw_uint(&jw, dimensions[d].chunk_size);
  jw_array_end(&jw);

  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  if (codec != CODEC_NONE) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    jw_string(&jw, codec == CODEC_LZ4 ? "lz4" : "zstd");
    jw_key(&jw, "configuration");
    jw_object_begin(&jw);
    jw_object_end(&jw);
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "index_codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "crc32c");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "index_location");
  jw_string(&jw, "end");

  jw_object_end(&jw); // configuration
  jw_object_end(&jw); // sharding_indexed
  jw_array_end(&jw);  // codecs

  jw_key(&jw, "fill_value");
  jw_float(&jw, fill_value);

  // dimension_names
  {
    int has_names = 0;
    for (int d = 0; d < rank; ++d) {
      if (dimensions[d].name) {
        has_names = 1;
        break;
      }
    }
    if (has_names) {
      jw_key(&jw, "dimension_names");
      jw_array_begin(&jw);
      for (int d = 0; d < rank; ++d) {
        if (dimensions[d].name)
          jw_string(&jw, dimensions[d].name);
        else
          jw_null(&jw);
      }
      jw_array_end(&jw);
    }
  }

  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return (int)jw_length(&jw);
}

int
zarr_shard_key(char* buf,
               size_t cap,
               uint8_t rank,
               const uint64_t* shard_count,
               uint64_t flat)
{
  int pos = snprintf(buf, cap, "c");
  if (pos < 0 || (size_t)pos >= cap)
    return -1;

  uint64_t coords[MAX_ZARR_RANK];
  uint64_t rem = flat;
  for (int d = rank - 1; d >= 0; --d) {
    if (d == 0) {
      coords[d] = rem;
    } else {
      coords[d] = rem % shard_count[d];
      rem /= shard_count[d];
    }
  }

  for (int d = 0; d < rank; ++d) {
    int n = snprintf(
      buf + pos, cap - (size_t)pos, "/%llu", (unsigned long long)coords[d]);
    if (n < 0 || (size_t)(pos + n) >= cap)
      return -1;
    pos += n;
  }
  return 0;
}

void
zarr_compute_geometry(struct zarr_geometry* g,
                      uint8_t rank,
                      const struct dimension* dimensions)
{
  g->shard_inner_count = 1;
  for (int d = 0; d < rank; ++d) {
    g->chunk_count[d] =
      (dimensions[d].size == 0)
        ? 1
        : ceildiv(dimensions[d].size, dimensions[d].chunk_size);
    uint64_t cps = dimensions[d].chunks_per_shard;
    g->chunks_per_shard[d] = (cps == 0) ? g->chunk_count[d] : cps;
    g->shard_count[d] = ceildiv(g->chunk_count[d], g->chunks_per_shard[d]);
    if (d > 0)
      g->shard_inner_count *= g->shard_count[d];
  }
}
