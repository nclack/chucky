#pragma once

#include "stream.h"
#include <stddef.h>
#include <stdint.h>

struct zstd_tile_writer
{
  struct tile_writer base;
  struct writer* sink; // not owned
  size_t tile_bytes;   // expected decompressed size
  uint8_t* decomp_buf; // scratch buffer (one tile)
  size_t total_compressed;
  size_t total_decompressed;
};

struct zstd_tile_writer
zstd_tile_writer_new(size_t tile_bytes, struct writer* sink);

void
zstd_tile_writer_free(struct zstd_tile_writer* w);
