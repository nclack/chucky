#pragma once

#include "stream.h"
#include <stddef.h>

struct null_writer
{
  struct writer base;
  struct tile_writer tile_base;
  size_t bytes_discarded;
  size_t tiles_discarded;
  size_t compressed_bytes;
};

struct null_writer
null_writer_new(void);
