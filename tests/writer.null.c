#include "writer.null.h"
#include "prelude.h"

#include <stdint.h>

static struct writer_result
null_writer_append(struct writer* self, struct slice data)
{
  struct null_writer* w = (struct null_writer*)self;
  size_t n = (const uint8_t*)data.end - (const uint8_t*)data.beg;
  w->bytes_discarded += n;
  return (struct writer_result){ 0 };
}

static struct writer_result
null_writer_flush(struct writer* self)
{
  (void)self;
  return (struct writer_result){ 0 };
}

static int
null_tile_writer_append(struct tile_writer* self,
                        const void* const* tiles,
                        const size_t* sizes,
                        size_t count)
{
  struct null_writer* w = container_of(self, struct null_writer, tile_base);
  w->tiles_discarded += count;
  for (size_t i = 0; i < count; ++i)
    w->compressed_bytes += sizes[i];
  (void)tiles;
  return 0;
}

static int
null_tile_writer_flush(struct tile_writer* self)
{
  (void)self;
  return 0;
}

struct null_writer
null_writer_new(void)
{
  return (struct null_writer){
    .base = { .append = null_writer_append, .flush = null_writer_flush },
    .tile_base = { .append = null_tile_writer_append,
                   .flush = null_tile_writer_flush },
  };
}
