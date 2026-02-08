#include "writer.mem.h"

#include "log/log.h"
#include <stdlib.h>
#include <string.h>

static struct writer_result
mem_writer_append(struct writer* self, struct slice data)
{
  struct mem_writer* w = (struct mem_writer*)self;
  size_t n = (const uint8_t*)data.end - (const uint8_t*)data.beg;
  if (w->cursor + n > w->capacity) {
    log_error("mem_writer: overflow (%zu + %zu > %zu)",
              w->cursor,
              n,
              w->capacity);
    return (struct writer_result){ .error = 1, .rest = data };
  }
  memcpy(w->buf + w->cursor, data.beg, n);
  w->cursor += n;
  return (struct writer_result){ 0 };
}

static struct writer_result
mem_writer_flush(struct writer* self)
{
  (void)self;
  return (struct writer_result){ 0 };
}

struct mem_writer
mem_writer_new(size_t capacity)
{
  struct mem_writer w = {
    .base = { .append = mem_writer_append, .flush = mem_writer_flush },
    .buf = (uint8_t*)malloc(capacity),
    .capacity = capacity,
    .cursor = 0,
  };
  return w;
}

void
mem_writer_free(struct mem_writer* w)
{
  free(w->buf);
  *w = (struct mem_writer){ 0 };
}
