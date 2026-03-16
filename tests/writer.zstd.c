#include "writer.zstd.h"

#include "log/log.h"
#include <stdlib.h>
#include <zstd.h>

static int
zstd_chunk_writer_append(struct tile_writer* self,
                         const void* const* tiles,
                         const size_t* sizes,
                         size_t count)
{
  struct zstd_chunk_writer* w = (struct zstd_chunk_writer*)self;

  for (size_t i = 0; i < count; ++i) {
    size_t result =
      ZSTD_decompress(w->decomp_buf, w->chunk_bytes, tiles[i], sizes[i]);
    if (ZSTD_isError(result)) {
      log_error("zstd_chunk_writer: ZSTD_decompress failed for chunk %zu: %s",
                i,
                ZSTD_getErrorName(result));
      return 1;
    }
    if (result != w->chunk_bytes) {
      log_error(
        "zstd_chunk_writer: decompressed size mismatch: expected %zu, got %zu",
        w->chunk_bytes,
        result);
      return 1;
    }

    struct slice data = { .beg = w->decomp_buf,
                          .end = w->decomp_buf + w->chunk_bytes };
    struct writer_result wr = writer_append_wait(w->sink, data);
    if (wr.error)
      return 1;

    w->total_compressed += sizes[i];
    w->total_decompressed += result;
  }

  return 0;
}

static int
zstd_chunk_writer_flush(struct tile_writer* self)
{
  struct zstd_chunk_writer* w = (struct zstd_chunk_writer*)self;
  return writer_flush(w->sink).error;
}

struct zstd_chunk_writer
zstd_chunk_writer_new(size_t chunk_bytes, struct writer* sink)
{
  return (struct zstd_chunk_writer){
    .base = { .append = zstd_chunk_writer_append,
              .flush = zstd_chunk_writer_flush },
    .sink = sink,
    .chunk_bytes = chunk_bytes,
    .decomp_buf = (uint8_t*)malloc(chunk_bytes),
  };
}

void
zstd_chunk_writer_free(struct zstd_chunk_writer* w)
{
  free(w->decomp_buf);
  *w = (struct zstd_chunk_writer){ 0 };
}
