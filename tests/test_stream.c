#include "index.ops.util.h"
#include "log/log.h"
#include "stream.h"
#include "writer.mem.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

#define countof(e) (sizeof(e) / sizeof(e[0]))

#define CU(lbl, e)                                                             \
  do {                                                                         \
    if (handle_curesult(e, __FILE__, __LINE__))                                \
      goto lbl;                                                                \
  } while (0)

#define CHECK(lbl, expr)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      log_error("%s(%d): Check failed: (%s)", __FILE__, __LINE__, #expr);      \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

static int
handle_curesult(CUresult ecode, const char* file, int line)
{
  if (ecode == CUDA_SUCCESS)
    return 0;
  const char *name, *desc;
  cuGetErrorName(ecode, &name);
  cuGetErrorString(ecode, &desc);
  if (name && desc) {
    log_error("%s(%d): CUDA error: %s %s", file, line, name, desc);
  } else {
    log_error("%s(%d): Failed to retrieve error info for CUresult: %d",
              file,
              line,
              ecode);
  }
  return 1;
}

// --- tile pool verification helpers ---

// Build expected tile pool for one epoch.
static uint16_t*
make_expected_tiles(uint64_t epoch_start,
                    uint64_t epoch_elements,
                    uint64_t slot_count,
                    uint64_t tile_elements,
                    uint8_t lifted_rank,
                    const uint64_t* lifted_shape,
                    const int64_t* lifted_strides)
{
  uint64_t pool_size = slot_count * tile_elements;
  uint16_t* expected = (uint16_t*)calloc(pool_size, sizeof(uint16_t));
  if (!expected)
    return NULL;

  for (uint64_t i = 0; i < epoch_elements; ++i) {
    uint64_t idx = epoch_start + i;
    uint64_t off = ravel(lifted_rank, lifted_shape, lifted_strides, idx);
    expected[off] = (uint16_t)(idx % 65536);
  }
  return expected;
}

// Compare tile pool contents against expected.
static int
verify_tiles(const uint16_t* actual,
             const uint16_t* expected,
             uint64_t pool_size,
             int epoch,
             const char* label)
{
  int ecode = 0;
  int mismatch_count = 0;
  for (uint64_t i = 0; i < pool_size; ++i) {
    if (actual[i] != expected[i]) {
      if (mismatch_count < 10) {
        log_error("%s epoch %d: mismatch at pool[%lu]: expected %u, got %u",
                  label,
                  epoch,
                  (unsigned long)i,
                  expected[i],
                  actual[i]);
      }
      mismatch_count++;
      ecode = 1;
    }
  }
  if (mismatch_count > 10) {
    log_warn("%s epoch %d: ... %d total mismatches",
             label,
             epoch,
             mismatch_count);
  }
  return ecode;
}

// Test: feed all data in one append call.
// Shape (4,4,6), tile (2,2,3) -> 2 epochs, 4 tiles/epoch, 12 elements/tile.
// Total 96 elements.
static int
test_stream_single_append(void)
{
  log_info("=== test_stream_single_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 }, // slowest (dim 0)
    { .size = 4, .tile_size = 2 }, // dim 1
    { .size = 6, .tile_size = 3 }, // fastest (dim 2)
  };

  // 2 epochs * 4 slots * 12 elements * 2 bytes = 192 bytes
  const size_t pool_bytes = 4 * 12 * sizeof(uint16_t);
  struct mem_writer mw = mem_writer_new(2 * pool_bytes);
  CHECK(Fail0, mw.buf);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .sink = &mw.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  // Verify computed layout
  log_info("  tile_elements=%lu  slot_count=%lu  epoch_elements=%lu",
           (unsigned long)s.tile_elements,
           (unsigned long)s.slot_count,
           (unsigned long)s.epoch_elements);
  CHECK(Fail, s.tile_elements == 12);
  CHECK(Fail, s.slot_count == 4);
  CHECK(Fail, s.epoch_elements == 48);

  {
    printf("  lifted_shape: ");
    println_vu64(s.lifted_rank, s.lifted_shape);
    printf("  lifted_strides: ");
    println_vi64(s.lifted_rank, s.lifted_strides);
  }

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data — this processes epoch 0, flushes it to the writer,
  // then processes epoch 1's data.
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);

  // Epoch 0 was flushed to mw during append.
  CHECK(Fail, mw.cursor >= pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(0,
                                             s.epoch_elements,
                                             s.slot_count,
                                             s.tile_elements,
                                             s.lifted_rank,
                                             s.lifted_shape,
                                             s.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)mw.buf,
                           expected,
                           s.slot_count * s.tile_elements,
                           0,
                           "single_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 0 verification");
      goto Fail;
    }
    log_info("  epoch 0: OK");
  }

  // Flush to get epoch 1
  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);

  CHECK(Fail, mw.cursor == 2 * pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(s.epoch_elements,
                                             s.epoch_elements,
                                             s.slot_count,
                                             s.tile_elements,
                                             s.lifted_rank,
                                             s.lifted_shape,
                                             s.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)(mw.buf + pool_bytes),
                           expected,
                           s.slot_count * s.tile_elements,
                           1,
                           "single_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 1 verification");
      goto Fail;
    }
    log_info("  epoch 1: OK");
  }

  transpose_stream_destroy(&s);
  mem_writer_free(&mw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  mem_writer_free(&mw);
  log_error("  FAIL");
  return 1;
}

// Test: feed data in small chunks (e.g., 7 elements at a time)
// to exercise buffer-fill + dispatch + epoch-crossing logic.
static int
test_stream_chunked_append(void)
{
  log_info("=== test_stream_chunked_append ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 },
    { .size = 4, .tile_size = 2 },
    { .size = 6, .tile_size = 3 },
  };

  const size_t pool_bytes = 4 * 12 * sizeof(uint16_t);
  struct mem_writer mw = mem_writer_new(2 * pool_bytes);
  CHECK(Fail0, mw.buf);

  // Small buffer: 10 elements worth
  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 10 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .sink = &mw.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  const int total = 96;
  uint16_t src[96];
  for (int i = 0; i < total; ++i)
    src[i] = (uint16_t)i;

  // Feed in chunks of 7 elements
  const int chunk_elements = 7;

  for (int off = 0; off < total; off += chunk_elements) {
    int n = chunk_elements;
    if (off + n > total)
      n = total - off;

    struct slice input = { .beg = src + off, .end = src + off + n };
    struct writer_result r = writer_append(&s.writer, input);
    CHECK(Fail, r.error == 0);
  }

  // Epoch 0 was flushed during append.
  CHECK(Fail, mw.cursor >= pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(0,
                                             s.epoch_elements,
                                             s.slot_count,
                                             s.tile_elements,
                                             s.lifted_rank,
                                             s.lifted_shape,
                                             s.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)mw.buf,
                           expected,
                           s.slot_count * s.tile_elements,
                           0,
                           "chunked_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 0 verification");
      goto Fail;
    }
    log_info("  epoch 0: OK");
  }

  // Flush remaining data (epoch 1)
  {
    struct writer_result r = writer_flush(&s.writer);
    CHECK(Fail, r.error == 0);
  }

  CHECK(Fail, mw.cursor == 2 * pool_bytes);
  {
    uint16_t* expected = make_expected_tiles(s.epoch_elements,
                                             s.epoch_elements,
                                             s.slot_count,
                                             s.tile_elements,
                                             s.lifted_rank,
                                             s.lifted_shape,
                                             s.lifted_strides);
    CHECK(Fail, expected);
    int err = verify_tiles((const uint16_t*)(mw.buf + pool_bytes),
                           expected,
                           s.slot_count * s.tile_elements,
                           1,
                           "chunked_append");
    free(expected);
    if (err) {
      log_error("  FAIL: epoch 1 verification");
      goto Fail;
    }
    log_info("  epoch 1: OK");
  }

  transpose_stream_destroy(&s);
  mem_writer_free(&mw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  mem_writer_free(&mw);
  log_error("  FAIL");
  return 1;
}

// --- Compressed tile collector for roundtrip test ---

struct comp_tile_collector
{
  struct tile_writer base;
  // Flat buffer storing all compressed tile data across epochs
  uint8_t* buf;
  size_t* sizes; // compressed size of each tile (all epochs)
  size_t capacity;
  size_t count;     // total tiles received so far
  size_t max_count; // max tiles we can store
  size_t max_chunk; // max compressed bytes per tile
};

static int
comp_collector_append(struct tile_writer* self,
                      const void* const* tiles,
                      const size_t* sizes,
                      size_t count)
{
  struct comp_tile_collector* c = (struct comp_tile_collector*)self;
  if (c->count + count > c->max_count) {
    log_error("comp_collector: overflow");
    return 1;
  }
  for (size_t i = 0; i < count; ++i) {
    size_t idx = c->count + i;
    memcpy(c->buf + idx * c->max_chunk, tiles[i], sizes[i]);
    c->sizes[idx] = sizes[i];
  }
  c->count += count;
  return 0;
}

static int
comp_collector_flush(struct tile_writer* self)
{
  (void)self;
  return 0;
}

static struct comp_tile_collector
comp_collector_new(size_t max_tiles, size_t max_chunk_bytes)
{
  struct comp_tile_collector c = {
    .base = { .append = comp_collector_append, .flush = comp_collector_flush },
    .buf = (uint8_t*)malloc(max_tiles * max_chunk_bytes),
    .sizes = (size_t*)calloc(max_tiles, sizeof(size_t)),
    .max_count = max_tiles,
    .max_chunk = max_chunk_bytes,
  };
  return c;
}

static void
comp_collector_free(struct comp_tile_collector* c)
{
  free(c->buf);
  free(c->sizes);
  *c = (struct comp_tile_collector){ 0 };
}

// Test: compressed roundtrip — compress tiles with nvcomp, decompress with
// libzstd, verify contents match expected tile pool.
static int
test_stream_compressed_roundtrip(void)
{
  log_info("=== test_stream_compressed_roundtrip ===");

  const struct dimension dims[] = {
    { .size = 4, .tile_size = 2 },
    { .size = 4, .tile_size = 2 },
    { .size = 6, .tile_size = 3 },
  };

  // 2 epochs, 4 tiles/epoch = 8 tiles total
  const size_t total_tiles = 8;

  // Create stream first to query max_comp_chunk_bytes, then build collector.
  // We need a two-pass approach or use a generous upper bound.
  // Use a generous upper bound for the collector.
  const size_t max_chunk_est = 4096; // generous for 24-byte tiles
  struct comp_tile_collector cc =
    comp_collector_new(total_tiles, max_chunk_est);
  CHECK(Fail0, cc.buf);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 96 * sizeof(uint16_t),
    .bytes_per_element = sizeof(uint16_t),
    .rank = 3,
    .dimensions = dims,
    .compress = 1,
    .compressed_sink = &cc.base,
  };

  struct transpose_stream s;
  CHECK(Fail0, transpose_stream_create(&config, &s) == 0);

  log_info("  tile_elements=%lu  tile_stride=%lu  slot_count=%lu  "
           "epoch_elements=%lu",
           (unsigned long)s.tile_elements,
           (unsigned long)s.tile_stride,
           (unsigned long)s.slot_count,
           (unsigned long)s.epoch_elements);
  log_info("  max_comp_chunk_bytes=%zu  tile_pool_bytes=%zu",
           s.max_comp_chunk_bytes,
           s.tile_pool_bytes);

  CHECK(Fail, s.tile_elements == 12);
  CHECK(Fail, s.slot_count == 4);
  CHECK(Fail, s.epoch_elements == 48);

  // Fill source with sequential u16 values
  uint16_t src[96];
  for (size_t i = 0; i < countof(src); ++i)
    src[i] = (uint16_t)i;

  // Append all data
  struct slice input = { .beg = src, .end = src + countof(src) };
  struct writer_result r = writer_append(&s.writer, input);
  CHECK(Fail, r.error == 0);

  // Epoch 0 should have been delivered
  CHECK(Fail, cc.count >= 4);

  r = writer_flush(&s.writer);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, cc.count == total_tiles);

  // Decompress each tile and verify against expected
  const size_t tile_bytes = s.tile_stride * sizeof(uint16_t);

  for (int epoch = 0; epoch < 2; ++epoch) {
    uint16_t* expected = make_expected_tiles(
      (uint64_t)epoch * s.epoch_elements,
      s.epoch_elements,
      s.slot_count,
      s.tile_elements,
      s.lifted_rank,
      s.lifted_shape,
      s.lifted_strides);
    CHECK(Fail, expected);

    int err = 0;
    for (uint64_t t = 0; t < s.slot_count; ++t) {
      size_t idx = (size_t)epoch * s.slot_count + t;
      const uint8_t* comp_data = cc.buf + idx * cc.max_chunk;
      size_t comp_size = cc.sizes[idx];

      CHECK(Fail, comp_size > 0);

      // Decompress with libzstd
      uint8_t* decomp = (uint8_t*)calloc(1, tile_bytes);
      CHECK(Fail, decomp);

      size_t result = ZSTD_decompress(decomp, tile_bytes, comp_data, comp_size);
      if (ZSTD_isError(result)) {
        log_error("  ZSTD_decompress failed for tile %lu epoch %d: %s",
                  (unsigned long)t,
                  epoch,
                  ZSTD_getErrorName(result));
        free(decomp);
        free(expected);
        goto Fail;
      }
      CHECK(Fail, result == tile_bytes);

      // Compare the first tile_elements of the decompressed data against
      // the expected tile pool at offset t * tile_stride.
      // With stride padding, tile data starts at t * tile_stride elements
      // in the expected pool (which uses tile_elements stride since
      // make_expected_tiles doesn't account for stride padding).
      // Actually, make_expected_tiles uses the lifted strides which already
      // incorporate tile_stride, so the offset is t * tile_elements in the
      // expected array, but the decompressed data uses tile_stride layout.
      //
      // The decompressed tile has tile_stride elements. The expected pool
      // is slot_count * tile_elements with no padding. We need to compare
      // the first tile_elements of the decompressed data against the
      // expected pool at offset t * tile_elements.
      const uint16_t* decomp_u16 = (const uint16_t*)decomp;
      const uint16_t* expected_tile = expected + t * s.tile_elements;
      for (uint64_t e = 0; e < s.tile_elements; ++e) {
        if (decomp_u16[e] != expected_tile[e]) {
          log_error("  epoch %d tile %lu elem %lu: expected %u, got %u",
                    epoch,
                    (unsigned long)t,
                    (unsigned long)e,
                    expected_tile[e],
                    decomp_u16[e]);
          err = 1;
        }
      }
      free(decomp);
    }
    free(expected);
    if (err) {
      log_error("  FAIL: epoch %d verification", epoch);
      goto Fail;
    }
    log_info("  epoch %d: OK", epoch);
  }

  transpose_stream_destroy(&s);
  comp_collector_free(&cc);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
Fail0:
  comp_collector_free(&cc);
  log_error("  FAIL");
  return 1;
}

int
main(int ac, char* av[])
{
  (void)ac;
  (void)av;

  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_stream_single_append();
  log_info("");
  ecode |= test_stream_chunked_append();
  log_info("");
  ecode |= test_stream_compressed_roundtrip();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  if (ctx)
    cuCtxDestroy(ctx);
  return 1;
}
