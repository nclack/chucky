#include "index.ops.util.h"
#include "log/log.h"
#include "stream.h"
#include "writer.mem.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  if (ctx)
    cuCtxDestroy(ctx);
  return 1;
}
