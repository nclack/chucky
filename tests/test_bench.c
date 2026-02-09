#include "compress.h"
#include "index.ops.util.h"
#include "log/log.h"
#include "platform.h"
#include "stream.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

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

// --- Deterministic source data ---
// Hash-based values: any element is reconstructable from its global index.

static uint16_t
source_value_at(size_t gi, size_t total)
{
  (void)total;
  // Mix of compressible and incompressible:
  //  First 1/3: linear ramp (very compressible)
  //  Middle 1/3: constant 42 (maximally compressible)
  //  Last 1/3: pseudo-random (incompressible)
  if (gi < total / 3)
    return (uint16_t)(gi);
  if (gi < 2 * total / 3)
    return 42;
  return (uint16_t)(gi ^ (gi >> 16));
}


// --- Minimal nvcomp round-trip test (no streaming infra) ---

static int
test_compress_roundtrip(void)
{
  log_info("=== test_compress_roundtrip ===");

  const size_t tile_bytes = 1048576; // 1 MiB
  const size_t num_tiles = 96;
  const size_t pool_bytes = num_tiles * tile_bytes;

  const size_t max_comp = compress_get_max_output_size(tile_bytes);
  const size_t comp_temp = compress_get_temp_size(num_tiles, tile_bytes);
  const size_t comp_pool = num_tiles * max_comp;

  log_info("  tile_bytes=%zu num_tiles=%zu max_comp=%zu comp_temp=%zu",
           tile_bytes, num_tiles, max_comp, comp_temp);

  // Host data
  uint16_t* h_data = (uint16_t*)malloc(pool_bytes);
  uint8_t* h_compressed = NULL;
  size_t* h_comp_sizes = NULL;
  uint8_t* decomp_buf = NULL;
  void* d_data = NULL;
  void* d_compressed = NULL;
  void* d_temp = NULL;
  size_t* d_comp_sizes = NULL;
  void** d_in_ptrs = NULL;
  void** d_out_ptrs = NULL;
  size_t* d_in_sizes = NULL;
  CUstream stream = 0;
  int ok = 0;

  CHECK(Fail, h_data);
  h_compressed = (uint8_t*)malloc(comp_pool);
  CHECK(Fail, h_compressed);
  h_comp_sizes = (size_t*)malloc(num_tiles * sizeof(size_t));
  CHECK(Fail, h_comp_sizes);
  decomp_buf = (uint8_t*)malloc(tile_bytes);
  CHECK(Fail, decomp_buf);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_data, pool_bytes));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_compressed, comp_pool));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_temp, comp_temp));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_comp_sizes, num_tiles * sizeof(size_t)));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_in_ptrs, num_tiles * sizeof(void*)));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_out_ptrs, num_tiles * sizeof(void*)));
  CU(Fail, cuMemAlloc((CUdeviceptr*)&d_in_sizes, num_tiles * sizeof(size_t)));

  // Set up pointer and size arrays
  {
    void** h_ptrs = (void**)malloc(num_tiles * sizeof(void*));
    size_t* h_sizes = (size_t*)malloc(num_tiles * sizeof(size_t));
    CHECK(Fail, h_ptrs && h_sizes);

    for (size_t i = 0; i < num_tiles; ++i) {
      h_ptrs[i] = (char*)d_data + i * tile_bytes;
      h_sizes[i] = tile_bytes;
    }
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_in_ptrs, h_ptrs, num_tiles * sizeof(void*)));
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_in_sizes, h_sizes, num_tiles * sizeof(size_t)));

    for (size_t i = 0; i < num_tiles; ++i)
      h_ptrs[i] = (char*)d_compressed + i * max_comp;
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_out_ptrs, h_ptrs, num_tiles * sizeof(void*)));

    free(h_ptrs);
    free(h_sizes);
  }

  // Run TWO rounds with different data
  for (int round = 0; round < 2; ++round) {
    // Fill host data with hash seeded by round
    const size_t elems = pool_bytes / sizeof(uint16_t);
    const size_t gi_offset = (size_t)round * elems;
    for (size_t i = 0; i < elems; ++i)
      h_data[i] = source_value_at(gi_offset + i, 2 * elems);

    // H2D
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)d_data, h_data, pool_bytes));

    // Compress
    CHECK(Fail,
          compress_batch_async(
            (const void* const*)d_in_ptrs,
            d_in_sizes,
            tile_bytes,
            num_tiles,
            d_temp,
            comp_temp,
            (void* const*)d_out_ptrs,
            d_comp_sizes,
            stream) == 0);

    // Wait for compress to finish, then D2H
    CU(Fail, cuStreamSynchronize(stream));
    CU(Fail, cuMemcpyDtoH(h_compressed, (CUdeviceptr)d_compressed, comp_pool));
    CU(Fail, cuMemcpyDtoH(h_comp_sizes, (CUdeviceptr)d_comp_sizes, num_tiles * sizeof(size_t)));

    // Verify: decompress each tile and compare
    int round_errors = 0;
    for (size_t t = 0; t < num_tiles; ++t) {
      const uint8_t* comp_tile = h_compressed + t * max_comp;
      size_t result = ZSTD_decompress(decomp_buf, tile_bytes, comp_tile, h_comp_sizes[t]);
      if (ZSTD_isError(result)) {
        log_error("  round %d tile %zu: ZSTD_decompress failed: %s",
                  round, t, ZSTD_getErrorName(result));
        round_errors++;
        continue;
      }
      if (result != tile_bytes) {
        log_error("  round %d tile %zu: size mismatch: expected %zu got %zu",
                  round, t, tile_bytes, result);
        round_errors++;
        continue;
      }

      const uint16_t* actual = (const uint16_t*)decomp_buf;
      const uint16_t* expected = h_data + t * (tile_bytes / sizeof(uint16_t));
      int mismatch = 0;
      for (size_t e = 0; e < tile_bytes / sizeof(uint16_t); ++e) {
        if (actual[e] != expected[e]) {
          if (mismatch == 0)
            log_error("  round %d tile %zu elem %zu: expected %u got %u",
                      round, t, e, expected[e], actual[e]);
          mismatch++;
        }
      }
      if (mismatch > 0) {
        log_error("  round %d tile %zu: %d mismatches", round, t, mismatch);
        round_errors++;
      }
    }

    if (round_errors > 0) {
      log_error("  round %d: %d tile errors", round, round_errors);
      goto Fail;
    }
    log_info("  round %d: OK", round);
  }

  ok = 1;

Fail:
  free(h_data);
  free(h_compressed);
  free(h_comp_sizes);
  free(decomp_buf);
  if (d_data) cuMemFree((CUdeviceptr)d_data);
  if (d_compressed) cuMemFree((CUdeviceptr)d_compressed);
  if (d_temp) cuMemFree((CUdeviceptr)d_temp);
  if (d_comp_sizes) cuMemFree((CUdeviceptr)d_comp_sizes);
  if (d_in_ptrs) cuMemFree((CUdeviceptr)d_in_ptrs);
  if (d_out_ptrs) cuMemFree((CUdeviceptr)d_out_ptrs);
  if (d_in_sizes) cuMemFree((CUdeviceptr)d_in_sizes);
  if (stream) cuStreamDestroy(stream);

  if (ok) {
    log_info("  PASS");
    return 0;
  }
  log_error("  FAIL");
  return 1;
}

static void
fill_chunk(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  for (size_t i = 0; i < count; ++i)
    buf[i] = source_value_at(offset + i, total);
}

// --- Bench tile_writer: sparse inline verification ---

struct bench_tile_writer
{
  struct tile_writer base;

  size_t total_compressed;
  size_t total_tiles;
  size_t epochs_seen;

  int verify_interval;
  int verify_errors;
  int verify_count;

  // Sink timing (wall clock spent in this callback)
  double sink_ms;

  // Layout info (copied from stream)
  size_t total_elements;
  uint64_t epoch_elements;
  uint64_t slot_count;
  uint64_t tile_stride;
  uint64_t tile_elements;
  uint8_t lifted_rank;
  uint64_t lifted_shape[2 * MAX_RANK];
  int64_t lifted_strides[2 * MAX_RANK];
  size_t tile_bytes;

  uint8_t* decomp_buf;
  struct platform_clock clock;
};

static int
bench_tile_writer_append(struct tile_writer* self,
                         const void* const* tiles,
                         const size_t* sizes,
                         size_t count)
{
  struct bench_tile_writer* w = (struct bench_tile_writer*)self;
  platform_toc(&w->clock); // start sink timing

  for (size_t i = 0; i < count; ++i)
    w->total_compressed += sizes[i];
  w->total_tiles += count;

  int do_verify =
    w->verify_interval > 0 && (w->epochs_seen % w->verify_interval) == 0;

  if (do_verify) {
    const uint64_t epoch_start = w->epochs_seen * w->epoch_elements;
    const uint64_t pool_size = w->slot_count * w->tile_stride;

    // Check compressed sizes
    size_t max_comp = 0;
    for (size_t i = 0; i < count; ++i) {
      if (sizes[i] > max_comp)
        max_comp = sizes[i];
    }
    log_info("  epoch %zu: max_comp_size=%zu tile_bytes=%zu",
             w->epochs_seen,
             max_comp,
             w->tile_bytes);

    uint16_t* expected = (uint16_t*)calloc(pool_size, sizeof(uint16_t));
    if (!expected) {
      log_error("bench: failed to allocate expected pool");
      w->verify_errors++;
      goto skip;
    }

    uint64_t epoch_elems = w->epoch_elements;
    if (epoch_start + epoch_elems > w->total_elements)
      epoch_elems = w->total_elements - epoch_start;

    for (uint64_t i = 0; i < epoch_elems; ++i) {
      uint64_t idx = epoch_start + i;
      uint64_t off =
        ravel(w->lifted_rank, w->lifted_shape, w->lifted_strides, idx);
      expected[off] = source_value_at(idx, w->total_elements);
    }

    int mismatch_total = 0;
    int bad_tile_count = 0;
    for (size_t t = 0; t < count; ++t) {
      size_t result =
        ZSTD_decompress(w->decomp_buf, w->tile_bytes, tiles[t], sizes[t]);
      if (ZSTD_isError(result)) {
        log_error("bench: ZSTD_decompress failed epoch %zu tile %zu: %s",
                  w->epochs_seen,
                  t,
                  ZSTD_getErrorName(result));
        w->verify_errors++;
        free(expected);
        goto skip;
      }

      const uint16_t* actual = (const uint16_t*)w->decomp_buf;
      const uint16_t* exp_tile = expected + t * w->tile_stride;
      int tile_mismatch = 0;
      for (uint64_t e = 0; e < w->tile_elements; ++e) {
        if (actual[e] != exp_tile[e]) {
          if (mismatch_total < 3)
            log_error(
              "  epoch %zu tile %zu elem %lu: expected %u, got %u (diff=%d)",
              w->epochs_seen,
              t,
              (unsigned long)e,
              exp_tile[e],
              actual[e],
              (int)actual[e] - (int)exp_tile[e]);
          mismatch_total++;
          tile_mismatch++;
        }
      }
      if (tile_mismatch > 0)
        bad_tile_count++;
    }
    free(expected);

    if (mismatch_total > 0) {
      log_error("  epoch %zu: %d mismatches in %d/%zu tiles (epoch_start=%lu)",
                w->epochs_seen,
                mismatch_total,
                bad_tile_count,
                count,
                (unsigned long)epoch_start);
      w->verify_errors++;
    } else {
      log_info("  epoch %zu: OK", w->epochs_seen);
    }
    w->verify_count++;
  }

skip:
  w->sink_ms += (double)platform_toc(&w->clock) * 1000.0;
  w->epochs_seen++;
  return 0;
}

static int
bench_tile_writer_flush(struct tile_writer* self)
{
  (void)self;
  return 0;
}

static struct bench_tile_writer
bench_tile_writer_new(const struct transpose_stream* s,
                      size_t total_elements,
                      int verify_interval)
{
  struct bench_tile_writer w = {
    .base = { .append = bench_tile_writer_append,
              .flush = bench_tile_writer_flush },
    .verify_interval = verify_interval,
    .total_elements = total_elements,
    .epoch_elements = s->epoch_elements,
    .slot_count = s->slot_count,
    .tile_stride = s->tile_stride,
    .tile_elements = s->tile_elements,
    .lifted_rank = s->lifted_rank,
    .tile_bytes = s->tile_stride * s->config.bytes_per_element,
    .decomp_buf =
      (uint8_t*)malloc(s->tile_stride * s->config.bytes_per_element),
  };
  memcpy(w.lifted_shape, s->lifted_shape, s->lifted_rank * sizeof(uint64_t));
  memcpy(
    w.lifted_strides, s->lifted_strides, s->lifted_rank * sizeof(int64_t));
  return w;
}

static void
bench_tile_writer_free(struct bench_tile_writer* w)
{
  free(w->decomp_buf);
  *w = (struct bench_tile_writer){ 0 };
}

// --- Small compressed test: same shape, same hash, 10 epochs ---

static int
test_compressed_small(void)
{
  log_info("=== test_compressed_small ===");

  const struct dimension dims[] = {
    { .size = 40, .tile_size = 4 },
    { .size = 2048, .tile_size = 256 },
    { .size = 2048, .tile_size = 512 },
    { .size = 3, .tile_size = 1 },
  };
  const size_t total_elements = (size_t)40 * 2048 * 2048 * 3;

  struct transpose_stream s = { 0 };
  struct bench_tile_writer btw = { 0 };

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .compressed_sink = NULL,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);

  const size_t num_epochs =
    (total_elements + s.epoch_elements - 1) / s.epoch_elements;
  log_info("  total: %zu elements, %zu epochs",
           total_elements,
           num_epochs);

  btw = bench_tile_writer_new(&s, total_elements, 1);
  CHECK(Fail, btw.decomp_buf);
  s.config.compressed_sink = &btw.base;

  const size_t chunk_elements = 4 * 1024 * 1024;
  uint16_t* chunk = (uint16_t*)malloc(chunk_elements * sizeof(uint16_t));
  CHECK(Fail, chunk);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;
    fill_chunk(chunk, n, offset, total_elements);
    struct slice input = { .beg = chunk, .end = chunk + n };
    struct writer_result r = writer_append_wait(&s.writer, input);
    if (r.error) {
      free(chunk);
      goto Fail;
    }
  }
  {
    struct writer_result r = writer_flush(&s.writer);
    if (r.error) {
      free(chunk);
      goto Fail;
    }
  }
  free(chunk);

  if (btw.verify_errors > 0) {
    log_error("  FAIL: %d errors across %d checked epochs",
              btw.verify_errors,
              btw.verify_count);
    goto Fail;
  }
  log_info("  verification: OK (%d/%zu epochs checked)",
           btw.verify_count,
           num_epochs);

  transpose_stream_destroy(&s);
  bench_tile_writer_free(&btw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  bench_tile_writer_free(&btw);
  log_error("  FAIL");
  return 1;
}

// --- Throughput helpers ---

static double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Benchmark ---

static int
test_bench(void)
{
  log_info("=== test_bench ===");

  // Shape: (2132, 2048, 2048, 3) with tiles (4, 256, 512, 1)
  //   tile: 4*256*512*1 = 524,288 elements = 1 MiB
  //   slot_count: 8*4*3 = 96
  //   epoch pool: 96 MiB
  //   epochs: 533
  //   total: ~50 GiB
  const struct dimension dims[] = {
    { .size = 2132, .tile_size = 4 },
    { .size = 2048, .tile_size = 256 },
    { .size = 2048, .tile_size = 512 },
    { .size = 3, .tile_size = 1 },
  };

  const size_t total_elements = (size_t)2132 * 2048 * 2048 * 3;
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct transpose_stream s = { 0 };
  struct bench_tile_writer btw = { 0 };

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20, // 8 MiB staging buffer
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .compressed_sink = NULL,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);

  const size_t num_epochs =
    (total_elements + s.epoch_elements - 1) / s.epoch_elements;

  log_info("  shape:       (%u, %u, %u, %u)  tiles: (%u, %u, %u, %u)",
           (unsigned)dims[0].size,
           (unsigned)dims[1].size,
           (unsigned)dims[2].size,
           (unsigned)dims[3].size,
           (unsigned)dims[0].tile_size,
           (unsigned)dims[1].tile_size,
           (unsigned)dims[2].tile_size,
           (unsigned)dims[3].tile_size);
  log_info("  total:       %.2f GiB (%zu elements, %zu epochs)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements,
           num_epochs);
  log_info("  tile:        %lu elements = %lu KiB  (stride=%lu)",
           (unsigned long)s.tile_elements,
           (unsigned long)(s.tile_stride * sizeof(uint16_t) / 1024),
           (unsigned long)s.tile_stride);
  log_info("  epoch:       %lu slots, %lu MiB pool",
           (unsigned long)s.slot_count,
           (unsigned long)(s.tile_pool_bytes / (1024 * 1024)));
  log_info("  compress:    max_chunk=%zu comp_pool=%zu MiB",
           s.max_comp_chunk_bytes,
           s.comp_pool_bytes / (1024 * 1024));

  // Verify frequently to find where drift starts
  int verify_interval = 50;
  btw = bench_tile_writer_new(&s, total_elements, verify_interval);
  CHECK(Fail, btw.decomp_buf);
  s.config.compressed_sink = &btw.base;

  // Chunk buffer for feeding
  const size_t chunk_elements = 32 * 1024 * 1024; // 32M elements = 64 MiB
  uint16_t* chunk = (uint16_t*)malloc(chunk_elements * sizeof(uint16_t));
  CHECK(Fail, chunk);

  // Run the pipeline, timing source fill separately
  struct platform_clock clock = { 0 };
  struct platform_clock fill_clock = { 0 };
  double fill_ms = 0;
  platform_toc(&clock);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;

    platform_toc(&fill_clock);
    fill_chunk(chunk, n, offset, total_elements);
    fill_ms += (double)platform_toc(&fill_clock) * 1000.0;

    struct slice input = { .beg = chunk, .end = chunk + n };
    struct writer_result r = writer_append_wait(&s.writer, input);
    if (r.error) {
      log_error("  append failed at offset %zu", offset);
      free(chunk);
      goto Fail;
    }
  }

  {
    struct writer_result r = writer_flush(&s.writer);
    if (r.error) {
      log_error("  flush failed");
      free(chunk);
      goto Fail;
    }
  }

  float wall_s = platform_toc(&clock);
  free(chunk);
  chunk = NULL;

  // Check cursor integrity
  if (s.cursor != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%zd)",
              total_elements,
              (size_t)s.cursor,
              (ssize_t)((int64_t)s.cursor - (int64_t)total_elements));
  }

  // Check verification
  if (btw.verify_errors > 0) {
    log_error("  FAIL: %d verification errors across %d checked epochs",
              btw.verify_errors,
              btw.verify_count);
    goto Fail;
  }
  log_info("  verification: OK (%d/%zu epochs checked)",
           btw.verify_count,
           num_epochs);

  // Report
  {
    const size_t total_decompressed = btw.total_tiles * btw.tile_bytes;
    const double comp_ratio =
      total_decompressed > 0
        ? (double)btw.total_compressed / (double)total_decompressed
        : 0.0;

    struct stream_metrics m = transpose_stream_get_metrics(&s);
    const double pool_bytes = (double)s.tile_pool_bytes;
    const double comp_pool = (double)s.comp_pool_bytes;
    const int N = m.epoch_count;

    log_info("");
    log_info("  --- Benchmark Results ---");
    log_info("  Input:        %.2f GiB (%zu elements)",
             (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
             total_elements);
    log_info("  Compressed:   %.2f GiB (ratio: %.3f)",
             (double)btw.total_compressed / (1024.0 * 1024.0 * 1024.0),
             comp_ratio);
    log_info("  Tiles:        %zu (%zu/epoch x %zu epochs)",
             btw.total_tiles,
             (size_t)s.slot_count,
             num_epochs);

    log_info("");
    log_info(
      "  %-12s %8s %8s %10s %10s", "GPU step", "avg GB/s", "best GB/s",
      "avg ms", "best ms");

    if (N > 0) {
      // H2D: data volume = tile_pool_bytes per epoch (input data scattered)
      {
        double avg_ms = m.h2d_ms / N;
        log_info("  %-12s %8.2f %8s %10.2f %10s",
                 "H2D",
                 gb_per_s(pool_bytes * N, m.h2d_ms),
                 "-",
                 avg_ms,
                 "-");
      }
      // scatter
      {
        double avg_ms = m.scatter_ms / N;
        log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
                 "scatter",
                 gb_per_s(pool_bytes * N, m.scatter_ms),
                 gb_per_s(pool_bytes, m.scatter_best_ms),
                 avg_ms,
                 m.scatter_best_ms);
      }
      // compress
      {
        double avg_ms = m.compress_ms / N;
        log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
                 "compress",
                 gb_per_s(pool_bytes * N, m.compress_ms),
                 gb_per_s(pool_bytes, m.compress_best_ms),
                 avg_ms,
                 m.compress_best_ms);
      }
      // D2H
      {
        double avg_ms = m.d2h_ms / N;
        log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
                 "D2H",
                 gb_per_s(comp_pool * N, m.d2h_ms),
                 gb_per_s(comp_pool, m.d2h_best_ms),
                 avg_ms,
                 m.d2h_best_ms);
      }
    }

    log_info("");
    log_info("  Source fill:   %.2f GB/s  (%.1f ms total)",
             gb_per_s((double)total_bytes, fill_ms),
             fill_ms);
    log_info("  Sink:          %.1f ms total  (%d epochs verified)",
             btw.sink_ms,
             btw.verify_count);

    double throughput_gib =
      wall_s > 0
        ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
        : 0.0;
    log_info("");
    log_info("  Wall time:     %.3f s", wall_s);
    log_info("  Throughput:    %.2f GiB/s", throughput_gib);
  }

  transpose_stream_destroy(&s);
  bench_tile_writer_free(&btw);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  bench_tile_writer_free(&btw);
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

  ecode |= test_compress_roundtrip();
  if (ecode)
    goto Done;
  ecode |= test_compressed_small();
  if (ecode)
    goto Done;
  ecode |= test_bench();

Done:
  cuCtxDestroy(ctx);
  return ecode;

Fail:
  if (ctx)
    cuCtxDestroy(ctx);
  return 1;
}
