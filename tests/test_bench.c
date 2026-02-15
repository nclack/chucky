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

// Deterministic source data
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

static void
fill_chunk(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  for (size_t i = 0; i < count; ++i)
    buf[i] = source_value_at(offset + i, total);
}

// --- Throughput helpers ---

static double
gb_per_s(double bytes, double ms)
{
  if (ms <= 0)
    return 0;
  return (bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// --- Discard shard_sink for benchmarks ---

struct discard_shard_writer
{
  struct shard_writer base;
  struct discard_shard_sink* parent;
};

struct discard_shard_sink
{
  struct shard_sink base;
  struct discard_shard_writer writer;
  size_t total_bytes;
  size_t shards_finalized;
  struct stream_metric sink;
  struct platform_clock clock;
};

static int
discard_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  (void)offset;
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  platform_toc(&w->parent->clock);
  w->parent->total_bytes += (size_t)((const char*)end - (const char*)beg);
  float ms = (float)(platform_toc(&w->parent->clock) * 1000.0);
  w->parent->sink.ms += ms;
  w->parent->sink.count++;
  if (ms < w->parent->sink.best_ms)
    w->parent->sink.best_ms = ms;
  return 0;
}

static int
discard_shard_finalize(struct shard_writer* self)
{
  struct discard_shard_writer* w = (struct discard_shard_writer*)self;
  w->parent->shards_finalized++;
  return 0;
}

static struct shard_writer*
discard_shard_open(struct shard_sink* self, uint64_t shard_index)
{
  (void)shard_index;
  struct discard_shard_sink* s = (struct discard_shard_sink*)self;
  return &s->writer.base;
}

static void
discard_shard_sink_init(struct discard_shard_sink* s)
{
  *s = (struct discard_shard_sink){
    .base = { .open = discard_shard_open },
    .sink = { .name = "Sink", .best_ms = 1e30f },
  };
  s->writer = (struct discard_shard_writer){
    .base = { .write = discard_shard_write, .finalize = discard_shard_finalize },
    .parent = s,
  };
}

// --- Small compressed+shard smoke test ---

static int
test_compressed_small(void)
{
  log_info("=== test_compressed_small ===");

  const struct dimension dims[] = {
    { .size = 40, .tile_size = 4, .tiles_per_shard = 5 },
    { .size = 2048, .tile_size = 256, .tiles_per_shard = 4 },
    { .size = 2048, .tile_size = 512, .tiles_per_shard = 2 },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3 },
  };
  const size_t total_elements = (size_t)40 * 2048 * 2048 * 3;

  struct transpose_stream s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);

  const size_t num_epochs =
    (total_elements + s.layout.epoch_elements - 1) / s.layout.epoch_elements;
  log_info("  total: %zu elements, %zu epochs", total_elements, num_epochs);

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

  CHECK(Fail, s.cursor == total_elements);
  log_info("  shards finalized: %zu, total bytes: %zu",
           dss.shards_finalized,
           dss.total_bytes);

  transpose_stream_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  log_error("  FAIL");
  return 1;
}

// --- Report helpers ---

static void
print_metric_row(const struct stream_metric* m, double bytes_per_unit)
{
  if (m->count <= 0)
    return;
  const int N = m->count;
  double avg_ms = (double)m->ms / N;
  double avg_gbs = gb_per_s(bytes_per_unit * N, (double)m->ms);
  int has_best = m->best_ms < 1e29f;

  if (has_best) {
    double best_gbs = gb_per_s(bytes_per_unit, (double)m->best_ms);
    log_info("  %-12s %8.2f %8.2f %10.2f %10.2f",
             m->name,
             avg_gbs,
             best_gbs,
             avg_ms,
             (double)m->best_ms);
  } else {
    log_info(
      "  %-12s %8.2f %8s %10.2f %10s", m->name, avg_gbs, "-", avg_ms, "-");
  }
}

static void
print_bench_report(const struct transpose_stream* s,
                   const struct discard_shard_sink* dss,
                   const struct stream_metric* src,
                   size_t total_bytes,
                   size_t total_elements,
                   size_t chunk_elements,
                   float wall_s)
{
  struct stream_metrics m = transpose_stream_get_metrics(s);
  const size_t tile_bytes =
    s->layout.tile_stride * s->config.bytes_per_element;
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;
  const size_t total_tiles = num_epochs * s->layout.slot_count;
  const size_t total_decompressed = total_tiles * tile_bytes;
  const double comp_ratio =
    total_decompressed > 0
      ? (double)dss->total_bytes / (double)total_decompressed
      : 0.0;

  const double pool_bytes = (double)s->layout.tile_pool_bytes;
  const double comp_pool = (double)s->comp.comp_pool_bytes;
  const double chunk_bytes = (double)chunk_elements * sizeof(uint16_t);

  log_info("");
  log_info("  --- Benchmark Results ---");
  log_info("  Input:        %.2f GiB (%zu elements)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements);
  log_info("  Compressed:   %.2f GiB (ratio: %.3f)",
           (double)dss->total_bytes / (1024.0 * 1024.0 * 1024.0),
           comp_ratio);
  log_info("  Tiles:        %zu (%zu/epoch x %zu epochs)",
           total_tiles,
           (size_t)s->layout.slot_count,
           num_epochs);

  log_info("");
  log_info("  %-12s %8s %8s %10s %10s",
           "Stage",
           "avg GB/s",
           "best GB/s",
           "avg ms",
           "best ms");

  double h2d_per_dispatch =
    m.h2d.count > 0 ? (double)total_bytes / m.h2d.count : 0;
  print_metric_row(src, chunk_bytes);
  double scatter_per_dispatch =
    m.scatter.count > 0 ? (double)total_bytes / m.scatter.count : 0;
  print_metric_row(&m.h2d, h2d_per_dispatch);
  print_metric_row(&m.scatter, scatter_per_dispatch);
  print_metric_row(&m.compress, pool_bytes);
  print_metric_row(&m.aggregate, comp_pool);
  print_metric_row(&m.d2h, comp_pool);
  print_metric_row(&dss->sink, comp_pool);

  double throughput_gib =
    wall_s > 0 ? ((double)total_bytes / (1024.0 * 1024.0 * 1024.0)) / wall_s
               : 0.0;
  log_info("");
  log_info("  Wall time:     %.3f s", wall_s);
  log_info("  Throughput:    %.2f GiB/s", throughput_gib);
}

// --- Benchmark ---

static int
test_bench(void)
{
  log_info("=== test_bench ===");

  // Shape: (2132, 2048, 2048, 3) with tiles (32, 128, 128, 1)
  //   tile: 32*128*128*1 = 0.5 Mi elements = 1 MiB
  //   slot_count: 16*16*3 = 768
  //   epoch pool: 768 MiB
  //   epochs: 66
  //   total: ~50 GiB
  const struct dimension dims[] = {
    { .size = 2132, .tile_size = 32, .tiles_per_shard = 16 },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4 },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4 },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3 },
  };

  const size_t total_elements = (size_t)2132 * 2048 * 2048 * 3;
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct transpose_stream s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20, // 128 MiB staging buffer
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, transpose_stream_create(&config, &s) == 0);

  const size_t num_epochs =
    (total_elements + s.layout.epoch_elements - 1) / s.layout.epoch_elements;

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
           (unsigned long)s.layout.tile_elements,
           (unsigned long)(s.layout.tile_stride * sizeof(uint16_t) / 1024),
           (unsigned long)s.layout.tile_stride);
  log_info("  epoch:       %lu slots, %lu MiB pool",
           (unsigned long)s.layout.slot_count,
           (unsigned long)(s.layout.tile_pool_bytes / (1024 * 1024)));
  log_info("  compress:    max_chunk=%zu comp_pool=%zu MiB",
           s.comp.max_comp_chunk_bytes,
           s.comp.comp_pool_bytes / (1024 * 1024));

  // Pre-generate one source chunk (reused each iteration)
  const size_t chunk_elements = 32 * 1024 * 1024; // 32M elements = 64 MiB
  uint16_t* chunk = (uint16_t*)malloc(chunk_elements * sizeof(uint16_t));
  CHECK(Fail, chunk);
  fill_chunk(chunk, chunk_elements, 0, total_elements);

  // Run the pipeline
  struct platform_clock clock = { 0 };
  struct platform_clock src_clock = { 0 };
  struct stream_metric src = { .name = "Source", .best_ms = 1e30f };
  platform_toc(&clock);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;

    struct slice input = { .beg = chunk, .end = chunk + n };
    platform_toc(&src_clock);
    struct writer_result r = writer_append_wait(&s.writer, input);
    float ms = (float)(platform_toc(&src_clock) * 1000.0);
    src.ms += ms;
    src.count++;
    if (ms < src.best_ms)
      src.best_ms = ms;
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
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  print_bench_report(
    &s, &dss, &src, total_bytes, total_elements, chunk_elements, wall_s);

  transpose_stream_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
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
