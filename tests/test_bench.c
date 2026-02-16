#define _USE_MATH_DEFINES
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"
#include "test_platform.h"
#include "zarr_sink.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// Deterministic source data — mix of compressibility levels.
//   First 1/3:  Gaussian noise in 12-bit range (partially compressible)
//   Middle 1/3: constant 42 (maximally compressible)
//   Last 1/3:   uniform random uint16 (incompressible)

static uint64_t
splitmix64(uint64_t* state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static double
splitmix64_uniform(uint64_t* state)
{
  return (double)(splitmix64(state) >> 11) * 0x1.0p-53;
}

static inline uint64_t
hash_cell(int gx, int gy, int gt, uint64_t seed)
{
  uint64_t h = seed;
  h ^= (uint64_t)(unsigned)gx * 0x9e3779b97f4a7c15ULL;
  h ^= (uint64_t)(unsigned)gy * 0xbf58476d1ce4e5b9ULL;
  h ^= (uint64_t)(unsigned)gt * 0x94d049bb133111ebULL;
  h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
  h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
  return h ^ (h >> 31);
}

// Synthetic fluorescence microscopy: sparse bright blobs on dark background.
// Hash-based, all integer math, one hash per pixel.
//   Grid: 32x32 px cells, ~3% density, radius 12 px, 64-frame temporal blocks.
//   RGB variation: R-dominant, G-dominant, B-dominant, or white per blob.
static void
fill_chunk_visual(uint16_t* buf, size_t count, size_t offset)
{
  // shape: (T, 2048, 2048, 3), row-major, c fastest
  const int W = 2048, C = 3;
  const size_t row = (size_t)C;
  const size_t plane = (size_t)W * row;
  const size_t vol = (size_t)W * plane;
  const int RADIUS_SQ = 12 * 12; // 144

  for (size_t i = 0; i < count; ++i) {
    size_t gi = offset + i;
    int c = (int)(gi % row);
    int x = (int)((gi / row) % W);
    int y = (int)((gi / plane) % W);
    int t = (int)(gi / vol);

    int gx = x >> 5; // / 32
    int gy = y >> 5;
    int gt = t >> 6; // / 64

    uint64_t h = hash_cell(gx, gy, gt, 0xdeadbeefULL);

    // ~3% density: low 5 bits == 0 → 1/32 ≈ 3.1%
    if ((h & 0x1F) != 0) {
      buf[i] = 0;
      continue;
    }

    // Cell center with ±4 wobble (stays within grid cell for radius 12)
    int cx = (gx << 5) + 16 + (int)((h >> 5) & 7) - 4;
    int cy = (gy << 5) + 16 + (int)((h >> 8) & 7) - 4;
    int dx = x - cx, dy = y - cy;
    int d2 = dx * dx + dy * dy;

    if (d2 >= RADIUS_SQ) {
      buf[i] = 0;
      continue;
    }

    int falloff = RADIUS_SQ - d2;                // [1, 144]
    int base = 8000 + (int)((h >> 11) & 0x1FFF); // 8000–16191
    int val = (base * falloff) / RADIUS_SQ;

    // Channel color: bits 24-25 select type (0=R, 1=G, 2=B, 3=white)
    int ctype = (int)((h >> 24) & 3);
    if (ctype < 3 && ctype != c)
      val >>= 2; // dim non-primary channels

    buf[i] = (uint16_t)val;
  }
}

static void
fill_chunk(uint16_t* buf, size_t count, size_t offset, size_t total)
{
  const size_t third1 = total / 3;
  const size_t third2 = 2 * total / 3;
  uint64_t rng = offset * 0x9e3779b97f4a7c15ULL + 1;

  for (size_t i = 0; i < count; ++i) {
    size_t gi = offset + i;
    if (gi < third1) {
      // Gaussian via Box-Muller, mu=2048 sigma=512, clamped to [0,4095]
      double u1 = splitmix64_uniform(&rng);
      double u2 = splitmix64_uniform(&rng);
      if (u1 < 1e-15)
        u1 = 1e-15;
      double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
      int val = (int)(2048.0 + 512.0 * z);
      if (val < 0)
        val = 0;
      if (val > 4095)
        val = 4095;
      buf[i] = (uint16_t)val;
    } else if (gi < third2) {
      buf[i] = 42;
    } else {
      buf[i] = (uint16_t)splitmix64(&rng);
    }
  }
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
  accumulate_metric_ms(&w->parent->sink,
                       (float)(platform_toc(&w->parent->clock) * 1000.0));
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
discard_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
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
    .base = { .write = discard_shard_write,
              .finalize = discard_shard_finalize },
    .parent = s,
  };
}

// --- Metering shard_sink wrapper ---
// Wraps any shard_sink and captures timing + byte counts.

struct metered_shard_writer
{
  struct shard_writer shard_writer;
  struct shard_writer* inner;
  struct metered_shard_sink* parent;
};

struct metered_shard_sink
{
  struct shard_sink shard_writer;
  struct shard_sink* inner;
  size_t total_bytes;
  size_t shards_finalized;
  struct stream_metric sink;
  struct platform_clock clock;
};

static int
metered_shard_write(struct shard_writer* self,
                    uint64_t offset,
                    const void* beg,
                    const void* end)
{
  struct metered_shard_writer* w = (struct metered_shard_writer*)self;
  platform_toc(&w->parent->clock);
  w->parent->total_bytes += (size_t)((const char*)end - (const char*)beg);
  int rc = w->inner->write(w->inner, offset, beg, end);
  accumulate_metric_ms(&w->parent->sink,
                       (float)(platform_toc(&w->parent->clock) * 1000.0));
  return rc;
}

static int
metered_shard_finalize(struct shard_writer* self)
{
  struct metered_shard_writer* w = (struct metered_shard_writer*)self;
  int rc = w->inner->finalize(w->inner);
  w->parent->shards_finalized++;
  free(w);
  return rc;
}

static struct shard_writer*
metered_shard_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  struct metered_shard_sink* s = (struct metered_shard_sink*)self;
  struct shard_writer* inner = s->inner->open(s->inner, level, shard_index);
  if (!inner)
    return NULL;
  struct metered_shard_writer* w = malloc(sizeof(*w));
  if (!w)
    return NULL;
  *w = (struct metered_shard_writer){
    .shard_writer = { .write = metered_shard_write,
                      .finalize = metered_shard_finalize },
    .inner = inner,
    .parent = s,
  };
  return &w->shard_writer;
}

static void
metered_shard_sink_init(struct metered_shard_sink* s, struct shard_sink* inner)
{
  *s = (struct metered_shard_sink){
    .shard_writer = { .open = metered_shard_open },
    .inner = inner,
    .sink = { .name = "Sink", .best_ms = 1e30f },
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

struct sink_stats
{
  size_t total_bytes;
  const struct stream_metric* sink;
};

static void
print_bench_report(const struct transpose_stream* s,
                   const struct sink_stats* ss,
                   size_t total_bytes,
                   size_t total_elements,
                   float wall_s)
{
  struct stream_metrics m = transpose_stream_get_metrics(s);
  const size_t tile_bytes = s->layout.tile_stride * s->config.bytes_per_element;
  const size_t num_epochs =
    (total_elements + s->layout.epoch_elements - 1) / s->layout.epoch_elements;
  const size_t total_tiles = num_epochs * s->layout.slot_count;
  const size_t total_decompressed = total_tiles * tile_bytes;
  const double comp_ratio =
    total_decompressed > 0
      ? (double)ss->total_bytes / (double)total_decompressed
      : 0.0;

  const double pool_bytes = (double)s->layout.tile_pool_bytes;
  const double comp_pool = (double)s->comp.comp_pool_bytes;

  log_info("");
  log_info("  --- Benchmark Results ---");
  log_info("  Input:        %.2f GiB (%zu elements)",
           (double)total_bytes / (1024.0 * 1024.0 * 1024.0),
           total_elements);
  log_info("  Compressed:   %.2f GiB (ratio: %.3f)",
           (double)ss->total_bytes / (1024.0 * 1024.0 * 1024.0),
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

  double memcpy_per =
    m.memcpy.count > 0 ? (double)total_bytes / m.memcpy.count : 0;
  print_metric_row(&m.memcpy, memcpy_per);
  double h2d_per_dispatch =
    m.h2d.count > 0 ? (double)total_bytes / m.h2d.count : 0;
  double scatter_per_dispatch =
    m.scatter.count > 0 ? (double)total_bytes / m.scatter.count : 0;
  print_metric_row(&m.h2d, h2d_per_dispatch);
  print_metric_row(&m.scatter, scatter_per_dispatch);
  print_metric_row(&m.compress, pool_bytes);
  double agg_per = m.aggregate.count > 0
                     ? m.aggregate.total_bytes / m.aggregate.count
                     : comp_pool;
  print_metric_row(&m.aggregate, agg_per);
  double d2h_per =
    m.d2h.count > 0 ? m.d2h.total_bytes / m.d2h.count : comp_pool;
  print_metric_row(&m.d2h, d2h_per);
  double sink_per_call =
    ss->sink->count > 0 ? (double)ss->total_bytes / ss->sink->count : 0;
  print_metric_row(ss->sink, sink_per_call);

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
  platform_toc(&clock);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;

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
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct sink_stats ss = { .total_bytes = dss.total_bytes,
                             .sink = &dss.sink };
    print_bench_report(&s, &ss, total_bytes, total_elements, wall_s);
  }

  transpose_stream_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  log_error("  FAIL");
  return 1;
}

// --- Benchmark with zarr file output ---

static int
test_bench_zarr(const char* output_path)
{
  log_info("=== test_bench_zarr ===");
  log_info("  output: %s", output_path);

  test_tmpdir_remove(output_path);

  const struct dimension dims[] = {
    { .size = 4264, .tile_size = 32, .tiles_per_shard = 16, .name = "t" },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4, .name = "y" },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4, .name = "x" },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3, .name = "c" },
  };

  const struct zarr_config zarr_cfg = {
    .store_path = output_path,
    .array_name = "0",
    .data_type = zarr_dtype_uint16,
    .fill_value = 0,
    .rank = 4,
    .dimensions = dims,
  };

  const size_t total_elements = (size_t)4264 * 2048 * 2048 * 3;
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct transpose_stream s = { 0 };
  struct zarr_sink* zs = zarr_sink_create(&zarr_cfg);
  CHECK(Fail, zs);

  struct metered_shard_sink mss;
  metered_shard_sink_init(&mss, zarr_sink_as_shard_sink(zs));

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &mss.shard_writer,
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

  const size_t chunk_elements = 32 * 1024 * 1024;
  uint16_t* chunk = (uint16_t*)malloc(chunk_elements * sizeof(uint16_t));
  CHECK(Fail, chunk);
  fill_chunk(chunk, chunk_elements, 0, total_elements);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;

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

  zarr_sink_flush(zs);

  float wall_s = platform_toc(&clock);
  free(chunk);
  chunk = NULL;

  if (s.cursor != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct sink_stats ss = { .total_bytes = mss.total_bytes,
                             .sink = &mss.sink };
    print_bench_report(&s, &ss, total_bytes, total_elements, wall_s);
  }

  transpose_stream_destroy(&s);
  zarr_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  zarr_sink_destroy(zs);
  log_error("  FAIL");
  return 1;
}

// --- Visual zarr output (synthetic fluorescence microscopy) ---

static int
test_visual_zarr(const char* output_path)
{
  log_info("=== test_visual_zarr ===");
  log_info("  output: %s", output_path);

  test_tmpdir_remove(output_path);

  const struct dimension dims[] = {
    { .size = 4264, .tile_size = 32, .tiles_per_shard = 16, .name = "t" },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4, .name = "y" },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 4, .name = "x" },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3, .name = "c" },
  };

  const struct zarr_config zarr_cfg = {
    .store_path = output_path,
    .array_name = "0",
    .data_type = zarr_dtype_uint16,
    .fill_value = 0,
    .rank = 4,
    .dimensions = dims,
  };

  const size_t total_elements = (size_t)4264 * 2048 * 2048 * 3;
  const size_t total_bytes = total_elements * sizeof(uint16_t);

  struct transpose_stream s = { 0 };
  struct zarr_sink* zs = zarr_sink_create(&zarr_cfg);
  CHECK(Fail, zs);

  struct metered_shard_sink mss;
  metered_shard_sink_init(&mss, zarr_sink_as_shard_sink(zs));

  const struct transpose_stream_configuration config = {
    .buffer_capacity_bytes = 128 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .compress = 1,
    .shard_sink = &mss.shard_writer,
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

  const size_t chunk_elements = 32 * 1024 * 1024;
  uint16_t* chunk = (uint16_t*)malloc(chunk_elements * sizeof(uint16_t));
  CHECK(Fail, chunk);

  struct platform_clock clock = { 0 };
  platform_toc(&clock);

  for (size_t offset = 0; offset < total_elements; offset += chunk_elements) {
    size_t n = chunk_elements;
    if (offset + n > total_elements)
      n = total_elements - offset;

    fill_chunk_visual(chunk, n, offset);

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

  zarr_sink_flush(zs);

  float wall_s = platform_toc(&clock);
  free(chunk);
  chunk = NULL;

  if (s.cursor != total_elements) {
    log_error("  cursor drift: expected %zu, got %zu (diff=%td)",
              total_elements,
              (size_t)s.cursor,
              (ptrdiff_t)((int64_t)s.cursor - (int64_t)total_elements));
    goto Fail;
  }

  {
    struct sink_stats ss = { .total_bytes = mss.total_bytes,
                             .sink = &mss.sink };
    print_bench_report(&s, &ss, total_bytes, total_elements, wall_s);
  }

  transpose_stream_destroy(&s);
  zarr_sink_destroy(zs);
  log_info("  PASS");
  return 0;

Fail:
  transpose_stream_destroy(&s);
  zarr_sink_destroy(zs);
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

  // --zarr <path>: run only the zarr file-writing benchmark
  // --visual <path>: write visually interesting synthetic microscopy zarr
  const char* zarr_path = NULL;
  const char* visual_path = NULL;
  for (int i = 1; i < ac; ++i) {
    if (strcmp(av[i], "--zarr") == 0 && i + 1 < ac) {
      zarr_path = av[i + 1];
    } else if (strcmp(av[i], "--visual") == 0 && i + 1 < ac) {
      visual_path = av[i + 1];
    }
  }

  if (visual_path) {
    ecode |= test_visual_zarr(visual_path);
  } else if (zarr_path) {
    ecode |= test_bench_zarr(zarr_path);
  } else {
    ecode |= test_compressed_small();
    if (!ecode)
      ecode |= test_bench();
  }

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
