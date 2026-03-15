#include "stream_internal.h"

#include "prelude.cuda.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Test shard sink
// ---------------------------------------------------------------------------

struct test_shard_sink
{
  struct shard_sink base;
  struct
  {
    struct shard_writer base;
    uint8_t* buf;
    size_t capacity;
    size_t size;
  } writer;
  int open_count;
  int finalize_count;
};

static int
test_sink_write(struct shard_writer* self,
                uint64_t offset,
                const void* beg,
                const void* end)
{
  struct test_shard_sink* sink =
    (struct test_shard_sink*)((char*)self -
                              offsetof(struct test_shard_sink, writer));
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > sink->writer.capacity)
    return 1;
  memcpy(sink->writer.buf + offset, beg, nbytes);
  if (offset + nbytes > sink->writer.size)
    sink->writer.size = offset + nbytes;
  return 0;
}

static int
test_sink_finalize(struct shard_writer* self)
{
  struct test_shard_sink* sink =
    (struct test_shard_sink*)((char*)self -
                              offsetof(struct test_shard_sink, writer));
  sink->finalize_count++;
  return 0;
}

static struct shard_writer*
test_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  (void)shard_index;
  struct test_shard_sink* s = (struct test_shard_sink*)self;
  s->open_count++;
  return &s->writer.base;
}

static void
test_sink_init(struct test_shard_sink* s, size_t capacity)
{
  memset(s, 0, sizeof(*s));
  s->base.open = test_sink_open;
  s->writer.base.write = test_sink_write;
  s->writer.base.finalize = test_sink_finalize;
  s->writer.buf = (uint8_t*)calloc(1, capacity);
  s->writer.capacity = capacity;
}

static void
test_sink_free(struct test_shard_sink* s)
{
  free(s->writer.buf);
  memset(s, 0, sizeof(*s));
}

// ---------------------------------------------------------------------------
// Helpers (from test_compress_agg.c)
// ---------------------------------------------------------------------------

static int
make_test_config(struct tile_stream_configuration* config,
                 struct dimension* dims,
                 uint8_t epochs_per_batch)
{
  dims[0] = (struct dimension){
    .size = 4, .tile_size = 2, .tiles_per_shard = 2, .storage_position = 0
  };
  dims[1] = (struct dimension){
    .size = 4, .tile_size = 2, .tiles_per_shard = 2, .storage_position = 1
  };
  dims[2] = (struct dimension){
    .size = 6, .tile_size = 3, .tiles_per_shard = 2, .storage_position = 2
  };

  memset(config, 0, sizeof(*config));
  config->rank = 3;
  config->dimensions = dims;
  config->bytes_per_element = 2;
  config->buffer_capacity_bytes = 4096;
  config->codec = CODEC_NONE;
  config->shard_alignment = 0;
  config->epochs_per_batch = epochs_per_batch;
  return 0;
}

static int
fill_pool_epoch(CUdeviceptr pool_buf,
                uint64_t tiles,
                uint64_t tile_stride,
                size_t bpe,
                uint16_t (*fill_fn)(uint64_t tile))
{
  size_t epoch_bytes = tiles * tile_stride * bpe;
  uint16_t* h = (uint16_t*)malloc(epoch_bytes);
  CHECK(Fail, h);
  memset(h, 0, epoch_bytes);

  for (uint64_t t = 0; t < tiles; ++t) {
    uint16_t val = fill_fn(t);
    uint16_t* tile_data = h + t * tile_stride;
    for (uint64_t e = 0; e < tile_stride; ++e)
      tile_data[e] = val;
  }

  CU(Fail, cuMemcpyHtoD(pool_buf, h, epoch_bytes));
  free(h);
  return 0;

Fail:
  free(h);
  return 1;
}

static uint16_t
fill_epoch0(uint64_t t)
{
  return (uint16_t)(t + 1);
}

static uint16_t
fill_epoch1(uint64_t t)
{
  return (uint16_t)(t + 100);
}

static uint16_t
fill_epoch2(uint64_t t)
{
  return (uint16_t)(t + 200);
}

static uint16_t
fill_epoch3(uint64_t t)
{
  return (uint16_t)(t + 300);
}

// ---------------------------------------------------------------------------
// Orchestration test context: assembles flush_context from individual parts
// ---------------------------------------------------------------------------

struct orch_ctx
{
  struct computed_stream_layouts cl;
  struct compress_agg_stage ca;
  struct d2h_deliver_stage d2h;
  struct flush_pipeline flush;
  struct pool_state pools;
  struct batch_state batch;
  struct lod_state lod;
  struct stream_metrics metrics;
  struct platform_clock metadata_clock;
  struct gpu_streams streams;

  int ca_inited;
  int d2h_inited;
};

static void
orch_ctx_init(struct orch_ctx* c)
{
  memset(c, 0, sizeof(*c));
}

static void
orch_ctx_destroy(struct orch_ctx* c)
{
  // Batch events
  for (uint32_t i = 0; i < c->batch.epochs_per_batch; ++i)
    cu_event_destroy(c->batch.pool_events[i]);

  if (c->d2h_inited)
    d2h_deliver_destroy(&c->d2h);
  if (c->ca_inited)
    compress_agg_destroy(&c->ca, c->cl.levels.nlod);

  // Pools
  for (int i = 0; i < 2; ++i) {
    cu_mem_free(c->pools.buf[i]);
    cu_event_destroy(c->pools.ready[i]);
  }

  computed_stream_layouts_free(&c->cl);

  cu_stream_destroy(c->streams.compute);
  cu_stream_destroy(c->streams.compress);
  cu_stream_destroy(c->streams.d2h);
}

static struct stream_metrics
init_test_metrics(void)
{
  struct stream_metrics m;
  memset(&m, 0, sizeof(m));
  m.compress.best_ms = 1e30f;
  m.aggregate.best_ms = 1e30f;
  m.d2h.best_ms = 1e30f;
  m.sink.best_ms = 1e30f;
  m.lod_gather.best_ms = 1e30f;
  return m;
}

// Set up all components for the flush orchestration test.
static int
orch_ctx_setup(struct orch_ctx* c,
               struct tile_stream_configuration* config)
{
  CHECK(Fail, compute_stream_layouts(config, &c->cl) == 0);

  const uint32_t K = c->cl.epochs_per_batch;
  const uint64_t total_tiles = c->cl.levels.total_tiles;
  const uint64_t tile_stride = c->cl.l0.tile_stride;
  const size_t bpe = config->bytes_per_element;
  const size_t pool_bytes = (uint64_t)K * total_tiles * tile_stride * bpe;

  // GPU streams
  CU(Fail, cuStreamCreate(&c->streams.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->streams.compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&c->streams.d2h, CU_STREAM_NON_BLOCKING));

  // Compress+aggregate stage
  CHECK(Fail, compress_agg_init(&c->ca, &c->cl, config, c->streams.compute) == 0);
  c->ca_inited = 1;

  // D2H+deliver stage
  CHECK(Fail,
        d2h_deliver_init(
          &c->d2h, c->ca.levels, c->cl.levels.nlod, c->streams.compute) == 0);
  c->d2h_inited = 1;

  // Double-buffered tile pools
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuMemAlloc(&c->pools.buf[i], pool_bytes));
    CU(Fail, cuMemsetD8Async(c->pools.buf[i], 0, pool_bytes,
                              c->streams.compute));
    CU(Fail, cuEventCreate(&c->pools.ready[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(c->pools.ready[i], c->streams.compute));
  }
  c->pools.current = 0;

  // Batch state + pool events
  c->batch.epochs_per_batch = K;
  c->batch.accumulated = 0;
  for (uint32_t i = 0; i < K; ++i) {
    CU(Fail, cuEventCreate(&c->batch.pool_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(c->batch.pool_events[i], c->streams.compute));
  }

  // Flush pipeline state
  memset(&c->flush, 0, sizeof(c->flush));

  // Non-multiscale: zeroed lod
  memset(&c->lod, 0, sizeof(c->lod));

  c->metrics = init_test_metrics();
  memset(&c->metadata_clock, 0, sizeof(c->metadata_clock));

  CU(Fail, cuStreamSynchronize(c->streams.compute));
  return 0;

Fail:
  return 1;
}

// Build flush_context from individual parts.
static struct flush_context
orch_ctx_make_fctx(struct orch_ctx* c,
                   const struct tile_stream_configuration* config)
{
  return (struct flush_context){
    .flush = &c->flush,
    .compress_agg = &c->ca,
    .d2h_deliver = &c->d2h,
    .levels = &c->cl.levels,
    .batch = &c->batch,
    .pools = &c->pools,
    .lod = &c->lod,
    .metrics = &c->metrics,
    .config = config,
    .layout = &c->cl.l0,
    .streams = c->streams,
    .metadata_update_clock = &c->metadata_clock,
  };
}

// Fill one epoch in the current pool. Syncs compute stream first to ensure
// any pending pool zeroing is complete.
static int
orch_ctx_fill_epoch(struct orch_ctx* c,
                    uint32_t epoch_in_batch,
                    const struct tile_stream_configuration* config,
                    uint16_t (*fill_fn)(uint64_t))
{
  CU(Fail, cuStreamSynchronize(c->streams.compute));

  const uint64_t total_tiles = c->cl.levels.total_tiles;
  const uint64_t tile_stride = c->cl.l0.tile_stride;
  const size_t bpe = config->bytes_per_element;
  CUdeviceptr epoch_ptr =
    c->pools.buf[c->pools.current] +
    (uint64_t)epoch_in_batch * total_tiles * tile_stride * bpe;
  return fill_pool_epoch(epoch_ptr, total_tiles, tile_stride, bpe, fill_fn);

Fail:
  return 1;
}

// ---------------------------------------------------------------------------
// Test 1: Accumulate one epoch into K=2 batch — no flush triggered
// ---------------------------------------------------------------------------
static int
test_accumulate_one_epoch(void)
{
  log_info("=== test_accumulate_one_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, 512 * 1024);
  config.shard_sink = &sink.base;

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config) == 0);
  CHECK(Fail, c.cl.epochs_per_batch == 2);

  // Fill epoch 0 in current pool
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);

  // Accumulate epoch
  struct flush_context fctx = orch_ctx_make_fctx(&c, &config);
  struct writer_result r = flush_accumulate_epoch(&fctx);
  CHECK(Fail, r.error == 0);

  // Verify: mid-batch, no flush triggered
  CHECK(Fail, c.batch.accumulated == 1);
  CHECK(Fail, c.pools.current == 0);
  CHECK(Fail, c.flush.pending == 0);

  // Epoch mask recorded
  CHECK(Fail, c.flush.slot[0].batch_active_masks[0] == 0x1);
  CHECK(Fail, c.flush.slot[0].active_levels_mask == 0x1);

  // Sink not touched
  CHECK(Fail, sink.open_count == 0);
  CHECK(Fail, sink.finalize_count == 0);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: Full batch triggers auto-flush + pool swap
// ---------------------------------------------------------------------------
static int
test_full_batch_auto_flush(void)
{
  log_info("=== test_full_batch_auto_flush ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, 512 * 1024);
  config.shard_sink = &sink.base;

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config) == 0);

  // Fill and accumulate 2 epochs
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  struct flush_context fctx = orch_ctx_make_fctx(&c, &config);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);
  CHECK(Fail, c.batch.accumulated == 1);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);

  // After full batch: drain_kick_and_swap fired
  CHECK(Fail, c.batch.accumulated == 0);
  CHECK(Fail, c.pools.current == 1);       // swapped to pool 1
  CHECK(Fail, c.flush.pending == 1);        // batch 1 pending delivery
  CHECK(Fail, c.flush.current == 0);        // batch 1 was on pool 0

  // Fresh pool slot is reset
  CHECK(Fail, c.flush.slot[1].active_levels_mask == 0);

  // Sink not yet written (D2H kicked but not drained)
  CHECK(Fail, sink.open_count == 0);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 3: Full batch + drain → data arrives in sink
// ---------------------------------------------------------------------------
static int
test_drain_delivers_data(void)
{
  log_info("=== test_drain_delivers_data ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, 512 * 1024);
  config.shard_sink = &sink.base;

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config) == 0);

  // Fill and accumulate 2 epochs (full batch → auto-kick)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  struct flush_context fctx = orch_ctx_make_fctx(&c, &config);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);

  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);
  CHECK(Fail, c.flush.pending == 1);

  // Drain the pending batch
  struct writer_result r = flush_drain_pending(&fctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.flush.pending == 0);

  // Data delivered: shard opened and finalized (tps_0=2, 2 epochs → complete)
  CHECK(Fail, sink.open_count >= 1);
  CHECK(Fail, sink.finalize_count == 1);
  CHECK(Fail, sink.writer.size > 0);

  // Sink metric always recorded (uses platform_toc, not CUDA events)
  CHECK(Fail, c.metrics.sink.count == 1);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: flush_accumulated_sync with partial batch (1 epoch, K=2)
// ---------------------------------------------------------------------------
static int
test_accumulated_sync_partial(void)
{
  log_info("=== test_accumulated_sync_partial ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, 512 * 1024);
  config.shard_sink = &sink.base;

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config) == 0);

  // Fill and accumulate 1 epoch (partial batch)
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  struct flush_context fctx = orch_ctx_make_fctx(&c, &config);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);
  CHECK(Fail, c.batch.accumulated == 1);

  // Sync flush: processes the partial batch (per-epoch path)
  struct writer_result r = flush_accumulated_sync(&fctx);
  CHECK(Fail, r.error == 0);

  // Batch drained
  CHECK(Fail, c.batch.accumulated == 0);

  // Data delivered to sink (1 epoch, shard not finalized since tps_0=2)
  CHECK(Fail, sink.open_count >= 1);
  CHECK(Fail, sink.writer.size > 0);

  // Sink metric recorded (platform_toc, not CUDA events — always fires)
  CHECK(Fail, c.metrics.sink.count == 1);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 5: Two full batch cycles — verifies pool swap dance
// ---------------------------------------------------------------------------
static int
test_two_batch_cycle(void)
{
  log_info("=== test_two_batch_cycle ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, 2);

  struct test_shard_sink sink;
  test_sink_init(&sink, 1024 * 1024);
  config.shard_sink = &sink.base;

  struct orch_ctx c;
  orch_ctx_init(&c);
  int ok = 0;

  CHECK(Fail, orch_ctx_setup(&c, &config) == 0);
  struct flush_context fctx = orch_ctx_make_fctx(&c, &config);

  // --- Batch 1: epochs 0,1 on pool 0 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch0) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch1) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);

  // Batch 1 kicked, pool swapped to 1
  CHECK(Fail, c.pools.current == 1);
  CHECK(Fail, c.flush.pending == 1);
  CHECK(Fail, c.flush.current == 0);
  CHECK(Fail, c.batch.accumulated == 0);

  // --- Batch 2: epochs 2,3 on pool 1 ---
  CHECK(Fail, orch_ctx_fill_epoch(&c, 0, &config, fill_epoch2) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);
  CHECK(Fail, orch_ctx_fill_epoch(&c, 1, &config, fill_epoch3) == 0);
  CHECK(Fail, flush_accumulate_epoch(&fctx).error == 0);

  // Batch 2 auto-drained batch 1 (pending was set), then kicked batch 2
  CHECK(Fail, c.pools.current == 0);       // swapped back to pool 0
  CHECK(Fail, c.flush.pending == 1);        // batch 2 now pending
  CHECK(Fail, c.flush.current == 1);        // batch 2 was on pool 1
  CHECK(Fail, c.batch.accumulated == 0);

  // Batch 1 was already drained (by drain_kick_and_swap during batch 2)
  // tps_0=2, so batch 1 (2 epochs) finalized one shard
  CHECK(Fail, sink.finalize_count >= 1);

  // Drain batch 2
  struct writer_result r = flush_drain_pending(&fctx);
  CHECK(Fail, r.error == 0);
  CHECK(Fail, c.flush.pending == 0);

  // Second shard finalized
  CHECK(Fail, sink.finalize_count >= 2);

  // Sink metric: 2 batch drains
  CHECK(Fail, c.metrics.sink.count == 2);

  ok = 1;

Fail:
  orch_ctx_destroy(&c);
  test_sink_free(&sink);
  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

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

  ecode |= test_accumulate_one_epoch();
  log_info("");
  ecode |= test_full_batch_auto_flush();
  log_info("");
  ecode |= test_drain_delivers_data();
  log_info("");
  ecode |= test_accumulated_sync_partial();
  log_info("");
  ecode |= test_two_batch_cycle();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
