#include "stream_internal.h"

#include "prelude.cuda.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>
#include <zstd.h>

// CPU reference: compute permutation P[i] using the same unravel-dot logic
// as the GPU kernel (from test_aggregate.c).
static uint32_t
cpu_perm(uint64_t i,
         uint8_t lifted_rank,
         const uint64_t* shape,
         const int64_t* strides)
{
  uint64_t out = 0;
  uint64_t rest = i;
  for (int d = lifted_rank - 1; d >= 0; --d) {
    uint64_t coord = rest % shape[d];
    rest /= shape[d];
    out += coord * (uint64_t)strides[d];
  }
  return (uint32_t)out;
}

// Build a tile_stream_configuration for testing.
// Shape: rank=3, dims {4,4,6}, tiles {2,2,3}
//   tile_elements=12, tiles_per_epoch=4, epoch_elements=48
static int
make_test_config(struct tile_stream_configuration* config,
                 struct dimension* dims,
                 enum compression_codec codec,
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
  config->codec = codec;
  config->shard_sink = NULL;
  config->shard_alignment = 0;
  config->epochs_per_batch = epochs_per_batch;
  return 0;
}

// Fill tile pool on device: tile t gets all elements set to (uint16_t)fill_val.
// pool_buf points to the epoch's tile data in the device pool.
// tiles * tile_stride * bpe bytes starting at pool_buf.
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

// ---------------------------------------------------------------------------
// Test 1: CODEC_NONE, K=1, single epoch
// ---------------------------------------------------------------------------
static int
test_compress_agg_single_epoch(void)
{
  log_info("=== test_compress_agg_single_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_NONE, 1);

  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));
  struct compress_agg_stage stage;
  memset(&stage, 0, sizeof(stage));
  CUstream compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent epoch_event = 0;
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail, compute_stream_layouts(&config, &cl) == 0);

  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const size_t bpe = config.bytes_per_element;
  const size_t tile_bytes = tile_stride * bpe;
  const size_t pool_bytes = total_tiles * tile_stride * bpe;

  log_info("  total_tiles=%lu tile_stride=%lu tile_bytes=%zu K=%u",
           (unsigned long)total_tiles, (unsigned long)tile_stride,
           tile_bytes, cl.epochs_per_batch);

  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, compress_agg_init(&stage, &cl, &config, compute) == 0);

  // Allocate tile pool (1 epoch)
  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CHECK(Fail, fill_pool_epoch(d_pool, total_tiles, tile_stride, bpe,
                              fill_epoch0) == 0);

  // Create and record epoch event
  CU(Fail, cuEventCreate(&epoch_event, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(epoch_event, compute));

  // Build input and kick
  struct batch_state batch = { .epochs_per_batch = 1, .accumulated = 0 };
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = 1,
    .active_levels_mask = 0x1,
    .pool_buf = d_pool,
    .epochs_per_batch = 1,
    .lod_done = 0,
  };
  in.batch_active_masks[0] = 0x1;
  in.epoch_events[0] = epoch_event;

  struct flush_handoff handoff;
  memset(&handoff, 0, sizeof(handoff));

  CHECK(Fail,
        compress_agg_kick(&stage, &in, &cl.levels, &batch, compute,
                          &handoff) == 0);
  CU(Fail, cuStreamSynchronize(compute));

  // Verify handoff
  CHECK(Fail, handoff.fc == 0);
  CHECK(Fail, handoff.n_epochs == 1);
  CHECK(Fail, handoff.active_levels_mask == 0x1);
  CHECK(Fail, handoff.t_aggregate_end != 0);
  CHECK(Fail, handoff.t_compress_start != 0);
  CHECK(Fail, handoff.t_compress_end != 0);
  CHECK(Fail, handoff.max_output_size == tile_bytes);
  CHECK(Fail, handoff.agg[0] != NULL);
  CHECK(Fail, handoff.agg_layout[0] != NULL);

  // D2H aggregate offsets and data
  {
    struct aggregate_slot* agg = handoff.agg[0];
    const struct aggregate_layout* al = handoff.agg_layout[0];
    uint64_t C = al->covering_count;

    CU(Fail,
       cuMemcpyDtoH(agg->h_offsets, (CUdeviceptr)agg->d_offsets,
                     (C + 1) * sizeof(size_t)));

    size_t total_data = agg->h_offsets[C];
    h_agg = malloc(total_data);
    CHECK(Fail, h_agg);
    CU(Fail,
       cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

    // Verify offsets: h_offsets[0]==0, monotonic, each tile == tile_bytes
    CHECK(Fail, agg->h_offsets[0] == 0);
    for (uint64_t j = 1; j <= C; ++j)
      CHECK(Fail, agg->h_offsets[j] >= agg->h_offsets[j - 1]);

    // For CODEC_NONE, each actual tile has size == tile_bytes.
    // Padding tiles have size 0.
    size_t expected_total = total_tiles * tile_bytes;
    CHECK(Fail, total_data == expected_total);

    // Byte-for-byte data verification
    uint16_t* tile_buf = (uint16_t*)malloc(tile_bytes);
    CHECK(Fail, tile_buf);

    int errors = 0;
    for (uint64_t t = 0; t < total_tiles; ++t) {
      uint32_t pi = cpu_perm(t, al->lifted_rank, al->lifted_shape,
                             al->lifted_strides);
      size_t off = agg->h_offsets[pi];
      size_t sz = agg->h_offsets[pi + 1] - off;
      if (sz != tile_bytes) {
        log_error("  tile %lu: size=%zu expected=%zu", (unsigned long)t,
                  sz, tile_bytes);
        errors++;
        continue;
      }

      uint16_t expected_val = fill_epoch0(t);
      const uint16_t* got = (const uint16_t*)((char*)h_agg + off);
      for (uint64_t e = 0; e < tile_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error("  tile %lu elem %lu: expected %u got %u",
                      (unsigned long)t, (unsigned long)e, expected_val,
                      got[e]);
          errors++;
        }
      }
    }
    free(tile_buf);
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(h_agg);
  compress_agg_destroy(&stage, cl.levels.nlod);
  computed_stream_layouts_free(&cl);
  cu_mem_free(d_pool);
  cu_event_destroy(epoch_event);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 2: CODEC_NONE, K=2, batch LUT path
// ---------------------------------------------------------------------------
static int
test_compress_agg_batch(void)
{
  log_info("=== test_compress_agg_batch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));
  struct compress_agg_stage stage;
  memset(&stage, 0, sizeof(stage));
  CUstream compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent epoch_events[2] = { 0 };
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail, compute_stream_layouts(&config, &cl) == 0);

  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const size_t bpe = config.bytes_per_element;
  const size_t tile_bytes = tile_stride * bpe;
  const size_t pool_bytes = 2 * total_tiles * tile_stride * bpe;

  log_info("  total_tiles=%lu tile_stride=%lu K=%u",
           (unsigned long)total_tiles, (unsigned long)tile_stride,
           cl.epochs_per_batch);
  CHECK(Fail, cl.epochs_per_batch == 2);

  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, compress_agg_init(&stage, &cl, &config, compute) == 0);

  // Allocate tile pool (2 epochs)
  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));

  // Epoch 0: tile t -> (t+1)
  CHECK(Fail, fill_pool_epoch(d_pool, total_tiles, tile_stride, bpe,
                              fill_epoch0) == 0);
  // Epoch 1: tile t -> (t+100)
  CHECK(Fail, fill_pool_epoch(
                d_pool + total_tiles * tile_stride * bpe,
                total_tiles, tile_stride, bpe, fill_epoch1) == 0);

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&epoch_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(epoch_events[i], compute));
  }

  struct batch_state batch = { .epochs_per_batch = 2, .accumulated = 0 };
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = 2,
    .active_levels_mask = 0x1,
    .pool_buf = d_pool,
    .epochs_per_batch = 2,
    .lod_done = 0,
  };
  in.batch_active_masks[0] = 0x1;
  in.batch_active_masks[1] = 0x1;
  in.epoch_events[0] = epoch_events[0];
  in.epoch_events[1] = epoch_events[1];

  struct flush_handoff handoff;
  memset(&handoff, 0, sizeof(handoff));

  CHECK(Fail,
        compress_agg_kick(&stage, &in, &cl.levels, &batch, compute,
                          &handoff) == 0);
  CU(Fail, cuStreamSynchronize(compute));

  // Verify handoff
  CHECK(Fail, handoff.fc == 0);
  CHECK(Fail, handoff.n_epochs == 2);
  CHECK(Fail, handoff.max_output_size == tile_bytes);

  // D2H aggregate
  {
    struct aggregate_slot* agg = handoff.agg[0];
    const struct aggregate_layout* al = handoff.agg_layout[0];
    uint64_t C = al->covering_count;
    // For batch, covering is per-epoch; the slot was sized for batch_count*C
    uint32_t batch_count = stage.levels[0].batch_active_count;
    uint64_t batch_covering = (uint64_t)batch_count * C;

    CU(Fail,
       cuMemcpyDtoH(agg->h_offsets, (CUdeviceptr)agg->d_offsets,
                     (batch_covering + 1) * sizeof(size_t)));

    size_t total_data = agg->h_offsets[batch_covering];

    // For CODEC_NONE K=2: 2 * total_tiles * tile_bytes
    size_t expected_total = 2 * total_tiles * tile_bytes;
    log_info("  total_data=%zu expected=%zu batch_covering=%lu",
             total_data, expected_total, (unsigned long)batch_covering);
    CHECK(Fail, total_data == expected_total);

    // Offsets monotonic
    CHECK(Fail, agg->h_offsets[0] == 0);
    for (uint64_t j = 1; j <= batch_covering; ++j)
      CHECK(Fail, agg->h_offsets[j] >= agg->h_offsets[j - 1]);

    h_agg = malloc(total_data);
    CHECK(Fail, h_agg);
    CU(Fail,
       cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

    // Verify data: need to use batch LUT permutation.
    // In the batch path, tile index = epoch * total_tiles + tile_in_level.
    // The perm maps (epoch_slot * tiles_lv + j) to shard-ordered position.
    // For the LUT path, h_perm[idx] = perm_pos * batch_count + epoch_slot.
    // We verify by decomposing: for each epoch a and tile j,
    // compute the aggregate position and check the data.
    uint64_t tiles_lv = cl.levels.tile_count[0];
    int errors = 0;

    for (uint32_t a = 0; a < batch_count; ++a) {
      uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
      for (uint64_t j = 0; j < tiles_lv; ++j) {
        // Compute expected perm position (same logic as compress_agg_init LUT)
        uint64_t perm_pos = 0;
        uint64_t rest = j;
        for (int d = al->lifted_rank - 1; d >= 0; --d) {
          uint64_t coord = rest % al->lifted_shape[d];
          rest /= al->lifted_shape[d];
          perm_pos += coord * (uint64_t)al->lifted_strides[d];
        }
        uint64_t out_idx = perm_pos * batch_count + a;

        size_t off = agg->h_offsets[out_idx];
        size_t sz = agg->h_offsets[out_idx + 1] - off;
        if (sz != tile_bytes) {
          if (errors < 5)
            log_error("  epoch %u tile %lu: size=%zu expected=%zu",
                      a, (unsigned long)j, sz, tile_bytes);
          errors++;
          continue;
        }

        uint16_t expected_val = fill_fn(j);
        const uint16_t* got = (const uint16_t*)((char*)h_agg + off);
        for (uint64_t e = 0; e < tile_stride; ++e) {
          if (got[e] != expected_val) {
            if (errors < 5)
              log_error("  epoch %u tile %lu elem %lu: expected %u got %u",
                        a, (unsigned long)j, (unsigned long)e,
                        expected_val, got[e]);
            errors++;
          }
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(h_agg);
  compress_agg_destroy(&stage, cl.levels.nlod);
  computed_stream_layouts_free(&cl);
  cu_mem_free(d_pool);
  for (int i = 0; i < 2; ++i)
    cu_event_destroy(epoch_events[i]);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 3: CODEC_NONE, K=2 but n_epochs=1 (partial batch, no LUTs)
// ---------------------------------------------------------------------------
static int
test_compress_agg_partial_batch(void)
{
  log_info("=== test_compress_agg_partial_batch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_NONE, 2);

  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));
  struct compress_agg_stage stage;
  memset(&stage, 0, sizeof(stage));
  CUstream compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent epoch_event = 0;
  void* h_agg = NULL;
  int ok = 0;

  CHECK(Fail, compute_stream_layouts(&config, &cl) == 0);

  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const size_t bpe = config.bytes_per_element;
  const size_t tile_bytes = tile_stride * bpe;
  // Allocate 2 epochs worth even though we only use 1
  const size_t pool_bytes = 2 * total_tiles * tile_stride * bpe;

  CHECK(Fail, cl.epochs_per_batch == 2);

  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, compress_agg_init(&stage, &cl, &config, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CHECK(Fail, fill_pool_epoch(d_pool, total_tiles, tile_stride, bpe,
                              fill_epoch0) == 0);

  CU(Fail, cuEventCreate(&epoch_event, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(epoch_event, compute));

  // Kick with n_epochs=1 even though K=2 -> partial batch, use_luts=false
  struct batch_state batch = { .epochs_per_batch = 2, .accumulated = 0 };
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = 1,
    .active_levels_mask = 0x1,
    .pool_buf = d_pool,
    .epochs_per_batch = 2,
    .lod_done = 0,
  };
  in.batch_active_masks[0] = 0x1;
  in.epoch_events[0] = epoch_event;

  struct flush_handoff handoff;
  memset(&handoff, 0, sizeof(handoff));

  CHECK(Fail,
        compress_agg_kick(&stage, &in, &cl.levels, &batch, compute,
                          &handoff) == 0);
  CU(Fail, cuStreamSynchronize(compute));

  CHECK(Fail, handoff.n_epochs == 1);

  // D2H and verify
  {
    struct aggregate_slot* agg = handoff.agg[0];
    const struct aggregate_layout* al = handoff.agg_layout[0];
    uint64_t C = al->covering_count;

    // Partial batch: only 1 epoch, per-epoch aggregate path
    CU(Fail,
       cuMemcpyDtoH(agg->h_offsets, (CUdeviceptr)agg->d_offsets,
                     (C + 1) * sizeof(size_t)));

    size_t total_data = agg->h_offsets[C];
    size_t expected_total = total_tiles * tile_bytes;
    log_info("  total_data=%zu expected=%zu", total_data, expected_total);
    CHECK(Fail, total_data == expected_total);

    CHECK(Fail, agg->h_offsets[0] == 0);
    for (uint64_t j = 1; j <= C; ++j)
      CHECK(Fail, agg->h_offsets[j] >= agg->h_offsets[j - 1]);

    h_agg = malloc(total_data);
    CHECK(Fail, h_agg);
    CU(Fail,
       cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

    int errors = 0;
    for (uint64_t t = 0; t < total_tiles; ++t) {
      uint32_t pi = cpu_perm(t, al->lifted_rank, al->lifted_shape,
                             al->lifted_strides);
      size_t off = agg->h_offsets[pi];
      size_t sz = agg->h_offsets[pi + 1] - off;
      CHECK(Fail, sz == tile_bytes);

      uint16_t expected_val = fill_epoch0(t);
      const uint16_t* got = (const uint16_t*)((char*)h_agg + off);
      for (uint64_t e = 0; e < tile_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error("  tile %lu elem %lu: expected %u got %u",
                      (unsigned long)t, (unsigned long)e, expected_val,
                      got[e]);
          errors++;
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(h_agg);
  compress_agg_destroy(&stage, cl.levels.nlod);
  computed_stream_layouts_free(&cl);
  cu_mem_free(d_pool);
  cu_event_destroy(epoch_event);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 4: CODEC_ZSTD, K=1, single epoch
// ---------------------------------------------------------------------------
static int
test_compress_agg_zstd_single_epoch(void)
{
  log_info("=== test_compress_agg_zstd_single_epoch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_ZSTD, 1);

  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));
  struct compress_agg_stage stage;
  memset(&stage, 0, sizeof(stage));
  CUstream compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent epoch_event = 0;
  void* h_agg = NULL;
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail, compute_stream_layouts(&config, &cl) == 0);

  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const size_t bpe = config.bytes_per_element;
  const size_t tile_bytes = tile_stride * bpe;
  const size_t pool_bytes = total_tiles * tile_stride * bpe;

  log_info("  total_tiles=%lu tile_bytes=%zu max_output_size=%zu",
           (unsigned long)total_tiles, tile_bytes, cl.max_output_size);

  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, compress_agg_init(&stage, &cl, &config, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CHECK(Fail, fill_pool_epoch(d_pool, total_tiles, tile_stride, bpe,
                              fill_epoch0) == 0);

  CU(Fail, cuEventCreate(&epoch_event, CU_EVENT_DEFAULT));
  CU(Fail, cuEventRecord(epoch_event, compute));

  struct batch_state batch = { .epochs_per_batch = 1, .accumulated = 0 };
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = 1,
    .active_levels_mask = 0x1,
    .pool_buf = d_pool,
    .epochs_per_batch = 1,
    .lod_done = 0,
  };
  in.batch_active_masks[0] = 0x1;
  in.epoch_events[0] = epoch_event;

  struct flush_handoff handoff;
  memset(&handoff, 0, sizeof(handoff));

  CHECK(Fail,
        compress_agg_kick(&stage, &in, &cl.levels, &batch, compute,
                          &handoff) == 0);
  CU(Fail, cuStreamSynchronize(compute));

  // D2H and verify with ZSTD_decompress
  {
    struct aggregate_slot* agg = handoff.agg[0];
    const struct aggregate_layout* al = handoff.agg_layout[0];
    uint64_t C = al->covering_count;

    CU(Fail,
       cuMemcpyDtoH(agg->h_offsets, (CUdeviceptr)agg->d_offsets,
                     (C + 1) * sizeof(size_t)));

    CHECK(Fail, agg->h_offsets[0] == 0);
    for (uint64_t j = 1; j <= C; ++j)
      CHECK(Fail, agg->h_offsets[j] >= agg->h_offsets[j - 1]);

    size_t total_data = agg->h_offsets[C];
    log_info("  total_agg_data=%zu", total_data);

    h_agg = malloc(total_data);
    CHECK(Fail, h_agg);
    CU(Fail,
       cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

    decomp_buf = (uint8_t*)malloc(tile_bytes);
    CHECK(Fail, decomp_buf);

    int errors = 0;
    for (uint64_t t = 0; t < total_tiles; ++t) {
      uint32_t pi = cpu_perm(t, al->lifted_rank, al->lifted_shape,
                             al->lifted_strides);
      size_t off = agg->h_offsets[pi];
      size_t comp_sz = agg->h_offsets[pi + 1] - off;

      CHECK(Fail, comp_sz > 0);
      CHECK(Fail, comp_sz <= handoff.max_output_size);

      size_t result =
        ZSTD_decompress(decomp_buf, tile_bytes, (char*)h_agg + off, comp_sz);
      if (ZSTD_isError(result)) {
        log_error("  tile %lu: ZSTD_decompress failed: %s",
                  (unsigned long)t, ZSTD_getErrorName(result));
        errors++;
        continue;
      }
      CHECK(Fail, result == tile_bytes);

      uint16_t expected_val = fill_epoch0(t);
      const uint16_t* got = (const uint16_t*)decomp_buf;
      for (uint64_t e = 0; e < tile_stride; ++e) {
        if (got[e] != expected_val) {
          if (errors < 5)
            log_error("  tile %lu elem %lu: expected %u got %u",
                      (unsigned long)t, (unsigned long)e, expected_val,
                      got[e]);
          errors++;
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(h_agg);
  free(decomp_buf);
  compress_agg_destroy(&stage, cl.levels.nlod);
  computed_stream_layouts_free(&cl);
  cu_mem_free(d_pool);
  cu_event_destroy(epoch_event);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Test 5: CODEC_ZSTD, K=2, batch LUT path
// ---------------------------------------------------------------------------
static int
test_compress_agg_zstd_batch(void)
{
  log_info("=== test_compress_agg_zstd_batch ===");

  struct dimension dims[3];
  struct tile_stream_configuration config;
  make_test_config(&config, dims, CODEC_ZSTD, 2);

  struct computed_stream_layouts cl;
  memset(&cl, 0, sizeof(cl));
  struct compress_agg_stage stage;
  memset(&stage, 0, sizeof(stage));
  CUstream compute = 0;
  CUdeviceptr d_pool = 0;
  CUevent epoch_events[2] = { 0 };
  void* h_agg = NULL;
  uint8_t* decomp_buf = NULL;
  int ok = 0;

  CHECK(Fail, compute_stream_layouts(&config, &cl) == 0);

  const uint64_t total_tiles = cl.levels.total_tiles;
  const uint64_t tile_stride = cl.l0.tile_stride;
  const size_t bpe = config.bytes_per_element;
  const size_t tile_bytes = tile_stride * bpe;
  const size_t pool_bytes = 2 * total_tiles * tile_stride * bpe;

  CHECK(Fail, cl.epochs_per_batch == 2);

  CU(Fail, cuStreamCreate(&compute, CU_STREAM_NON_BLOCKING));
  CHECK(Fail, compress_agg_init(&stage, &cl, &config, compute) == 0);

  CU(Fail, cuMemAlloc(&d_pool, pool_bytes));
  CHECK(Fail, fill_pool_epoch(d_pool, total_tiles, tile_stride, bpe,
                              fill_epoch0) == 0);
  CHECK(Fail, fill_pool_epoch(
                d_pool + total_tiles * tile_stride * bpe,
                total_tiles, tile_stride, bpe, fill_epoch1) == 0);

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&epoch_events[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventRecord(epoch_events[i], compute));
  }

  struct batch_state batch = { .epochs_per_batch = 2, .accumulated = 0 };
  struct compress_agg_input in = {
    .fc = 0,
    .n_epochs = 2,
    .active_levels_mask = 0x1,
    .pool_buf = d_pool,
    .epochs_per_batch = 2,
    .lod_done = 0,
  };
  in.batch_active_masks[0] = 0x1;
  in.batch_active_masks[1] = 0x1;
  in.epoch_events[0] = epoch_events[0];
  in.epoch_events[1] = epoch_events[1];

  struct flush_handoff handoff;
  memset(&handoff, 0, sizeof(handoff));

  CHECK(Fail,
        compress_agg_kick(&stage, &in, &cl.levels, &batch, compute,
                          &handoff) == 0);
  CU(Fail, cuStreamSynchronize(compute));

  // D2H and verify
  {
    struct aggregate_slot* agg = handoff.agg[0];
    const struct aggregate_layout* al = handoff.agg_layout[0];
    uint64_t C = al->covering_count;
    uint32_t batch_count = stage.levels[0].batch_active_count;
    uint64_t batch_covering = (uint64_t)batch_count * C;

    CU(Fail,
       cuMemcpyDtoH(agg->h_offsets, (CUdeviceptr)agg->d_offsets,
                     (batch_covering + 1) * sizeof(size_t)));

    CHECK(Fail, agg->h_offsets[0] == 0);
    for (uint64_t j = 1; j <= batch_covering; ++j)
      CHECK(Fail, agg->h_offsets[j] >= agg->h_offsets[j - 1]);

    size_t total_data = agg->h_offsets[batch_covering];
    log_info("  total_agg_data=%zu batch_covering=%lu",
             total_data, (unsigned long)batch_covering);

    h_agg = malloc(total_data);
    CHECK(Fail, h_agg);
    CU(Fail,
       cuMemcpyDtoH(h_agg, (CUdeviceptr)agg->d_aggregated, total_data));

    decomp_buf = (uint8_t*)malloc(tile_bytes);
    CHECK(Fail, decomp_buf);

    uint64_t tiles_lv = cl.levels.tile_count[0];
    int errors = 0;

    for (uint32_t a = 0; a < batch_count; ++a) {
      uint16_t (*fill_fn)(uint64_t) = (a == 0) ? fill_epoch0 : fill_epoch1;
      for (uint64_t j = 0; j < tiles_lv; ++j) {
        uint64_t perm_pos = 0;
        uint64_t rest = j;
        for (int d = al->lifted_rank - 1; d >= 0; --d) {
          uint64_t coord = rest % al->lifted_shape[d];
          rest /= al->lifted_shape[d];
          perm_pos += coord * (uint64_t)al->lifted_strides[d];
        }
        uint64_t out_idx = perm_pos * batch_count + a;

        size_t off = agg->h_offsets[out_idx];
        size_t comp_sz = agg->h_offsets[out_idx + 1] - off;

        CHECK(Fail, comp_sz > 0);
        CHECK(Fail, comp_sz <= handoff.max_output_size);

        size_t result =
          ZSTD_decompress(decomp_buf, tile_bytes, (char*)h_agg + off,
                          comp_sz);
        if (ZSTD_isError(result)) {
          log_error("  epoch %u tile %lu: ZSTD_decompress failed: %s",
                    a, (unsigned long)j, ZSTD_getErrorName(result));
          errors++;
          continue;
        }
        CHECK(Fail, result == tile_bytes);

        uint16_t expected_val = fill_fn(j);
        const uint16_t* got = (const uint16_t*)decomp_buf;
        for (uint64_t e = 0; e < tile_stride; ++e) {
          if (got[e] != expected_val) {
            if (errors < 5)
              log_error("  epoch %u tile %lu elem %lu: expected %u got %u",
                        a, (unsigned long)j, (unsigned long)e,
                        expected_val, got[e]);
            errors++;
          }
        }
      }
    }
    CHECK(Fail, errors == 0);
  }

  ok = 1;

Fail:
  free(h_agg);
  free(decomp_buf);
  compress_agg_destroy(&stage, cl.levels.nlod);
  computed_stream_layouts_free(&cl);
  cu_mem_free(d_pool);
  for (int i = 0; i < 2; ++i)
    cu_event_destroy(epoch_events[i]);
  cu_stream_destroy(compute);

  log_info("  %s", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
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

  ecode |= test_compress_agg_single_epoch();
  ecode |= test_compress_agg_batch();
  ecode |= test_compress_agg_partial_batch();
  ecode |= test_compress_agg_zstd_single_epoch();
  ecode |= test_compress_agg_zstd_batch();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  cuCtxDestroy(ctx);
  return 1;
}
