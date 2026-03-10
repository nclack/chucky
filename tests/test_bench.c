#include "bench_util.h"
#include "prelude.cuda.h"
#include "prelude.h"

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
  const size_t total_elements = dim_total_elements(dims, 4);

  struct tile_stream_gpu s = { 0 };
  struct discard_shard_sink dss;
  discard_shard_sink_init(&dss);

  const struct tile_stream_configuration config = {
    .buffer_capacity_bytes = 8 << 20,
    .bytes_per_element = sizeof(uint16_t),
    .rank = 4,
    .dimensions = dims,
    .codec = CODEC_ZSTD,
    .shard_sink = &dss.base,
  };

  CHECK(Fail, tile_stream_gpu_create(&config, &s));

  const size_t num_epochs =
    (total_elements + s.layout.epoch_elements - 1) / s.layout.epoch_elements;
  log_info("  total: %zu elements, %zu epochs", total_elements, num_epochs);

  CHECK(Fail, pump_data(&s.writer, total_elements, fill_zeros) == 0);

  CHECK(Fail, s.cursor == total_elements);
  log_info("  shards finalized: %zu, total bytes: %zu",
           dss.shards_finalized,
           dss.total_bytes);

  tile_stream_gpu_destroy(&s);
  log_info("  PASS");
  return 0;

Fail:
  tile_stream_gpu_destroy(&s);
  log_error("  FAIL");
  return 1;
}

int
main(void)
{
  int ecode = 0;
  CUcontext ctx = 0;
  CUdevice dev;

  CU(Fail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  ecode |= test_compressed_small();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  printf("FAIL\n");
  cuCtxDestroy(ctx);
  return 1;
}
