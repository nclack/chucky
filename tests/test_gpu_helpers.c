#include "test_gpu_helpers.h"

#include "prelude.cuda.h"
#include "prelude.h"

#include <stdlib.h>
#include <string.h>

int
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
  config->shard_alignment = 0;
  config->epochs_per_batch = epochs_per_batch;
  return 0;
}

int
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

uint16_t
fill_epoch0(uint64_t t)
{
  return (uint16_t)(t + 1);
}

uint16_t
fill_epoch1(uint64_t t)
{
  return (uint16_t)(t + 100);
}

uint16_t
fill_epoch2(uint64_t t)
{
  return (uint16_t)(t + 200);
}

uint16_t
fill_epoch3(uint64_t t)
{
  return (uint16_t)(t + 300);
}
