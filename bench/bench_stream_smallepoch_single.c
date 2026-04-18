#include "bench_util.h"
#include "dimension.h"

int
main(int ac, char* av[])
{
  struct dimension dims[3];
  uint64_t sizes[] = { 65536, 16, 16 };
  uint8_t rank = dims_create(dims, "tyx", sizes);

  uint8_t ratios[] = { 1, 0, 0 };

  return bench_stream_main(ac,
                           av,
                           (struct bench_spec){
                             .label = "single",
                             .dims = dims,
                             .rank = rank,
                             .chunk_ratios = ratios,
                             .default_chunk_bytes = 1 << 16,
                             .min_chunk_bytes = 1 << 16,
                             .min_shard_bytes = 1ull << 30,
                             .max_concurrent_shards = 1,
                           });
}
