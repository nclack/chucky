#include "bench_util.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  struct dimension dims[] = {
    { .size = 1000, .tile_size = 16, .tiles_per_shard = 128, .name = "t", .storage_position = 0 },
    { .size = 2, .tile_size = 1, .tiles_per_shard = 2, .name = "c", .storage_position = 1 },
    { .size = 2048, .tile_size = 128, .tiles_per_shard = 9, .name = "y", .storage_position = 2 },
    { .size = 2304, .tile_size = 128, .tiles_per_shard = 9, .name = "x", .storage_position = 3 },
  };
  return bench_stream_main(ac, av, "single", dims, countof(dims));
}
