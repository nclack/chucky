#include "bench_util.h"
#include "prelude.h"

int
main(int ac, char* av[])
{
  struct dimension dims[] = {
    { .size = 1000, .tile_size = 2, .tiles_per_shard = 32, .name = "t" },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "z" },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "y" },
    { .size = 256, .tile_size = 16, .tiles_per_shard = 8, .name = "x" },
    { .size = 3, .tile_size = 1, .tiles_per_shard = 3, .name = "c" },
  };
  return bench_stream_main(ac, av, "single", dims, countof(dims));
}
