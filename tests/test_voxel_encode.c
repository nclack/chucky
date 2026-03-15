#include "test_voxel_encode.h"

#include <stdlib.h>

uint32_t
encode_voxel(int s0, int s1, int s2,
             int t0, int t1, int t2,
             int v0, int v1, int v2)
{
  return ((uint32_t)s0 << 24) | ((uint32_t)s1 << 21) | ((uint32_t)s2 << 18) |
         ((uint32_t)t0 << 15) | ((uint32_t)t1 << 12) | ((uint32_t)t2 << 9) |
         ((uint32_t)v0 << 6) | ((uint32_t)v1 << 3) | ((uint32_t)v2);
}

uint32_t*
generate_encoded_volume(const int size[3],
                        const int tile_size[3],
                        const int tiles_per_shard[3])
{
  int total = size[0] * size[1] * size[2];
  uint32_t* src = (uint32_t*)malloc((size_t)total * sizeof(uint32_t));
  if (!src)
    return NULL;

  for (int x0 = 0; x0 < size[0]; ++x0) {
    for (int x1 = 0; x1 < size[1]; ++x1) {
      for (int x2 = 0; x2 < size[2]; ++x2) {
        int gi = x0 * size[1] * size[2] + x1 * size[2] + x2;
        int s0 = x0 / (tile_size[0] * tiles_per_shard[0]);
        int s1 = x1 / (tile_size[1] * tiles_per_shard[1]);
        int s2 = x2 / (tile_size[2] * tiles_per_shard[2]);
        int t0 = (x0 / tile_size[0]) % tiles_per_shard[0];
        int t1 = (x1 / tile_size[1]) % tiles_per_shard[1];
        int t2 = (x2 / tile_size[2]) % tiles_per_shard[2];
        int v0 = x0 % tile_size[0];
        int v1 = x1 % tile_size[1];
        int v2 = x2 % tile_size[2];
        src[gi] = encode_voxel(s0, s1, s2, t0, t1, t2, v0, v1, v2);
      }
    }
  }
  return src;
}
