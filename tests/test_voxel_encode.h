#ifndef TEST_VOXEL_ENCODE_H
#define TEST_VOXEL_ENCODE_H

#include <stdint.h>

// Fixture constants for the 3D voxel-encoded volume tests.
// dim0=12, dim1=8, dim2=12. tile: 2,4,3. tps: 3,2,2.
static const int voxel_fixture_size[3] = { 12, 8, 12 };
static const int voxel_fixture_tile_size[3] = { 2, 4, 3 };
static const int voxel_fixture_tps[3] = { 3, 2, 2 };

// Encode a voxel's 3D shard/tile/voxel coordinates into a u32 value.
// Bit layout (27 of 32 bits):
//   [26:24] = shard_coord[0]     [23:21] = shard_coord[1]     [20:18] = shard_coord[2]
//   [17:15] = tile_in_shard[0]   [14:12] = tile_in_shard[1]   [11:9]  = tile_in_shard[2]
//   [8:6]   = voxel_in_tile[0]   [5:3]   = voxel_in_tile[1]   [2:0]   = voxel_in_tile[2]
uint32_t
encode_voxel(int s0, int s1, int s2,
             int t0, int t1, int t2,
             int v0, int v1, int v2);

// Generate source data for the voxel fixture: raster-order u32 with encoded coords.
// Caller owns returned pointer; free() when done. Returns NULL on failure.
uint32_t*
generate_encoded_volume(const int size[3],
                        const int tile_size[3],
                        const int tiles_per_shard[3]);

#endif // TEST_VOXEL_ENCODE_H
