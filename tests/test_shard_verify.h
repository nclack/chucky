#ifndef TEST_SHARD_VERIFY_H
#define TEST_SHARD_VERIFY_H

#include <stddef.h>
#include <stdint.h>

// Parse the shard index from the end of a shard buffer.
// Layout: [tile_data...][offset0 u64][size0 u64]...[offsetN u64][sizeN u64][crc32c u32]
// Returns 0 on success, 1 on failure.
int
shard_index_parse(const uint8_t* buf,
                  size_t shard_size,
                  size_t tiles_per_shard,
                  uint64_t* offsets,
                  uint64_t* sizes);

// Verify that offsets are monotonically non-decreasing, starting at 0.
// offsets[0..n] inclusive (n+1 entries).
// Returns 0 on success, 1 on failure.
int
verify_offsets_monotonic(const size_t* offsets, uint64_t n);

// Decompress a ZSTD-compressed tile and verify all u16 elements equal expected_val.
// Returns 0 on success, 1 on failure.
int
tile_decompress_verify_u16(const uint8_t* comp_data,
                           size_t comp_size,
                           size_t tile_bytes,
                           uint64_t tile_stride,
                           uint16_t expected_val);

// Decompress a ZSTD-compressed tile into caller-provided buffer.
// Returns 0 on success, 1 on failure.
int
tile_decompress(const uint8_t* comp_data,
                size_t comp_size,
                void* out,
                size_t out_bytes);

#endif // TEST_SHARD_VERIFY_H
