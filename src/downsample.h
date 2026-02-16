#pragma once

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Downsample by 2x along dimensions marked in downsample_mask.
  //
  // Each thread computes one output element as the mean of 2^popcount(mask)
  // source values. For dim 0 downsampling, d_src_a holds the even epoch and
  // d_src_b holds the odd epoch. If dim 0 is not downsampled, d_src_b is
  // ignored (pass 0).
  //
  // Parameters:
  //   d_dst             - destination tile pool (level L+1)
  //   d_src_a           - source tile pool slot A (even epoch)
  //   d_src_b           - source tile pool slot B (odd epoch), 0 if dim0 not
  //                       downsampled
  //   rank              - number of dimensions
  //   downsample_mask   - bit d set => halve dim d
  //   d_dst_tile_size   - [rank] tile sizes at destination level (on device)
  //   d_src_tile_size   - [rank] tile sizes at source level (on device)
  //   d_src_extent      - [rank] array extent at source level (on device)
  //   d_src_pool_strides - [rank] tile-pool strides at source level (on device)
  //   d_dst_pool_strides - [rank] tile-pool strides at dest level (on device)
  //   dst_total_elements - total number of output elements (threads)
  //   stream            - CUDA stream

  void downsample_mean_u8(CUdeviceptr d_dst,
                          CUdeviceptr d_src_a,
                          CUdeviceptr d_src_b,
                          uint8_t rank,
                          uint8_t downsample_mask,
                          const uint64_t* d_dst_tile_size,
                          const uint64_t* d_src_tile_size,
                          const uint64_t* d_src_extent,
                          const int64_t* d_src_pool_strides,
                          const int64_t* d_dst_pool_strides,
                          uint64_t dst_total_elements,
                          CUstream stream);

  void downsample_mean_u16(CUdeviceptr d_dst,
                           CUdeviceptr d_src_a,
                           CUdeviceptr d_src_b,
                           uint8_t rank,
                           uint8_t downsample_mask,
                           const uint64_t* d_dst_tile_size,
                           const uint64_t* d_src_tile_size,
                           const uint64_t* d_src_extent,
                           const int64_t* d_src_pool_strides,
                           const int64_t* d_dst_pool_strides,
                           uint64_t dst_total_elements,
                           CUstream stream);

  void downsample_mean_u32(CUdeviceptr d_dst,
                           CUdeviceptr d_src_a,
                           CUdeviceptr d_src_b,
                           uint8_t rank,
                           uint8_t downsample_mask,
                           const uint64_t* d_dst_tile_size,
                           const uint64_t* d_src_tile_size,
                           const uint64_t* d_src_extent,
                           const int64_t* d_src_pool_strides,
                           const int64_t* d_dst_pool_strides,
                           uint64_t dst_total_elements,
                           CUstream stream);

  void downsample_mean_u64(CUdeviceptr d_dst,
                           CUdeviceptr d_src_a,
                           CUdeviceptr d_src_b,
                           uint8_t rank,
                           uint8_t downsample_mask,
                           const uint64_t* d_dst_tile_size,
                           const uint64_t* d_src_tile_size,
                           const uint64_t* d_src_extent,
                           const int64_t* d_src_pool_strides,
                           const int64_t* d_dst_pool_strides,
                           uint64_t dst_total_elements,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
