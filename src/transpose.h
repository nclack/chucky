#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_RANK (64)

  void fill_u16(CUdeviceptr d_beg,
                CUdeviceptr d_end,
                int grid_size,
                int block_size,
                CUstream stream);

  void transpose_indices(CUdeviceptr d_beg,
                         CUdeviceptr d_end,
                         uint64_t i_offset,
                         uint8_t rank,
                         const uint64_t* d_shape,
                         const int64_t* d_strides,
                         CUstream stream);

  void transpose_u8_v0(CUdeviceptr d_dst_beg,
                       CUdeviceptr d_dst_end,
                       CUdeviceptr d_src_beg,
                       CUdeviceptr d_src_end,
                       uint64_t i_offset,
                       uint8_t rank,
                       const uint64_t* d_shape,
                       const int64_t* d_strides,
                       CUstream stream);

  void transpose_u16_v0(CUdeviceptr d_dst_beg,
                        CUdeviceptr d_dst_end,
                        CUdeviceptr d_src_beg,
                        CUdeviceptr d_src_end,
                        uint64_t i_offset,
                        uint8_t rank,
                        const uint64_t* d_shape,
                        const int64_t* d_strides,
                        CUstream stream);

  void transpose_u32_v0(CUdeviceptr d_dst_beg,
                        CUdeviceptr d_dst_end,
                        CUdeviceptr d_src_beg,
                        CUdeviceptr d_src_end,
                        uint64_t i_offset,
                        uint8_t rank,
                        const uint64_t* d_shape,
                        const int64_t* d_strides,
                        CUstream stream);
  void transpose_u64_v0(CUdeviceptr d_dst_beg,
                        CUdeviceptr d_dst_end,
                        CUdeviceptr d_src_beg,
                        CUdeviceptr d_src_end,
                        uint64_t i_offset,
                        uint8_t rank,
                        const uint64_t* d_shape,
                        const int64_t* d_strides,
                        CUstream stream);
#ifdef __cplusplus
}
#endif
