#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

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
                         const uint64_t* __restrict__ shape,
                         const int64_t* __restrict__ strides,
                         CUstream stream);

  void transpose_u16_v0(CUdeviceptr d_dst_beg,
                        CUdeviceptr d_dst_end,
                        CUdeviceptr d_src_beg,
                        CUdeviceptr d_src_end,
                        uint64_t i_offset,
                        uint8_t rank,
                        const uint64_t* __restrict__ shape,
                        const int64_t* __restrict__ strides,
                        CUstream stream);
#ifdef __cplusplus
}
#endif
