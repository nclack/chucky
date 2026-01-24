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

  void transpose_next(CUdeviceptr d_beg,
                      CUdeviceptr d_end,
                      uint8_t bytes_per_element,
                      uint64_t i_offset,
                      uint64_t o_offset,
                      uint8_t rank,
                      uint64_t shape[MAX_RANK],
                      int64_t strides[MAX_RANK]);

#ifdef __cplusplus
}
#endif
