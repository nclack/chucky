#include "transpose.h"
#include <stdint.h>

__global__ void
fill_k(uint16_t* beg, uint16_t* end)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (beg + i < end) {
    beg[i] = i;
  }
}

extern "C" void
fill_u16(CUdeviceptr d_beg,
         CUdeviceptr d_end,
         int grid_size,
         int block_size,
         CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  fill_k<<<grid_size, block_size, 0, cuda_stream>>>((uint16_t*)d_beg,
                                                    (uint16_t*)d_end);
}
