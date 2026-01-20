#include <stdint.h>
#include <cuda_runtime.h>

__global__ void
fill(uint16_t* beg, uint16_t* end)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (beg + i < end) {
    beg[i] = i;
  }
}

// C-callable wrapper to launch the fill kernel
extern "C" cudaError_t
launch_fill(uint16_t* d_beg, uint16_t* d_end, int grid_size, int block_size)
{
  fill<<<grid_size, block_size>>>(d_beg, d_end);
  return cudaGetLastError();
}
