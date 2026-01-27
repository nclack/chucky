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

// Transpose indices kernel
// Each thread independently computes one output index using the add() algorithm
__global__ void
transpose_indices_k(uint64_t* d_out,
                    uint64_t beg,
                    uint64_t end,
                    uint8_t rank,
                    const uint64_t* shape,
                    const int64_t* strides)
{
  const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t input_idx = beg + gid;

  if (input_idx < end) {
    // Convert input index to output index using add() logic
    // Decompose input_idx into coordinates, then compute output offset
    uint64_t out = 0;
    uint64_t rest = input_idx;

    for (int d = rank - 1; d >= 0; --d) {
      const uint64_t coord = rest % shape[d];
      rest /= shape[d];
      out += coord * strides[d];
    }

    // Note: base_output_offset (o_offset) is unused in this simple
    // implementation It's provided for potential future optimization
    d_out[gid] = out;
  }
}

extern "C" void
transpose_indices(CUdeviceptr d_beg,
                  CUdeviceptr d_end,
                  uint64_t i_offset,
                  uint8_t rank,
                  const uint64_t* __restrict__ shape,
                  const int64_t* __restrict__ strides,
                  CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  // Compute grid dimensions
  uint64_t* out_beg = (uint64_t*)d_beg;
  uint64_t* out_end = (uint64_t*)d_end;
  const uint64_t n = out_end - out_beg;

  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  // Async copy shape and strides to device
  uint64_t* d_shape;
  int64_t* d_strides;
  cudaMallocAsync(&d_shape, rank * sizeof(uint64_t), cuda_stream);
  cudaMallocAsync(&d_strides, rank * sizeof(int64_t), cuda_stream);
  cudaMemcpyAsync(d_shape,
                  shape,
                  rank * sizeof(uint64_t),
                  cudaMemcpyHostToDevice,
                  cuda_stream);
  cudaMemcpyAsync(d_strides,
                  strides,
                  rank * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  cuda_stream);

  // Launch kernel
  transpose_indices_k<<<grid_size, block_size, 0, cuda_stream>>>(
    out_beg, i_offset, i_offset + n, rank, d_shape, d_strides);

  cudaFreeAsync(d_shape, cuda_stream);
  cudaFreeAsync(d_strides, cuda_stream);
}
