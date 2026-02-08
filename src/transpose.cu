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
                  const uint64_t* d_shape,
                  const int64_t* d_strides,
                  CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;

  uint64_t* out_beg = (uint64_t*)d_beg;
  uint64_t* out_end = (uint64_t*)d_end;
  const uint64_t n = out_end - out_beg;

  const int block_size = 256;
  const int grid_size = (n + block_size - 1) / block_size;

  transpose_indices_k<<<grid_size, block_size, 0, cuda_stream>>>(
    out_beg, i_offset, i_offset + n, rank, d_shape, d_strides);
}

// Transpose u16 data kernel - v0
// Uses shared memory to stage data before computing output positions
template<typename T>
__global__ void
transpose_v0_k(T* d_dst,
               const T* d_src,
               uint64_t src_size,
               uint64_t i_offset,
               uint8_t rank,
               const uint64_t* shape,
               const int64_t* strides)
{
  // Shared memory buffer: sized for the block
  constexpr int ELEMENTS_PER_BLOCK = (1 << 12) / sizeof(T); // 4KB
  __shared__ T shared_buf[ELEMENTS_PER_BLOCK];

  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;

  // Block-strided load
  for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
    int global_idx = block_offset + i;
    if (global_idx < src_size) {
      shared_buf[i] = d_src[global_idx];
    }
  }
  __syncthreads();

  for (int i = tid; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
    int global_idx = block_offset + i;

    if (global_idx < src_size) {
      uint64_t input_idx = i_offset + global_idx;
      uint64_t out_offset = 0;
      uint64_t rest = input_idx;

      for (int d = rank - 1; d >= 0; --d) {
        const uint64_t coord = rest % shape[d];
        rest /= shape[d];
        out_offset += coord * strides[d];
      }

      // Store
      d_dst[out_offset] = shared_buf[i];
    }
  }
}

extern "C" void
transpose_u16_v0(CUdeviceptr d_dst_beg,
                 CUdeviceptr d_dst_end,
                 CUdeviceptr d_src_beg,
                 CUdeviceptr d_src_end,
                 uint64_t i_offset,
                 uint8_t rank,
                 const uint64_t* d_shape,
                 const int64_t* d_strides,
                 CUstream stream)
{
  (void)d_dst_end; // UNUSED

  cudaStream_t cuda_stream = (cudaStream_t)stream;

  uint16_t* src_beg = (uint16_t*)d_src_beg;
  uint16_t* src_end = (uint16_t*)d_src_end;
  const uint64_t src_size = src_end - src_beg;

  if (src_size == 0)
    return;

  const int block_size = 256;
  const int elements_per_block = (1 << 12) / sizeof(uint16_t);
  const int grid_size =
    (src_size + elements_per_block - 1) / elements_per_block;

  transpose_v0_k<uint16_t>
    <<<grid_size, block_size, 0, cuda_stream>>>((uint16_t*)d_dst_beg,
                                                src_beg,
                                                src_size,
                                                i_offset,
                                                rank,
                                                d_shape,
                                                d_strides);
}
