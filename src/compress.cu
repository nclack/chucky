#include "compress.h"
#include <nvcomp/zstd.h>

extern "C" size_t
compress_get_input_alignment(void)
{
  return nvcompZstdRequiredCompressionAlignment;
}

extern "C" size_t
compress_get_max_output_size(size_t uncompressed_bytes)
{
  size_t max_compressed = 0;
  nvcompStatus_t st = nvcompBatchedZstdCompressGetMaxOutputChunkSize(
    uncompressed_bytes, nvcompBatchedZstdCompressDefaultOpts, &max_compressed);
  if (st != nvcompSuccess)
    return 0;
  return max_compressed;
}

extern "C" size_t
compress_get_temp_size(size_t batch_size, size_t max_uncompressed_bytes)
{
  size_t temp = 0;
  nvcompStatus_t st = nvcompBatchedZstdCompressGetTempSizeAsync(
    batch_size,
    max_uncompressed_bytes,
    nvcompBatchedZstdCompressDefaultOpts,
    &temp,
    batch_size * max_uncompressed_bytes);
  if (st != nvcompSuccess)
    return 0;
  return temp;
}

extern "C" int
compress_batch_async(const void* const* d_uncomp_ptrs,
                     const size_t* d_uncomp_sizes,
                     size_t max_uncompressed_bytes,
                     size_t num_chunks,
                     void* d_temp,
                     size_t temp_bytes,
                     void* const* d_comp_ptrs,
                     size_t* d_comp_sizes,
                     CUstream stream)
{
  cudaStream_t cuda_stream = (cudaStream_t)stream;
  nvcompStatus_t st = nvcompBatchedZstdCompressAsync(
    d_uncomp_ptrs,
    d_uncomp_sizes,
    max_uncompressed_bytes,
    num_chunks,
    d_temp,
    temp_bytes,
    d_comp_ptrs,
    d_comp_sizes,
    nvcompBatchedZstdCompressDefaultOpts,
    NULL,
    cuda_stream);
  return st != nvcompSuccess;
}
