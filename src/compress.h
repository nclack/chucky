#pragma once

#include <cuda.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Minimum alignment required for input chunk addresses.
  size_t compress_get_input_alignment(void);

  // Maximum compressed output size for a single chunk of uncompressed_bytes.
  size_t compress_get_max_output_size(size_t uncompressed_bytes);

  // Temporary workspace size for compressing a batch.
  size_t compress_get_temp_size(size_t batch_size,
                                size_t max_uncompressed_bytes);

  // Compress a batch of chunks asynchronously on the given stream.
  // All pointer arrays and size arrays must reside in device memory.
  // Returns 0 on success, non-zero on error.
  int compress_batch_async(const void* const* d_uncomp_ptrs,
                           const size_t* d_uncomp_sizes,
                           size_t max_uncompressed_bytes,
                           size_t num_chunks,
                           void* d_temp,
                           size_t temp_bytes,
                           void* const* d_comp_ptrs,
                           size_t* d_comp_sizes,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
