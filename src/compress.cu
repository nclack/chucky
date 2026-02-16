#include "compress.h"
#include "log/log.h"
#include <nvcomp/zstd.h>

static const char*
nvcomp_status_name(nvcompStatus_t st)
{
  switch (st) {
    case nvcompSuccess:
      return "nvcompSuccess";
    case nvcompErrorInvalidValue:
      return "nvcompErrorInvalidValue";
    case nvcompErrorNotSupported:
      return "nvcompErrorNotSupported";
    case nvcompErrorCannotDecompress:
      return "nvcompErrorCannotDecompress";
    case nvcompErrorBadChecksum:
      return "nvcompErrorBadChecksum";
    case nvcompErrorCannotVerifyChecksums:
      return "nvcompErrorCannotVerifyChecksums";
    case nvcompErrorCudaError:
      return "nvcompErrorCudaError";
    case nvcompErrorInternal:
      return "nvcompErrorInternal";
    default:
      return "nvcompUnknown";
  }
}

static int
handle_nvcomp(int level,
              nvcompStatus_t st,
              const char* file,
              int line,
              const char* expr)
{
  if (st == nvcompSuccess)
    return 0;
  log_log(level,
          file,
          line,
          "nvcomp error: %s (%d) %s",
          nvcomp_status_name(st),
          (int)st,
          expr);
  return 1;
}

#define NVCOMP(lbl, e)                                                         \
  do {                                                                         \
    nvcompStatus_t st_ = (e);                                                  \
    if (st_ != nvcompSuccess &&                                                \
        handle_nvcomp(LOG_ERROR, st_, __FILE__, __LINE__, #e)) {               \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

extern "C" size_t
compress_get_input_alignment(void)
{
  return nvcompZstdRequiredCompressionAlignment;
}

extern "C" size_t
compress_get_max_output_size(size_t uncompressed_bytes)
{
  size_t max_compressed = 0;
  NVCOMP(Fail,
         nvcompBatchedZstdCompressGetMaxOutputChunkSize(
           uncompressed_bytes,
           nvcompBatchedZstdCompressDefaultOpts,
           &max_compressed));
  return max_compressed;

Fail:
  return 0;
}

extern "C" size_t
compress_get_temp_size(size_t batch_size, size_t max_uncompressed_bytes)
{
  size_t temp = 0;
  NVCOMP(Fail,
         nvcompBatchedZstdCompressGetTempSizeAsync(
           batch_size,
           max_uncompressed_bytes,
           nvcompBatchedZstdCompressDefaultOpts,
           &temp,
           batch_size * max_uncompressed_bytes));
  return temp;

Fail:
  return 0;
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
  NVCOMP(Fail,
         nvcompBatchedZstdCompressAsync(d_uncomp_ptrs,
                                        d_uncomp_sizes,
                                        max_uncompressed_bytes,
                                        num_chunks,
                                        d_temp,
                                        temp_bytes,
                                        d_comp_ptrs,
                                        d_comp_sizes,
                                        nvcompBatchedZstdCompressDefaultOpts,
                                        NULL,
                                        cuda_stream));
  return 0;

Fail:
  return 1;
}
