#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>

#define CU(lbl, e)                                                             \
  do {                                                                         \
    if (handle_curesult(e, __FILE__, __LINE__))                                \
      goto lbl;                                                                \
  } while (0)

#define CUDA(lbl, e)                                                           \
  do {                                                                         \
    if (handle_cudaerror(e, __FILE__, __LINE__))                               \
      goto lbl;                                                                \
  } while (0)

#define CHECK_EQ(lbl, a, b)                                                    \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      fprintf(stderr,                                                          \
              "%s(%d): Check failed: %s != %s (%u != %u)\n",                   \
              __FILE__,                                                        \
              __LINE__,                                                        \
              #a,                                                              \
              #b,                                                              \
              (unsigned)(a),                                                   \
              (unsigned)(b));                                                  \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

static int
handle_curesult(CUresult ecode, const char* file, int line)
{
  if (ecode == CUDA_SUCCESS)
    return 0;
  const char *name, *desc;
  cuGetErrorName(ecode, &name);
  cuGetErrorString(ecode, &desc);
  if (name && desc) {
    fprintf(stderr, "%s(%d): CUDA error: %s %s\n", file, line, name, desc);
  } else {
    fprintf(stderr,
            "%s(%d): Failed to retrieve error info for CUresult: %d\n",
            file,
            line,
            ecode);
  }
  return 1;
}

static int
handle_cudaerror(cudaError_t ecode, const char* file, int line)
{
  if (ecode == cudaSuccess)
    return 0;
  fprintf(stderr,
          "%s(%d): CUDA Runtime error: %s\n",
          file,
          line,
          cudaGetErrorString(ecode));
  return 1;
}

// External wrapper function that launches the fill kernel
extern cudaError_t
launch_fill(uint16_t* d_beg, uint16_t* d_end, int grid_size, int block_size);

int
main(int argc, char* argv[])
{
  int ecode = 0;
  CUcontext ctx = NULL;
  CUdeviceptr d_data = 0;
  uint16_t* h_data = NULL;
  CUdevice dev = -1;

  CU(InitFail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  const size_t nbytes = 1ULL << 30;
  const size_t nelems = nbytes / sizeof(uint16_t);

  CU(Fail, cuMemAlloc(&d_data, nbytes));

  CU(Fail, cuMemAllocHost((void**)&h_data, nbytes));

  const int block_size = 256;
  const int grid_size = (nelems + block_size - 1) / block_size;

  uint16_t* d_end = (uint16_t*)d_data + nelems;
  CUDA(Fail, launch_fill((uint16_t*)d_data, d_end, grid_size, block_size));

  CU(Fail, cuCtxSynchronize());

  CU(Fail, cuMemcpyDtoH(h_data, d_data, nbytes));

  CHECK_EQ(Fail, h_data[0], 0);
  CHECK_EQ(Fail, h_data[100], 100);
  CHECK_EQ(Fail, h_data[1000], 1000);
  CHECK_EQ(Fail, h_data[65535], 65535);

  printf("OK\n");

Cleanup:
  cuMemFreeHost(h_data);
  cuMemFree(d_data);
  cuCtxDestroy(ctx);
  return ecode;

Fail:
  ecode = 1;
  printf("FAIL!\n");
  goto Cleanup;

InitFail:
  return 1;
}
