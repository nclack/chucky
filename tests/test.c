#include "transpose.h"
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#define CU(lbl, e)                                                             \
  do {                                                                         \
    if (handle_curesult(e, __FILE__, __LINE__))                                \
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

int
main(int argc, char* argv[])
{
  (void)argc; // unused
  (void)argv; // unused

  int ecode = 0;
  CUcontext ctx = NULL;
  CUdeviceptr d_data[2] = { 0 };
  uint16_t* h_data[2] = { 0 };
  CUstream stream_compute = NULL, stream_transfer = NULL;
  CUevent event_kernel_start[2] = { 0 }, event_kernel_end[2] = { 0 },
          event_transfer_start[2] = { 0 }, event_transfer_end[2] = { 0 };
  CUdevice dev = -1;

  CU(InitFail, cuInit(0));
  CU(Fail, cuDeviceGet(&dev, 0));
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  const size_t nbytes = 1ULL << 30;
  const size_t nelems = nbytes / sizeof(uint16_t);
  const size_t total_bytes = 10ULL << 30;
  const int niter = (int)(total_bytes / nbytes);

  CU(Fail, cuMemAlloc(&d_data[0], nbytes));
  CU(Fail, cuMemAlloc(&d_data[1], nbytes));

  CU(Fail,
     cuMemHostAlloc((void**)&h_data[0], nbytes, CU_MEMHOSTALLOC_WRITECOMBINED));
  CU(Fail,
     cuMemHostAlloc((void**)&h_data[1], nbytes, CU_MEMHOSTALLOC_WRITECOMBINED));

  CU(Fail, cuStreamCreate(&stream_compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&stream_transfer, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; i++) {
    CU(Fail, cuEventCreate(&event_kernel_start[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&event_kernel_end[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&event_transfer_start[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&event_transfer_end[i], CU_EVENT_DEFAULT));
  }

  const int block_size = 1024;
  const int grid_size = (int)((nelems + block_size - 1) / block_size);

  printf("Processing %d iterations of 1GB (%.2f GB total)\n",
         niter,
         total_bytes / (1024.0 * 1024.0 * 1024.0));

  float total_kernel_time = 0.0f;
  float total_transfer_time = 0.0f;

  for (int iter = 0; iter < niter; iter++) {
    int buf_idx = iter % 2;

    CU(Fail, cuEventRecord(event_kernel_start[buf_idx], stream_compute));
    CUdeviceptr d_end = d_data[buf_idx] + nbytes;
    fill_u16(d_data[buf_idx], d_end, grid_size, block_size, stream_compute);
    CU(Fail, cuEventRecord(event_kernel_end[buf_idx], stream_compute));

    CU(Fail, cuStreamWaitEvent(stream_transfer, event_kernel_end[buf_idx], 0));
    CU(Fail, cuEventRecord(event_transfer_start[buf_idx], stream_transfer));
    CU(Fail,
       cuMemcpyDtoHAsync(
         h_data[buf_idx], d_data[buf_idx], nbytes, stream_transfer));
    CU(Fail, cuEventRecord(event_transfer_end[buf_idx], stream_transfer));

    CU(Fail, cuStreamSynchronize(stream_transfer));

    float kernel_time, transfer_time;
    CU(Fail,
       cuEventElapsedTime(
         &kernel_time, event_kernel_start[buf_idx], event_kernel_end[buf_idx]));
    CU(Fail,
       cuEventElapsedTime(&transfer_time,
                          event_transfer_start[buf_idx],
                          event_transfer_end[buf_idx]));

    total_kernel_time += kernel_time;
    total_transfer_time += transfer_time;

    CHECK_EQ(Fail, h_data[buf_idx][0], 0);
    CHECK_EQ(Fail, h_data[buf_idx][100], 100);
    CHECK_EQ(Fail, h_data[buf_idx][1000], 1000);
    CHECK_EQ(Fail, h_data[buf_idx][65535], 65535);
  }

  char device_name[256];
  CU(Fail, cuDeviceGetName(device_name, sizeof(device_name), dev));

  float measured_bw_gbs =
    (float)((total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_transfer_time / 1000.0));

  printf("OK\n");
  printf("Device: %s\n", device_name);
  printf("Total: kernel=%.2fms, transfer=%.2fms\n",
         total_kernel_time,
         total_transfer_time);
  printf("Throughput: kernel=%.2f GB/s, transfer=%.2f GB/s\n",
         (total_bytes / (1024.0 * 1024.0 * 1024.0)) /
           (total_kernel_time / 1000.0),
         measured_bw_gbs);

Cleanup:
  for (int i = 0; i < 2; i++) {
    cuEventDestroy(event_transfer_end[i]);
    cuEventDestroy(event_transfer_start[i]);
    cuEventDestroy(event_kernel_end[i]);
    cuEventDestroy(event_kernel_start[i]);
  }
  cuStreamDestroy(stream_transfer);
  cuStreamDestroy(stream_compute);
  cuMemFreeHost(h_data[1]);
  cuMemFreeHost(h_data[0]);
  cuMemFree(d_data[1]);
  cuMemFree(d_data[0]);
  cuCtxDestroy(ctx);
  return ecode;

Fail:
  ecode = 1;
  printf("FAIL!\n");
  goto Cleanup;

InitFail:
  return 1;
}
