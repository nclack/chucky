/// Test: downsampling kernels
#include "downsample.h"

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CU(lbl, e)                                                             \
  do {                                                                         \
    if (handle_curesult(e, __FILE__, __LINE__))                                \
      goto lbl;                                                                \
  } while (0)

#define CHECK(lbl, e)                                                          \
  do {                                                                         \
    if (!(e)) {                                                                \
      fprintf(stderr, "%s(%d): Check failed: %s\n", __FILE__, __LINE__, #e);   \
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
  if (name && desc)
    fprintf(stderr, "%s(%d): CUDA error: %s %s\n", file, line, name, desc);
  else
    fprintf(stderr,
            "%s(%d): Failed to retrieve error info for CUresult: %d\n",
            file,
            line,
            ecode);
  return 1;
}

static uint64_t
ceildiv(uint64_t a, uint64_t b)
{
  return (a + b - 1) / b;
}

// -----------------------------------------------------------------------
// Test 1: 2D spatial-only downsample (dims 1,2 downsampled, dim 0 not)
//
// Source: 4x8 tile pool with tile_size=(2,4), so tile_count=(2,2).
//   slot_count = tile_count[0]*tile_count[1] = 4 (but for epoch: skip dim0)
//   Actually for the pool: slot_count = tile_count[1] = 2 (if dim0=epoch)
//
// Simplify: rank=2, size=(8,8), tile_size=(4,4), downsample_mask=0x3.
// Source has tile_count=(2,2), slot_count=4, tile_stride=16, pool_bytes=64*2B.
// Dest has size=(4,4), tile_count=(1,1), slot_count=1, tile_stride=16.
// Each dest element = mean of 4 source elements.
// -----------------------------------------------------------------------
static int
test_2d_spatial(void)
{
  printf("--- test_2d_spatial ---\n");
  int ok = 0;

  const uint8_t rank = 2;
  const uint64_t src_size[2] = { 8, 8 };
  const uint64_t src_tile_size[2] = { 4, 4 };
  const uint64_t dst_size[2] = { 4, 4 };
  const uint64_t dst_tile_size[2] = { 4, 4 };
  const uint8_t downsample_mask = 0x3; // both dims

  // Source tile pool layout
  uint64_t src_tile_count[2];
  for (int d = 0; d < rank; ++d)
    src_tile_count[d] = ceildiv(src_size[d], src_tile_size[d]);
  // src_tile_count = (2, 2)

  uint64_t src_tile_elements = 1;
  for (int d = 0; d < rank; ++d)
    src_tile_elements *= src_tile_size[d];
  // src_tile_elements = 16

  uint64_t src_tile_stride = src_tile_elements; // no padding
  uint64_t src_slot_count = 1;
  for (int d = 0; d < rank; ++d)
    src_slot_count *= src_tile_count[d];
  // src_slot_count = 4

  // Pool strides: pool_stride[d] = tile_stride * prod(tile_count[j] for j>d)
  int64_t src_pool_strides[2];
  {
    int64_t s = (int64_t)src_tile_stride;
    for (int d = rank - 1; d >= 0; --d) {
      src_pool_strides[d] = s;
      s *= (int64_t)src_tile_count[d];
    }
  }
  // src_pool_strides = (32, 16)

  // Destination tile pool layout
  uint64_t dst_tile_count[2];
  for (int d = 0; d < rank; ++d)
    dst_tile_count[d] = ceildiv(dst_size[d], dst_tile_size[d]);
  // dst_tile_count = (1, 1)

  uint64_t dst_tile_elements = 1;
  for (int d = 0; d < rank; ++d)
    dst_tile_elements *= dst_tile_size[d];
  // dst_tile_elements = 16

  uint64_t dst_tile_stride = dst_tile_elements;
  uint64_t dst_slot_count = 1;
  for (int d = 0; d < rank; ++d)
    dst_slot_count *= dst_tile_count[d];
  // dst_slot_count = 1

  int64_t dst_pool_strides[2];
  {
    int64_t s = (int64_t)dst_tile_stride;
    for (int d = rank - 1; d >= 0; --d) {
      dst_pool_strides[d] = s;
      s *= (int64_t)dst_tile_count[d];
    }
  }
  // dst_pool_strides = (16, 16)

  uint64_t src_pool_elements = src_slot_count * src_tile_stride;
  uint64_t dst_pool_elements = dst_slot_count * dst_tile_stride;

  // Fill source data: src[global_y][global_x] = global_y * 8 + global_x
  // Pool layout: for tile (ty, tx) at position (ey, ex) within tile:
  //   pool offset = ty * pool_stride[0] + tx * pool_stride[1] + ey * 4 + ex
  uint16_t* h_src = NULL;
  uint16_t* h_expected = NULL;
  uint16_t* h_result = NULL;
  CUdeviceptr d_src = 0, d_dst = 0;
  CUdeviceptr d_src_ts = 0, d_dst_ts = 0, d_src_ext = 0;
  CUdeviceptr d_src_ps = 0, d_dst_ps = 0;

  h_src = (uint16_t*)calloc(src_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_src);

  for (uint64_t gy = 0; gy < src_size[0]; ++gy) {
    for (uint64_t gx = 0; gx < src_size[1]; ++gx) {
      uint64_t ty = gy / src_tile_size[0];
      uint64_t ey = gy % src_tile_size[0];
      uint64_t tx = gx / src_tile_size[1];
      uint64_t ex = gx % src_tile_size[1];

      uint64_t pool_off = ty * (uint64_t)src_pool_strides[0] +
                          tx * (uint64_t)src_pool_strides[1] +
                          ey * src_tile_size[1] + ex;
      h_src[pool_off] = (uint16_t)(gy * src_size[1] + gx);
    }
  }

  // Compute expected output
  h_expected = (uint16_t*)calloc(dst_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_expected);

  for (uint64_t dy = 0; dy < dst_size[0]; ++dy) {
    for (uint64_t dx = 0; dx < dst_size[1]; ++dx) {
      uint64_t sy0 = 2 * dy;
      uint64_t sy1 = 2 * dy + 1;
      uint64_t sx0 = 2 * dx;
      uint64_t sx1 = 2 * dx + 1;
      if (sy1 >= src_size[0])
        sy1 = src_size[0] - 1;
      if (sx1 >= src_size[1])
        sx1 = src_size[1] - 1;

      uint32_t sum = (uint32_t)(sy0 * src_size[1] + sx0) +
                     (uint32_t)(sy0 * src_size[1] + sx1) +
                     (uint32_t)(sy1 * src_size[1] + sx0) +
                     (uint32_t)(sy1 * src_size[1] + sx1);
      uint16_t mean = (uint16_t)((sum + 2) / 4);

      uint64_t dty = dy / dst_tile_size[0];
      uint64_t dey = dy % dst_tile_size[0];
      uint64_t dtx = dx / dst_tile_size[1];
      uint64_t dex = dx % dst_tile_size[1];
      uint64_t dst_off = dty * (uint64_t)dst_pool_strides[0] +
                         dtx * (uint64_t)dst_pool_strides[1] +
                         dey * dst_tile_size[1] + dex;
      h_expected[dst_off] = mean;
    }
  }

  // GPU execution
  CU(Cleanup, cuMemAlloc(&d_src, src_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemAlloc(&d_dst, dst_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemsetD8(d_dst, 0, dst_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src, h_src, src_pool_elements * sizeof(uint16_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ts, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ts, src_tile_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_dst_ts, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_dst_ts, dst_tile_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ext, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ext, src_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ps, rank * sizeof(int64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ps, src_pool_strides, rank * sizeof(int64_t)));

  CU(Cleanup, cuMemAlloc(&d_dst_ps, rank * sizeof(int64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_dst_ps, dst_pool_strides, rank * sizeof(int64_t)));

  downsample_mean_u16(d_dst,
                      d_src,
                      0, // no pool_b (no dim0 downsample within epochs)
                      rank,
                      downsample_mask,
                      (const uint64_t*)d_dst_ts,
                      (const uint64_t*)d_src_ts,
                      (const uint64_t*)d_src_ext,
                      (const int64_t*)d_src_ps,
                      (const int64_t*)d_dst_ps,
                      dst_pool_elements,
                      0);
  CU(Cleanup, cuCtxSynchronize());

  // Read back
  h_result = (uint16_t*)calloc(dst_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_result);
  CU(Cleanup,
     cuMemcpyDtoH(h_result, d_dst, dst_pool_elements * sizeof(uint16_t)));

  // Verify
  {
    int mismatches = 0;
    for (uint64_t i = 0; i < dst_pool_elements; ++i) {
      if (h_result[i] != h_expected[i]) {
        if (mismatches < 10)
          fprintf(stderr,
                  "  mismatch at %lu: expected %u, got %u\n",
                  (unsigned long)i,
                  h_expected[i],
                  h_result[i]);
        mismatches++;
      }
    }

    if (mismatches == 0) {
      printf("  PASS\n");
      ok = 1;
    } else {
      fprintf(stderr, "  FAIL: %d mismatches\n", mismatches);
    }
  }

Cleanup:
  free(h_src);
  free(h_expected);
  free(h_result);
  if (d_src)
    cuMemFree(d_src);
  if (d_dst)
    cuMemFree(d_dst);
  if (d_src_ts)
    cuMemFree(d_src_ts);
  if (d_dst_ts)
    cuMemFree(d_dst_ts);
  if (d_src_ext)
    cuMemFree(d_src_ext);
  if (d_src_ps)
    cuMemFree(d_src_ps);
  if (d_dst_ps)
    cuMemFree(d_dst_ps);
  return ok ? 0 : 1;
}

// -----------------------------------------------------------------------
// Test 2: 3D downsample including dim 0 (two pools)
//
// rank=3, src_size=(2,4,4), tile_size=(1,2,2), downsample_mask=0x7 (all dims)
// Source: tile_count=(2,2,2), but per-epoch (dim0=1) slot_count=4.
// Two epochs in pool_a (epoch 0) and pool_b (epoch 1).
// Dest: size=(1,2,2), tile_count=(1,1,1), slot_count=1.
// Each dest element = mean of 8 source elements.
// -----------------------------------------------------------------------
static int
test_3d_with_dim0(void)
{
  printf("--- test_3d_with_dim0 ---\n");
  int ok = 0;

  const uint8_t rank = 3;
  const uint64_t src_size[3] = { 2, 4, 4 };
  const uint64_t src_tile_size[3] = { 1, 2, 2 };
  const uint64_t dst_size[3] = { 1, 2, 2 };
  const uint64_t dst_tile_size[3] = { 1, 2, 2 };
  const uint8_t downsample_mask = 0x7;

  // Source: per-epoch pool. Dim 0 has 1 tile per epoch (epoch = 1 slice).
  // tile_count within epoch: (1, 2, 2) since dim0 has 1 tile in pool
  uint64_t src_tile_count_in_pool[3] = { 1, 2, 2 };

  uint64_t src_tile_elements = 1;
  for (int d = 0; d < rank; ++d)
    src_tile_elements *= src_tile_size[d];
  // = 4

  uint64_t src_tile_stride = src_tile_elements;
  uint64_t src_slot_count = 1;
  for (int d = 0; d < rank; ++d)
    src_slot_count *= src_tile_count_in_pool[d];
  // = 4

  int64_t src_pool_strides[3];
  {
    int64_t s = (int64_t)src_tile_stride;
    for (int d = rank - 1; d >= 0; --d) {
      src_pool_strides[d] = s;
      s *= (int64_t)src_tile_count_in_pool[d];
    }
  }
  // src_pool_strides = (16, 8, 4)

  // Destination
  uint64_t dst_tile_count[3];
  for (int d = 0; d < rank; ++d)
    dst_tile_count[d] = ceildiv(dst_size[d], dst_tile_size[d]);
  // = (1, 1, 1)

  uint64_t dst_tile_elements = 1;
  for (int d = 0; d < rank; ++d)
    dst_tile_elements *= dst_tile_size[d];
  // = 4

  uint64_t dst_tile_stride = dst_tile_elements;
  uint64_t dst_slot_count = 1;
  for (int d = 0; d < rank; ++d)
    dst_slot_count *= dst_tile_count[d];
  // = 1

  int64_t dst_pool_strides[3];
  {
    int64_t s = (int64_t)dst_tile_stride;
    for (int d = rank - 1; d >= 0; --d) {
      dst_pool_strides[d] = s;
      s *= (int64_t)dst_tile_count[d];
    }
  }
  // dst_pool_strides = (4, 4, 4)

  uint64_t src_pool_elements = src_slot_count * src_tile_stride;
  uint64_t dst_pool_elements = dst_slot_count * dst_tile_stride;

  // Fill pool_a (epoch 0, z=0) and pool_b (epoch 1, z=1)
  // Value = z * 16 + y * 4 + x
  uint16_t* h_pool_a = NULL;
  uint16_t* h_pool_b = NULL;
  uint16_t* h_expected = NULL;
  uint16_t* h_result = NULL;
  CUdeviceptr d_pool_a = 0, d_pool_b = 0, d_dst = 0;
  CUdeviceptr d_src_ts = 0, d_dst_ts = 0, d_src_ext = 0;
  CUdeviceptr d_src_ps = 0, d_dst_ps = 0;

  h_pool_a = (uint16_t*)calloc(src_pool_elements, sizeof(uint16_t));
  h_pool_b = (uint16_t*)calloc(src_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_pool_a && h_pool_b);

  for (int z = 0; z < 1; ++z) {
    for (uint64_t y = 0; y < src_size[1]; ++y) {
      for (uint64_t x = 0; x < src_size[2]; ++x) {
        uint64_t ty = y / src_tile_size[1];
        uint64_t ey = y % src_tile_size[1];
        uint64_t tx = x / src_tile_size[2];
        uint64_t ex = x % src_tile_size[2];

        uint64_t off = (uint64_t)z * (uint64_t)src_pool_strides[0] +
                       ty * (uint64_t)src_pool_strides[1] +
                       tx * (uint64_t)src_pool_strides[2] +
                       ey * src_tile_size[2] + ex;

        // pool_a = epoch 0 (z=0)
        h_pool_a[off] = (uint16_t)(0 * 16 + y * 4 + x);
        // pool_b = epoch 1 (z=1)
        h_pool_b[off] = (uint16_t)(1 * 16 + y * 4 + x);
      }
    }
  }

  // Expected output: mean of 8 neighbors
  h_expected = (uint16_t*)calloc(dst_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_expected);

  for (uint64_t dy = 0; dy < dst_size[1]; ++dy) {
    for (uint64_t dx = 0; dx < dst_size[2]; ++dx) {
      uint32_t sum = 0;
      for (int dz = 0; dz < 2; ++dz) {
        for (int offy = 0; offy < 2; ++offy) {
          for (int offx = 0; offx < 2; ++offx) {
            uint64_t sy = 2 * dy + (uint64_t)offy;
            uint64_t sx = 2 * dx + (uint64_t)offx;
            if (sy >= src_size[1])
              sy = src_size[1] - 1;
            if (sx >= src_size[2])
              sx = src_size[2] - 1;
            sum += (uint32_t)(dz * 16 + sy * 4 + sx);
          }
        }
      }
      uint16_t mean = (uint16_t)((sum + 4) / 8);
      uint64_t dty = dy / dst_tile_size[1];
      uint64_t dey = dy % dst_tile_size[1];
      uint64_t dtx = dx / dst_tile_size[2];
      uint64_t dex = dx % dst_tile_size[2];
      uint64_t off = dty * (uint64_t)dst_pool_strides[1] +
                     dtx * (uint64_t)dst_pool_strides[2] +
                     dey * dst_tile_size[2] + dex;
      h_expected[off] = mean;
    }
  }

  CU(Cleanup, cuMemAlloc(&d_pool_a, src_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemAlloc(&d_pool_b, src_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemAlloc(&d_dst, dst_pool_elements * sizeof(uint16_t)));
  CU(Cleanup, cuMemsetD8(d_dst, 0, dst_pool_elements * sizeof(uint16_t)));
  CU(Cleanup,
     cuMemcpyHtoD(d_pool_a, h_pool_a, src_pool_elements * sizeof(uint16_t)));
  CU(Cleanup,
     cuMemcpyHtoD(d_pool_b, h_pool_b, src_pool_elements * sizeof(uint16_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ts, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ts, src_tile_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_dst_ts, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_dst_ts, dst_tile_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ext, rank * sizeof(uint64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ext, src_size, rank * sizeof(uint64_t)));

  CU(Cleanup, cuMemAlloc(&d_src_ps, rank * sizeof(int64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_src_ps, src_pool_strides, rank * sizeof(int64_t)));

  CU(Cleanup, cuMemAlloc(&d_dst_ps, rank * sizeof(int64_t)));
  CU(Cleanup, cuMemcpyHtoD(d_dst_ps, dst_pool_strides, rank * sizeof(int64_t)));

  downsample_mean_u16(d_dst,
                      d_pool_a,
                      d_pool_b,
                      rank,
                      downsample_mask,
                      (const uint64_t*)d_dst_ts,
                      (const uint64_t*)d_src_ts,
                      (const uint64_t*)d_src_ext,
                      (const int64_t*)d_src_ps,
                      (const int64_t*)d_dst_ps,
                      dst_pool_elements,
                      0);
  CU(Cleanup, cuCtxSynchronize());

  h_result = (uint16_t*)calloc(dst_pool_elements, sizeof(uint16_t));
  CHECK(Cleanup, h_result);
  CU(Cleanup,
     cuMemcpyDtoH(h_result, d_dst, dst_pool_elements * sizeof(uint16_t)));

  {
    int mismatches = 0;
    for (uint64_t i = 0; i < dst_pool_elements; ++i) {
      if (h_result[i] != h_expected[i]) {
        if (mismatches < 10)
          fprintf(stderr,
                  "  mismatch at %lu: expected %u, got %u\n",
                  (unsigned long)i,
                  h_expected[i],
                  h_result[i]);
        mismatches++;
      }
    }

    if (mismatches == 0) {
      printf("  PASS\n");
      ok = 1;
    } else {
      fprintf(stderr, "  FAIL: %d mismatches\n", mismatches);
    }
  }

Cleanup:
  free(h_pool_a);
  free(h_pool_b);
  free(h_expected);
  free(h_result);
  if (d_pool_a)
    cuMemFree(d_pool_a);
  if (d_pool_b)
    cuMemFree(d_pool_b);
  if (d_dst)
    cuMemFree(d_dst);
  if (d_src_ts)
    cuMemFree(d_src_ts);
  if (d_dst_ts)
    cuMemFree(d_dst_ts);
  if (d_src_ext)
    cuMemFree(d_src_ext);
  if (d_src_ps)
    cuMemFree(d_src_ps);
  if (d_dst_ps)
    cuMemFree(d_dst_ps);
  return ok ? 0 : 1;
}

int
main(void)
{
  CU(Fail, cuInit(0));

  CUdevice dev;
  CU(Fail, cuDeviceGet(&dev, 0));

  CUcontext ctx;
  CU(Fail, cuCtxCreate(&ctx, 0, dev));

  int ecode = 0;
  ecode |= test_2d_spatial();
  ecode |= test_3d_with_dim0();

  cuCtxDestroy(ctx);
  return ecode;

Fail:
  fprintf(stderr, "Failed to initialize CUDA\n");
  return 1;
}
