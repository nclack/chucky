#include "lod.h"
#include "lod_plan.h"
#include "metric.cuda.h"
#include "prelude.cuda.h"
#include "prelude.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LOD LOD_MAX_LEVELS

static int
upload(CUdeviceptr* d_ptr, const void* h_ptr, size_t bytes)
{
  CUresult r = cuMemAlloc(d_ptr, bytes);
  if (r != CUDA_SUCCESS)
    return 0;
  r = cuMemcpyHtoD(*d_ptr, h_ptr, bytes);
  if (r != CUDA_SUCCESS) {
    cuMemFree(*d_ptr);
    *d_ptr = 0;
    return 0;
  }
  return 1;
}

static void
report_metric(const struct stream_metric* m)
{
  if (m->count == 0)
    return;
  float avg_ms = m->ms / (float)m->count;
  double avg_bytes = m->total_bytes / (double)m->count;
  printf("  %-8s %7.3f ms  %6.2f GB/s  (%d iters)\n",
         m->name,
         avg_ms,
         avg_bytes / ((double)avg_ms * 1e6),
         m->count);
}

static int
bench_scatter_lut(const char* label,
                  int ndim,
                  const uint64_t* shape,
                  uint8_t lod_mask,
                  int niter)
{
  printf("--- %s ---\n", label);
  int ok = 0;
  struct lod_plan plan = { 0 };
  CUstream stream = NULL;
  CUevent ev0 = NULL, ev1 = NULL;
  CUdeviceptr d_src = 0, d_dst = 0, d_dst2 = 0, d_dst3 = 0;
  CUdeviceptr d_full_shape = 0, d_lod_shape = 0;
  CUdeviceptr d_lut = 0;
  CUdeviceptr d_src_lut = 0, d_lod_strides = 0, d_batch_offsets = 0;

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, NULL, lod_mask, MAX_LOD));

  uint64_t n_elements = 1;
  for (int d = 0; d < ndim; ++d)
    n_elements *= shape[d];
  uint64_t lod_count = plan.lod_counts[0];

  printf("  n_elements=%llu  lod_count=%llu  lod_ndim=%d  nlod=%d\n",
         (unsigned long long)n_elements,
         (unsigned long long)lod_count,
         plan.lod_ndim,
         plan.nlod);

  size_t src_bytes = n_elements * sizeof(uint16_t);
  size_t dst_bytes = plan.batch_count * lod_count * sizeof(uint16_t);
  size_t lut_bytes = lod_count * sizeof(uint32_t);

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuEventCreate(&ev0, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev1, CU_EVENT_DEFAULT));

  CU(Fail, cuMemAlloc(&d_src, src_bytes));
  CU(Fail, cuMemAlloc(&d_dst, dst_bytes));
  CU(Fail, cuMemAlloc(&d_dst2, dst_bytes));
  CU(Fail, cuMemAlloc(&d_lut, lut_bytes));

  CHECK(Fail, upload(&d_full_shape, plan.shapes[0], ndim * sizeof(uint64_t)));
  CHECK(
    Fail,
    upload(&d_lod_shape, plan.lod_shapes[0], plan.lod_ndim * sizeof(uint64_t)));

  // Fill src with pattern
  CU(Fail, cuMemsetD16(d_src, 0x1234, n_elements));

  // Warmup
  lod_scatter(d_dst,
              d_src,
              lod_dtype_u16,
              ndim,
              n_elements,
              d_full_shape,
              d_lod_shape,
              plan.lod_ndim,
              plan.lod_shapes[0],
              plan.lod_mask,
              lod_count,
              stream);
  CU(Fail, cuStreamSynchronize(stream));

  // Bench: original scatter
  struct stream_metric m_orig = { .name = "original", .best_ms = 1e30f };
  for (int i = 0; i < niter; ++i) {
    CU(Fail, cuEventRecord(ev0, stream));
    lod_scatter(d_dst,
                d_src,
                lod_dtype_u16,
                ndim,
                n_elements,
                d_full_shape,
                d_lod_shape,
                plan.lod_ndim,
                plan.lod_shapes[0],
                plan.lod_mask,
                lod_count,
                stream);
    CU(Fail, cuEventRecord(ev1, stream));
    CU(Fail, cuStreamSynchronize(stream));
    accumulate_metric_cu(&m_orig, ev0, ev1, src_bytes);
  }

  // Build forward LUT (time it)
  struct stream_metric m_build = { .name = "lut_build", .best_ms = 1e30f };
  for (int i = 0; i < niter; ++i) {
    CU(Fail, cuEventRecord(ev0, stream));
    lod_build_scatter_lut(
      d_lut, d_lod_shape, plan.lod_ndim, plan.lod_shapes[0], lod_count, stream);
    CU(Fail, cuEventRecord(ev1, stream));
    CU(Fail, cuStreamSynchronize(stream));
    accumulate_metric_cu(&m_build, ev0, ev1, lut_bytes);
  }

  // Bench: LUT scatter
  struct stream_metric m_lut = { .name = "lut_scat", .best_ms = 1e30f };
  for (int i = 0; i < niter; ++i) {
    CU(Fail, cuEventRecord(ev0, stream));
    lod_scatter_lut(d_dst2,
                    d_src,
                    d_lut,
                    lod_dtype_u16,
                    ndim,
                    n_elements,
                    d_full_shape,
                    d_lod_shape,
                    plan.lod_ndim,
                    plan.lod_mask,
                    lod_count,
                    stream);
    CU(Fail, cuEventRecord(ev1, stream));
    CU(Fail, cuStreamSynchronize(stream));
    accumulate_metric_cu(&m_lut, ev0, ev1, src_bytes);
  }

  // Build gather (inverse) LUT
  // Compute LOD strides: for each LOD dim, its C-order stride in the full array
  {
    uint64_t lod_strides[LOD_MAX_NDIM];
    {
      // Full C-order strides: stride[d] = prod(shape[k] for k > d)
      uint64_t full_strides[LOD_MAX_NDIM];
      full_strides[ndim - 1] = 1;
      for (int d = ndim - 2; d >= 0; --d)
        full_strides[d] = full_strides[d + 1] * shape[d + 1];

      // Extract LOD strides matching lod_coords ordering:
      // lod_coords[0] = highest full dim in mask, ..., [lod_ndim-1] = lowest
      int lod_dim = plan.lod_ndim - 1;
      for (int d = ndim - 1; d >= 0; --d) {
        if ((lod_mask >> d) & 1) {
          lod_strides[lod_dim] = full_strides[d];
          lod_dim--;
        }
      }
    }

    // Compute batch_offsets[batch_index] = C-order offset from batch dims
    uint32_t* batch_offsets =
      (uint32_t*)calloc(plan.batch_count, sizeof(uint32_t));
    CHECK(Fail, batch_offsets);
    {
      uint64_t full_strides[LOD_MAX_NDIM];
      full_strides[ndim - 1] = 1;
      for (int d = ndim - 2; d >= 0; --d)
        full_strides[d] = full_strides[d + 1] * shape[d + 1];

      for (uint64_t bi = 0; bi < plan.batch_count; ++bi) {
        uint64_t remainder = bi;
        uint64_t offset = 0;
        // batch dims in same order as batch_shape:
        // batch_map[k] is the full dim index
        for (int k = plan.batch_ndim - 1; k >= 0; --k) {
          uint64_t coord = remainder % plan.batch_shape[k];
          remainder /= plan.batch_shape[k];
          offset += coord * full_strides[plan.batch_map[k]];
        }
        batch_offsets[bi] = (uint32_t)offset;
      }
    }

    CU(CleanupOnFail, cuMemAlloc(&d_src_lut, lod_count * sizeof(uint32_t)));
    CU(CleanupOnFail,
       cuMemAlloc(&d_lod_strides, plan.lod_ndim * sizeof(uint64_t)));
    CU(CleanupOnFail,
       cuMemcpyHtoD(
         d_lod_strides, lod_strides, plan.lod_ndim * sizeof(uint64_t)));
    CU(CleanupOnFail,
       cuMemAlloc(&d_batch_offsets, plan.batch_count * sizeof(uint32_t)));
    CU(CleanupOnFail,
       cuMemcpyHtoD(
         d_batch_offsets, batch_offsets, plan.batch_count * sizeof(uint32_t)));
    free(batch_offsets);
    batch_offsets = NULL;

    CU(Fail, cuMemAlloc(&d_dst3, dst_bytes));

    // Build gather LUT
    lod_build_gather_lut(d_src_lut,
                         d_lod_shape,
                         d_lod_strides,
                         plan.lod_ndim,
                         plan.lod_shapes[0],
                         lod_count,
                         stream);
    CU(Fail, cuStreamSynchronize(stream));

    if (0) {
    CleanupOnFail:
      free(batch_offsets);
      goto Fail;
    }
  }

  // Bench: gather (coalesced writes)
  struct stream_metric m_gather = { .name = "gather", .best_ms = 1e30f };
  for (int i = 0; i < niter; ++i) {
    CU(Fail, cuEventRecord(ev0, stream));
    lod_gather_lut(d_dst3,
                   d_src,
                   d_src_lut,
                   d_batch_offsets,
                   lod_dtype_u16,
                   lod_count,
                   plan.batch_count,
                   stream);
    CU(Fail, cuEventRecord(ev1, stream));
    CU(Fail, cuStreamSynchronize(stream));
    accumulate_metric_cu(&m_gather, ev0, ev1, src_bytes);
  }

  // Verify all approaches match original
  {
    uint16_t* h_orig = NULL;
    uint16_t* h_cmp = NULL;
    uint64_t total = plan.batch_count * lod_count;
    h_orig = (uint16_t*)malloc(dst_bytes);
    h_cmp = (uint16_t*)malloc(dst_bytes);
    if (!h_orig || !h_cmp)
      goto Fail2;
    if (cuMemcpyDtoH(h_orig, d_dst, dst_bytes) != CUDA_SUCCESS)
      goto Fail2;

    // Check LUT scatter
    if (cuMemcpyDtoH(h_cmp, d_dst2, dst_bytes) != CUDA_SUCCESS)
      goto Fail2;
    for (uint64_t i = 0; i < total; ++i) {
      if (h_orig[i] != h_cmp[i]) {
        printf("  LUT MISMATCH at %llu: orig=%u lut=%u\n",
               (unsigned long long)i,
               (unsigned)h_orig[i],
               (unsigned)h_cmp[i]);
        goto Fail2;
      }
    }

    // Check gather
    if (cuMemcpyDtoH(h_cmp, d_dst3, dst_bytes) != CUDA_SUCCESS)
      goto Fail2;
    for (uint64_t i = 0; i < total; ++i) {
      if (h_orig[i] != h_cmp[i]) {
        printf("  GATHER MISMATCH at %llu: orig=%u gather=%u\n",
               (unsigned long long)i,
               (unsigned)h_orig[i],
               (unsigned)h_cmp[i]);
        goto Fail2;
      }
    }

    free(h_orig);
    free(h_cmp);
    goto Verified;
  Fail2:
    free(h_orig);
    free(h_cmp);
    goto Fail;
  Verified:;
  }

  report_metric(&m_orig);
  report_metric(&m_build);
  report_metric(&m_lut);
  report_metric(&m_gather);
  printf("  PASS\n");
  ok = 1;

Fail:
  cuMemFree(d_src);
  cuMemFree(d_dst);
  cuMemFree(d_dst2);
  cuMemFree(d_dst3);
  cuMemFree(d_lut);
  cuMemFree(d_src_lut);
  cuMemFree(d_lod_strides);
  cuMemFree(d_batch_offsets);
  cuMemFree(d_full_shape);
  cuMemFree(d_lod_shape);
  cuEventDestroy(ev0);
  cuEventDestroy(ev1);
  cuStreamDestroy(stream);
  lod_plan_free(&plan);
  return ok;
}

int
main(void)
{
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf("CUDA init failed\n");
    return 1;
  }

  int nfail = 0;

  nfail += !bench_scatter_lut(
    "bench_scatter_lut_3d_256", 3, (uint64_t[]){ 256, 256, 256 }, 0x7, 10);
  nfail += !bench_scatter_lut("bench_scatter_lut_3d_mixed",
                              5,
                              (uint64_t[]){ 2, 256, 256, 256, 3 },
                              0xE,
                              10);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);

  cuCtxDestroy(ctx);
  return nfail ? 1 : 0;
}
