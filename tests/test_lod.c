// GPU LOD test: compare GPU kernels against CPU reference from morton.util.c

#include "morton.util.c"

#include "lod.h"
#include "metric.cuda.h"
#include "prelude.cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
report_metric(const struct stream_metric* m, uint64_t n_elements)
{
  if (m->count == 0)
    return;
  double bytes = (double)n_elements * 4.0;
  float avg_ms = m->ms / (float)m->count;
  printf("  %-8s %7.3f ms  %6.2f GB/s\n",
         m->name, avg_ms, bytes / ((double)avg_ms * 1e6));
}

struct lod_metrics
{
  struct stream_metric scatter;
  struct stream_metric pyramid;
  struct stream_metric total;
};

static void
lod_metrics_init(struct lod_metrics* m)
{
  *m = (struct lod_metrics){
    .scatter = { .name = "scatter", .best_ms = 1e30f },
    .pyramid = { .name = "pyramid", .best_ms = 1e30f },
    .total = { .name = "total", .best_ms = 1e30f },
  };
}

static void
lod_metrics_report(const struct lod_metrics* m, uint64_t n_elements)
{
  report_metric(&m->scatter, n_elements);
  report_metric(&m->pyramid, n_elements);
  report_metric(&m->total, n_elements);
}

// Run LOD computation on GPU: scatter + fill_ends + reduce.
// Returns allocated host buffer with results, or NULL on failure.
static float*
lod_compute_gpu(const struct lod_plan* p,
                const float* src,
                struct lod_metrics* metrics)
{
  float* result = NULL;
  CUstream stream = NULL;
  CUevent ev_start = NULL, ev_scatter = NULL, ev_done = NULL;

  CUdeviceptr d_src = 0, d_values = 0;
  CUdeviceptr d_full_shape = 0, d_ds_shape = 0;
  CUdeviceptr d_ends = 0;
  CUdeviceptr d_child_shape = 0, d_parent_shape = 0;

  uint64_t n_elements = slice_len(spans_at(&p->levels, 0));
  uint64_t total_vals = p->levels.ends[p->nlev - 1];

  CU(Fail, cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  CU(Fail, cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_scatter, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&ev_done, CU_EVENT_DEFAULT));

  CHECK(Fail, upload(&d_src, src, n_elements * sizeof(float)));
  CU(Fail, cuMemAlloc(&d_values, total_vals * sizeof(float)));

  CHECK(Fail,
        upload(&d_full_shape, p->shapes[0], p->ndim * sizeof(uint64_t)));
  if (p->ds_ndim > 0)
    CHECK(Fail,
          upload(&d_ds_shape, p->ds_shapes[0],
                 p->ds_ndim * sizeof(uint64_t)));

  CU(Fail, cuEventRecord(ev_start, stream));

  lod_scatter_f32(d_values, d_src, p->ndim, n_elements,
                  d_full_shape, d_ds_shape, p->ds_ndim,
                  p->ds_shapes[0], p->ds_mask, p->ds_counts[0], stream);

  CU(Fail, cuEventRecord(ev_scatter, stream));

  for (int l = 0; l < p->nlev - 1; ++l) {
    struct slice seg = lod_segment(p, l);
    uint64_t n_parents = slice_len(seg);

    cuMemFree(d_child_shape);
    cuMemFree(d_parent_shape);
    d_child_shape = 0;
    d_parent_shape = 0;

    CHECK(Fail,
          upload(&d_child_shape, p->ds_shapes[l],
                 p->ds_ndim * sizeof(uint64_t)));
    CHECK(Fail,
          upload(&d_parent_shape, p->ds_shapes[l + 1],
                 p->ds_ndim * sizeof(uint64_t)));

    cuMemFree(d_ends);
    d_ends = 0;
    CU(Fail, cuMemAlloc(&d_ends, n_parents * sizeof(uint64_t)));

    lod_fill_ends_gpu(d_ends, p->ds_ndim,
                      d_child_shape, d_parent_shape,
                      p->ds_shapes[l], p->ds_shapes[l + 1],
                      n_parents, stream);

    struct slice src_level = spans_at(&p->levels, l);
    struct slice dst_level = spans_at(&p->levels, l + 1);

    lod_reduce_f32(d_values, d_ends,
                   src_level.beg, dst_level.beg,
                   p->ds_counts[l], p->ds_counts[l + 1],
                   p->batch_count, stream);
  }

  CU(Fail, cuEventRecord(ev_done, stream));
  CU(Fail, cuStreamSynchronize(stream));

  if (metrics) {
    accumulate_metric_cu(&metrics->scatter, ev_start, ev_scatter);
    accumulate_metric_cu(&metrics->pyramid, ev_scatter, ev_done);
    accumulate_metric_cu(&metrics->total, ev_start, ev_done);
  }

  result = (float*)malloc(total_vals * sizeof(float));
  CHECK(Fail, result);
  CU(Fail, cuMemcpyDtoH(result, d_values, total_vals * sizeof(float)));

Fail:
  cuMemFree(d_src);
  cuMemFree(d_values);
  cuMemFree(d_full_shape);
  cuMemFree(d_ds_shape);
  cuMemFree(d_ends);
  cuMemFree(d_child_shape);
  cuMemFree(d_parent_shape);
  cuEventDestroy(ev_start);
  cuEventDestroy(ev_scatter);
  cuEventDestroy(ev_done);
  cuStreamDestroy(stream);
  return result;
}

static int
test_lod_gpu(const char* label,
             int ndim,
             const uint64_t* shape,
             uint8_t ds_mask)
{
  printf("--- %s ---\n", label);
  int ok = 0;
  float* src = NULL;
  float* cpu_values = NULL;
  float* gpu_values = NULL;
  struct lod_plan plan = { 0 };

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, ds_mask, MAX_LOD));
  printf("  ds_mask=0x%x  ds_ndim=%d  batch_ndim=%d  batch_count=%llu  nlev=%d\n",
         ds_mask, plan.ds_ndim, plan.batch_ndim,
         (unsigned long long)plan.batch_count, plan.nlev);

  CHECK(Fail, lod_compute(&plan, src, &cpu_values));

  struct lod_metrics metrics;
  lod_metrics_init(&metrics);
  gpu_values = lod_compute_gpu(&plan, src, &metrics);
  CHECK(Fail, gpu_values);
  lod_metrics_report(&metrics, n);

  {
    uint64_t total = plan.levels.ends[plan.nlev - 1];
    for (uint64_t i = 0; i < total; ++i) {
      if (fabsf(gpu_values[i] - cpu_values[i]) > 1e-5f) {
        printf("  FAIL at i=%llu: gpu=%f cpu=%f\n",
               (unsigned long long)i, gpu_values[i], cpu_values[i]);
        goto Fail;
      }
    }
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(cpu_values);
  free(gpu_values);
  lod_plan_free(&plan);
  return ok;
}

int
main(void)
{
  CUdevice dev;
  CUcontext ctx;
  if (cuInit(0) != CUDA_SUCCESS ||
      cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
      cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
    printf("CUDA init failed\n");
    return 1;
  }

  int nfail = 0;

  // All dims downsampled
  nfail +=
    !test_lod_gpu("gpu_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7);

  // Mixed: only some dims downsampled
  nfail +=
    !test_lod_gpu("gpu_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2);
  nfail +=
    !test_lod_gpu("gpu_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1);
  nfail +=
    !test_lod_gpu("gpu_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2);

  // No dims downsampled (trivial: nlev=1)
  nfail +=
    !test_lod_gpu("gpu_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0);
  // 1D
  nfail +=
    !test_lod_gpu("gpu_lod_1d", 1, (uint64_t[]){ 9 }, 0x1);

  // Larger mixed
  nfail +=
    !test_lod_gpu("gpu_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA);

  // Larger cases for throughput estimation
  nfail +=
    !test_lod_gpu("gpu_lod_3d_256", 3, (uint64_t[]){ 256, 256, 256 }, 0x7);
  nfail +=
    !test_lod_gpu("gpu_lod_3d_mixed_large", 3,
                  (uint64_t[]){ 64, 256, 256 }, 0x6);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);

  cuCtxDestroy(ctx);
  return nfail ? 1 : 0;
}
