#include "morton.util.c"

static int
test_3d(void)
{
  printf("--- test_3d ---\n");
  int ok = 0;
  const int ndim = 3;
  const uint64_t shape[] = { 3, 2, 5 };

  int p = ceil_log2(max_shape(ndim, shape));
  uint64_t box = 1ull << (ndim * p);
  for (uint64_t k = 0; k <= box; ++k) {
    uint64_t coords[MAX_NDIM];
    morton_decode(ndim, k, coords);
    uint64_t r = morton_rank(ndim, shape, coords, 0);
    uint64_t r_ref = morton_rank_ref(ndim, shape, k);
    if (r != r_ref) {
      printf("  FAIL at k=%llu: got %llu, expected %llu\n",
             (unsigned long long)k,
             (unsigned long long)r,
             (unsigned long long)r_ref);
      goto Fail;
    }
  }
  uint64_t total = morton_rank_ref(ndim, shape, 1ull << (ndim * p));
  printf("  total valid in 3x2x5 = %llu\n", (unsigned long long)total);
  CHECK(Fail, total == 30);
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_1d(void)
{
  printf("--- test_1d ---\n");
  int ok = 0;
  const int ndim = 1;
  const uint64_t shape[] = { 7 };
  for (uint64_t k = 0; k <= 8; ++k) {
    uint64_t coords[MAX_NDIM];
    morton_decode(ndim, k, coords);
    uint64_t r = morton_rank(ndim, shape, coords, 0);
    uint64_t r_ref = morton_rank_ref(ndim, shape, k);
    CHECK(Fail, r == r_ref);
  }
  printf("  PASS\n");
  ok = 1;
Fail:
  return ok;
}

static int
test_lod(const char* label,
         int ndim,
         const uint64_t* shape,
         uint8_t ds_mask)
{
  printf("--- %s ---\n", label);
  int ok = 0;
  float* src = NULL;
  float* values = NULL;
  struct lod_plan plan = { 0 };
  float* prev_rm = NULL;
  float* ref = NULL;
  float* cur_rm = NULL;

  uint64_t n = 1;
  for (int d = 0; d < ndim; ++d)
    n *= shape[d];

  src = (float*)malloc(n * sizeof(float));
  CHECK(Fail, src);
  for (uint64_t i = 0; i < n; ++i)
    src[i] = (float)(i + 1);

  CHECK(Fail, lod_plan_init(&plan, ndim, shape, ds_mask, MAX_LOD));
  printf("  ds_mask=0x%x  ds_ndim=%d  batch_ndim=%d  batch_count=%llu\n",
         ds_mask, plan.ds_ndim, plan.batch_ndim,
         (unsigned long long)plan.batch_count);

  CHECK(Fail, lod_compute(&plan, src, &values));
  printf("  levels: %d\n", plan.nlev);

  if (plan.nlev < 2) {
    prev_rm = (float*)malloc(n * sizeof(float));
    CHECK(Fail, prev_rm);
    morton_unshuffle(&plan, 0, values, prev_rm);
    for (uint64_t i = 0; i < n; ++i) {
      if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
        printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
               (unsigned long long)i, prev_rm[i], src[i]);
        goto Fail;
      }
    }
    printf("  level 0 scatter: ok (no downsample levels)\n");
    printf("  PASS\n");
    ok = 1;
    goto Fail;
  }

  CHECK(Fail, plan.nlev >= 2);

  prev_rm = (float*)malloc(n * sizeof(float));
  CHECK(Fail, prev_rm);
  morton_unshuffle(&plan, 0, values, prev_rm);
  for (uint64_t i = 0; i < n; ++i) {
    if (fabsf(prev_rm[i] - src[i]) > 1e-6f) {
      printf("  FAIL level 0 unshuffle at i=%llu: got %f, expected %f\n",
             (unsigned long long)i, prev_rm[i], src[i]);
      goto Fail;
    }
  }
  printf("  level 0 scatter: ok\n");

  for (int l = 1; l < plan.nlev; ++l) {
    const uint64_t* prev_shape = plan.shapes[l - 1];
    const uint64_t* cur_shape = plan.shapes[l];
    struct slice lev = spans_at(&plan.levels, l);
    uint64_t cur_n = slice_len(lev);

    ref = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, ref);
    downsample_ref(ndim, ds_mask, prev_shape, cur_shape, prev_rm, ref);

    cur_rm = (float*)malloc(cur_n * sizeof(float));
    CHECK(Fail, cur_rm);
    morton_unshuffle(&plan, l, values + lev.beg, cur_rm);

    for (uint64_t i = 0; i < cur_n; ++i) {
      if (fabsf(cur_rm[i] - ref[i]) > 1e-5f) {
        uint64_t coords[MAX_NDIM];
        linear_to_coords(ndim, cur_shape, i, coords);
        printf("  FAIL level %d at (", l);
        for (int d = 0; d < ndim; ++d)
          printf("%s%llu", d ? "," : "", (unsigned long long)coords[d]);
        printf("): got %f, expected %f\n", cur_rm[i], ref[i]);
        goto Fail;
      }
    }
    printf("  level %d: ok\n", l);

    free(ref);
    ref = NULL;
    free(prev_rm);
    prev_rm = cur_rm;
    cur_rm = NULL;
  }

  printf("  PASS\n");
  ok = 1;
Fail:
  free(src);
  free(values);
  lod_plan_free(&plan);
  free(prev_rm);
  free(ref);
  free(cur_rm);
  return ok;
}

int
main(void)
{
  int nfail = 0;
  nfail += !test_3d();
  nfail += !test_1d();

  nfail +=
    !test_lod("test_lod_2d_all", 2, (uint64_t[]){ 3, 5 }, 0x3);
  nfail +=
    !test_lod("test_lod_3d_all", 3, (uint64_t[]){ 3, 2, 5 }, 0x7);

  nfail +=
    !test_lod("test_lod_3d_d02", 3, (uint64_t[]){ 6, 3, 5 }, 0x5);
  nfail +=
    !test_lod("test_lod_3d_d1", 3, (uint64_t[]){ 4, 6, 3 }, 0x2);
  nfail +=
    !test_lod("test_lod_2d_d0", 2, (uint64_t[]){ 5, 3 }, 0x1);
  nfail +=
    !test_lod("test_lod_2d_d1", 2, (uint64_t[]){ 3, 7 }, 0x2);

  nfail +=
    !test_lod("test_lod_3d_none", 3, (uint64_t[]){ 3, 2, 5 }, 0x0);
  nfail +=
    !test_lod("test_lod_1d", 1, (uint64_t[]){ 9 }, 0x1);

  nfail +=
    !test_lod("test_lod_4d_d13", 4, (uint64_t[]){ 3, 8, 2, 6 }, 0xA);

  printf("\n%s (%d failures)\n", nfail ? "FAIL" : "ALL PASSED", nfail);
  return nfail ? 1 : 0;
}
