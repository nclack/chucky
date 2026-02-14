#include "compress.h"
#include "log/log.h"
#include "platform.h"
#include "stream.h"

#include <stdlib.h>
#include <string.h>

#define container_of(ptr, type, member)                                        \
  ((type*)((char*)(ptr) - offsetof(type, member)))

#define CU(lbl, e)                                                             \
  do {                                                                         \
    CUresult res_ = (e);                                                       \
    if (res_ != CUDA_SUCCESS &&                                                \
        handle_curesult(LOG_ERROR, res_, __FILE__, __LINE__, #e)) {            \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define CUWARN(e)                                                              \
  do {                                                                         \
    handle_curesult(LOG_WARN, (e), __FILE__, __LINE__, #e);                    \
  } while (0)

#define CHECK(lbl, e)                                                          \
  do {                                                                         \
    if (!(e)) {                                                                \
      log_error("%s:%d check failed: %s", __FILE__, __LINE__, #e);             \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

static int
handle_curesult(int level,
                CUresult ecode,
                const char* file,
                int line,
                const char* expr)
{
  if (ecode == CUDA_SUCCESS)
    return 0;
  const char *name, *desc;
  cuGetErrorName(ecode, &name);
  cuGetErrorString(ecode, &desc);
  if (name && desc) {
    log_log(level, file, line, "CUDA error: %s %s %s\n", name, desc, expr);
  } else {
    log_log(level,
            file,
            line,
            "%s. Failed to retrieve error info for CUresult: %d\n",
            expr,
            ecode);
  }
  return 1;
}

static void
buffer_free(struct buffer* buffer)
{
  if (!buffer || !buffer->data) {
    return;
  }

  if (buffer->ready) {
    CUresult res = cuEventDestroy(buffer->ready);
    if (res != CUDA_SUCCESS) {
      const char* err_str = NULL;
      cuGetErrorString(res, &err_str);
      log_warn("Failed to destroy event: %s", err_str ? err_str : "unknown");
    }
    buffer->ready = NULL;
  }

  switch (buffer->domain) {
    case host:
      cuMemFreeHost(buffer->data);
      break;
    case device:
      cuMemFree((CUdeviceptr)buffer->data);
      break;
    default:
      log_error("Invalid domain during buffer_free: %d", buffer->domain);
      return;
  }

  buffer->data = NULL;
}

static struct buffer
buffer_new(size_t capacity, enum domain domain, unsigned int host_flags)
{
  struct buffer buf = { 0 };
  buf.domain = domain;

  switch (domain) {
    case host:
      CU(Fail, cuMemHostAlloc(&buf.data, capacity, host_flags));
      break;
    case device:
      CU(Fail, cuMemAlloc((CUdeviceptr*)&buf.data, capacity));
      break;
    default:
      log_error("Invalid domain: %d", domain);
      goto Fail;
  }
  CU(Fail, cuEventCreate(&buf.ready, CU_EVENT_DEFAULT));
  return buf;

Fail:
  buffer_free(&buf);
  return (struct buffer){ 0 };
}

static void
accumulate_metric(struct stream_metric* m, CUevent start, CUevent end)
{
  float ms = 0;
  cuEventElapsedTime(&ms, start, end);
  if (ms < 1e-2f)
    return; // skip bogus measurements from seeded events
  m->ms += ms;
  m->count++;
  if (ms < m->best_ms)
    m->best_ms = ms;
}

static uint64_t
ceildiv(uint64_t a, uint64_t b)
{
  return (a + b - 1) / b;
}

static size_t
align_up(size_t x, size_t alignment)
{
  return (x + alignment - 1) / alignment * alignment;
}

// Free a device pointer if non-NULL.
static void
device_free(void* ptr)
{
  if (ptr)
    CUWARN(cuMemFree((CUdeviceptr)ptr));
}

// Free a host-pinned pointer if non-NULL.
static void
host_free(void* ptr)
{
  if (ptr)
    cuMemFreeHost(ptr);
}

// Dispatch staged data: H2D transfer + scatter kernel
// Returns 0 on success, 1 on error.
static int
dispatch_scatter(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 0;

  const uint64_t elements = s->stage.fill / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage.current;
  const int tidx = s->tiles.current;
  struct staging_slot* ss = &s->stage.slot[idx];
  struct tile_pool_slot* ts = &s->tiles.slot[tidx];

  // H2D
  CU(Error, cuEventRecord(ss->t_h2d_start, s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)ss->d_in.data,
                       ss->h_in.data,
                       s->stage.fill,
                       s->h2d));
  CU(Error, cuEventRecord(ss->h_in.ready, s->h2d));

  // Kernel waits for H2D, then scatters into tile pool
  CU(Error, cuStreamWaitEvent(s->compute, ss->h_in.ready, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, s->compute));
  // FIXME: dispatch based on element type
  transpose_u16_v0((CUdeviceptr)ts->d_tiles.data,
                   (CUdeviceptr)ts->d_tiles.data + s->layout.tile_pool_bytes,
                   (CUdeviceptr)ss->d_in.data,
                   (CUdeviceptr)ss->d_in.data + s->stage.fill,
                   s->cursor,
                   s->layout.lifted_rank,
                   s->layout.d_lifted_shape,
                   s->layout.d_lifted_strides,
                   s->compute);
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));
  CU(Error, cuEventRecord(ts->d_tiles.ready, s->compute));

  s->cursor += elements;
  s->stage.current ^= 1;
  return 0;

Error:
  return 1;
}

struct writer_result
writer_append(struct writer* w, struct slice data)
{
  return w->append(w, data);
}

struct writer_result
writer_flush(struct writer* w)
{
  return w->flush(w);
}

struct writer_result
writer_append_wait(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct writer_result r = writer_append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_append_wait: no progress after %d retries", stalls);
        return writer_error_at(data.beg, data.end);
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return writer_ok();
}

// Deliver compressed tiles from the previous epoch to the compressed_sink.
static int
deliver_compressed_tiles(struct transpose_stream* s, int pool)
{
  const uint64_t M = s->layout.slot_count;
  const struct compression_slot* cs = &s->tiles.slot[pool].comp;
  const void** tile_ptrs = (const void**)malloc(M * sizeof(void*));
  CHECK(Error, tile_ptrs);

  for (uint64_t i = 0; i < M; ++i)
    tile_ptrs[i] =
      (char*)cs->h_compressed.data + i * s->comp.max_comp_chunk_bytes;

  int err = s->config.compressed_sink->append(
    s->config.compressed_sink, tile_ptrs, cs->h_comp_sizes, M);
  free((void*)tile_ptrs);
  return err;

Error:
  return 1;
}

// Wait for source, record timing, memcpy D2H, record completion.
// Returns 0 on success.
static int
d2h_memcpy_async(CUstream stream,
                 CUevent src_ready,
                 CUevent t_start,
                 void* h_dst,
                 CUdeviceptr d_src,
                 size_t bytes,
                 CUevent dst_ready)
{
  CU(Error, cuStreamWaitEvent(stream, src_ready, 0));
  CU(Error, cuEventRecord(t_start, stream));
  CU(Error, cuMemcpyDtoHAsync(h_dst, d_src, bytes, stream));
  CU(Error, cuEventRecord(dst_ready, stream));
  return 0;

Error:
  return 1;
}

// Enqueue compress (if enabled) + D2H for the given tile pool.
// Returns 0 on success; the D2H is left in-flight.
static int
kick_epoch_d2h(struct transpose_stream* s, int pool)
{
  struct tile_pool_slot* ts = &s->tiles.slot[pool];

  if (s->config.compress && s->config.compressed_sink) {
    struct compression_slot* cs = &ts->comp;

    // Compress on compute (ordered after scatter on same stream)
    CU(Error, cuEventRecord(ts->t_compress_start, s->compute));
    CHECK(Error,
          compress_batch_async((const void* const*)cs->d_uncomp_ptrs,
                               s->comp.d_uncomp_sizes,
                               s->layout.tile_stride * s->config.bytes_per_element,
                               s->layout.slot_count,
                               s->comp.d_comp_temp,
                               s->comp.comp_temp_bytes,
                               cs->d_comp_ptrs,
                               cs->d_comp_sizes,
                               s->compute) == 0);
    CU(Error, cuEventRecord(cs->d_compressed.ready, s->compute));

    // Synchronize compute: nvcomp may use internal streams not captured by
    // the event we record on compute.  A full stream sync ensures the
    // batched compress is truly finished before D2H.
    // FIXME: this shouldn't be necessary
    CU(Error, cuStreamSynchronize(s->compute));

    // D2H waits for compress, then transfers compressed data + sizes
    CHECK(Error,
          d2h_memcpy_async(s->d2h,
                           cs->d_compressed.ready,
                           ts->t_d2h_start,
                           cs->h_compressed.data,
                           (CUdeviceptr)cs->d_compressed.data,
                           s->comp.comp_pool_bytes,
                           cs->h_compressed.ready) == 0);
    // Sizes piggyback on the same stream (no separate event pair)
    CU(Error,
       cuMemcpyDtoHAsync(cs->h_comp_sizes,
                         (CUdeviceptr)cs->d_comp_sizes,
                         s->layout.slot_count * sizeof(size_t),
                         s->d2h));
    CU(Error, cuEventRecord(cs->h_compressed.ready, s->d2h));
  } else {
    CHECK(Error,
          d2h_memcpy_async(s->d2h,
                           ts->d_tiles.ready,
                           ts->t_d2h_start,
                           ts->h_tiles.data,
                           (CUdeviceptr)ts->d_tiles.data,
                           s->layout.tile_pool_bytes,
                           ts->h_tiles.ready) == 0);
  }

  return 0;

Error:
  return 1;
}

// Wait for D2H on the given pool, accumulate metrics, and deliver to sink.
static struct writer_result
wait_and_deliver(struct transpose_stream* s, int pool)
{
  struct tile_pool_slot* ts = &s->tiles.slot[pool];

  if (s->config.compress && s->config.compressed_sink) {
    struct compression_slot* cs = &ts->comp;
    CU(Error, cuEventSynchronize(cs->h_compressed.ready));
    accumulate_metric(
      &s->metrics.compress, ts->t_compress_start, cs->d_compressed.ready);
    accumulate_metric(
      &s->metrics.d2h, ts->t_d2h_start, cs->h_compressed.ready);
    if (deliver_compressed_tiles(s, pool))
      goto Error;
  } else {
    CU(Error, cuEventSynchronize(ts->h_tiles.ready));
    accumulate_metric(&s->metrics.d2h, ts->t_d2h_start, ts->h_tiles.ready);
    if (s->config.sink) {
      struct slice tiles = {
        .beg = ts->h_tiles.data,
        .end = (char*)ts->h_tiles.data + s->layout.tile_pool_bytes,
      };
      return writer_append_wait(s->config.sink, tiles);
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

// Deliver the previous epoch's tiles if an async D2H is still pending.
static struct writer_result
drain_pending_flush(struct transpose_stream* s)
{
  if (!s->tiles.flush_pending)
    return writer_ok();

  s->tiles.flush_pending = 0;
  return wait_and_deliver(s, s->tiles.current ^ 1);
}

// Synchronously flush the current tile pool (used for the final partial epoch).
static struct writer_result
flush_epoch_sync(struct transpose_stream* s)
{
  if (kick_epoch_d2h(s, s->tiles.current))
    return writer_error();
  return wait_and_deliver(s, s->tiles.current);
}

// Flush the current epoch's tile pool: async D2H, swap pools, zero next.
static struct writer_result
flush_epoch(struct transpose_stream* s)
{
  const int cur = s->tiles.current;

  // Deliver the previous epoch if its D2H is still in flight
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  if (kick_epoch_d2h(s, cur))
    return writer_error();

  // Switch to other pool and zero it for next epoch
  s->tiles.current ^= 1;
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)s->tiles.slot[s->tiles.current].d_tiles.data,
                     0,
                     s->layout.tile_pool_bytes,
                     s->compute));

  s->tiles.flush_pending = 1;
  return writer_ok();

Error:
  return writer_error();
}

struct stream_metrics
transpose_stream_get_metrics(const struct transpose_stream* s)
{
  return s->metrics;
}

void
transpose_stream_destroy(struct transpose_stream* stream)
{
  if (!stream)
    return;

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->d2h));

  if (stream->layout.d_lifted_shape)
    CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_shape));
  if (stream->layout.d_lifted_strides)
    CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_strides));

  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stream->stage.slot[i];
    if (ss->t_h2d_start)
      CUWARN(cuEventDestroy(ss->t_h2d_start));
    if (ss->t_scatter_start)
      CUWARN(cuEventDestroy(ss->t_scatter_start));
    if (ss->t_scatter_end)
      CUWARN(cuEventDestroy(ss->t_scatter_end));
    buffer_free(&ss->h_in);
    buffer_free(&ss->d_in);

    struct tile_pool_slot* ts = &stream->tiles.slot[i];
    if (ts->t_compress_start)
      CUWARN(cuEventDestroy(ts->t_compress_start));
    if (ts->t_d2h_start)
      CUWARN(cuEventDestroy(ts->t_d2h_start));
    buffer_free(&ts->d_tiles);
    buffer_free(&ts->h_tiles);

    buffer_free(&ts->comp.d_compressed);
    buffer_free(&ts->comp.h_compressed);
    device_free(ts->comp.d_uncomp_ptrs);
    device_free(ts->comp.d_comp_ptrs);
    device_free(ts->comp.d_comp_sizes);
    host_free(ts->comp.h_comp_sizes);
  }

  device_free(stream->comp.d_uncomp_sizes);
  device_free(stream->comp.d_comp_temp);

  *stream = (struct transpose_stream){ 0 };
}

// Allocate compression buffers and build device pointer arrays.
// Assumes s->layout, s->tiles, and s->config are already initialized.
// Returns 0 on success, 1 on error.
// On failure, partially-allocated state is safe for transpose_stream_destroy.
static int
init_compression_state(struct transpose_stream* s)
{
  const uint64_t M = s->layout.slot_count;
  const size_t bpe = s->config.bytes_per_element;
  const size_t tile_bytes = s->layout.tile_stride * bpe;

  s->comp.max_comp_chunk_bytes = align_up(
    compress_get_max_output_size(tile_bytes), compress_get_input_alignment());
  CHECK(Error, s->comp.max_comp_chunk_bytes > 0);
  s->comp.comp_pool_bytes = M * s->comp.max_comp_chunk_bytes;

  s->comp.comp_temp_bytes = compress_get_temp_size(M, tile_bytes);
  if (s->comp.comp_temp_bytes > 0)
    CU(Error, cuMemAlloc((CUdeviceptr*)&s->comp.d_comp_temp, s->comp.comp_temp_bytes));

  for (int i = 0; i < 2; ++i) {
    struct compression_slot* cs = &s->tiles.slot[i].comp;
    CHECK(Error,
          (cs->d_compressed = buffer_new(s->comp.comp_pool_bytes, device, 0)).data);
    CHECK(Error,
          (cs->h_compressed = buffer_new(s->comp.comp_pool_bytes, host, 0)).data);
    CU(Error,
       cuMemAlloc((CUdeviceptr*)&cs->d_comp_sizes, M * sizeof(size_t)));
    CU(Error,
       cuMemHostAlloc((void**)&cs->h_comp_sizes, M * sizeof(size_t), 0));
  }

  // Build device pointer arrays for nvcomp batch API
  void** h_ptrs = (void**)malloc(M * sizeof(void*));
  CHECK(Error, h_ptrs);

  for (int i = 0; i < 2; ++i) {
    struct compression_slot* cs = &s->tiles.slot[i].comp;
    CU(Free, cuMemAlloc((CUdeviceptr*)&cs->d_uncomp_ptrs, M * sizeof(void*)));
    CU(Free, cuMemAlloc((CUdeviceptr*)&cs->d_comp_ptrs, M * sizeof(void*)));

    for (uint64_t k = 0; k < M; ++k)
      h_ptrs[k] = (char*)s->tiles.slot[i].d_tiles.data + k * tile_bytes;
    CU(Free,
       cuMemcpyHtoD(
         (CUdeviceptr)cs->d_uncomp_ptrs, h_ptrs, M * sizeof(void*)));

    for (uint64_t k = 0; k < M; ++k)
      h_ptrs[k] =
        (char*)cs->d_compressed.data + k * s->comp.max_comp_chunk_bytes;
    CU(Free,
       cuMemcpyHtoD(
         (CUdeviceptr)cs->d_comp_ptrs, h_ptrs, M * sizeof(void*)));
  }

  // Uncompressed sizes: all the same
  {
    size_t* h_sizes = (size_t*)malloc(M * sizeof(size_t));
    if (!h_sizes)
      goto Free;
    for (uint64_t k = 0; k < M; ++k)
      h_sizes[k] = tile_bytes;

    CU(Free, cuMemAlloc((CUdeviceptr*)&s->comp.d_uncomp_sizes, M * sizeof(size_t)));
    CUresult rc = cuMemcpyHtoD(
      (CUdeviceptr)s->comp.d_uncomp_sizes, h_sizes, M * sizeof(size_t));
    free(h_sizes);
    CU(Free, rc);
  }

  free(h_ptrs);

  // Record initial events for compressed host buffers
  CU(Error, cuEventRecord(s->tiles.slot[0].comp.h_compressed.ready, s->compute));
  CU(Error, cuEventRecord(s->tiles.slot[1].comp.h_compressed.ready, s->compute));

  return 0;

Free:
  free(h_ptrs);
Error:
  return 1;
}

// Forward declarations for vtable
static struct writer_result
transpose_stream_append(struct writer* self, struct slice input);
static struct writer_result
transpose_stream_flush(struct writer* self);

int
transpose_stream_create(const struct transpose_stream_configuration* config,
                        struct transpose_stream* out)
{
  CHECK(Fail, config);
  CHECK(Fail, out);
  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  CHECK(Fail, (config->buffer_capacity_bytes & 4095) == 0); // must be 4KB-aligned for vectorized kernel loads
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= MAX_RANK / 2); // lifted rank = 2 * rank
  CHECK(Fail, config->dimensions);

  *out = (struct transpose_stream){
    .writer = { .append = transpose_stream_append,
                .flush = transpose_stream_flush },
    .config = *config,
  };

  CU(Fail, cuStreamCreate(&out->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_scatter_end, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->tiles.slot[i].t_compress_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->tiles.slot[i].t_d2h_start, CU_EVENT_DEFAULT));
  }

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;

  // Lifted shape (row-major, slowest first): (t_{D-1}, n_{D-1}, ..., t_0, n_0)
  out->layout.lifted_rank = 2 * rank;
  out->layout.tile_elements = 1;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    out->layout.lifted_shape[2 * i] = tile_count[i];
    out->layout.lifted_shape[2 * i + 1] = dims[i].tile_size;
    out->layout.tile_elements *= dims[i].tile_size;
  }

  // Compute tile_stride: pad tile_elements so each tile starts at an aligned
  // byte address when compression is enabled.
  {
    size_t alignment = config->compress ? compress_get_input_alignment() : 1;
    size_t tile_bytes = out->layout.tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    out->layout.tile_stride = padded_bytes / bpe;
  }

  // Build lifted strides
  //   n_stride: within-tile element stride, accumulates tile_size
  //   t_stride: tile-pool element stride, accumulates tile_count
  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)out->layout.tile_stride;

    for (int i = rank - 1; i >= 0; --i) {
      out->layout.lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      out->layout.lifted_strides[2 * i] = t_stride;
      t_stride *= (int64_t)tile_count[i];
    }
  }

  // An epoch is one slice of the array along the outermost tile dimension —
  // all the tiles excluding that slowest-varying tile index. The tile pool
  // holds exactly one epoch, so we flush after every tile_count[0] steps.
  out->layout.slot_count = out->layout.lifted_strides[0] / out->layout.tile_stride;
  out->layout.epoch_elements = out->layout.slot_count * out->layout.tile_elements;

  // Collapse epoch dimension: the outermost tile index wraps via flush,
  // so its stride is zero in the kernel.
  out->layout.lifted_strides[0] = 0;

  out->layout.tile_pool_bytes = out->layout.slot_count * out->layout.tile_stride * bpe;

  // Allocate device copies of lifted shape and strides
  {
    const size_t shape_bytes = out->layout.lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = out->layout.lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->layout.d_lifted_shape, shape_bytes));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->layout.d_lifted_strides, strides_bytes));
    CU(Fail,
       cuMemcpyHtoD(
         (CUdeviceptr)out->layout.d_lifted_shape, out->layout.lifted_shape, shape_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)out->layout.d_lifted_strides,
                    out->layout.lifted_strides,
                    strides_bytes));
  }

  // Allocate input staging buffers (double-buffered)
  // h_in: host writes sequentially, GPU reads via H2D -> write-combined
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (out->stage.slot[i].h_in = buffer_new(config->buffer_capacity_bytes,
                                                host,
                                                CU_MEMHOSTALLOC_WRITECOMBINED))
            .data);
    CHECK(Fail,
          (out->stage.slot[i].d_in = buffer_new(config->buffer_capacity_bytes, device, 0))
            .data);
  }

  // Allocate tile pools (double-buffered)
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (out->tiles.slot[i].d_tiles = buffer_new(out->layout.tile_pool_bytes, device, 0)).data);
    // h_tiles: GPU writes via D2H, host reads -> normal pinned (no WC)
    CHECK(Fail,
          (out->tiles.slot[i].h_tiles = buffer_new(out->layout.tile_pool_bytes, host, 0)).data);
    CU(Fail,
       cuMemsetD8Async((CUdeviceptr)out->tiles.slot[i].d_tiles.data,
                       0,
                       out->layout.tile_pool_bytes,
                       out->compute));
  }

  // Compression buffers
  if (config->compress)
    CHECK(Fail, init_compression_state(out) == 0);

  // Record initial events so first cuEventSynchronize / cuStreamWaitEvent
  // calls succeed immediately, and seed timing events so cuEventElapsedTime
  // never sees unrecorded events.
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(out->stage.slot[i].h_in.ready, out->compute));
    CU(Fail, cuEventRecord(out->tiles.slot[i].d_tiles.ready, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_h2d_start, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_scatter_start, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_scatter_end, out->compute));
    CU(Fail, cuEventRecord(out->tiles.slot[i].t_compress_start, out->compute));
    CU(Fail, cuEventRecord(out->tiles.slot[i].t_d2h_start, out->compute));
  }

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics.h2d = (struct stream_metric){ .name = "H2D", .best_ms = 1e30f };
  out->metrics.scatter =
    (struct stream_metric){ .name = "Scatter", .best_ms = 1e30f };
  out->metrics.compress =
    (struct stream_metric){ .name = "Compress", .best_ms = 1e30f };
  out->metrics.d2h = (struct stream_metric){ .name = "D2H", .best_ms = 1e30f };

  out->cursor = 0;
  out->stage.fill = 0;
  out->stage.current = 0;
  out->tiles.current = 0;
  out->tiles.flush_pending = 0;

  return 0;

Fail:
  transpose_stream_destroy(out);
  return 1;
}

static struct writer_result
transpose_stream_append(struct writer* self, struct slice input)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  const size_t bpe = s->config.bytes_per_element;
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  while (src < end) {
    // How many elements remain in the current epoch?
    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    const uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    const uint64_t bytes_this_pass = elements_this_pass * bpe;

    // Fill h_in, dispatch H2D + kernel in buffer-sized payloads
    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - s->stage.fill;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        // Wait for this staging buffer's prior H2D to finish
        if (s->stage.fill == 0) {
          const int si = s->stage.current;
          struct staging_slot* ss = &s->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->h_in.ready));

          // Accumulate H2D and scatter time from previous dispatch on this slot
          if (s->cursor > 0) {
            accumulate_metric(
              &s->metrics.h2d, ss->t_h2d_start, ss->h_in.ready);
            CU(Error, cuEventSynchronize(ss->t_scatter_end));
            accumulate_metric(
              &s->metrics.scatter, ss->t_scatter_start, ss->t_scatter_end);
          }
        }

        memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in.data + s->stage.fill,
               src + written,
               payload);
        s->stage.fill += payload;
        written += payload;

        if (s->stage.fill == buffer_capacity || written == bytes_this_pass) {
          if (dispatch_scatter(s))
            goto Error;
          s->stage.fill = 0;
        }
      }
    }
    src += bytes_this_pass;

    // If we just completed an epoch, flush it
    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_epoch(s);
      if (fr.error)
        return writer_error_at(src, end);
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
transpose_stream_flush(struct writer* self)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  // Dispatch any remaining staged data
  if (s->stage.fill > 0) {
    if (dispatch_scatter(s))
      return writer_error();
    s->stage.fill = 0;
  }

  // Drain any pending async epoch flush
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Flush current partial epoch synchronously (skip if cursor is on an
  // epoch boundary — that epoch was already flushed during append)
  if (s->cursor % s->layout.epoch_elements != 0 || s->cursor == 0) {
    r = flush_epoch_sync(s);
    if (r.error)
      return r;
  }

  if (s->config.compress && s->config.compressed_sink)
    return s->config.compressed_sink->flush(s->config.compressed_sink)
             ? writer_error()
             : writer_ok();

  if (s->config.sink)
    return writer_flush(s->config.sink);

  return writer_ok();
}
