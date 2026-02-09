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
  const uint64_t elements = s->stage_fill / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage_idx;
  const int tidx = s->tile_idx;

  // H2D
  CU(Error, cuEventRecord(s->t_h2d_start[idx], s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync((CUdeviceptr)s->d_in[idx].data,
                        s->h_in[idx].data,
                        s->stage_fill,
                        s->h2d));
  CU(Error, cuEventRecord(s->h_in[idx].ready, s->h2d));

  // Kernel waits for H2D, then scatters into tile pool
  CU(Error, cuStreamWaitEvent(s->compute, s->h_in[idx].ready, 0));
  CU(Error, cuEventRecord(s->t_scatter_start[tidx], s->compute));
  transpose_u16_v0((CUdeviceptr)s->d_tiles[tidx].data,
                   (CUdeviceptr)s->d_tiles[tidx].data + s->tile_pool_bytes,
                   (CUdeviceptr)s->d_in[idx].data,
                   (CUdeviceptr)s->d_in[idx].data + s->stage_fill,
                   s->cursor,
                   s->lifted_rank,
                   s->d_lifted_shape,
                   s->d_lifted_strides,
                   s->compute);
  CU(Error, cuEventRecord(s->d_tiles[tidx].ready, s->compute));

  s->cursor += elements;
  s->stage_idx ^= 1;
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
        return (struct writer_result){ .error = 1, .rest = data };
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return (struct writer_result){ 0 };
}

// Deliver compressed tiles from the previous epoch to the compressed_sink.
static int
deliver_compressed_tiles(struct transpose_stream* s, int pool)
{
  const uint64_t M = s->slot_count;
  const void** tile_ptrs = (const void**)malloc(M * sizeof(void*));
  CHECK(Error, tile_ptrs);

  for (uint64_t i = 0; i < M; ++i)
    tile_ptrs[i] =
      (char*)s->h_compressed[pool].data + i * s->max_comp_chunk_bytes;

  int err = s->config.compressed_sink->append(
    s->config.compressed_sink, tile_ptrs, s->h_comp_sizes[pool], M);
  free((void*)tile_ptrs);
  return err;

Error:
  return 1;
}

// Deliver the previous epoch's tiles if an async D2H is still pending.
static struct writer_result
drain_pending_flush(struct transpose_stream* s)
{
  if (!s->flush_pending)
    return (struct writer_result){ 0 };

  const int prev = s->tile_idx ^ 1;

  if (s->config.compress && s->config.compressed_sink) {
    CU(Error, cuEventSynchronize(s->h_compressed[prev].ready));
    // Accumulate scatter, compress, and D2H timing
    {
      float sc = 0, co = 0, dh = 0;
      cuEventElapsedTime(
        &sc, s->t_scatter_start[prev], s->d_tiles[prev].ready);
      s->metrics.scatter_ms += sc;
      cuEventElapsedTime(
        &co, s->t_compress_start[prev], s->d_compressed[prev].ready);
      s->metrics.compress_ms += co;
      cuEventElapsedTime(
        &dh, s->t_d2h_start[prev], s->h_compressed[prev].ready);
      s->metrics.d2h_ms += dh;
      if (sc < s->metrics.scatter_best_ms)
        s->metrics.scatter_best_ms = sc;
      if (co < s->metrics.compress_best_ms)
        s->metrics.compress_best_ms = co;
      if (dh < s->metrics.d2h_best_ms)
        s->metrics.d2h_best_ms = dh;
      s->metrics.epoch_count++;
    }
    s->flush_pending = 0;
    if (deliver_compressed_tiles(s, prev))
      goto Error;
  } else {
    CU(Error, cuEventSynchronize(s->h_tiles[prev].ready));
    // Accumulate scatter and D2H timing
    {
      float sc = 0, dh = 0;
      cuEventElapsedTime(
        &sc, s->t_scatter_start[prev], s->d_tiles[prev].ready);
      s->metrics.scatter_ms += sc;
      cuEventElapsedTime(
        &dh, s->t_d2h_start[prev], s->h_tiles[prev].ready);
      s->metrics.d2h_ms += dh;
      if (sc < s->metrics.scatter_best_ms)
        s->metrics.scatter_best_ms = sc;
      if (dh < s->metrics.d2h_best_ms)
        s->metrics.d2h_best_ms = dh;
      s->metrics.epoch_count++;
    }
    s->flush_pending = 0;
    if (s->config.sink) {
      struct slice tiles = {
        .beg = s->h_tiles[prev].data,
        .end = (char*)s->h_tiles[prev].data + s->tile_pool_bytes,
      };
      return writer_append_wait(s->config.sink, tiles);
    }
  }

  return (struct writer_result){ 0 };

Error:
  return (struct writer_result){ .error = 1 };
}

// Synchronously flush the current tile pool (used for the final partial epoch).
static struct writer_result
flush_epoch_sync(struct transpose_stream* s)
{
  const int cur = s->tile_idx;

  if (s->config.compress && s->config.compressed_sink) {
    // Compress on compute (ordered after scatter on same stream)
    CU(Error, cuEventRecord(s->t_compress_start[cur], s->compute));
    CHECK(Error,
          compress_batch_async(
            (const void* const*)s->d_uncomp_ptrs[cur],
            s->d_uncomp_sizes,
            s->tile_stride * s->config.bytes_per_element,
            s->slot_count,
            s->d_comp_temp,
            s->comp_temp_bytes,
            s->d_comp_ptrs[cur],
            s->d_comp_sizes[cur],
            s->compute) == 0);
    CU(Error, cuEventRecord(s->d_compressed[cur].ready, s->compute));

    // D2H waits for compress, then transfers
    CU(Error,
       cuStreamWaitEvent(s->d2h, s->d_compressed[cur].ready, 0));
    CU(Error, cuEventRecord(s->t_d2h_start[cur], s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_compressed[cur].data,
                          (CUdeviceptr)s->d_compressed[cur].data,
                          s->comp_pool_bytes,
                          s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_comp_sizes[cur],
                          (CUdeviceptr)s->d_comp_sizes[cur],
                          s->slot_count * sizeof(size_t),
                          s->d2h));
    CU(Error, cuEventRecord(s->h_compressed[cur].ready, s->d2h));
    CU(Error, cuStreamSynchronize(s->d2h));

    // Accumulate scatter, compress, and D2H timing
    {
      float sc = 0, co = 0, dh = 0;
      cuEventElapsedTime(
        &sc, s->t_scatter_start[cur], s->d_tiles[cur].ready);
      s->metrics.scatter_ms += sc;
      cuEventElapsedTime(
        &co, s->t_compress_start[cur], s->d_compressed[cur].ready);
      s->metrics.compress_ms += co;
      cuEventElapsedTime(
        &dh, s->t_d2h_start[cur], s->h_compressed[cur].ready);
      s->metrics.d2h_ms += dh;
      if (sc < s->metrics.scatter_best_ms)
        s->metrics.scatter_best_ms = sc;
      if (co < s->metrics.compress_best_ms)
        s->metrics.compress_best_ms = co;
      if (dh < s->metrics.d2h_best_ms)
        s->metrics.d2h_best_ms = dh;
      s->metrics.epoch_count++;
    }

    if (deliver_compressed_tiles(s, cur))
      goto Error;
  } else {
    CU(Error, cuStreamWaitEvent(s->d2h, s->d_tiles[cur].ready, 0));
    CU(Error, cuEventRecord(s->t_d2h_start[cur], s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_tiles[cur].data,
                          (CUdeviceptr)s->d_tiles[cur].data,
                          s->tile_pool_bytes,
                          s->d2h));
    CU(Error, cuEventRecord(s->h_tiles[cur].ready, s->d2h));
    CU(Error, cuStreamSynchronize(s->d2h));

    // Accumulate scatter and D2H timing
    {
      float sc = 0, dh = 0;
      cuEventElapsedTime(
        &sc, s->t_scatter_start[cur], s->d_tiles[cur].ready);
      s->metrics.scatter_ms += sc;
      cuEventElapsedTime(
        &dh, s->t_d2h_start[cur], s->h_tiles[cur].ready);
      s->metrics.d2h_ms += dh;
      if (sc < s->metrics.scatter_best_ms)
        s->metrics.scatter_best_ms = sc;
      if (dh < s->metrics.d2h_best_ms)
        s->metrics.d2h_best_ms = dh;
      s->metrics.epoch_count++;
    }

    if (s->config.sink) {
      struct slice tiles = {
        .beg = s->h_tiles[cur].data,
        .end = (char*)s->h_tiles[cur].data + s->tile_pool_bytes,
      };
      return writer_append_wait(s->config.sink, tiles);
    }
  }

  return (struct writer_result){ 0 };

Error:
  return (struct writer_result){ .error = 1 };
}

// Flush the current epoch's tile pool: async D2H, swap pools, zero next.
static struct writer_result
flush_epoch(struct transpose_stream* s)
{
  const int cur = s->tile_idx;

  // Deliver the previous epoch if its D2H is still in flight
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  if (s->config.compress && s->config.compressed_sink) {
    // Compress on compute (ordered after scatter on same stream)
    CU(Error, cuEventRecord(s->t_compress_start[cur], s->compute));
    CHECK(Error,
          compress_batch_async(
            (const void* const*)s->d_uncomp_ptrs[cur],
            s->d_uncomp_sizes,
            s->tile_stride * s->config.bytes_per_element,
            s->slot_count,
            s->d_comp_temp,
            s->comp_temp_bytes,
            s->d_comp_ptrs[cur],
            s->d_comp_sizes[cur],
            s->compute) == 0);
    CU(Error, cuEventRecord(s->d_compressed[cur].ready, s->compute));

    // D2H waits for compress, then transfers
    CU(Error,
       cuStreamWaitEvent(s->d2h, s->d_compressed[cur].ready, 0));
    CU(Error, cuEventRecord(s->t_d2h_start[cur], s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_compressed[cur].data,
                          (CUdeviceptr)s->d_compressed[cur].data,
                          s->comp_pool_bytes,
                          s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_comp_sizes[cur],
                          (CUdeviceptr)s->d_comp_sizes[cur],
                          s->slot_count * sizeof(size_t),
                          s->d2h));
    CU(Error, cuEventRecord(s->h_compressed[cur].ready, s->d2h));
  } else {
    CU(Error, cuStreamWaitEvent(s->d2h, s->d_tiles[cur].ready, 0));
    CU(Error, cuEventRecord(s->t_d2h_start[cur], s->d2h));
    CU(Error,
       cuMemcpyDtoHAsync(s->h_tiles[cur].data,
                          (CUdeviceptr)s->d_tiles[cur].data,
                          s->tile_pool_bytes,
                          s->d2h));
    CU(Error, cuEventRecord(s->h_tiles[cur].ready, s->d2h));
  }

  // Switch to other pool and zero it for next epoch
  s->tile_idx ^= 1;
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)s->d_tiles[s->tile_idx].data,
                      0,
                      s->tile_pool_bytes,
                      s->compute));

  s->flush_pending = 1;
  return (struct writer_result){ 0 };

Error:
  return (struct writer_result){ .error = 1 };
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

  for (int i = 0; i < 2; ++i) {
    if (stream->t_h2d_start[i])
      CUWARN(cuEventDestroy(stream->t_h2d_start[i]));
    if (stream->t_scatter_start[i])
      CUWARN(cuEventDestroy(stream->t_scatter_start[i]));
    if (stream->t_compress_start[i])
      CUWARN(cuEventDestroy(stream->t_compress_start[i]));
    if (stream->t_d2h_start[i])
      CUWARN(cuEventDestroy(stream->t_d2h_start[i]));
  }

  if (stream->d_lifted_shape)
    CUWARN(cuMemFree((CUdeviceptr)stream->d_lifted_shape));
  if (stream->d_lifted_strides)
    CUWARN(cuMemFree((CUdeviceptr)stream->d_lifted_strides));

  buffer_free(&stream->h_in[0]);
  buffer_free(&stream->h_in[1]);
  buffer_free(&stream->d_in[0]);
  buffer_free(&stream->d_in[1]);
  buffer_free(&stream->d_tiles[0]);
  buffer_free(&stream->d_tiles[1]);
  buffer_free(&stream->h_tiles[0]);
  buffer_free(&stream->h_tiles[1]);

  // Compression buffers
  buffer_free(&stream->d_compressed[0]);
  buffer_free(&stream->d_compressed[1]);
  buffer_free(&stream->h_compressed[0]);
  buffer_free(&stream->h_compressed[1]);

  for (int i = 0; i < 2; ++i) {
    device_free(stream->d_uncomp_ptrs[i]);
    device_free(stream->d_comp_ptrs[i]);
    device_free(stream->d_comp_sizes[i]);
    host_free(stream->h_comp_sizes[i]);
  }
  device_free(stream->d_uncomp_sizes);
  device_free(stream->d_comp_temp);

  *stream = (struct transpose_stream){ 0 };
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
    CU(Fail, cuEventCreate(&out->t_h2d_start[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->t_scatter_start[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->t_compress_start[i], CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->t_d2h_start[i], CU_EVENT_DEFAULT));
  }

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;

  // Lifted shape (row-major, slowest first): (t_{D-1}, n_{D-1}, ..., t_0, n_0)
  out->lifted_rank = 2 * rank;
  out->tile_elements = 1;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    out->lifted_shape[2 * i] = tile_count[i];
    out->lifted_shape[2 * i + 1] = dims[i].tile_size;
    out->tile_elements *= dims[i].tile_size;
  }

  // Compute tile_stride: pad tile_elements so each tile starts at an aligned
  // byte address when compression is enabled.
  {
    size_t alignment = config->compress ? compress_get_input_alignment() : 1;
    size_t tile_bytes = out->tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    out->tile_stride = padded_bytes / bpe;
  }

  // Build lifted strides
  //   n_stride: within-tile element stride, accumulates tile_size
  //   t_stride: tile-pool element stride, accumulates tile_count
  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)out->tile_stride;

    for (int i = rank - 1; i >= 0; --i) {
      out->lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      out->lifted_strides[2 * i] = t_stride;
      t_stride *= (int64_t)tile_count[i];
    }
  }

  // An epoch is one slice of the array along the outermost tile dimension —
  // all the tiles excluding that slowest-varying tile index. The tile pool
  // holds exactly one epoch, so we flush after every tile_count[0] steps.
  out->slot_count = out->lifted_strides[0] / out->tile_stride;
  out->epoch_elements = out->slot_count * out->tile_elements;

  // Collapse epoch dimension: the outermost tile index wraps via flush,
  // so its stride is zero in the kernel.
  out->lifted_strides[0] = 0;

  out->tile_pool_bytes = out->slot_count * out->tile_stride * bpe;

  // Allocate device copies of lifted shape and strides
  {
    const size_t shape_bytes = out->lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = out->lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_lifted_shape, shape_bytes));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_lifted_strides, strides_bytes));
    CU(Fail,
       cuMemcpyHtoD(
         (CUdeviceptr)out->d_lifted_shape, out->lifted_shape, shape_bytes));
    CU(Fail,
       cuMemcpyHtoD((CUdeviceptr)out->d_lifted_strides,
                     out->lifted_strides,
                     strides_bytes));
  }

  // Allocate input staging buffers (double-buffered)
  // h_in: host writes sequentially, GPU reads via H2D -> write-combined
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (out->h_in[i] = buffer_new(config->buffer_capacity_bytes,
                                      host,
                                      CU_MEMHOSTALLOC_WRITECOMBINED))
            .data);
    CHECK(Fail,
          (out->d_in[i] = buffer_new(config->buffer_capacity_bytes, device, 0))
            .data);
  }

  // Allocate tile pools (double-buffered)
  for (int i = 0; i < 2; ++i) {
    CHECK(
      Fail,
      (out->d_tiles[i] = buffer_new(out->tile_pool_bytes, device, 0)).data);
    // h_tiles: GPU writes via D2H, host reads -> normal pinned (no WC)
    CHECK(Fail,
          (out->h_tiles[i] = buffer_new(out->tile_pool_bytes, host, 0)).data);
    CU(Fail,
       cuMemsetD8Async((CUdeviceptr)out->d_tiles[i].data,
                        0,
                        out->tile_pool_bytes,
                        out->compute));
  }

  // Compression buffers
  if (config->compress) {
    const uint64_t M = out->slot_count;
    const size_t tile_bytes = out->tile_stride * bpe;

    out->max_comp_chunk_bytes = align_up(
      compress_get_max_output_size(tile_bytes),
      compress_get_input_alignment());
    CHECK(Fail, out->max_comp_chunk_bytes > 0);
    out->comp_pool_bytes = M * out->max_comp_chunk_bytes;

    out->comp_temp_bytes = compress_get_temp_size(M, tile_bytes);
    if (out->comp_temp_bytes > 0) {
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&out->d_comp_temp, out->comp_temp_bytes));
    }

    for (int i = 0; i < 2; ++i) {
      CHECK(Fail,
            (out->d_compressed[i] =
               buffer_new(out->comp_pool_bytes, device, 0))
              .data);
      CHECK(
        Fail,
        (out->h_compressed[i] = buffer_new(out->comp_pool_bytes, host, 0))
          .data);

      // Per-chunk compressed sizes
      CU(Fail,
         cuMemAlloc((CUdeviceptr*)&out->d_comp_sizes[i],
                     M * sizeof(size_t)));
      CU(Fail,
         cuMemHostAlloc(
           (void**)&out->h_comp_sizes[i], M * sizeof(size_t), 0));
    }

    // Build device pointer arrays for nvcomp batch API
    {
      void** h_ptrs = (void**)malloc(M * sizeof(void*));
      CHECK(Fail, h_ptrs);

      for (int i = 0; i < 2; ++i) {
        CU(Fail2,
           cuMemAlloc((CUdeviceptr*)&out->d_uncomp_ptrs[i],
                       M * sizeof(void*)));
        CU(Fail2,
           cuMemAlloc((CUdeviceptr*)&out->d_comp_ptrs[i],
                       M * sizeof(void*)));

        // Uncompressed pointers: offsets into d_tiles[i]
        for (uint64_t k = 0; k < M; ++k)
          h_ptrs[k] = (char*)out->d_tiles[i].data + k * tile_bytes;
        CU(Fail2,
           cuMemcpyHtoD((CUdeviceptr)out->d_uncomp_ptrs[i],
                         h_ptrs,
                         M * sizeof(void*)));

        // Compressed pointers: offsets into d_compressed[i]
        for (uint64_t k = 0; k < M; ++k)
          h_ptrs[k] =
            (char*)out->d_compressed[i].data + k * out->max_comp_chunk_bytes;
        CU(Fail2,
           cuMemcpyHtoD((CUdeviceptr)out->d_comp_ptrs[i],
                         h_ptrs,
                         M * sizeof(void*)));
      }

      // Uncompressed sizes: all the same
      {
        size_t* h_sizes = (size_t*)malloc(M * sizeof(size_t));
        if (!h_sizes) {
          free(h_ptrs);
          goto Fail;
        }
        for (uint64_t k = 0; k < M; ++k)
          h_sizes[k] = tile_bytes;

        CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_uncomp_sizes,
                              M * sizeof(size_t)));
        CUresult rc = cuMemcpyHtoD(
          (CUdeviceptr)out->d_uncomp_sizes, h_sizes, M * sizeof(size_t));
        free(h_sizes);
        free(h_ptrs);
        h_ptrs = NULL;
        CU(Fail, rc);
      }

      if (h_ptrs)
        free(h_ptrs);
    }

    // Record initial events for compressed host buffers
    CU(Fail, cuEventRecord(out->h_compressed[0].ready, out->compute));
    CU(Fail, cuEventRecord(out->h_compressed[1].ready, out->compute));
  }

  // Record initial events so first cuEventSynchronize / cuStreamWaitEvent
  // calls succeed immediately.
  CU(Fail, cuEventRecord(out->h_in[0].ready, out->compute));
  CU(Fail, cuEventRecord(out->h_in[1].ready, out->compute));
  CU(Fail, cuEventRecord(out->d_tiles[0].ready, out->compute));
  CU(Fail, cuEventRecord(out->d_tiles[1].ready, out->compute));

  // Seed timing start events so cuEventElapsedTime never sees unrecorded events
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(out->t_h2d_start[i], out->compute));
    CU(Fail, cuEventRecord(out->t_scatter_start[i], out->compute));
    CU(Fail, cuEventRecord(out->t_compress_start[i], out->compute));
    CU(Fail, cuEventRecord(out->t_d2h_start[i], out->compute));
  }

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics.scatter_best_ms = 1e30f;
  out->metrics.compress_best_ms = 1e30f;
  out->metrics.d2h_best_ms = 1e30f;

  out->cursor = 0;
  out->stage_fill = 0;
  out->stage_idx = 0;
  out->tile_idx = 0;
  out->flush_pending = 0;

  return 0;

Fail2:
  // h_ptrs leaked if we jump here — but the destroy path will clean up
  // all device allocations, and destroy is called from Fail.
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
      s->epoch_elements - (s->cursor % s->epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    const uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    const uint64_t bytes_this_pass = elements_this_pass * bpe;

    // Fill h_in, dispatch H2D + kernel in buffer-sized chunks
    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - s->stage_fill;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t chunk = space < remaining ? space : (size_t)remaining;

        // Wait for this staging buffer's prior H2D to finish
        if (s->stage_fill == 0) {
          const int si = s->stage_idx;
          CU(Error, cuEventSynchronize(s->h_in[si].ready));
          // Accumulate H2D time from previous dispatch on this slot
          if (s->cursor > 0) {
            float ms = 0;
            cuEventElapsedTime(&ms, s->t_h2d_start[si], s->h_in[si].ready);
            s->metrics.h2d_ms += ms;
          }
        }

        memcpy(
          (uint8_t*)s->h_in[s->stage_idx].data + s->stage_fill,
          src + written,
          chunk);
        s->stage_fill += chunk;
        written += chunk;

        if (s->stage_fill == buffer_capacity || written == bytes_this_pass) {
          if (dispatch_scatter(s))
            goto Error;
          s->stage_fill = 0;
        }
      }
    }
    src += bytes_this_pass;

    // If we just completed an epoch, flush it
    if (s->cursor % s->epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_epoch(s);
      if (fr.error)
        return (struct writer_result){ .error = 1,
                                       .rest = { .beg = src, .end = end } };
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return (struct writer_result){ .error = 1,
                                 .rest = { .beg = src, .end = end } };
}

static struct writer_result
transpose_stream_flush(struct writer* self)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  // Dispatch any remaining staged data
  if (s->stage_fill > 0) {
    if (dispatch_scatter(s))
      return (struct writer_result){ .error = 1 };
    s->stage_fill = 0;
  }

  // Drain any pending async epoch flush
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Flush current partial epoch synchronously (skip if cursor is on an
  // epoch boundary — that epoch was already flushed during append)
  if (s->cursor % s->epoch_elements != 0 || s->cursor == 0) {
    r = flush_epoch_sync(s);
    if (r.error)
      return r;
  }

  if (s->config.compress && s->config.compressed_sink)
    return s->config.compressed_sink->flush(s->config.compressed_sink)
             ? (struct writer_result){ .error = 1 }
             : (struct writer_result){ 0 };

  if (s->config.sink)
    return writer_flush(s->config.sink);

  return (struct writer_result){ 0 };
}
