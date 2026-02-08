#include "log/log.h"
#include "platform.h"
#include "stream.h"

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

// Dispatch staged data: H2D transfer + scatter kernel
static void
dispatch_scatter(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const uint64_t elements = s->stage_fill / bpe;
  if (elements == 0)
    return;

  const size_t tile_pool_bytes = s->slot_count * s->tile_elements * bpe;

  // H2D
  CUWARN(cuMemcpyHtoDAsync(
    (CUdeviceptr)s->d_in.data, s->h_in.data, s->stage_fill, s->h2d));
  CUWARN(cuEventRecord(s->h_in.ready, s->h2d));

  // Kernel waits for H2D, then scatters into tile pool
  CUWARN(cuStreamWaitEvent(s->compute, s->h_in.ready, 0));
  transpose_u16_v0((CUdeviceptr)s->d_tiles.data,
                   (CUdeviceptr)s->d_tiles.data + tile_pool_bytes,
                   (CUdeviceptr)s->d_in.data,
                   (CUdeviceptr)s->d_in.data + s->stage_fill,
                   s->cursor,
                   s->lifted_rank,
                   s->d_lifted_shape,
                   s->d_lifted_strides,
                   s->compute);
  CUWARN(cuEventRecord(s->d_tiles.ready, s->compute));

  s->cursor += elements;
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

// Flush the current epoch's tile pool: D2H, sync, zero
static struct writer_result
flush_epoch(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const size_t tile_pool_bytes = s->slot_count * s->tile_elements * bpe;

  // Wait for compute to finish, then D2H
  CUWARN(cuStreamWaitEvent(s->d2h, s->d_tiles.ready, 0));
  CUWARN(cuMemcpyDtoHAsync(
    s->h_tiles.data, (CUdeviceptr)s->d_tiles.data, tile_pool_bytes, s->d2h));
  CUWARN(cuStreamSynchronize(s->d2h));

  // Deliver tiles to downstream writer
  struct writer_result r = { 0 };
  if (s->config.sink) {
    struct slice tiles = {
      .beg = s->h_tiles.data,
      .end = (char*)s->h_tiles.data + tile_pool_bytes,
    };
    r = writer_append_wait(s->config.sink, tiles);
  }

  // Zero tile pool for next epoch
  CUWARN(cuMemsetD8Async(
    (CUdeviceptr)s->d_tiles.data, 0, tile_pool_bytes, s->compute));

  return r;
}

void
transpose_stream_destroy(struct transpose_stream* stream)
{
  if (!stream)
    return;

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->d2h));

  if (stream->d_lifted_shape)
    CUWARN(cuMemFree((CUdeviceptr)stream->d_lifted_shape));
  if (stream->d_lifted_strides)
    CUWARN(cuMemFree((CUdeviceptr)stream->d_lifted_strides));

  buffer_free(&stream->h_in);
  buffer_free(&stream->d_in);
  buffer_free(&stream->d_tiles);
  buffer_free(&stream->h_tiles);

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
  *out = (struct transpose_stream){
    .writer = { .append = transpose_stream_append,
                .flush = transpose_stream_flush },
    .config = *config,
  };

  CU(Fail, cuStreamCreate(&out->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->d2h, CU_STREAM_NON_BLOCKING));

  const uint8_t rank = config->rank;
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

  // Build lifted strides
  //   n_stride: within-tile element stride, accumulates tile_size
  //   t_stride: tile-pool element stride, accumulates tile_count
  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)out->tile_elements;

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
  out->slot_count = out->lifted_strides[0] / out->tile_elements;
  out->epoch_elements = (uint64_t)out->lifted_strides[0];

  // Collapse epoch dimension: the outermost tile index wraps via flush,
  // so its stride is zero in the kernel.
  out->lifted_strides[0] = 0;

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

  // Allocate input staging buffers
  // h_in: host writes sequentially, GPU reads via H2D -> write-combined
  CHECK(
    Fail,
    (out->h_in =
       buffer_new(config->buffer_capacity_bytes, host, CU_MEMHOSTALLOC_WRITECOMBINED))
      .data);
  CHECK(Fail,
        (out->d_in = buffer_new(config->buffer_capacity_bytes, device, 0))
          .data);

  // Allocate tile pool
  const size_t tile_pool_bytes =
    out->slot_count * out->tile_elements * config->bytes_per_element;
  CHECK(Fail,
        (out->d_tiles = buffer_new(tile_pool_bytes, device, 0)).data);
  // h_tiles: GPU writes via D2H, host reads -> normal pinned (no WC)
  CHECK(Fail, (out->h_tiles = buffer_new(tile_pool_bytes, host, 0)).data);

  // Zero the tile pool initially
  CU(Fail,
     cuMemsetD8Async(
       (CUdeviceptr)out->d_tiles.data, 0, tile_pool_bytes, out->compute));
  CU(Fail, cuStreamSynchronize(out->compute));

  out->cursor = 0;
  out->stage_fill = 0;

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

        memcpy((uint8_t*)s->h_in.data + s->stage_fill, src + written, chunk);
        s->stage_fill += chunk;
        written += chunk;

        if (s->stage_fill == buffer_capacity || written == bytes_this_pass) {
          dispatch_scatter(s);
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
}

static struct writer_result
transpose_stream_flush(struct writer* self)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  // Dispatch any remaining staged data
  if (s->stage_fill > 0) {
    dispatch_scatter(s);
    s->stage_fill = 0;
  }

  // Flush partial epoch (skip if cursor is on an epoch boundary — that
  // epoch was already flushed during append)
  if (s->cursor % s->epoch_elements != 0 || s->cursor == 0) {
    struct writer_result r = flush_epoch(s);
    if (r.error)
      return r;
  }

  if (s->config.sink)
    return writer_flush(s->config.sink);

  return (struct writer_result){ 0 };
}
