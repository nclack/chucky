#include "log/log.h"
#include "stream.h"

#include <string.h>
#include <time.h>

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

// Domain-specific allocation
static CUresult
alloc_host(void** data, size_t capacity)
{
  return cuMemHostAlloc(data, capacity, CU_MEMHOSTALLOC_WRITECOMBINED);
}

static CUresult
alloc_device(void** data, size_t capacity)
{
  return cuMemAlloc((CUdeviceptr*)data, capacity);
}

// Domain-specific free
static void
free_host(void* data)
{
  cuMemFreeHost(data);
}

static void
free_device(void* data)
{
  cuMemFree((CUdeviceptr)data);
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
      free_host(buffer->data);
      break;
    case device:
      free_device(buffer->data);
      break;
    default:
      log_error("Invalid domain during buffer_free: %d", buffer->domain);
      return;
  }

  buffer->data = NULL;
}

static struct buffer
buffer_new(size_t capacity, enum domain domain)
{
  struct buffer buf = { 0 };
  buf.domain = domain;

  switch (domain) {
    case host:
      CU(Fail, alloc_host(&buf.data, capacity));
      break;
    case device:
      CU(Fail, alloc_device(&buf.data, capacity));
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

  const size_t tile_pool_bytes =
    s->slot_count * s->tile_elements * bpe;

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
                   s->lifted_shape,
                   s->lifted_strides,
                   s->compute);
  CUWARN(cuEventRecord(s->d_tiles.ready, s->compute));

  s->cursor += elements;
}

// Drain a slice into a writer, retrying with exponential back-off on stall.
static struct stream_result
writer_drain(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct stream_result r = w->append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_drain: no progress after %d retries", stalls);
        return (struct stream_result){ .error = 1, .rest = data };
      }
      log_warn("writer_drain: stall %d/%d, backing off", stalls, max_stalls);
      struct timespec ts = {
        .tv_sec = 0,
        .tv_nsec = 1000000L << (stalls < 6 ? stalls : 6), // 1ms..64ms
      };
      nanosleep(&ts, NULL);
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return (struct stream_result){ 0 };
}

// Flush the current epoch's tile pool: D2H, sync, zero
static struct stream_result
flush_epoch(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  const size_t tile_pool_bytes =
    s->slot_count * s->tile_elements * bpe;

  // Wait for compute to finish, then D2H
  CUWARN(cuStreamWaitEvent(s->d2h, s->d_tiles.ready, 0));
  CUWARN(cuMemcpyDtoHAsync(
    s->h_tiles.data, (CUdeviceptr)s->d_tiles.data, tile_pool_bytes, s->d2h));
  CUWARN(cuStreamSynchronize(s->d2h));

  // Deliver tiles to downstream writer
  struct stream_result r = { 0 };
  if (s->config.sink) {
    struct slice tiles = {
      .beg = s->h_tiles.data,
      .end = (char*)s->h_tiles.data + tile_pool_bytes,
    };
    r = writer_drain(s->config.sink, tiles);
  }

  // Zero tile pool for next epoch
  CUWARN(
    cuMemsetD8Async((CUdeviceptr)s->d_tiles.data, 0, tile_pool_bytes, s->compute));

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

  buffer_free(&stream->h_in);
  buffer_free(&stream->d_in);
  buffer_free(&stream->d_tiles);
  buffer_free(&stream->h_tiles);

  *stream = (struct transpose_stream){ 0 };
}

// Writer vtable wrappers — cast writer* to transpose_stream* and delegate
static struct stream_result
writer_append(struct writer* self, struct slice data)
{
  return transpose_stream_append((struct transpose_stream*)self, data);
}

static struct stream_result
writer_flush(struct writer* self)
{
  return transpose_stream_flush((struct transpose_stream*)self);
}

int
transpose_stream_create(const struct transpose_stream_configuration* config,
                        struct transpose_stream* out)
{
  *out = (struct transpose_stream){
    .base = { .append = writer_append, .flush = writer_flush },
    .config = *config,
  };

  CU(Fail, cuStreamCreate(&out->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->d2h, CU_STREAM_NON_BLOCKING));

  const uint8_t rank = config->rank;
  const struct dimension* dims = config->dimensions;

  // Compute tile_count[i] and build lifted shape
  // Lifted shape (row-major, slowest first): (t_{D-1}, n_{D-1}, ..., t_0, n_0)
  // lifted_shape[2*i]   = tile_count[i]
  // lifted_shape[2*i+1] = tile_size[i]
  out->lifted_rank = 2 * rank;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    out->lifted_shape[2 * i] = tile_count[i];
    out->lifted_shape[2 * i + 1] = dims[i].tile_size;
  }

  // tile_elements = product of all tile_size[i]
  out->tile_elements = 1;
  for (int i = 0; i < rank; ++i)
    out->tile_elements *= dims[i].tile_size;

  // slot_count M = product of tile_count[i] for i > 0 (all except slowest)
  out->slot_count = 1;
  for (int i = 1; i < rank; ++i)
    out->slot_count *= tile_count[i];

  // epoch_elements = M * tile_elements
  out->epoch_elements = out->slot_count * out->tile_elements;

  // Build lifted strides
  //
  // For dimension i (from rank-1 down to 0):
  //   lifted_strides[2*i+1] (for n_i) = n_stride, then n_stride *= tile_size[i]
  //   lifted_strides[2*i]   (for t_i) = t_stride, then t_stride *= tile_count[i]
  //   except lifted_strides[0] = 0 (collapse epoch dimension)
  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)out->tile_elements;

    for (int i = rank - 1; i >= 0; --i) {
      out->lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      if (i == 0) {
        out->lifted_strides[0] = 0;  // collapse epoch dimension
      } else {
        out->lifted_strides[2 * i] = t_stride;
        t_stride *= (int64_t)tile_count[i];
      }
    }
  }

  // Allocate input staging buffers
  out->h_in = buffer_new(config->buffer_capacity_bytes, host);
  if (!out->h_in.data)
    goto Fail;
  out->d_in = buffer_new(config->buffer_capacity_bytes, device);
  if (!out->d_in.data)
    goto Fail;

  // Allocate tile pool
  const size_t tile_pool_bytes =
    out->slot_count * out->tile_elements * config->bytes_per_element;
  out->d_tiles = buffer_new(tile_pool_bytes, device);
  if (!out->d_tiles.data)
    goto Fail;
  out->h_tiles = buffer_new(tile_pool_bytes, host);
  if (!out->h_tiles.data)
    goto Fail;

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

struct stream_result
transpose_stream_append(struct transpose_stream* s, struct slice input)
{
  const size_t bpe = s->config.bytes_per_element;
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  while (src < end) {
    // How many elements remain in the current epoch?
    uint64_t epoch_remaining =
      s->epoch_elements - (s->cursor % s->epoch_elements);
    uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    uint64_t bytes_this_pass = elements_this_pass * bpe;

    // Fill h_in, dispatch H2D + kernel in buffer-sized chunks
    uint64_t written = 0;
    while (written < bytes_this_pass) {
      size_t space = buffer_capacity - s->stage_fill;
      uint64_t remaining = bytes_this_pass - written;
      size_t chunk = space < remaining ? space : (size_t)remaining;

      memcpy((uint8_t*)s->h_in.data + s->stage_fill, src + written, chunk);
      s->stage_fill += chunk;
      written += chunk;

      if (s->stage_fill == buffer_capacity ||
          written == bytes_this_pass) {
        dispatch_scatter(s);
        s->stage_fill = 0;
      }
    }

    src += bytes_this_pass;

    // If we just completed an epoch, flush it
    if (s->cursor % s->epoch_elements == 0 && s->cursor > 0) {
      struct stream_result fr = flush_epoch(s);
      if (fr.error)
        return (struct stream_result){ .error = 1, .rest = { .beg = src, .end = end } };
    }
  }

  return (struct stream_result){ .error = 0, .rest = { .beg = src, .end = end } };
}

struct stream_result
transpose_stream_flush(struct transpose_stream* s)
{
  // Dispatch any remaining staged data
  if (s->stage_fill > 0) {
    dispatch_scatter(s);
    s->stage_fill = 0;
  }

  // Flush partial epoch (skip if cursor is on an epoch boundary — that
  // epoch was already flushed during append)
  if (s->cursor % s->epoch_elements != 0 || s->cursor == 0) {
    struct stream_result r = flush_epoch(s);
    if (r.error)
      return r;
  }

  if (s->config.sink)
    return s->config.sink->flush(s->config.sink);

  return (struct stream_result){ 0 };
}
