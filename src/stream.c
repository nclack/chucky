#include "log/log.h"
#include "stream.h"

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
handle_curesult(CUresult ecode,
                int level,
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

  // Destroy the event
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

static struct transpose_pipeline
transpose_pipeline_new(size_t capacity)
{
  struct transpose_pipeline out = {
    .h_in = buffer_new(capacity, host),
    .d_in = buffer_new(capacity, device),
    .d_out = buffer_new(capacity, device),
    .h_out = buffer_new(capacity, host),
  };
  return out;
}

static void
transpose_pipeline_free(struct transpose_pipeline* self)
{
  buffer_free(&self->h_in);
  buffer_free(&self->d_in);
  buffer_free(&self->d_out);
  buffer_free(&self->h_out);
  *self = (struct transpose_pipeline){ 0 };
}

void
transpose_stream_destroy(struct transpose_stream* stream)
{
  if (!stream) {
    return;
  }

  // TODO: trigger flush

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->d2h));

  transpose_pipeline_free(&stream->pipeline[0]);
  transpose_pipeline_free(&stream->pipeline[1]);

  *stream = (struct transpose_stream){ 0 };
}

struct transpose_stream_result
transpose_stream_create(const struct transpose_stream_configuration* config)
{
  struct transpose_stream stream = { .config = *config };

  CU(Fail, cuStreamCreate(&stream.h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&stream.compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&stream.d2h, CU_STREAM_NON_BLOCKING));

  stream.pipeline[0] = transpose_pipeline_new(config->buffer_capacity_bytes);
  stream.pipeline[1] = transpose_pipeline_new(config->buffer_capacity_bytes);

  return (struct transpose_stream_result){
    .tag = result_ok,
    .stream = stream,
  };

Fail:
  transpose_stream_destroy(&stream);
  return (struct transpose_stream_result){
    .tag = result_err,
    .ecode = error_code_fail,
  };
}
