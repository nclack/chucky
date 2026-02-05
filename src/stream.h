#pragma once

#include <cuda.h>

enum result_status
{
  result_ok = 0,
  result_err = 1
};

enum error_code
{
  error_code_ok = 0,
  error_code_fail = 1
};

enum domain
{
  host,
  device
};

struct buffer
{
  void* data;
  CUevent ready;
  enum domain domain;
};

struct transpose_pipeline
{
  struct buffer h_in, d_in, d_out, h_out;
};

struct transpose_stream_configuration
{
  size_t buffer_capacity_bytes;
  uint8_t rank;
  const uint64_t* shape;
  const int64_t* strides;
};

struct transpose_stream
{
  CUstream h2d, compute, d2h;
  struct transpose_pipeline pipeline[2];
  struct transpose_stream_configuration config;
};

struct transpose_stream_result
{
  enum result_status tag;
  union
  {
    struct transpose_stream stream;
    enum error_code ecode;
  };
};

struct transpose_stream_result
transpose_stream_create(const struct transpose_stream_configuration* config);

void
transpose_stream_destroy(struct transpose_stream* stream);

// # stream
// Buffered writer using host-pinned memory for streaming to a cuda device.
struct writer
{
  void* __restrict__ d_out;
  void* __restrict__ h_in[2];
  size_t capacity, bytes_written;
  CUstream stream;
  CUevent done[2], // done[i] signals when h_in[i] is done transfering, ready to
                   // be used again
    ready;         // ready indicates when d_out
};

struct writer_result
{
  enum result_status tag;
  union
  {
    size_t nbytes;
    int ecode;
  };
};

// Allocates resources and returns the writer.
struct writer
writer_create(size_t capacity);

// Flushes and releases resources.
void
writer_destroy(struct writer* writer);

// Trys to appends the bytes [beg,end) to the stream.
// Returns the number of bytes appended to the writer.
// Only appends up to the available buffering capacity.
struct writer_result
writer_append(struct writer* writer, void* beg, void* end);

// Empties the writer of any buffered bytes by flushing them to their
// destination. Returns the number of bytes written if any.
// Will block until all buffered bytes are flushed.
struct writer_result
writer_flush(struct writer* writer);

/*
# Notes

## Single transfer stream
Back-pressure.
- Only two buffers so need to be able to queue the next one while one finishes
  and in enough time to avoid any scheduling jitter.

No back-pressure
- will see some scheduling latency

At 64GB/s, and 100 ms jitter, would want something like 6.4 GB buffers.

Instead of making bigger double buffers, more buffers is probably better.
I bet there's an interesting analysis here.

## Writer composition

struct transposer {
  struct writer writer;
  struct CUstream stream;
  ...
};

struct writer_result
tranposer_append(self, void *beg, void *end) {
  struct writer_result res = writer_write(writer,beg,end); // fills up d_dst
sometimes? if(writer indicates another "capacity" worth of bytes was written) {
    barrier()
  }

}


*/
