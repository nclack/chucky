#ifndef TEST_SHARD_SINK_H
#define TEST_SHARD_SINK_H

#include "writer.h"

#include <stddef.h>
#include <stdint.h>

#define TEST_SHARD_SINK_MAX_SHARDS 16

struct test_shard_writer
{
  struct shard_writer base;
  struct test_shard_sink* sink; // back-pointer for finalize
  uint8_t* buf;
  size_t capacity;
  size_t size;
};

struct test_shard_sink
{
  struct shard_sink base;
  struct test_shard_writer writers[TEST_SHARD_SINK_MAX_SHARDS];
  size_t per_shard_capacity;
  int open_count;
  int finalize_count;
};

void
test_sink_init(struct test_shard_sink* s, size_t per_shard_capacity);

void
test_sink_free(struct test_shard_sink* s);

#endif // TEST_SHARD_SINK_H
