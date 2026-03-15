#include "test_shard_sink.h"

#include <stdlib.h>
#include <string.h>

static int
test_sink_write(struct shard_writer* self,
                uint64_t offset,
                const void* beg,
                const void* end)
{
  struct test_shard_writer* w = (struct test_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);
  if (offset + nbytes > w->capacity)
    return 1;
  memcpy(w->buf + offset, beg, nbytes);
  if (offset + nbytes > w->size)
    w->size = offset + nbytes;
  return 0;
}

static int
test_sink_finalize(struct shard_writer* self)
{
  struct test_shard_writer* w = (struct test_shard_writer*)self;
  w->sink->finalize_count++;
  return 0;
}

static struct shard_writer*
test_sink_open(struct shard_sink* self, uint8_t level, uint64_t shard_index)
{
  (void)level;
  struct test_shard_sink* s = (struct test_shard_sink*)self;
  if (shard_index >= TEST_SHARD_SINK_MAX_SHARDS)
    return NULL;
  s->open_count++;
  struct test_shard_writer* w = &s->writers[shard_index];
  if (!w->buf) {
    w->buf = (uint8_t*)calloc(1, s->per_shard_capacity);
    w->capacity = s->per_shard_capacity;
    w->base.write = test_sink_write;
    w->base.finalize = test_sink_finalize;
    w->sink = s;
  }
  w->size = 0;
  return &w->base;
}

void
test_sink_init(struct test_shard_sink* s, size_t per_shard_capacity)
{
  memset(s, 0, sizeof(*s));
  s->base.open = test_sink_open;
  s->per_shard_capacity = per_shard_capacity;
}

void
test_sink_free(struct test_shard_sink* s)
{
  for (int i = 0; i < TEST_SHARD_SINK_MAX_SHARDS; ++i)
    free(s->writers[i].buf);
  memset(s, 0, sizeof(*s));
}
