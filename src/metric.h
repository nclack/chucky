/// PRIVATE: never include in other headers.
#pragma once

struct stream_metric
{
  const char* name;
  float ms;           // cumulative
  float best_ms;      // best single measurement (1e30f = not yet measured)
  double total_bytes; // cumulative bytes (for throughput from real data)
  int count;
};

static inline void
accumulate_metric_ms(struct stream_metric* m, float ms)
{
  m->ms += ms;
  m->count++;
  if (ms < m->best_ms)
    m->best_ms = ms;
}
