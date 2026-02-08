#include "platform.h"
#include <time.h>

void
platform_sleep_ns(int64_t ns)
{
  struct timespec ts = {
    .tv_sec = ns / 1000000000LL,
    .tv_nsec = ns % 1000000000LL,
  };
  nanosleep(&ts, NULL);
}

static int64_t
monotonic_ns(void)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (now - clock->last_ns) / 1e9f;
  clock->last_ns = now;
  return elapsed;
}
