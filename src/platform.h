#pragma once

#include <stdint.h>

// Sleep for the given number of nanoseconds.
void
platform_sleep_ns(int64_t ns);

// Monotonic clock for timing. Returns elapsed seconds since last call.
struct platform_clock
{
  int64_t last_ns;
};

float
platform_toc(struct platform_clock* clock);
