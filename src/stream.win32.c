#include "platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void
platform_sleep_ns(int64_t ns)
{
  /* Sleep() has millisecond granularity; round up so we never sleep short. */
  DWORD ms = (DWORD)((ns + 999999LL) / 1000000LL);
  Sleep(ms);
}

static int64_t
monotonic_ns(void)
{
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER cnt;
  if (freq.QuadPart == 0)
    QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&cnt);
  /* Convert to nanoseconds: cnt * 1e9 / freq, avoiding overflow. */
  return (int64_t)(cnt.QuadPart / freq.QuadPart) * 1000000000LL
       + (int64_t)(cnt.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (now - clock->last_ns) / 1e9f;
  clock->last_ns = now;
  return elapsed;
}
