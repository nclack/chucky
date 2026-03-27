#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  static inline uint64_t ceildiv(uint64_t a, uint64_t b)
  {
    return (a + b - 1) / b;
  }

#ifdef __cplusplus
}
#endif
