#include <stdint.h>
#include <type_traits>

extern "C" int
f()
{
  return 5;
}

#define MAX_RANK (64)

struct layout
{
  uint8_t rank;
  uint32_t shape[MAX_RANK];  // 256 bytes
  int64_t strides[MAX_RANK]; // 512 bytes
};

