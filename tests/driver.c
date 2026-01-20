#include <stdio.h>

extern int
f(void);

#define ERR(...) fprintf(stderr, __VA_ARGS__)
#define EXPECT(e, fmt, ...)                                                    \
  do {                                                                         \
    if (!(e)) {                                                                \
      ERR("FAIL: " fmt, __VA_ARGS__);                                          \
      goto Fail;                                                               \
    }                                                                          \
  } while (0)
#define EXPECT_INT_EQ(a, b)                                                    \
  do {                                                                         \
    const int va = (a);                                                        \
    const int vb = (b);                                                        \
    EXPECT(                                                                    \
      va == vb, "Expected \"" #a "\" == \"" #b "\". Got %d != %d", va, vb);    \
  } while (0)

int
main(void)
{

  EXPECT_INT_EQ(f(), 5);

  return 0;
Fail:
  return 1;
}
