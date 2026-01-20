#include <cuda.h>
#include <stdio.h>

int main() {
  CUresult res;
  const char *name = NULL, *desc = NULL;

  res = cuInit(0);
  printf("cuInit(0) = %d\n", res);

  CUdevice dev;
  res = cuDeviceGet(&dev, 0);
  printf("cuDeviceGet(&dev, 0) = %d\n", res);

  printf("\nTesting cuCtxDestroy(NULL):\n");
  res = cuCtxDestroy(NULL);
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &desc);

  printf("cuCtxDestroy(NULL) returned: %d\n", res);
  if (name) printf("Error name: %s\n", name);
  if (desc) printf("Error desc: %s\n", desc);

  return 0;
}
