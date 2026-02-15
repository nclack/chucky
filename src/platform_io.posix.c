#include "platform_io.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
platform_mkdir(const char* path)
{
  if (mkdir(path, 0755) != 0 && errno != EEXIST)
    return -1;
  return 0;
}

platform_fd
platform_open_write(const char* path)
{
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}

int
platform_pwrite(platform_fd fd, const void* buf, size_t nbytes, uint64_t offset)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    ssize_t n = pwrite(fd, p, remaining, (off_t)(offset + (nbytes - remaining)));
    if (n < 0)
      return -1;
    p += n;
    remaining -= (size_t)n;
  }
  return 0;
}

int
platform_write(platform_fd fd, const void* buf, size_t nbytes)
{
  const char* p = (const char*)buf;
  size_t remaining = nbytes;
  while (remaining > 0) {
    ssize_t n = write(fd, p, remaining);
    if (n < 0)
      return -1;
    p += n;
    remaining -= (size_t)n;
  }
  return 0;
}

void
platform_close(platform_fd fd)
{
  close(fd);
}
