#include "test_platform.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int
test_tmpdir_create(char* buf, size_t cap)
{
  const char* tmpl = "/tmp/test_XXXXXX";
  size_t len = strlen(tmpl);
  if (len + 1 > cap)
    return -1;
  memcpy(buf, tmpl, len + 1);
  return mkdtemp(buf) ? 0 : -1;
}

int
test_tmpdir_remove(const char* path)
{
  char cmd[4200];
  snprintf(cmd, sizeof(cmd), "rm -rf '%s'", path);
  return system(cmd);
}

int
test_mkdir(const char* path)
{
  if (mkdir(path, 0755) != 0 && errno != EEXIST)
    return -1;
  return 0;
}

int
test_file_exists(const char* path)
{
  struct stat st;
  return stat(path, &st) == 0;
}
