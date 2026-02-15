#include "test_platform.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
test_tmpdir_create(char* buf, size_t cap)
{
  char tmp[MAX_PATH];
  DWORD len = GetTempPathA(sizeof(tmp), tmp);
  if (len == 0 || len >= sizeof(tmp))
    return -1;

  // Generate a unique subdirectory name
  char name[MAX_PATH];
  if (GetTempFileNameA(tmp, "tst", 0, name) == 0)
    return -1;

  // GetTempFileNameA creates a file — delete it, then create a directory
  DeleteFileA(name);
  if (!CreateDirectoryA(name, NULL))
    return -1;

  size_t nlen = strlen(name);
  if (nlen + 1 > cap)
    return -1;
  memcpy(buf, name, nlen + 1);
  return 0;
}

int
test_tmpdir_remove(const char* path)
{
  char cmd[4200];
  snprintf(cmd, sizeof(cmd), "rmdir /s /q \"%s\"", path);
  return system(cmd);
}

int
test_mkdir(const char* path)
{
  if (CreateDirectoryA(path, NULL))
    return 0;
  if (GetLastError() == ERROR_ALREADY_EXISTS)
    return 0;
  return -1;
}

int
test_file_exists(const char* path)
{
  DWORD attr = GetFileAttributesA(path);
  return attr != INVALID_FILE_ATTRIBUTES;
}
