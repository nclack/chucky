#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum dtype
  {
    dtype_u8,
    dtype_u16,
    dtype_u32,
    dtype_u64,
    dtype_i8,
    dtype_i16,
    dtype_i32,
    dtype_i64,
    dtype_f16,
    dtype_f32,
    dtype_f64,
  };

  static inline size_t dtype_bpe(enum dtype dt)
  {
    switch (dt) {
      case dtype_u8:
      case dtype_i8:
        return 1;
      case dtype_u16:
      case dtype_i16:
      case dtype_f16:
        return 2;
      case dtype_u32:
      case dtype_i32:
      case dtype_f32:
        return 4;
      case dtype_u64:
      case dtype_i64:
      case dtype_f64:
        return 8;
    }
    return 0;
  }

  // Zarr v3 data_type string for metadata JSON.
  static inline const char* dtype_zarr_string(enum dtype dt)
  {
    switch (dt) {
      case dtype_u8:
        return "uint8";
      case dtype_u16:
        return "uint16";
      case dtype_u32:
        return "uint32";
      case dtype_u64:
        return "uint64";
      case dtype_i8:
        return "int8";
      case dtype_i16:
        return "int16";
      case dtype_i32:
        return "int32";
      case dtype_i64:
        return "int64";
      case dtype_f16:
        return "float16";
      case dtype_f32:
        return "float32";
      case dtype_f64:
        return "float64";
    }
    return "unknown";
  }

#ifdef __cplusplus
}
#endif
