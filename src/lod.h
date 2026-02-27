#pragma once

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void lod_scatter_f32(CUdeviceptr d_dst,
                       CUdeviceptr d_src,
                       int ndim,
                       uint64_t n_elements,
                       CUdeviceptr d_full_shape,
                       CUdeviceptr d_lod_shape,
                       int lod_ndim,
                       const uint64_t* lod_shape_host,
                       uint32_t lod_mask,
                       uint64_t lod_count,
                       CUstream stream);

  void lod_fill_ends_gpu(CUdeviceptr d_ends,
                         int ndim,
                         CUdeviceptr d_child_shape,
                         CUdeviceptr d_parent_shape,
                         const uint64_t* child_shape_host,
                         const uint64_t* parent_shape_host,
                         uint64_t n_parents,
                         CUstream stream);

  void lod_reduce_f32(CUdeviceptr d_values,
                      CUdeviceptr d_ends,
                      uint64_t src_offset,
                      uint64_t dst_offset,
                      uint64_t src_lod_count,
                      uint64_t dst_lod_count,
                      uint64_t batch_count,
                      CUstream stream);

#ifdef __cplusplus
}
#endif
