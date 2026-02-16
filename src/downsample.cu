#include "downsample.h"
#include <assert.h>
#include <stdint.h>

// Downsample kernel: each thread produces one output element by averaging
// 2^popcount(downsample_mask) source values from the tile pool.
//
// The tile pool uses per-dimension strides: for dimension d, the pool stride
// gives the element offset between consecutive tiles along d, and elements
// within a tile are stored contiguously (innermost-last / row-major within
// each tile).
//
// Pool strides encode: pool_stride[d] = product(tile_count[j] for j>d) *
// tile_stride (elements between tiles along dim d). Within-tile addressing
// uses tile_size strides.

template<typename T, typename Acc>
__global__ void
downsample_mean_k(T* __restrict__ d_dst,
                  const T* __restrict__ d_src_a,
                  const T* __restrict__ d_src_b,
                  uint8_t rank,
                  uint8_t downsample_mask,
                  const uint64_t* __restrict__ dst_tile_size,
                  const uint64_t* __restrict__ src_tile_size,
                  const uint64_t* __restrict__ src_extent,
                  const int64_t* __restrict__ src_pool_strides,
                  const int64_t* __restrict__ dst_pool_strides,
                  uint64_t dst_total_elements)
{
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_total_elements)
    return;

  // Decompose gid into (dst_tile[d], dst_elem[d]) for each dimension.
  // dst_pool_strides[d] is the element offset per tile along dimension d.
  // Within a tile, elements are stored in row-major order with tile_size
  // extents.
  //
  // The pool layout for the destination is:
  //   linear_offset = sum_d(tile[d] * pool_stride[d] + elem[d] * within_stride[d])
  // where within_stride is the row-major product of tile_size for dims after d.
  //
  // We decompose by first finding the tile index and within-tile offset
  // along each dimension, then mapping to source coordinates.

  // Compute within-tile strides (row-major) for destination
  uint64_t dst_within_stride[32];
  {
    uint64_t s = 1;
    for (int d = rank - 1; d >= 0; --d) {
      dst_within_stride[d] = s;
      s *= dst_tile_size[d];
    }
  }

  // Compute within-tile strides for source
  uint64_t src_within_stride[32];
  {
    uint64_t s = 1;
    for (int d = rank - 1; d >= 0; --d) {
      src_within_stride[d] = s;
      s *= src_tile_size[d];
    }
  }

  // Decompose gid into tile coords and element coords at destination level
  uint64_t dst_tile[32];
  uint64_t dst_elem[32];
  {
    uint64_t rest = gid;
    for (int d = 0; d < rank; ++d) {
      // pool_stride gives elements-per-tile-step along this dim
      uint64_t tile_and_rest = rest / dst_within_stride[d];
      uint64_t e = rest % dst_within_stride[d];

      // tile_and_rest = tile_idx * (pool_stride/within_stride is tile count
      // product) but we want: tile_idx and within-tile elem. The pool_stride
      // encodes tile_stride * prod(tile_count[j>d]). However, the total
      // elements layout is: for each tile combination, tile_elements
      // contiguous elements.
      //
      // Actually, the total element count is slot_count * tile_elements.
      // The pool_stride[d] / tile_stride gives prod(tile_count[j>d]).
      // So decomposing with tile_size works like a mixed-radix number.
      //
      // Use dst_tile_size to decompose:
      dst_tile[d] = tile_and_rest / dst_tile_size[d];
      dst_elem[d] = tile_and_rest % dst_tile_size[d];
      rest = e;
      (void)rest; // used in next iteration
    }
    // Actually, let's redo this more carefully using the pool stride layout.
    // The destination tile pool linear address for (tile[0..D-1], elem[0..D-1])
    // is: sum(tile[d] * pool_stride[d]) + sum(elem[d] * within_stride[d])
    //
    // So gid represents the linear index through all pool slots, and we need
    // to invert that. The pool is structured as tiles addressed by pool_strides
    // with within-tile elements addressed by within_strides.
    //
    // A simpler decomposition: compute tile_elements and slot_count from the
    // strides, then:
    //   slot_index = gid / tile_elements
    //   elem_index = gid % tile_elements
    // Then decompose each separately.
  }

  // Simpler decomposition:
  uint64_t tile_elements = 1;
  for (int d = 0; d < rank; ++d)
    tile_elements *= dst_tile_size[d];

  uint64_t slot_index = gid / tile_elements;
  uint64_t elem_index = gid % tile_elements;

  // Decompose slot_index into per-dim tile indices using pool_strides.
  // pool_stride[d] encodes: tile_stride * prod(tile_count[j] for j>d)
  // But for the slot (without tile_stride factor), the slot_index is
  // simply decomposed by tile_count along each dimension.
  // slot_index = sum(tile[d] * prod(tile_count[j] for j>d))
  //
  // We can recover tile_count[d] = pool_stride[d-1] / pool_stride[d]
  // (with pool_stride[-1] = slot_count * tile_stride).
  // But it's simpler to just decompose with the strides directly.
  //
  // Since pool_stride[d] in elements includes the tile_stride factor:
  //   pool_stride[d] = tile_stride * prod(tile_count[j] for j > d)
  // and tile_stride >= tile_elements, we have:
  //   slot_stride[d] = pool_stride[d] / tile_stride
  // which is the slot-level stride.
  //
  // But we don't have tile_stride here. Instead, let's compute slot strides:
  // slot_stride[d] = prod(tile_count[j] for j > d) where tile_count is
  // implicit from pool_strides ratio.
  //
  // Actually, the pool_strides passed in are per-element strides. So for the
  // slot decomposition, we divide by the tile_stride (which is
  // pool_stride[rank-1] since the innermost dimension has 1 tile count after
  // it, giving pool_stride[rank-1] = tile_stride).
  //
  // Wait - pool_stride[d] represents the element offset between tiles along
  // dim d. For the innermost dim (d=rank-1), pool_stride = tile_stride
  // (the padded tile size). For d=rank-2, pool_stride = tile_stride *
  // tile_count[rank-1]. Etc.
  //
  // So slot_stride[d] = pool_stride[d] / pool_stride[rank-1], and
  // tile_count[d] = pool_stride[d-1] / pool_stride[d] (or total slots /
  // pool_stride[0] for d=0... wait, that's not right either since dim 0
  // stride is 0 for the stream layout).
  //
  // For LOD levels, dim 0 stride won't be 0 since we're addressing full
  // pools. Let me just use a straightforward approach:

  {
    // Recompute: pool_stride[d] = tile_stride * prod(tile_count[j] for j>d)
    // tile_stride = pool_stride[rank-1] (when tile_count[rank-1] has no
    // further dims)
    // Actually for rank-1: pool_stride[rank-1] = tile_stride * 1 = tile_stride
    int64_t tile_stride = dst_pool_strides[rank - 1];
    uint64_t rest = slot_index;
    for (int d = 0; d < rank; ++d) {
      uint64_t slot_stride = (uint64_t)dst_pool_strides[d] / (uint64_t)tile_stride;
      dst_tile[d] = rest / slot_stride;
      rest = rest % slot_stride;
    }
  }

  // Decompose elem_index into per-dim element coords
  {
    uint64_t rest = elem_index;
    for (int d = 0; d < rank; ++d) {
      dst_elem[d] = rest / dst_within_stride[d];
      rest = rest % dst_within_stride[d];
    }
  }

  // Now compute the mean over source elements.
  // For each downsampled dim d:
  //   global_dst_coord[d] = dst_tile[d] * dst_tile_size[d] + dst_elem[d]
  //   Two source coords: 2 * global_dst_coord[d] and 2 * global_dst_coord[d]+1
  //   Each maps to (src_tile, src_elem) = divmod by src_tile_size[d]
  // For non-downsampled dims: source coords = dest coords.
  //
  // For dim 0 if downsampled: one coord from pool_a, other from pool_b.
  // For dim > 0 if downsampled: both coords from same pool.
  //
  // We iterate over all 2^k combinations of lo/hi per downsampled dim.

  int num_ds = __popc((unsigned int)downsample_mask);
  int num_samples = 1 << num_ds;

  Acc accum = 0;
  int actual_samples = 0;

  for (int combo = 0; combo < num_samples; ++combo) {
    // Map combo bits to per-dimension offsets (0 or 1)
    uint64_t src_tile_coord[32];
    uint64_t src_elem_coord[32];
    int use_pool_b = 0; // set if dim 0 is downsampled and this combo uses +1

    int bit = 0;
    for (int d = 0; d < rank; ++d) {
      if (downsample_mask & (1 << d)) {
        uint64_t global_dst = dst_tile[d] * dst_tile_size[d] + dst_elem[d];
        uint64_t src_global = 2 * global_dst + ((combo >> bit) & 1);

        // Clamp to source extent
        if (src_global >= src_extent[d])
          src_global = src_extent[d] - 1;

        if (d == 0 && d_src_b != nullptr) {
          // Dim 0 downsampled: even goes to pool_a, odd goes to pool_b
          if (src_global & 1)
            use_pool_b = 1;
          // Both pool_a and pool_b have the same tile layout but represent
          // consecutive epochs. The src_tile/elem within the pool is just
          // the same as the non-dim0 address (pool holds one epoch).
          // For dim 0 in the pool, the tile index is always 0 (epoch is
          // one slice along dim 0), so src_tile[0] = 0, src_elem[0] = 0.
          src_tile_coord[d] = 0;
          src_elem_coord[d] = 0;
        } else {
          src_tile_coord[d] = src_global / src_tile_size[d];
          src_elem_coord[d] = src_global % src_tile_size[d];
        }
        bit++;
      } else {
        src_tile_coord[d] = dst_tile[d];
        src_elem_coord[d] = dst_elem[d];
      }
    }

    // Compute source pool offset
    int64_t src_offset = 0;
    for (int d = 0; d < rank; ++d) {
      src_offset += (int64_t)src_tile_coord[d] * src_pool_strides[d];
      src_offset += (int64_t)src_elem_coord[d] * (int64_t)src_within_stride[d];
    }

    const T* pool = use_pool_b ? d_src_b : d_src_a;
    accum += (Acc)pool[src_offset];
    actual_samples++;
  }

  // Compute destination pool offset
  int64_t dst_offset = 0;
  for (int d = 0; d < rank; ++d) {
    dst_offset += (int64_t)dst_tile[d] * dst_pool_strides[d];
    dst_offset += (int64_t)dst_elem[d] * (int64_t)dst_within_stride[d];
  }

  d_dst[dst_offset] = (T)((accum + actual_samples / 2) / actual_samples);
}

extern "C" void
downsample_mean_u8(CUdeviceptr d_dst,
                   CUdeviceptr d_src_a,
                   CUdeviceptr d_src_b,
                   uint8_t rank,
                   uint8_t downsample_mask,
                   const uint64_t* d_dst_tile_size,
                   const uint64_t* d_src_tile_size,
                   const uint64_t* d_src_extent,
                   const int64_t* d_src_pool_strides,
                   const int64_t* d_dst_pool_strides,
                   uint64_t dst_total_elements,
                   CUstream stream)
{
  if (dst_total_elements == 0)
    return;

  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int block_size = 256;
  const int grid_size =
    (int)((dst_total_elements + block_size - 1) / block_size);

  downsample_mean_k<uint8_t, uint32_t>
    <<<grid_size, block_size, 0, cuda_stream>>>(
      (uint8_t*)d_dst,
      (const uint8_t*)d_src_a,
      (const uint8_t*)d_src_b,
      rank,
      downsample_mask,
      d_dst_tile_size,
      d_src_tile_size,
      d_src_extent,
      d_src_pool_strides,
      d_dst_pool_strides,
      dst_total_elements);
}

extern "C" void
downsample_mean_u16(CUdeviceptr d_dst,
                    CUdeviceptr d_src_a,
                    CUdeviceptr d_src_b,
                    uint8_t rank,
                    uint8_t downsample_mask,
                    const uint64_t* d_dst_tile_size,
                    const uint64_t* d_src_tile_size,
                    const uint64_t* d_src_extent,
                    const int64_t* d_src_pool_strides,
                    const int64_t* d_dst_pool_strides,
                    uint64_t dst_total_elements,
                    CUstream stream)
{
  if (dst_total_elements == 0)
    return;

  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int block_size = 256;
  const int grid_size =
    (int)((dst_total_elements + block_size - 1) / block_size);

  downsample_mean_k<uint16_t, uint32_t>
    <<<grid_size, block_size, 0, cuda_stream>>>(
      (uint16_t*)d_dst,
      (const uint16_t*)d_src_a,
      (const uint16_t*)d_src_b,
      rank,
      downsample_mask,
      d_dst_tile_size,
      d_src_tile_size,
      d_src_extent,
      d_src_pool_strides,
      d_dst_pool_strides,
      dst_total_elements);
}

extern "C" void
downsample_mean_u32(CUdeviceptr d_dst,
                    CUdeviceptr d_src_a,
                    CUdeviceptr d_src_b,
                    uint8_t rank,
                    uint8_t downsample_mask,
                    const uint64_t* d_dst_tile_size,
                    const uint64_t* d_src_tile_size,
                    const uint64_t* d_src_extent,
                    const int64_t* d_src_pool_strides,
                    const int64_t* d_dst_pool_strides,
                    uint64_t dst_total_elements,
                    CUstream stream)
{
  if (dst_total_elements == 0)
    return;

  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int block_size = 256;
  const int grid_size =
    (int)((dst_total_elements + block_size - 1) / block_size);

  downsample_mean_k<uint32_t, uint64_t>
    <<<grid_size, block_size, 0, cuda_stream>>>(
      (uint32_t*)d_dst,
      (const uint32_t*)d_src_a,
      (const uint32_t*)d_src_b,
      rank,
      downsample_mask,
      d_dst_tile_size,
      d_src_tile_size,
      d_src_extent,
      d_src_pool_strides,
      d_dst_pool_strides,
      dst_total_elements);
}

extern "C" void
downsample_mean_u64(CUdeviceptr d_dst,
                    CUdeviceptr d_src_a,
                    CUdeviceptr d_src_b,
                    uint8_t rank,
                    uint8_t downsample_mask,
                    const uint64_t* d_dst_tile_size,
                    const uint64_t* d_src_tile_size,
                    const uint64_t* d_src_extent,
                    const int64_t* d_src_pool_strides,
                    const int64_t* d_dst_pool_strides,
                    uint64_t dst_total_elements,
                    CUstream stream)
{
  if (dst_total_elements == 0)
    return;

  cudaStream_t cuda_stream = (cudaStream_t)stream;
  const int block_size = 256;
  const int grid_size =
    (int)((dst_total_elements + block_size - 1) / block_size);

  downsample_mean_k<uint64_t, uint64_t>
    <<<grid_size, block_size, 0, cuda_stream>>>(
      (uint64_t*)d_dst,
      (const uint64_t*)d_src_a,
      (const uint64_t*)d_src_b,
      rank,
      downsample_mask,
      d_dst_tile_size,
      d_src_tile_size,
      d_src_extent,
      d_src_pool_strides,
      d_dst_pool_strides,
      dst_total_elements);
}
