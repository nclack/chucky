#include "lod.h"

#include <assert.h>
#include <stdint.h>

#define MAX_LOD_NDIM 16
#define LOD_BLOCK 256

static int
ceil_log2_h(uint64_t v)
{
  if (v <= 1)
    return 0;
  return 64 - __builtin_clzll(v - 1);
}

static uint64_t
max_shape_h(int ndim, const uint64_t* shape)
{
  uint64_t max = 0;
  for (int d = 0; d < ndim; ++d)
    if (shape[d] > max)
      max = shape[d];
  return max;
}

// Number of elements in a tile of `tile_size` starting at `start`
// within a dimension of `dimension_size`, clamped to the boundary.
__device__ static uint64_t
clamped_tile_extent_d(uint64_t dimension_size,
                      uint64_t start,
                      uint64_t tile_size)
{
  if (start >= dimension_size)
    return 0;
  uint64_t remaining = dimension_size - start;
  return (remaining < tile_size) ? remaining : tile_size;
}

__device__ static int
imad(int a, int b, int c)
{
  return a * b + c;
}

// Index of the low child node for dimension d at the given tree level,
// derived from the coordinate without maintaining a prefix array.
//   level < nlod:  top bits of coord with LSB cleared
//   level >= nlod: coord shifted into the child grid
__device__ static uint64_t
tree_child_low(uint64_t coord, int level, int nlod)
{
  if (level < nlod) {
    int bit_index = nlod - 1 - level;
    return (coord >> (bit_index + 1)) << 1;
  }
  return coord << (level - nlod + 1);
}

__device__ static void
morton_next_d(int ndim, int nlod, uint64_t* s_coords, int stride, int tid)
{
  for (int bit = 0; bit < nlod; ++bit) {
    for (int d = 0; d < ndim; ++d) {
      uint64_t mask = 1ull << bit;
      s_coords[imad(d, stride, tid)] ^= mask;
      if (s_coords[imad(d, stride, tid)] & mask)
        return;
    }
  }
  for (int d = 0; d < ndim; ++d)
    s_coords[imad(d, stride, tid)] = 0;
  s_coords[imad(0, stride, tid)] = 1ull << nlod;
}

// Morton rank from coordinates in shared memory.
//
// s_shape:     [ndim]             shape (shared across block)
// s_coords:    [ndim][stride]     per-thread coordinates
// s_products:  [(ndim+1)][stride] per-thread subtree-width products
__device__ static uint64_t
morton_rank_d(int ndim,
              const uint64_t* s_shape,
              int nlod,
              const uint64_t* s_coords,
              uint64_t* s_products,
              int stride,
              int tid,
              int depth)
{
  for (int d = 0; d < ndim; ++d) {
    uint64_t coord = s_coords[imad(d, stride, tid)];
    int coord_bits = coord > 0 ? (64 - __clzll(coord)) : 0;
    if (coord_bits > nlod)
      nlod = coord_bits;
  }

  int total_levels = nlod + depth;
  uint64_t rank = 0;

  for (int level = 0; level < total_levels; ++level) {
    uint64_t tile_size = 1ull << (total_levels - 1 - level);

    int digit = 0;

    // Compute subtree-width products and extract the Morton digit.
    s_products[imad(0, stride, tid)] = 1;
    for (int d = 0; d < ndim; ++d) {
      uint64_t coord = s_coords[imad(d, stride, tid)];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      if (level < nlod)
        digit |= (int)((coord >> (nlod - 1 - level)) & 1) << d;
      uint64_t extent_low =
        clamped_tile_extent_d(s_shape[d], node_low * tile_size, tile_size);
      uint64_t extent_high = clamped_tile_extent_d(
        s_shape[d], (node_low + 1) * tile_size, tile_size);
      s_products[imad(d + 1, stride, tid)] =
        s_products[imad(d, stride, tid)] * (extent_low + extent_high);
    }

    // For each set bit in digit, count elements in the low subtree.
    uint64_t suffix = 1;
    for (int d = ndim - 1; d >= 0; --d) {
      uint64_t coord = s_coords[imad(d, stride, tid)];
      uint64_t node_low = tree_child_low(coord, level, nlod);
      int bit = (digit >> d) & 1;
      uint64_t extent_low =
        clamped_tile_extent_d(s_shape[d], node_low * tile_size, tile_size);
      if (bit == 1)
        rank += suffix * extent_low * s_products[imad(d, stride, tid)];
      uint64_t extent_chosen = clamped_tile_extent_d(
        s_shape[d], (node_low + (uint64_t)bit) * tile_size, tile_size);
      suffix *= extent_chosen;
    }
  }

  return rank;
}

__global__ void
lod_scatter_k(float* __restrict__ dst,
              const float* __restrict__ src,
              int ndim,
              uint64_t n_elements,
              const uint64_t* __restrict__ full_shape,
              const uint64_t* __restrict__ lod_shape,
              int lod_ndim,
              int lod_nlod,
              uint32_t lod_mask,
              uint64_t lod_count)
{
  extern __shared__ uint64_t smem[];
  uint64_t* s_full_shape = smem;
  uint64_t* s_lod_shape = s_full_shape + ndim;
  uint64_t* s_coords = s_lod_shape + lod_ndim;
  uint64_t* s_products = s_coords + lod_ndim * LOD_BLOCK;

  const int tid = threadIdx.x;

  for (int i = tid; i < ndim; i += LOD_BLOCK)
    s_full_shape[i] = full_shape[i];
  for (int i = tid; i < lod_ndim; i += LOD_BLOCK)
    s_lod_shape[i] = lod_shape[i];
  __syncthreads();

  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + tid;
  if (gid >= n_elements)
    return;

  uint64_t remainder = gid;
  uint64_t batch_index = 0, batch_stride = 1;
  int lod_dim = 0;
  for (int d = 0; d < ndim; ++d) {
    uint64_t coord = remainder % s_full_shape[d];
    remainder /= s_full_shape[d];
    if ((lod_mask >> d) & 1) {
      s_coords[imad(lod_dim, LOD_BLOCK, tid)] = coord;
      lod_dim++;
    } else {
      batch_index += coord * batch_stride;
      batch_stride *= s_full_shape[d];
    }
  }

  uint64_t pos = morton_rank_d(
    lod_ndim, s_lod_shape, lod_nlod, s_coords, s_products, LOD_BLOCK, tid, 0);

  dst[batch_index * lod_count + pos] = src[gid];
}

__global__ void
lod_fill_ends_k(uint64_t* __restrict__ ends,
                int ndim,
                const uint64_t* __restrict__ child_shape,
                const uint64_t* __restrict__ parent_shape,
                int parent_nlod,
                int child_nlod,
                uint64_t n_parents)
{
  extern __shared__ uint64_t smem[];
  uint64_t* s_parent_shape = smem;
  uint64_t* s_child_shape = s_parent_shape + ndim;
  uint64_t* s_coords = s_child_shape + ndim;
  uint64_t* s_products = s_coords + ndim * LOD_BLOCK;

  const int tid = threadIdx.x;

  for (int i = tid; i < ndim; i += LOD_BLOCK)
    s_parent_shape[i] = parent_shape[i];
  for (int i = tid; i < ndim; i += LOD_BLOCK)
    s_child_shape[i] = child_shape[i];
  __syncthreads();

  const uint64_t gid = (uint64_t)blockIdx.x * LOD_BLOCK + tid;
  if (gid >= n_parents)
    return;

  {
    uint64_t remainder = gid;
    for (int d = 0; d < ndim; ++d) {
      s_coords[imad(d, LOD_BLOCK, tid)] = remainder % s_parent_shape[d];
      remainder /= s_parent_shape[d];
    }
  }

  uint64_t pos = morton_rank_d(
    ndim, s_parent_shape, parent_nlod, s_coords, s_products, LOD_BLOCK, tid, 0);

  morton_next_d(ndim, parent_nlod, s_coords, LOD_BLOCK, tid);

  uint64_t val = morton_rank_d(
    ndim, s_child_shape, child_nlod, s_coords, s_products, LOD_BLOCK, tid, 1);

  ends[pos] = val;
}

__global__ void
lod_reduce_k(float* __restrict__ values,
             const uint64_t* __restrict__ ends,
             uint64_t src_offset,
             uint64_t dst_offset,
             uint64_t src_lod_count,
             uint64_t dst_lod_count,
             uint64_t batch_count)
{
  const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t total = batch_count * dst_lod_count;
  if (gid >= total)
    return;

  const uint64_t batch = gid / dst_lod_count;
  const uint64_t element = gid % dst_lod_count;

  const uint64_t src_base = src_offset + batch * src_lod_count;
  const uint64_t dst_base = dst_offset + batch * dst_lod_count;

  uint64_t start = (element > 0) ? ends[element - 1] : 0;
  uint64_t end = ends[element];
  uint64_t len = end - start;

  float sum = 0.0f;
  for (uint64_t j = start; j < end; ++j)
    sum += values[src_base + j];

  values[dst_base + element] = sum / (float)len;
}

static constexpr size_t
scatter_smem_bytes(int ndim, int lod_ndim)
{
  return ((size_t)ndim + lod_ndim + ((size_t)2 * lod_ndim + 1) * LOD_BLOCK) *
         sizeof(uint64_t);
}

static constexpr size_t
fill_ends_smem_bytes(int ndim)
{
  return (2 * (size_t)ndim + ((size_t)2 * ndim + 1) * LOD_BLOCK) *
         sizeof(uint64_t);
}

extern "C" void
lod_scatter_f32(CUdeviceptr d_dst,
                CUdeviceptr d_src,
                int ndim,
                uint64_t n_elements,
                CUdeviceptr d_full_shape,
                CUdeviceptr d_lod_shape,
                int lod_ndim,
                const uint64_t* lod_shape_host,
                uint32_t lod_mask,
                uint64_t lod_count,
                CUstream stream)
{
  int lod_nlod = ceil_log2_h(max_shape_h(lod_ndim, lod_shape_host));

  const int grid_size = (int)((n_elements + LOD_BLOCK - 1) / LOD_BLOCK);
  size_t smem = scatter_smem_bytes(ndim, lod_ndim);

  lod_scatter_k<<<grid_size, LOD_BLOCK, smem, stream>>>(
    (float*)d_dst,
    (const float*)d_src,
    ndim,
    n_elements,
    (const uint64_t*)d_full_shape,
    (const uint64_t*)d_lod_shape,
    lod_ndim,
    lod_nlod,
    lod_mask,
    lod_count);
}

extern "C" void
lod_fill_ends_gpu(CUdeviceptr d_ends,
                  int ndim,
                  CUdeviceptr d_child_shape,
                  CUdeviceptr d_parent_shape,
                  const uint64_t* child_shape_host,
                  const uint64_t* parent_shape_host,
                  uint64_t n_parents,
                  CUstream stream)
{
  int parent_nlod = ceil_log2_h(max_shape_h(ndim, parent_shape_host));
  int child_nlod = ceil_log2_h(max_shape_h(ndim, child_shape_host));

  const int grid_size = (int)((n_parents + LOD_BLOCK - 1) / LOD_BLOCK);
  size_t smem = fill_ends_smem_bytes(ndim);

  lod_fill_ends_k<<<grid_size, LOD_BLOCK, smem, stream>>>(
    (uint64_t*)d_ends,
    ndim,
    (const uint64_t*)d_child_shape,
    (const uint64_t*)d_parent_shape,
    parent_nlod,
    child_nlod,
    n_parents);
}

extern "C" void
lod_reduce_f32(CUdeviceptr d_values,
               CUdeviceptr d_ends,
               uint64_t src_offset,
               uint64_t dst_offset,
               uint64_t src_lod_count,
               uint64_t dst_lod_count,
               uint64_t batch_count,
               CUstream stream)
{
  const uint64_t total = batch_count * dst_lod_count;
  const int block_size = 256;
  const int grid_size = (int)((total + block_size - 1) / block_size);

  lod_reduce_k<<<grid_size, block_size, 0, stream>>>((float*)d_values,
                                                     (const uint64_t*)d_ends,
                                                     src_offset,
                                                     dst_offset,
                                                     src_lod_count,
                                                     dst_lod_count,
                                                     batch_count);
}
