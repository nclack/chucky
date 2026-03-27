#include "stream/dim_info.h"

#include "util/prelude.h"

#include <string.h>

int
dim_info_init(struct dim_info* info,
              const struct dimension* dims,
              uint8_t rank)
{
  CHECK(Fail, info);
  CHECK(Fail, dims_validate(dims, rank) == 0);

  memset(info, 0, sizeof(*info));

  uint8_t na = dims_n_append(dims, rank);

  info->append = (struct dim_slice){ .beg = dims, .end = dims + na };
  info->inner = (struct dim_slice){ .beg = dims + na, .end = dims + rank };

  // --- Precompute derived values ---

  // LOD mask: only inner dims with downsample
  info->lod_mask = 0;
  for (const struct dimension* d = info->inner.beg; d < info->inner.end; ++d) {
    if (d->downsample)
      info->lod_mask |= (1u << dim_index(info, d));
  }

  // Append downsample: rightmost append dim has downsample
  info->append_downsample = (na > 0) && dims[na - 1].downsample;

  // inner_append_count: product of chunk_count for bounded append dims 1..na-1
  info->inner_append_count = 1;
  for (int d = 1; d < na; ++d)
    info->inner_append_count *= ceildiv(dims[d].size, dims[d].chunk_size);

  return 0;
Fail:
  memset(info, 0, sizeof(*info));
  return 1;
}
