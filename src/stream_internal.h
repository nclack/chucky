#pragma once

#include "stream.h"
#include "stream_flush.h"
#include "stream_ingest.h"
#include "stream_lod.h"

_Static_assert(LOD_MAX_LEVELS <= 32,
               "active_levels_mask is uint32_t; LOD_MAX_LEVELS > 32 overflows");

// Set writer vtable (append/flush).
// Called from tile_stream_gpu_create after zeroing *out.
void
tile_stream_gpu_init_vtable(struct tile_stream_gpu* s);
