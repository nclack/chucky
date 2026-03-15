#pragma once

// Included via stream_internal.h — all internal types available above.

#include "compress.h"
#include "flush_handoff.h"

struct computed_stream_layouts;

// Input to the compress+aggregate stage.
struct compress_agg_input
{
  int fc;                                        // flush slot index (0 or 1)
  uint32_t n_epochs;                             // epochs in this batch
  uint32_t active_levels_mask;                   // union of per-epoch masks
  uint32_t batch_active_masks[MAX_BATCH_EPOCHS]; // per-epoch active level masks
  CUdeviceptr pool_buf;                          // tile pool for this slot
  CUevent epoch_events[MAX_BATCH_EPOCHS];        // per-epoch pool-ready signals
  CUevent lod_done;                              // NULL if no multiscale
  uint32_t epochs_per_batch;                     // K, for LUT path decisions
};

// Compress+aggregate stage. Owns codec, d_compressed buffers, compress/agg
// events, per-level aggregate_layout + batch LUTs.
struct compress_agg_stage
{
  struct codec codec;
  CUdeviceptr d_compressed[2];     // per flush slot
  CUevent t_compress_start[2];
  CUevent t_compress_end[2];
  CUevent t_aggregate_end[2];

  struct level_flush_state levels[LOD_MAX_LEVELS]; // per-level agg+shard+LUTs
};

// Initialize the compress+aggregate stage. Returns 0 on success.
int
compress_agg_init(struct compress_agg_stage* stage,
                  const struct computed_stream_layouts* cl,
                  const struct tile_stream_configuration* config,
                  CUstream compute);

// Destroy the compress+aggregate stage.
void
compress_agg_destroy(struct compress_agg_stage* stage, int nlod);

// Kick compress+aggregate for a batch. Populates handoff for D2H stage.
// Returns 0 on success.
int
compress_agg_kick(struct compress_agg_stage* stage,
                  const struct compress_agg_input* in,
                  const struct level_geometry* levels,
                  const struct batch_state* batch,
                  CUstream compress_stream,
                  struct flush_handoff* out);
