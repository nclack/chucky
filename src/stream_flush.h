#pragma once

// Included via stream_internal.h — all internal types available above.

// Context for flush pipeline operations.
// Stack-allocated by the orchestrator; contains only pointers/copies.
struct flush_context
{
  struct flush_pipeline* flush;
  struct compress_agg_stage* compress_agg;
  struct d2h_deliver_stage* d2h_deliver;
  const struct level_geometry* levels;
  struct batch_state* batch;
  struct pool_state* pools;
  struct lod_state* lod;
  struct stream_metrics* metrics;
  const struct tile_stream_configuration* config;
  const struct stream_layout* layout;
  struct gpu_streams streams;
  struct platform_clock* metadata_update_clock;
};

// Drain pending flush from the previous batch.
struct writer_result
flush_drain_pending(struct flush_context* ctx);

// Kick compress->aggregate->D2H for a batch of n_epochs epochs.
// fc: flush slot index (matches pool index before swap).
int
flush_kick_batch(struct flush_context* ctx, int fc, uint32_t n_epochs);

// Synchronously flush accumulated epochs (partial or full batch).
struct writer_result
flush_accumulated_sync(struct flush_context* ctx);

// Drain partial dim0 accumulators on final flush.
struct writer_result
flush_partial_dim0(struct flush_context* ctx);

// Compute active mask for current epoch (runs LOD if multiscale).
// Updates flush slot active masks. Returns 0 on success.
int
flush_run_epoch_lod(struct flush_context* ctx);

// Accumulate one epoch into the current batch:
// 1. Compute epoch mask (LOD or all-active)
// 2. Record pool event
// 3. Increment batch.accumulated
// 4. If batch full: drain pending, kick new flush, swap pools, reset
// Returns writer_result.
struct writer_result
flush_accumulate_epoch(struct flush_context* ctx);
