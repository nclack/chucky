#include "compress.h"
#include "downsample.h"
#include "metric.cuda.h"
#include "platform.h"
#include "prelude.cuda.h"
#include "prelude.h"
#include "stream.h"

#include <stdlib.h>
#include <string.h>

static struct writer_result
writer_ok(void)
{
  return (struct writer_result){ 0 };
}

static struct writer_result
writer_error(void)
{
  return (struct writer_result){ .error = 1 };
}

static struct writer_result
writer_error_at(const void* beg, const void* end)
{
  return (struct writer_result){ .error = 1, .rest = { beg, end } };
}

static void
buffer_free(struct buffer* buffer)
{
  if (!buffer || !buffer->data) {
    return;
  }

  if (buffer->ready) {
    CUresult res = cuEventDestroy(buffer->ready);
    if (res != CUDA_SUCCESS) {
      const char* err_str = NULL;
      cuGetErrorString(res, &err_str);
      log_warn("Failed to destroy event: %s", err_str ? err_str : "unknown");
    }
    buffer->ready = NULL;
  }

  switch (buffer->domain) {
    case host:
      cuMemFreeHost(buffer->data);
      break;
    case device:
      cuMemFree((CUdeviceptr)buffer->data);
      break;
    default:
      log_error("Invalid domain during buffer_free: %d", buffer->domain);
      return;
  }

  buffer->data = NULL;
}

static struct buffer
buffer_new(size_t capacity, enum domain domain, unsigned int host_flags)
{
  struct buffer buf = { 0 };
  buf.domain = domain;

  switch (domain) {
    case host:
      CU(Fail, cuMemHostAlloc(&buf.data, capacity, host_flags));
      break;
    case device:
      CU(Fail, cuMemAlloc((CUdeviceptr*)&buf.data, capacity));
      break;
    default:
      log_error("Invalid domain: %d", domain);
      goto Fail;
  }
  CU(Fail, cuEventCreate(&buf.ready, CU_EVENT_DEFAULT));
  return buf;

Fail:
  buffer_free(&buf);
  return (struct buffer){ 0 };
}

// FIXME: don't need this function
static void
device_free(void* ptr)
{
  CUWARN(cuMemFree((CUdeviceptr)ptr));
}

// --- Helpers ---

// Return pointer to the current L0 tile pool (A if pool_current==0, B if 1).
static inline void*
current_pool(struct transpose_stream* s)
{
  return s->pool_current ? s->pool_B.data : s->pool_A.data;
}

// Dispatch downsample by bpe. Returns 0 on success, -1 on error.
static int
dispatch_downsample(CUdeviceptr dst,
                    CUdeviceptr src_a,
                    CUdeviceptr src_b,
                    size_t bpe,
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
  switch (bpe) {
    case 1:
      downsample_mean_u8(dst, src_a, src_b, rank, downsample_mask,
                         d_dst_tile_size, d_src_tile_size, d_src_extent,
                         d_src_pool_strides, d_dst_pool_strides,
                         dst_total_elements, stream);
      break;
    case 2:
      downsample_mean_u16(dst, src_a, src_b, rank, downsample_mask,
                          d_dst_tile_size, d_src_tile_size, d_src_extent,
                          d_src_pool_strides, d_dst_pool_strides,
                          dst_total_elements, stream);
      break;
    case 4:
      downsample_mean_u32(dst, src_a, src_b, rank, downsample_mask,
                          d_dst_tile_size, d_src_tile_size, d_src_extent,
                          d_src_pool_strides, d_dst_pool_strides,
                          dst_total_elements, stream);
      break;
    case 8:
      downsample_mean_u64(dst, src_a, src_b, rank, downsample_mask,
                          d_dst_tile_size, d_src_tile_size, d_src_extent,
                          d_src_pool_strides, d_dst_pool_strides,
                          dst_total_elements, stream);
      break;
    default:
      log_error("dispatch_downsample: unsupported bytes_per_element=%zu", bpe);
      return -1;
  }
  return 0;
}

// Dispatch staged data: H2D transfer + scatter kernel
// Returns 0 on success, 1 on error.
static int
dispatch_scatter(struct transpose_stream* s)
{
  const size_t bpe = s->config.bytes_per_element;
  if (bpe == 0)
    return 0;

  const uint64_t elements = s->stage.fill / bpe;
  if (elements == 0)
    return 0;

  const int idx = s->stage.current;
  struct staging_slot* ss = &s->stage.slot[idx];

  void* pool = current_pool(s);

  // H2D — wait for prior scatter to finish reading d_in before overwriting
  CU(Error, cuStreamWaitEvent(s->h2d, ss->t_scatter_end, 0));
  CU(Error, cuEventRecord(ss->t_h2d_start, s->h2d));
  CU(Error,
     cuMemcpyHtoDAsync(
       (CUdeviceptr)ss->d_in.data, ss->h_in.data, s->stage.fill, s->h2d));
  CU(Error, cuEventRecord(ss->h_in.ready, s->h2d));

  // Kernel waits for H2D, then scatters into tile pool
  CU(Error, cuStreamWaitEvent(s->compute, ss->h_in.ready, 0));
  CU(Error, cuEventRecord(ss->t_scatter_start, s->compute));
  switch (bpe) {
    case 1:
      transpose_u8_v0((CUdeviceptr)pool,
                      (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                      (CUdeviceptr)ss->d_in.data,
                      (CUdeviceptr)ss->d_in.data + s->stage.fill,
                      s->cursor,
                      s->layout.lifted_rank,
                      s->layout.d_lifted_shape,
                      s->layout.d_lifted_strides,
                      s->compute);
      break;
    case 2:
      transpose_u16_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    case 4:
      transpose_u32_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    case 8:
      transpose_u64_v0((CUdeviceptr)pool,
                       (CUdeviceptr)pool + s->layout.tile_pool_bytes,
                       (CUdeviceptr)ss->d_in.data,
                       (CUdeviceptr)ss->d_in.data + s->stage.fill,
                       s->cursor,
                       s->layout.lifted_rank,
                       s->layout.d_lifted_shape,
                       s->layout.d_lifted_strides,
                       s->compute);
      break;
    default:
      log_error("dispatch_scatter: unsupported bytes_per_element=%zu", bpe);
      goto Error;
  }
  CU(Error, cuEventRecord(ss->t_scatter_end, s->compute));
  // Record pool ready on whichever pool we're writing to
  if (s->pool_current == 0) {
    CU(Error, cuEventRecord(s->pool_A.ready, s->compute));
  } else {
    CU(Error, cuEventRecord(s->pool_B.ready, s->compute));
  }

  s->cursor += elements;
  s->stage.current ^= 1;
  return 0;

Error:
  return 1;
}

struct writer_result
writer_append(struct writer* w, struct slice data)
{
  return w->append(w, data);
}

struct writer_result
writer_flush(struct writer* w)
{
  return w->flush(w);
}

struct writer_result
writer_append_wait(struct writer* w, struct slice data)
{
  int stalls = 0;
  const int max_stalls = 10;

  while (data.beg < data.end) {
    struct writer_result r = writer_append(w, data);
    if (r.error)
      return r;

    if (r.rest.beg == data.beg) {
      if (++stalls >= max_stalls) {
        log_error("writer_append_wait: no progress after %d retries", stalls);
        return writer_error_at(data.beg, data.end);
      }
      log_warn(
        "writer_append_wait: stall %d/%d, backing off", stalls, max_stalls);
      platform_sleep_ns(1000000LL << (stalls < 6 ? stalls : 6)); // 1ms..64ms
    } else {
      stalls = 0;
    }

    data = r.rest;
  }

  return writer_ok();
}

// Software CRC32C (Castagnoli) computed at runtime via a generated table.
static uint32_t crc32c_table[256];
static int crc32c_table_ready;

static void
crc32c_init_table(void)
{
  if (crc32c_table_ready)
    return;
  for (int i = 0; i < 256; ++i) {
    uint32_t crc = (uint32_t)i;
    for (int j = 0; j < 8; ++j)
      crc = (crc >> 1) ^ (0x82F63B78 & (0u - (crc & 1)));
    crc32c_table[i] = crc;
  }
  crc32c_table_ready = 1;
}

static uint32_t
crc32c(const void* data, size_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i)
    crc = crc32c_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFF;
}

// --- Unified shard delivery ---

// Emit completed shards (write index block + finalize).
// Works for both L0 and LOD levels.
static int
emit_shards(struct shard_state* ss)
{
  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    struct active_shard* sh = &ss->shards[si];
    if (!sh->writer)
      continue;

    size_t index_data_bytes = ss->tiles_per_shard_total * 2 * sizeof(uint64_t);
    size_t index_total_bytes = index_data_bytes + 4;
    uint8_t* index_buf = (uint8_t*)malloc(index_total_bytes);
    CHECK(Error, index_buf);

    memcpy(index_buf, sh->index, index_data_bytes);

    uint32_t crc_val = crc32c(index_buf, index_data_bytes);
    memcpy(index_buf + index_data_bytes, &crc_val, 4);

    int wrc = sh->writer->write(
      sh->writer, sh->data_cursor, index_buf, index_buf + index_total_bytes);
    free(index_buf);
    CHECK(Error, wrc == 0);

    CHECK(Error, sh->writer->finalize(sh->writer) == 0);

    sh->writer = NULL;
    sh->data_cursor = 0;
    memset(sh->index, 0xFF, ss->tiles_per_shard_total * 2 * sizeof(uint64_t));
  }

  ss->epoch_in_shard = 0;
  ss->shard_epoch++;
  return 0;

Error:
  return 1;
}

// Deliver compressed tile data from an aggregate slot to shards.
// level: 0 for L0, 1+ for LOD levels.
static int
deliver_to_shards(struct transpose_stream* s,
                  uint8_t level,
                  struct shard_state* ss,
                  struct aggregate_slot* agg_slot)
{
  const uint64_t tps_inner = ss->tiles_per_shard_inner;
  const uint64_t epoch_in_shard = ss->epoch_in_shard;

  for (uint64_t si = 0; si < ss->shard_inner_count; ++si) {
    uint64_t j_start = si * tps_inner;
    uint64_t j_end = j_start + tps_inner;

    struct active_shard* sh = &ss->shards[si];

    if (!sh->writer) {
      uint64_t flat = ss->shard_epoch * ss->shard_inner_count + si;
      sh->writer = s->config.shard_sink->open(s->config.shard_sink, level, flat);
      CHECK(Error, sh->writer);
    }

    size_t shard_bytes = agg_slot->h_offsets[j_end] - agg_slot->h_offsets[j_start];
    if (shard_bytes > 0) {
      const void* src =
        (const char*)agg_slot->h_aggregated + agg_slot->h_offsets[j_start];
      CHECK(Error,
            sh->writer->write(sh->writer,
                              sh->data_cursor,
                              src,
                              (const char*)src + shard_bytes) == 0);
    }

    for (uint64_t j = j_start; j < j_end; ++j) {
      size_t tile_size = agg_slot->h_offsets[j + 1] - agg_slot->h_offsets[j];
      if (tile_size > 0) {
        uint64_t within_inner = j - j_start;
        uint64_t slot = epoch_in_shard * tps_inner + within_inner;
        size_t tile_off =
          sh->data_cursor + (agg_slot->h_offsets[j] - agg_slot->h_offsets[j_start]);
        sh->index[2 * slot] = tile_off;
        sh->index[2 * slot + 1] = tile_size;
      }
    }
    sh->data_cursor += shard_bytes;
  }

  ss->epoch_in_shard++;

  if (ss->epoch_in_shard >= ss->tiles_per_shard_0)
    return emit_shards(ss);

  return 0;

Error:
  return 1;
}

// Wait for D2H on the given flush slot, record timing, deliver to sinks.
static struct writer_result
wait_and_deliver(struct transpose_stream* s, int fc)
{
  struct flush_slot* fs = &s->flush[fc];

  if (s->config.compress && s->config.shard_sink) {
    CU(Error, cuEventSynchronize(fs->ready));

    accumulate_metric_cu(&s->metrics.downsample, s->t_downsample_start,
                         s->t_downsample_end);
    accumulate_metric_cu(&s->metrics.compress, fs->t_compress_start,
                         s->flush[fc].d_compressed.ready);
    accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start, fs->ready);

    // Deliver each firing level
    for (int i = 0; i < fs->num_firing; ++i) {
      uint8_t level = fs->firing_levels[i];
      if (level == 0) {
        // L0
        struct aggregate_slot* agg = &s->agg[fc];
        if (deliver_to_shards(s, 0, &s->shard, agg))
          goto Error;
      } else {
        // LOD level
        struct level_state* lev = &s->levels[level - 1];
        struct aggregate_slot* agg = &lev->agg[fc];
        if (deliver_to_shards(s, level, &lev->shard, agg))
          goto Error;
      }
    }
  } else {
    // Uncompressed path: wait for host pool ready
    if (fc == 0) {
      CU(Error, cuEventSynchronize(s->pool_A_host.ready));
      accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start,
                           s->pool_A_host.ready);
      if (s->config.sink) {
        struct slice tiles = {
          .beg = s->pool_A_host.data,
          .end = (char*)s->pool_A_host.data + s->layout.tile_pool_bytes,
        };
        return writer_append_wait(s->config.sink, tiles);
      }
    } else {
      CU(Error, cuEventSynchronize(s->pool_B_host.ready));
      accumulate_metric_cu(&s->metrics.d2h, fs->t_d2h_start,
                           s->pool_B_host.ready);
      if (s->config.sink) {
        struct slice tiles = {
          .beg = s->pool_B_host.data,
          .end = (char*)s->pool_B_host.data + s->layout.tile_pool_bytes,
        };
        return writer_append_wait(s->config.sink, tiles);
      }
    }
  }

  return writer_ok();

Error:
  return writer_error();
}

// Drain pending flush from the previous epoch.
static struct writer_result
drain_pending_flush(struct transpose_stream* s)
{
  if (!s->flush_pending)
    return writer_ok();

  s->flush_pending = 0;
  return wait_and_deliver(s, s->flush_current);
}

// --- LOD cascade ---

// Cascade downsampling through LOD levels starting from level 0.
// pa, pb are device pointers to the parent's A and B pools (L0 data).
// Returns number of firing LOD levels (0 if none fired), or -1 on error.
//
// Uses two scratch buffers alternating like a stack of depth 2:
//   level 1: save → scratch[0], downsample from (pa, pb)
//   level 2: save → scratch[1], downsample using scratch[0]  → scratch[0] freed
//   level 3: save → scratch[0], downsample using scratch[1]  → scratch[1] freed
//   ...
static int
lod_cascade(struct transpose_stream* s, CUdeviceptr pa, CUdeviceptr pb)
{
  CUdeviceptr scratch[2] = {
    (CUdeviceptr)s->scratch[0].data,
    (CUdeviceptr)s->scratch[1].data,
  };
  int si = 0; // alternates 0, 1, 0, 1, ...
  int num_fired = 0;

  for (int lv = 0; lv < s->num_levels; ++lv) {
    struct level_state* lev = &s->levels[lv];
    lev->epoch_count++;

    if (lev->needs_two_epochs && (lev->epoch_count & 1))
      break; // wait for pair

    if (!lev->needs_two_epochs)
      pa = pb;

    CUdeviceptr dst = (CUdeviceptr)s->pool_B.data + s->level_offset[lv + 1];

    // Save old data so the next level can pair it with our new output.
    int has_deeper = (lv + 1 < s->num_levels);
    if (lev->needs_two_epochs && has_deeper) {
      size_t dst_bytes = lev->layout.slot_count * lev->layout.tile_stride *
                         s->config.bytes_per_element;
      CU(Error, cuMemcpyDtoDAsync(scratch[si], dst, dst_bytes, s->compute));
    }

    if (dispatch_downsample(
          dst, pa, pb,
          s->config.bytes_per_element, s->config.rank,
          lev->downsample_mask,
          lev->d_dst_tile_size, lev->d_src_tile_size, lev->d_src_extent,
          lev->d_src_pool_strides, lev->d_dst_pool_strides,
          lev->layout.slot_count * lev->layout.tile_elements,
          s->compute) < 0)
      return -1;

    // Next level's parents: old = scratch[si], new = dst
    pa = scratch[si];
    pb = dst;
    si ^= 1;
    num_fired++;
  }

  return num_fired;

Error:
  return -1;
}

// --- Epoch flush pipeline ---

// Helper: count total tiles for firing LOD levels (excludes L0).
static uint64_t
lod_firing_tiles(struct transpose_stream* s,
                 const uint8_t* firing,
                 int num_firing)
{
  uint64_t total = 0;
  for (int i = 0; i < num_firing; ++i) {
    if (firing[i] == 0)
      continue; // L0 counted separately
    total += s->levels[firing[i] - 1].layout.slot_count;
  }
  return total;
}

// Kick compress + aggregate + D2H for the current epoch.
// fc: flush slot index (0 or 1, matches pool_current before swap).
// num_chunks: number of tiles to compress.
// firing_levels[]: array of level indices that fired.
// num_firing: length of firing_levels.
static int
kick_epoch(struct transpose_stream* s,
           int fc,
           uint64_t num_chunks,
           const uint8_t* firing_levels,
           int num_firing)
{
  struct flush_slot* fs = &s->flush[fc];
  fs->num_firing = num_firing;
  memcpy(fs->firing_levels, firing_levels, num_firing);

  if (s->config.compress && s->config.shard_sink) {
    // Wait for scatter to finish
    if (fc == 0) {
      CU(Error, cuStreamWaitEvent(s->compress, s->pool_A.ready, 0));
    } else {
      CU(Error, cuStreamWaitEvent(s->compress, s->pool_B.ready, 0));
    }

    CU(Error, cuEventRecord(fs->t_compress_start, s->compress));
    CHECK(Error,
          compress_batch_async(
            (const void* const*)fs->d_uncomp_ptrs,
            s->d_uncomp_sizes,
            s->layout.tile_stride * s->config.bytes_per_element,
            num_chunks,
            s->d_comp_temp,
            s->comp_temp_bytes,
            fs->d_comp_ptrs,
            s->d_comp_sizes,
            s->compress) == 0);

    // nvcomp may use internal CUDA operations not captured by stream events.
    // Synchronize before reading compressed output.
    CU(Error, cuStreamSynchronize(s->compress));
    CU(Error, cuEventRecord(fs->d_compressed.ready, s->compress));

    // Per-level aggregate + D2H
    uint64_t tile_offset = 0;
    CU(Error, cuStreamWaitEvent(s->d2h, fs->d_compressed.ready, 0));
    CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));

    for (int i = 0; i < num_firing; ++i) {
      uint8_t level = firing_levels[i];
      uint64_t M;
      struct aggregate_layout* al;
      struct aggregate_slot* agg;
      size_t level_comp_pool_bytes;

      if (level == 0) {
        M = s->layout.slot_count;
        al = &s->agg_layout;
        agg = &s->agg[fc];
        level_comp_pool_bytes = s->comp_pool_bytes;
      } else {
        struct level_state* lev = &s->levels[level - 1];
        M = lev->layout.slot_count;
        al = &lev->agg_layout;
        agg = &lev->agg[fc];
        level_comp_pool_bytes = M * s->max_comp_chunk_bytes;
      }

      // d_compressed base for this level
      void* d_comp_base =
        (char*)fs->d_compressed.data + tile_offset * s->max_comp_chunk_bytes;
      size_t* d_sizes_base = s->d_comp_sizes + tile_offset;

      CHECK(Error,
            aggregate_by_shard_async(al, d_comp_base, d_sizes_base, agg,
                                     s->compress) == 0);
      CU(Error, cuEventRecord(agg->ready, s->compress));

      // D2H: aggregated data + offsets
      CU(Error, cuStreamWaitEvent(s->d2h, agg->ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(agg->h_aggregated,
                           (CUdeviceptr)agg->d_aggregated,
                           level_comp_pool_bytes,
                           s->d2h));
      CU(Error,
         cuMemcpyDtoHAsync(agg->h_offsets,
                           (CUdeviceptr)agg->d_offsets,
                           (al->covering_count + 1) * sizeof(size_t),
                           s->d2h));
      CU(Error, cuEventRecord(agg->ready, s->d2h));

      tile_offset += M;
    }

    CU(Error, cuEventRecord(fs->ready, s->d2h));
  } else {
    // Uncompressed path: D2H the L0 pool
    CU(Error, cuEventRecord(fs->t_d2h_start, s->d2h));
    if (fc == 0) {
      CU(Error, cuStreamWaitEvent(s->d2h, s->pool_A.ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(s->pool_A_host.data,
                           (CUdeviceptr)s->pool_A.data,
                           s->layout.tile_pool_bytes,
                           s->d2h));
      CU(Error, cuEventRecord(s->pool_A_host.ready, s->d2h));
    } else {
      CU(Error, cuStreamWaitEvent(s->d2h, s->pool_B.ready, 0));
      CU(Error,
         cuMemcpyDtoHAsync(s->pool_B_host.data,
                           (CUdeviceptr)s->pool_B.data,
                           s->layout.tile_pool_bytes,
                           s->d2h));
      CU(Error, cuEventRecord(s->pool_B_host.ready, s->d2h));
    }
    CU(Error, cuEventRecord(fs->ready, s->d2h));
  }

  return 0;

Error:
  return 1;
}

// Flush the current epoch's tile pool: cascade LODs, compress, D2H, swap.
static struct writer_result
flush_epoch(struct transpose_stream* s)
{
  const int fc = s->pool_current; // 0=A, 1=B

  // Deliver the previous epoch if its D2H is still in flight
  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  // Build list of firing levels
  uint8_t firing[MAX_LOD_LEVELS + 1];
  int num_firing = 0;
  firing[num_firing++] = 0; // L0 always fires

  // Cascade LODs
  if (s->num_levels > 0) {
    CUdeviceptr pa, pb;
    if (s->levels[0].needs_two_epochs) {
      pa = (CUdeviceptr)s->pool_A.data;
      pb = (CUdeviceptr)s->pool_B.data;
    } else {
      pa = (CUdeviceptr)current_pool(s);
      pb = pa;
    }

    CU(Error, cuEventRecord(s->t_downsample_start, s->compute));
    int num_lod_fired = lod_cascade(s, pa, pb);
    if (num_lod_fired < 0)
      return writer_error();
    CU(Error, cuEventRecord(s->t_downsample_end, s->compute));

    // Re-record pool event after cascade so compress waits for LOD writes too.
    // The pool event was originally recorded after scatter, but cascade also
    // writes to pool_B (LOD regions) on s->compute.
    if (num_lod_fired > 0) {
      struct buffer* pool = (fc == 0) ? &s->pool_A : &s->pool_B;
      CU(Error, cuEventRecord(pool->ready, s->compute));
    }

    // Add firing LOD levels
    for (int lv = 0; lv < num_lod_fired; ++lv)
      firing[num_firing++] = (uint8_t)(lv + 1);
  }

  if (kick_epoch(s, fc, s->layout.slot_count + lod_firing_tiles(s, firing, num_firing),
                 firing, num_firing))
    return writer_error();

  // Swap pool and zero next L0 region
  s->pool_current ^= 1;
  void* next = current_pool(s);
  CU(Error,
     cuMemsetD8Async((CUdeviceptr)next, 0, s->layout.tile_pool_bytes,
                     s->compute));

  s->flush_pending = 1;
  s->flush_current = fc;
  return writer_ok();

Error:
  return writer_error();
}

// Synchronously flush the current tile pool (used for the final partial epoch).
static struct writer_result
flush_epoch_sync(struct transpose_stream* s)
{
  const int fc = s->pool_current;

  // Only L0 fires for sync flush (no LOD cascade for partial epochs)
  uint8_t firing[1] = { 0 };
  if (kick_epoch(s, fc, s->layout.slot_count, firing, 1))
    return writer_error();
  return wait_and_deliver(s, fc);
}

struct stream_metrics
transpose_stream_get_metrics(const struct transpose_stream* s)
{
  return s->metrics;
}

// --- Destroy ---

void
transpose_stream_destroy(struct transpose_stream* stream)
{
  if (!stream)
    return;

  CUWARN(cuStreamDestroy(stream->h2d));
  CUWARN(cuStreamDestroy(stream->compute));
  CUWARN(cuStreamDestroy(stream->compress));
  CUWARN(cuStreamDestroy(stream->d2h));

  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_shape));
  CUWARN(cuMemFree((CUdeviceptr)stream->layout.d_lifted_strides));

  for (int i = 0; i < 2; ++i) {
    struct staging_slot* ss = &stream->stage.slot[i];
    CUWARN(cuEventDestroy(ss->t_h2d_start));
    CUWARN(cuEventDestroy(ss->t_scatter_start));
    CUWARN(cuEventDestroy(ss->t_scatter_end));
    buffer_free(&ss->h_in);
    buffer_free(&ss->d_in);
  }

  // Tile pools
  buffer_free(&stream->pool_A);
  buffer_free(&stream->pool_B);
  buffer_free(&stream->pool_A_host);
  buffer_free(&stream->pool_B_host);
  buffer_free(&stream->scratch[0]);
  buffer_free(&stream->scratch[1]);

  // Flush slots
  for (int i = 0; i < 2; ++i) {
    struct flush_slot* fs = &stream->flush[i];
    buffer_free(&fs->d_compressed);
    device_free(fs->d_uncomp_ptrs);
    device_free(fs->d_comp_ptrs);
    CUWARN(cuEventDestroy(fs->t_compress_start));
    CUWARN(cuEventDestroy(fs->t_d2h_start));
    CUWARN(cuEventDestroy(fs->ready));
  }

  // Downsample timing events
  CUWARN(cuEventDestroy(stream->t_downsample_start));
  CUWARN(cuEventDestroy(stream->t_downsample_end));

  // Shared compress state
  device_free(stream->d_comp_sizes);
  device_free(stream->d_uncomp_sizes);
  device_free(stream->d_comp_temp);

  // L0 aggregate + shard
  aggregate_layout_destroy(&stream->agg_layout);
  for (int i = 0; i < 2; ++i)
    aggregate_slot_destroy(&stream->agg[i]);

  if (stream->shard.shards) {
    for (uint64_t i = 0; i < stream->shard.shard_inner_count; ++i)
      free(stream->shard.shards[i].index);
    free(stream->shard.shards);
  }

  // LOD levels
  for (int lv = 0; lv < stream->num_levels; ++lv) {
    struct level_state* lev = &stream->levels[lv];

    device_free(lev->layout.d_lifted_shape);
    device_free(lev->layout.d_lifted_strides);

    aggregate_layout_destroy(&lev->agg_layout);
    for (int i = 0; i < 2; ++i)
      aggregate_slot_destroy(&lev->agg[i]);

    if (lev->shard.shards) {
      for (uint64_t i = 0; i < lev->shard.shard_inner_count; ++i)
        free(lev->shard.shards[i].index);
      free(lev->shard.shards);
    }

    device_free(lev->d_dst_tile_size);
    device_free(lev->d_src_tile_size);
    device_free(lev->d_src_extent);
    device_free(lev->d_src_pool_strides);
    device_free(lev->d_dst_pool_strides);
  }

  *stream = (struct transpose_stream){ 0 };
}

// --- Create ---

// Forward declarations for vtable
static struct writer_result
transpose_stream_append(struct writer* self, struct slice input);
static struct writer_result
transpose_stream_flush(struct writer* self);

int
transpose_stream_create(const struct transpose_stream_configuration* config,
                        struct transpose_stream* out)
{
  CHECK(Fail, config);
  CHECK(Fail, out);

  *out = (struct transpose_stream){
    .writer = { .append = transpose_stream_append,
                .flush = transpose_stream_flush },
    .config = *config,
  };

  CHECK(Fail, config->bytes_per_element > 0);
  CHECK(Fail, config->buffer_capacity_bytes > 0);
  out->config.buffer_capacity_bytes =
    (config->buffer_capacity_bytes + 4095) & ~(size_t)4095;
  CHECK(Fail, config->rank > 0);
  CHECK(Fail, config->rank <= MAX_RANK / 2);
  CHECK(Fail, config->dimensions);

  CU(Fail, cuStreamCreate(&out->h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->compress, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&out->d2h, CU_STREAM_NON_BLOCKING));

  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_h2d_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_scatter_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->stage.slot[i].t_scatter_end, CU_EVENT_DEFAULT));
  }

  // Flush slot events
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventCreate(&out->flush[i].t_compress_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->flush[i].t_d2h_start, CU_EVENT_DEFAULT));
    CU(Fail, cuEventCreate(&out->flush[i].ready, CU_EVENT_DEFAULT));
  }

  // Downsample timing events
  CU(Fail, cuEventCreate(&out->t_downsample_start, CU_EVENT_DEFAULT));
  CU(Fail, cuEventCreate(&out->t_downsample_end, CU_EVENT_DEFAULT));

  const uint8_t rank = config->rank;
  const size_t bpe = config->bytes_per_element;
  const struct dimension* dims = config->dimensions;

  // Lifted shape (row-major, slowest first): (t_{D-1}, n_{D-1}, ..., t_0, n_0)
  out->layout.lifted_rank = 2 * rank;
  out->layout.tile_elements = 1;

  uint64_t tile_count[MAX_RANK];
  for (int i = 0; i < rank; ++i) {
    tile_count[i] = ceildiv(dims[i].size, dims[i].tile_size);
    out->layout.lifted_shape[2 * i] = tile_count[i];
    out->layout.lifted_shape[2 * i + 1] = dims[i].tile_size;
    out->layout.tile_elements *= dims[i].tile_size;
  }

  // Compute tile_stride
  {
    size_t alignment = config->compress ? compress_get_input_alignment() : 1;
    size_t tile_bytes = out->layout.tile_elements * bpe;
    size_t padded_bytes = align_up(tile_bytes, alignment);
    out->layout.tile_stride = padded_bytes / bpe;
  }

  // Build lifted strides
  {
    int64_t n_stride = 1;
    int64_t t_stride = (int64_t)out->layout.tile_stride;

    for (int i = rank - 1; i >= 0; --i) {
      out->layout.lifted_strides[2 * i + 1] = n_stride;
      n_stride *= (int64_t)dims[i].tile_size;

      out->layout.lifted_strides[2 * i] = t_stride;
      t_stride *= (int64_t)tile_count[i];
    }
  }

  out->layout.slot_count =
    out->layout.lifted_strides[0] / out->layout.tile_stride;
  out->layout.epoch_elements =
    out->layout.slot_count * out->layout.tile_elements;
  out->layout.lifted_strides[0] = 0; // collapse epoch dim
  out->layout.tile_pool_bytes =
    out->layout.slot_count * out->layout.tile_stride * bpe;

  // Allocate device copies of lifted shape and strides
  {
    const size_t shape_bytes = out->layout.lifted_rank * sizeof(uint64_t);
    const size_t strides_bytes = out->layout.lifted_rank * sizeof(int64_t);
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->layout.d_lifted_shape, shape_bytes));
    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->layout.d_lifted_strides, strides_bytes));
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)out->layout.d_lifted_shape,
                           out->layout.lifted_shape, shape_bytes));
    CU(Fail, cuMemcpyHtoD((CUdeviceptr)out->layout.d_lifted_strides,
                           out->layout.lifted_strides, strides_bytes));
  }

  // Allocate input staging buffers
  for (int i = 0; i < 2; ++i) {
    CHECK(Fail,
          (out->stage.slot[i].h_in =
             buffer_new(config->buffer_capacity_bytes, host, 0))
            .data);
    CHECK(Fail,
          (out->stage.slot[i].d_in =
             buffer_new(config->buffer_capacity_bytes, device, 0))
            .data);
  }

  // --- LOD level computation (determines M_total, level_offset, etc.) ---
  const uint64_t M0 = out->layout.slot_count;
  out->M_total = M0;
  out->level_offset[0] = 0; // L0 starts at offset 0 in B

  int num_levels = 0;
  int needs_two_epochs = 0;

  uint8_t ds_mask = 0;
  if (config->enable_lod && config->compress && config->shard_sink) {
    for (int d = 0; d < rank; ++d) {
      if (dims[d].downsample)
        ds_mask |= (1 << d);
    }
  }

  if (ds_mask) {
    needs_two_epochs = (ds_mask & 1) ? 1 : 0;

    // Count LOD levels and compute their slot_counts
    struct dimension test_dims[MAX_RANK / 2];
    for (int d = 0; d < rank; ++d)
      test_dims[d] = dims[d];

    for (int lv = 0; lv < MAX_LOD_LEVELS; ++lv) {
      struct dimension next[MAX_RANK / 2];
      int all_downsampleable = 1;
      for (int d = 0; d < rank; ++d) {
        next[d] = test_dims[d];
        if (test_dims[d].downsample)
          next[d].size = ceildiv(test_dims[d].size, 2);
        if (next[d].downsample && next[d].size <= next[d].tile_size)
          all_downsampleable = 0;
      }
      num_levels++;

      if (!all_downsampleable)
        break;

      for (int d = 0; d < rank; ++d)
        test_dims[d] = next[d];
    }
  }

  out->num_levels = num_levels;

  // Build per-level state and compute M_total
  if (num_levels > 0) {
    const struct dimension* parent_dims = dims;
    const struct stream_layout* parent_layout = &out->layout;

    for (int lv = 0; lv < num_levels; ++lv) {
      struct level_state* lev = &out->levels[lv];
      lev->downsample_mask = ds_mask;
      lev->needs_two_epochs = needs_two_epochs;

      // Compute dimensions at this level
      for (int d = 0; d < rank; ++d) {
        lev->dimensions[d] = parent_dims[d];
        if (parent_dims[d].downsample)
          lev->dimensions[d].size = ceildiv(parent_dims[d].size, 2);
      }

      // Build tile_count, tiles_per_shard
      uint64_t lod_tile_count[MAX_RANK];
      uint64_t lod_tiles_per_shard[MAX_RANK];
      for (int d = 0; d < rank; ++d) {
        lod_tile_count[d] =
          ceildiv(lev->dimensions[d].size, lev->dimensions[d].tile_size);
        uint64_t tps = lev->dimensions[d].tiles_per_shard;
        lod_tiles_per_shard[d] = (tps == 0) ? lod_tile_count[d] : tps;
      }

      // Build stream_layout
      lev->layout.lifted_rank = 2 * rank;
      lev->layout.tile_elements = 1;
      for (int d = 0; d < rank; ++d) {
        lev->layout.lifted_shape[2 * d] = lod_tile_count[d];
        lev->layout.lifted_shape[2 * d + 1] = lev->dimensions[d].tile_size;
        lev->layout.tile_elements *= lev->dimensions[d].tile_size;
      }

      {
        size_t alignment = config->compress ? compress_get_input_alignment() : 1;
        size_t tile_bytes = lev->layout.tile_elements * bpe;
        size_t padded_bytes = align_up(tile_bytes, alignment);
        lev->layout.tile_stride = padded_bytes / bpe;
      }

      {
        int64_t n_stride = 1;
        int64_t t_stride = (int64_t)lev->layout.tile_stride;
        for (int d = rank - 1; d >= 0; --d) {
          lev->layout.lifted_strides[2 * d + 1] = n_stride;
          n_stride *= (int64_t)lev->dimensions[d].tile_size;
          lev->layout.lifted_strides[2 * d] = t_stride;
          t_stride *= (int64_t)lod_tile_count[d];
        }
      }

      lev->layout.slot_count =
        lev->layout.lifted_strides[0] / lev->layout.tile_stride;
      lev->layout.epoch_elements =
        lev->layout.slot_count * lev->layout.tile_elements;
      lev->layout.lifted_strides[0] = 0;
      lev->layout.tile_pool_bytes =
        lev->layout.slot_count * lev->layout.tile_stride * bpe;

      // Device copies of lifted shape/strides
      {
        const size_t shape_bytes = lev->layout.lifted_rank * sizeof(uint64_t);
        const size_t strides_bytes = lev->layout.lifted_rank * sizeof(int64_t);
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->layout.d_lifted_shape, shape_bytes));
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->layout.d_lifted_strides, strides_bytes));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->layout.d_lifted_shape,
                               lev->layout.lifted_shape, shape_bytes));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->layout.d_lifted_strides,
                               lev->layout.lifted_strides, strides_bytes));
      }

      // Level offset in B
      out->level_offset[lv + 1] =
        out->level_offset[lv] +
        (lv == 0 ? M0 : out->levels[lv - 1].layout.slot_count) *
          out->layout.tile_stride * bpe;
      out->M_total += lev->layout.slot_count;

      // Downsample kernel params
      {
        const size_t sz = rank * sizeof(uint64_t);
        const size_t ssz = rank * sizeof(int64_t);

        uint64_t h_dst_ts[MAX_RANK / 2];
        for (int d = 0; d < rank; ++d)
          h_dst_ts[d] = lev->dimensions[d].tile_size;
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->d_dst_tile_size, sz));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->d_dst_tile_size, h_dst_ts, sz));

        uint64_t h_src_ts[MAX_RANK / 2];
        for (int d = 0; d < rank; ++d)
          h_src_ts[d] = parent_dims[d].tile_size;
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->d_src_tile_size, sz));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->d_src_tile_size, h_src_ts, sz));

        uint64_t h_src_ext[MAX_RANK / 2];
        for (int d = 0; d < rank; ++d)
          h_src_ext[d] = parent_dims[d].size;
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->d_src_extent, sz));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->d_src_extent, h_src_ext, sz));

        int64_t h_src_ps[MAX_RANK / 2];
        for (int d = 0; d < rank; ++d) {
          if (d == 0)
            h_src_ps[d] =
              (int64_t)(parent_layout->slot_count * parent_layout->tile_stride);
          else
            h_src_ps[d] = parent_layout->lifted_strides[2 * d];
        }
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->d_src_pool_strides, ssz));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->d_src_pool_strides, h_src_ps, ssz));

        int64_t h_dst_ps[MAX_RANK / 2];
        for (int d = 0; d < rank; ++d) {
          if (d == 0)
            h_dst_ps[d] =
              (int64_t)(lev->layout.slot_count * lev->layout.tile_stride);
          else
            h_dst_ps[d] = lev->layout.lifted_strides[2 * d];
        }
        CU(Fail, cuMemAlloc((CUdeviceptr*)&lev->d_dst_pool_strides, ssz));
        CU(Fail, cuMemcpyHtoD((CUdeviceptr)lev->d_dst_pool_strides, h_dst_ps, ssz));
      }

      lev->epoch_count = 0;

      // Aggregate layout for this LOD level
      CHECK(Fail,
            aggregate_layout_init(&lev->agg_layout, rank, lod_tile_count,
                                  lod_tiles_per_shard, lev->layout.slot_count,
                                  0 /* filled below */) == 0);

      // Shard state
      lev->shard.tiles_per_shard_0 = lod_tiles_per_shard[0];
      lev->shard.tiles_per_shard_inner = 1;
      for (int d = 1; d < rank; ++d)
        lev->shard.tiles_per_shard_inner *= lod_tiles_per_shard[d];
      lev->shard.tiles_per_shard_total =
        lev->shard.tiles_per_shard_0 * lev->shard.tiles_per_shard_inner;

      lev->shard.shard_inner_count = 1;
      for (int d = 1; d < rank; ++d)
        lev->shard.shard_inner_count *=
          ceildiv(lod_tile_count[d], lod_tiles_per_shard[d]);

      lev->shard.shards = (struct active_shard*)calloc(
        lev->shard.shard_inner_count, sizeof(struct active_shard));
      CHECK(Fail, lev->shard.shards);

      size_t index_bytes = 2 * lev->shard.tiles_per_shard_total * sizeof(uint64_t);
      for (uint64_t i = 0; i < lev->shard.shard_inner_count; ++i) {
        lev->shard.shards[i].index = (uint64_t*)malloc(index_bytes);
        CHECK(Fail, lev->shard.shards[i].index);
        memset(lev->shard.shards[i].index, 0xFF, index_bytes);
      }

      lev->shard.epoch_in_shard = 0;
      lev->shard.shard_epoch = 0;

      parent_dims = lev->dimensions;
      parent_layout = &lev->layout;
    }
  }

  // --- Allocate unified tile pools ---
  {
    const size_t A_bytes = M0 * out->layout.tile_stride * bpe;
    const size_t B_bytes = out->M_total * out->layout.tile_stride * bpe;

    CHECK(Fail, (out->pool_A = buffer_new(A_bytes, device, 0)).data);
    CHECK(Fail, (out->pool_B = buffer_new(B_bytes, device, 0)).data);
    CU(Fail, cuMemsetD8Async((CUdeviceptr)out->pool_A.data, 0, A_bytes, out->compute));
    CU(Fail, cuMemsetD8Async((CUdeviceptr)out->pool_B.data, 0, B_bytes, out->compute));

    // Host pools for uncompressed path
    CHECK(Fail, (out->pool_A_host = buffer_new(out->layout.tile_pool_bytes, host, 0)).data);
    CHECK(Fail, (out->pool_B_host = buffer_new(out->layout.tile_pool_bytes, host, 0)).data);

    // Two scratch buffers for LOD cascade. Each must hold the largest level's
    // tile pool. The cascade alternates between them like a 2-element stack.
    if (needs_two_epochs && num_levels > 1) {
      uint64_t max_Mk = 0;
      for (int lv = 0; lv < num_levels; ++lv) {
        if (out->levels[lv].layout.slot_count > max_Mk)
          max_Mk = out->levels[lv].layout.slot_count;
      }
      size_t scratch_bytes = max_Mk * out->layout.tile_stride * bpe;
      CHECK(Fail, (out->scratch[0] = buffer_new(scratch_bytes, device, 0)).data);
      CHECK(Fail, (out->scratch[1] = buffer_new(scratch_bytes, device, 0)).data);
    }
  }

  // --- Compression state ---
  if (config->compress) {
    const size_t tile_bytes = out->layout.tile_stride * bpe;

    out->max_comp_chunk_bytes = align_up(
      compress_get_max_output_size(tile_bytes), compress_get_input_alignment());
    CHECK(Fail, out->max_comp_chunk_bytes > 0);
    out->comp_pool_bytes = M0 * out->max_comp_chunk_bytes;

    out->comp_temp_bytes = compress_get_temp_size(out->M_total, tile_bytes);
    if (out->comp_temp_bytes > 0)
      CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_comp_temp, out->comp_temp_bytes));

    CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_comp_sizes,
                         out->M_total * sizeof(size_t)));

    // d_uncomp_sizes: all same
    {
      size_t* h_sizes = (size_t*)malloc(out->M_total * sizeof(size_t));
      CHECK(Fail, h_sizes);
      for (uint64_t k = 0; k < out->M_total; ++k)
        h_sizes[k] = tile_bytes;
      CU(Fail, cuMemAlloc((CUdeviceptr*)&out->d_uncomp_sizes,
                           out->M_total * sizeof(size_t)));
      CUresult rc = cuMemcpyHtoD((CUdeviceptr)out->d_uncomp_sizes, h_sizes,
                                  out->M_total * sizeof(size_t));
      free(h_sizes);
      CU(Fail, rc);
    }

    // Allocate flush slots and pre-build pointer arrays
    for (int fc = 0; fc < 2; ++fc) {
      struct flush_slot* fs = &out->flush[fc];

      CHECK(Fail,
            (fs->d_compressed =
               buffer_new(out->M_total * out->max_comp_chunk_bytes, device, 0))
              .data);
      CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_uncomp_ptrs,
                           out->M_total * sizeof(void*)));
      CU(Fail, cuMemAlloc((CUdeviceptr*)&fs->d_comp_ptrs,
                           out->M_total * sizeof(void*)));

      // Build pointer arrays on host, then upload
      void** h_ptrs = (void**)malloc(out->M_total * sizeof(void*));
      CHECK(Fail, h_ptrs);

      // Uncomp pointers
      if (fc == 0) {
        // A-epoch: L0 from pool_A, LOD from pool_B
        for (uint64_t k = 0; k < M0; ++k)
          h_ptrs[k] = (char*)out->pool_A.data + k * tile_bytes;
        uint64_t off = M0;
        for (int lv = 0; lv < num_levels; ++lv) {
          uint64_t Mk = out->levels[lv].layout.slot_count;
          size_t base = out->level_offset[lv + 1];
          for (uint64_t k = 0; k < Mk; ++k)
            h_ptrs[off + k] = (char*)out->pool_B.data + base + k * tile_bytes;
          off += Mk;
        }
      } else {
        // B-epoch: everything from pool_B (contiguous)
        for (uint64_t k = 0; k < out->M_total; ++k)
          h_ptrs[k] = (char*)out->pool_B.data + k * tile_bytes;
      }
      CU(Fail, cuMemcpyHtoD((CUdeviceptr)fs->d_uncomp_ptrs, h_ptrs,
                              out->M_total * sizeof(void*)));

      // Comp pointers: into d_compressed
      for (uint64_t k = 0; k < out->M_total; ++k)
        h_ptrs[k] = (char*)fs->d_compressed.data + k * out->max_comp_chunk_bytes;
      CU(Fail, cuMemcpyHtoD((CUdeviceptr)fs->d_comp_ptrs, h_ptrs,
                              out->M_total * sizeof(void*)));
      free(h_ptrs);
    }
  }

  // --- Aggregate + shard state ---
  if (config->compress && config->shard_sink) {
    crc32c_init_table();

    uint64_t tiles_per_shard[MAX_RANK];
    for (int d = 0; d < rank; ++d) {
      uint64_t tps = dims[d].tiles_per_shard;
      tiles_per_shard[d] = (tps == 0) ? tile_count[d] : tps;
    }

    CHECK(Fail,
          aggregate_layout_init(&out->agg_layout, rank, tile_count,
                                tiles_per_shard, out->layout.slot_count,
                                out->max_comp_chunk_bytes) == 0);

    for (int i = 0; i < 2; ++i)
      CHECK(Fail,
            aggregate_slot_init(&out->agg[i], &out->agg_layout,
                                out->comp_pool_bytes) == 0);

    // Shard geometry
    out->shard.tiles_per_shard_0 = tiles_per_shard[0];
    out->shard.tiles_per_shard_inner = 1;
    for (int d = 1; d < rank; ++d)
      out->shard.tiles_per_shard_inner *= tiles_per_shard[d];
    out->shard.tiles_per_shard_total =
      out->shard.tiles_per_shard_0 * out->shard.tiles_per_shard_inner;

    out->shard.shard_inner_count = 1;
    for (int d = 1; d < rank; ++d)
      out->shard.shard_inner_count *= ceildiv(tile_count[d], tiles_per_shard[d]);

    out->shard.shards = (struct active_shard*)calloc(
      out->shard.shard_inner_count, sizeof(struct active_shard));
    CHECK(Fail, out->shard.shards);

    size_t index_bytes = 2 * out->shard.tiles_per_shard_total * sizeof(uint64_t);
    for (uint64_t i = 0; i < out->shard.shard_inner_count; ++i) {
      out->shard.shards[i].index = (uint64_t*)malloc(index_bytes);
      CHECK(Fail, out->shard.shards[i].index);
      memset(out->shard.shards[i].index, 0xFF, index_bytes);
    }

    out->shard.epoch_in_shard = 0;
    out->shard.shard_epoch = 0;

    // Seed aggregate slot events
    for (int i = 0; i < 2; ++i)
      CU(Fail, cuEventRecord(out->agg[i].ready, out->compute));

    // LOD level aggregate slots + shard events
    for (int lv = 0; lv < num_levels; ++lv) {
      struct level_state* lev = &out->levels[lv];

      // Re-init aggregate layout with correct max_comp_chunk_bytes
      aggregate_layout_destroy(&lev->agg_layout);

      uint64_t lod_tile_count[MAX_RANK];
      uint64_t lod_tiles_per_shard[MAX_RANK];
      for (int d = 0; d < rank; ++d) {
        lod_tile_count[d] =
          ceildiv(lev->dimensions[d].size, lev->dimensions[d].tile_size);
        uint64_t tps = lev->dimensions[d].tiles_per_shard;
        lod_tiles_per_shard[d] = (tps == 0) ? lod_tile_count[d] : tps;
      }

      CHECK(Fail,
            aggregate_layout_init(&lev->agg_layout, rank, lod_tile_count,
                                  lod_tiles_per_shard, lev->layout.slot_count,
                                  out->max_comp_chunk_bytes) == 0);

      size_t lev_comp_pool_bytes = lev->layout.slot_count * out->max_comp_chunk_bytes;
      for (int i = 0; i < 2; ++i) {
        CHECK(Fail,
              aggregate_slot_init(&lev->agg[i], &lev->agg_layout,
                                  lev_comp_pool_bytes) == 0);
        CU(Fail, cuEventRecord(lev->agg[i].ready, out->compute));
      }
    }
  }

  // Seed initial events
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(out->stage.slot[i].h_in.ready, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_h2d_start, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_scatter_start, out->compute));
    CU(Fail, cuEventRecord(out->stage.slot[i].t_scatter_end, out->compute));
  }
  CU(Fail, cuEventRecord(out->pool_A.ready, out->compute));
  CU(Fail, cuEventRecord(out->pool_B.ready, out->compute));
  CU(Fail, cuEventRecord(out->pool_A_host.ready, out->compute));
  CU(Fail, cuEventRecord(out->pool_B_host.ready, out->compute));
  for (int i = 0; i < 2; ++i) {
    CU(Fail, cuEventRecord(out->flush[i].t_compress_start, out->compute));
    CU(Fail, cuEventRecord(out->flush[i].t_d2h_start, out->compute));
    CU(Fail, cuEventRecord(out->flush[i].ready, out->compute));
  }
  CU(Fail, cuEventRecord(out->t_downsample_start, out->compute));
  CU(Fail, cuEventRecord(out->t_downsample_end, out->compute));

  CU(Fail, cuStreamSynchronize(out->compute));

  out->metrics.memcpy =
    (struct stream_metric){ .name = "Memcpy", .best_ms = 1e30f };
  out->metrics.h2d = (struct stream_metric){ .name = "H2D", .best_ms = 1e30f };
  out->metrics.scatter =
    (struct stream_metric){ .name = "Scatter", .best_ms = 1e30f };
  out->metrics.downsample =
    (struct stream_metric){ .name = "Downsample", .best_ms = 1e30f };
  out->metrics.compress =
    (struct stream_metric){ .name = "Compress", .best_ms = 1e30f };
  out->metrics.aggregate =
    (struct stream_metric){ .name = "Aggregate", .best_ms = 1e30f };
  out->metrics.d2h = (struct stream_metric){ .name = "D2H", .best_ms = 1e30f };

  out->cursor = 0;
  out->stage.fill = 0;
  out->stage.current = 0;
  out->pool_current = 0;
  out->flush_current = 0;
  out->flush_pending = 0;

  return 0;

Fail:
  transpose_stream_destroy(out);
  return 1;
}

static struct writer_result
transpose_stream_append(struct writer* self, struct slice input)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);
  const size_t bpe = s->config.bytes_per_element;
  const size_t buffer_capacity = s->config.buffer_capacity_bytes;
  const uint8_t* src = (const uint8_t*)input.beg;
  const uint8_t* end = (const uint8_t*)input.end;

  while (src < end) {
    const uint64_t epoch_remaining =
      s->layout.epoch_elements - (s->cursor % s->layout.epoch_elements);
    const uint64_t input_remaining = (uint64_t)(end - src) / bpe;
    const uint64_t elements_this_pass =
      epoch_remaining < input_remaining ? epoch_remaining : input_remaining;
    const uint64_t bytes_this_pass = elements_this_pass * bpe;

    {
      uint64_t written = 0;
      while (written < bytes_this_pass) {
        const size_t space = buffer_capacity - s->stage.fill;
        const uint64_t remaining = bytes_this_pass - written;
        const size_t payload = space < remaining ? space : (size_t)remaining;

        if (s->stage.fill == 0) {
          const int si = s->stage.current;
          struct staging_slot* ss = &s->stage.slot[si];
          CU(Error, cuEventSynchronize(ss->h_in.ready));

          if (s->cursor > 0) {
            accumulate_metric_cu(&s->metrics.h2d, ss->t_h2d_start, ss->h_in.ready);
            accumulate_metric_cu(&s->metrics.scatter, ss->t_scatter_start, ss->t_scatter_end);
          }
        }

        {
          struct platform_clock mc = { 0 };
          platform_toc(&mc);
          memcpy((uint8_t*)s->stage.slot[s->stage.current].h_in.data +
                   s->stage.fill,
                 src + written,
                 payload);
          accumulate_metric_ms(&s->metrics.memcpy,
                               (float)(platform_toc(&mc) * 1000.0));
        }
        s->stage.fill += payload;
        written += payload;

        if (s->stage.fill == buffer_capacity || written == bytes_this_pass) {
          if (dispatch_scatter(s))
            goto Error;
          s->stage.fill = 0;
        }
      }
    }
    src += bytes_this_pass;

    if (s->cursor % s->layout.epoch_elements == 0 && s->cursor > 0) {
      struct writer_result fr = flush_epoch(s);
      if (fr.error)
        return writer_error_at(src, end);
    }
  }

  return (struct writer_result){ .error = 0,
                                 .rest = { .beg = src, .end = end } };

Error:
  return writer_error_at(src, end);
}

static struct writer_result
transpose_stream_flush(struct writer* self)
{
  struct transpose_stream* s =
    container_of(self, struct transpose_stream, writer);

  if (s->stage.fill > 0) {
    if (dispatch_scatter(s))
      return writer_error();
    s->stage.fill = 0;
  }

  struct writer_result r = drain_pending_flush(s);
  if (r.error)
    return r;

  if (s->cursor % s->layout.epoch_elements != 0 || s->cursor == 0) {
    r = flush_epoch_sync(s);
    if (r.error)
      return r;
  }

  if (s->config.compress && s->config.shard_sink) {
    // Flush LOD levels: handle unpaired epochs
    for (int lv = 0; lv < s->num_levels; ++lv) {
      struct level_state* lev = &s->levels[lv];

      if (lev->needs_two_epochs && (lev->epoch_count & 1)) {
        // Self-pair: force cascade with pool_a = pool_b
        lev->epoch_count++;

        // Get the most recent parent pool
        CUdeviceptr pool_src;
        if (lv == 0) {
          pool_src = (CUdeviceptr)current_pool(s);
        } else {
          // Parent is LOD level lv-1's region in pool_B
          pool_src = (CUdeviceptr)s->pool_B.data + s->level_offset[lv];
        }

        CUdeviceptr dst = (CUdeviceptr)s->pool_B.data + s->level_offset[lv + 1];

        CU(Error, cuEventRecord(s->t_downsample_start, s->compute));

        if (dispatch_downsample(
              dst, pool_src, pool_src,
              s->config.bytes_per_element, s->config.rank,
              lev->downsample_mask,
              lev->d_dst_tile_size, lev->d_src_tile_size, lev->d_src_extent,
              lev->d_src_pool_strides, lev->d_dst_pool_strides,
              lev->layout.slot_count * lev->layout.tile_elements,
              s->compute) < 0)
          return writer_error();

        // Compress + D2H this level
        uint8_t firing[MAX_LOD_LEVELS + 1];
        int num_firing = 0;
        firing[num_firing++] = (uint8_t)(lv + 1);

        // Check deeper levels
        for (int dlv = lv + 1; dlv < s->num_levels; ++dlv) {
          struct level_state* dlev = &s->levels[dlv];
          dlev->epoch_count++;
          if (dlev->needs_two_epochs && (dlev->epoch_count & 1))
            break;

          CUdeviceptr dsrc = (CUdeviceptr)s->pool_B.data + s->level_offset[dlv + 1 - 1];
          CUdeviceptr ddst = (CUdeviceptr)s->pool_B.data + s->level_offset[dlv + 1];

          if (dispatch_downsample(
                ddst, dsrc, dsrc,
                s->config.bytes_per_element, s->config.rank,
                dlev->downsample_mask,
                dlev->d_dst_tile_size, dlev->d_src_tile_size, dlev->d_src_extent,
                dlev->d_src_pool_strides, dlev->d_dst_pool_strides,
                dlev->layout.slot_count * dlev->layout.tile_elements,
                s->compute) < 0)
            return writer_error();

          firing[num_firing++] = (uint8_t)(dlv + 1);
        }

        CU(Error, cuEventRecord(s->t_downsample_end, s->compute));

        // Batch compress + D2H for firing levels
        // Use flush slot 1 (B pool) since LOD data is always in B
        int flush_fc = 1;
        uint64_t total_chunks = 0;
        for (int fi = 0; fi < num_firing; ++fi)
          total_chunks += s->levels[firing[fi] - 1].layout.slot_count;

        if (kick_epoch(s, flush_fc, total_chunks, firing, num_firing))
          return writer_error();
        struct writer_result wr = wait_and_deliver(s, flush_fc);
        if (wr.error)
          return wr;
      }
    }

    // Emit partial shards
    for (int lv = 0; lv < s->num_levels; ++lv) {
      struct level_state* lev = &s->levels[lv];
      if (lev->shard.epoch_in_shard > 0) {
        if (emit_shards(&lev->shard))
          return writer_error();
      }
    }

    if (s->shard.epoch_in_shard > 0)
      return emit_shards(&s->shard) ? writer_error() : writer_ok();
    return writer_ok();
  }

  if (s->config.sink)
    return writer_flush(s->config.sink);

  return writer_ok();

Error:
  return writer_error();
}
