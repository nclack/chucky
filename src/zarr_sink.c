#include "zarr_sink.h"
#include "io_queue.h"
#include "json_writer.h"
#include "log/log.h"
#include "platform_io.h"

#include <stdlib.h>
#include <string.h>

#define CHECK(lbl, expr)                                                       \
  do {                                                                         \
    if (!(expr)) {                                                             \
      log_error("%s:%d check failed: %s", __FILE__, __LINE__, #expr);          \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define MAX_ZARR_RANK 8

static uint64_t
ceildiv(uint64_t a, uint64_t b)
{
  return (a + b - 1) / b;
}

// --- Directory creation ---

static int
mkdirp(const char* path)
{
  char tmp[4096];
  size_t len = strlen(path);
  if (len >= sizeof(tmp))
    return -1;
  memcpy(tmp, path, len + 1);

  for (size_t i = 1; i < len; ++i) {
    if (tmp[i] == '/' || tmp[i] == '\\') {
      char saved = tmp[i];
      tmp[i] = '\0';
      if (platform_mkdir(tmp) != 0)
        return -1;
      tmp[i] = saved;
    }
  }
  return platform_mkdir(tmp);
}

// --- Writer for a single shard file ---

struct zarr_shard_writer
{
  struct shard_writer base;
  platform_fd fd;
  struct io_queue* queue;
};

struct pwrite_job
{
  platform_fd fd;
  uint64_t offset;
  size_t nbytes;
  uint8_t data[];
};

static void
pwrite_fn(void* arg)
{
  struct pwrite_job* j = (struct pwrite_job*)arg;
  if (platform_pwrite(j->fd, j->data, j->nbytes, j->offset) != 0)
    log_error("zarr pwrite failed");
}

static int
zarr_shard_write(struct shard_writer* self,
                 uint64_t offset,
                 const void* beg,
                 const void* end)
{
  struct zarr_shard_writer* w = (struct zarr_shard_writer*)self;
  size_t nbytes = (size_t)((const char*)end - (const char*)beg);

  if (w->queue) {
    struct pwrite_job* j =
      (struct pwrite_job*)malloc(sizeof(struct pwrite_job) + nbytes);
    CHECK(Error, j);
    j->fd = w->fd;
    j->offset = offset;
    j->nbytes = nbytes;
    memcpy(j->data, beg, nbytes);
    io_queue_post(w->queue, pwrite_fn, j, free);
  } else {
    CHECK(Error, platform_pwrite(w->fd, beg, nbytes, offset) == 0);
  }
  return 0;

Error:
  return 1;
}

struct close_job
{
  platform_fd fd;
};

static void
close_fn(void* arg)
{
  struct close_job* j = (struct close_job*)arg;
  platform_close(j->fd);
}

static int
zarr_shard_finalize(struct shard_writer* self)
{
  struct zarr_shard_writer* w = (struct zarr_shard_writer*)self;
  if (w->fd == PLATFORM_FD_INVALID)
    return 0;

  if (w->queue) {
    struct close_job* j = (struct close_job*)malloc(sizeof(struct close_job));
    CHECK(Error, j);
    j->fd = w->fd;
    io_queue_post(w->queue, close_fn, j, free);
  } else {
    platform_close(w->fd);
  }

  w->fd = PLATFORM_FD_INVALID;
  return 0;

Error:
  return 1;
}

// --- Zarr sink ---

struct zarr_sink
{
  struct shard_sink base;
  struct io_queue* queue;

  // Geometry
  uint8_t rank;
  uint64_t tile_count[MAX_ZARR_RANK];
  uint64_t tiles_per_shard[MAX_ZARR_RANK];
  uint64_t shard_count[MAX_ZARR_RANK];
  uint64_t shard_inner_count; // prod(shard_count[d] for d > 0)

  // Paths
  char array_dir[4096]; // "{store_path}/{array_name}"

  // Writer pool: one per inner shard, reused across shard epochs
  struct zarr_shard_writer* writers;
  uint64_t num_writers;
};

// --- Path computation ---

static int
shard_path(char* buf,
           size_t cap,
           const char* array_dir,
           uint8_t rank,
           const uint64_t* shard_count,
           uint64_t flat)
{
  int pos = snprintf(buf, cap, "%s/c", array_dir);
  if (pos < 0 || (size_t)pos >= cap)
    return -1;

  // Unravel flat index into coordinates (row-major)
  uint64_t coords[MAX_ZARR_RANK];
  uint64_t rem = flat;
  for (int d = rank - 1; d >= 0; --d) {
    coords[d] = rem % shard_count[d];
    rem /= shard_count[d];
  }

  for (int d = 0; d < rank; ++d) {
    int n = snprintf(buf + pos, cap - (size_t)pos, "/%llu",
                     (unsigned long long)coords[d]);
    if (n < 0 || (size_t)(pos + n) >= cap)
      return -1;
    pos += n;
  }
  return 0;
}

// --- shard_sink open ---

static struct shard_writer*
zarr_sink_open(struct shard_sink* self, uint64_t shard_index)
{
  struct zarr_sink* zs = (struct zarr_sink*)self;

  // Map flat shard_index to inner writer index
  // flat = s0 * shard_inner_count + inner_flat
  uint64_t inner = shard_index % zs->shard_inner_count;
  struct zarr_shard_writer* w = &zs->writers[inner];

  // Build path
  char path[4096];
  if (shard_path(
        path, sizeof(path), zs->array_dir, zs->rank, zs->shard_count,
        shard_index) != 0) {
    log_error("zarr_sink_open: path too long for shard %llu",
              (unsigned long long)shard_index);
    return NULL;
  }

  w->fd = platform_open_write(path);
  if (w->fd == PLATFORM_FD_INVALID) {
    log_error("zarr_sink_open: open(%s) failed", path);
    return NULL;
  }

  return &w->base;
}

// --- Metadata writing ---

static int
write_file(const char* path, const char* data, size_t len)
{
  platform_fd fd = platform_open_write(path);
  if (fd == PLATFORM_FD_INVALID)
    return -1;
  int rc = platform_write(fd, data, len);
  platform_close(fd);
  return rc;
}

static int
write_root_metadata(const char* store_path)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", store_path);

  char buf[256];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);
  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);
  jw_key(&jw, "node_type");
  jw_string(&jw, "group");
  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return write_file(path, buf, jw_length(&jw));
}

static const char*
zarr_dtype_string(enum zarr_dtype dt)
{
  switch (dt) {
    case zarr_dtype_uint8:   return "uint8";
    case zarr_dtype_uint16:  return "uint16";
    case zarr_dtype_uint32:  return "uint32";
    case zarr_dtype_uint64:  return "uint64";
    case zarr_dtype_int8:    return "int8";
    case zarr_dtype_int16:   return "int16";
    case zarr_dtype_int32:   return "int32";
    case zarr_dtype_int64:   return "int64";
    case zarr_dtype_float32: return "float32";
    case zarr_dtype_float64: return "float64";
  }
  return "unknown";
}

static int
write_array_metadata(const char* array_dir, const struct zarr_config* cfg,
                     const uint64_t* tiles_per_shard)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", array_dir);

  char buf[4096];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "array");

  jw_key(&jw, "shape");
  jw_array_begin(&jw);
  for (int d = 0; d < cfg->rank; ++d)
    jw_uint(&jw, cfg->dimensions[d].size);
  jw_array_end(&jw);

  jw_key(&jw, "data_type");
  jw_string(&jw, zarr_dtype_string(cfg->data_type));

  // chunk_grid: shard shape = tile_size * tiles_per_shard
  jw_key(&jw, "chunk_grid");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "regular");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < cfg->rank; ++d)
    jw_uint(&jw, cfg->dimensions[d].tile_size * tiles_per_shard[d]);
  jw_array_end(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);

  jw_key(&jw, "chunk_key_encoding");
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "default");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "separator");
  jw_string(&jw, "/");
  jw_object_end(&jw);
  jw_object_end(&jw);

  // codecs: sharding_indexed
  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "sharding_indexed");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);

  jw_key(&jw, "chunk_shape");
  jw_array_begin(&jw);
  for (int d = 0; d < cfg->rank; ++d)
    jw_uint(&jw, cfg->dimensions[d].tile_size);
  jw_array_end(&jw);

  jw_key(&jw, "codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "zstd");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "index_codecs");
  jw_array_begin(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "bytes");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_key(&jw, "endian");
  jw_string(&jw, "little");
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_object_begin(&jw);
  jw_key(&jw, "name");
  jw_string(&jw, "crc32c");
  jw_key(&jw, "configuration");
  jw_object_begin(&jw);
  jw_object_end(&jw);
  jw_object_end(&jw);
  jw_array_end(&jw);

  jw_key(&jw, "index_location");
  jw_string(&jw, "end");

  jw_object_end(&jw); // configuration
  jw_object_end(&jw); // sharding_indexed codec
  jw_array_end(&jw);  // codecs

  jw_key(&jw, "fill_value");
  jw_float(&jw, cfg->fill_value);

  // dimension_names (optional)
  {
    int has_names = 0;
    for (int d = 0; d < cfg->rank; ++d) {
      if (cfg->dimensions[d].name) {
        has_names = 1;
        break;
      }
    }
    if (has_names) {
      jw_key(&jw, "dimension_names");
      jw_array_begin(&jw);
      for (int d = 0; d < cfg->rank; ++d) {
        if (cfg->dimensions[d].name)
          jw_string(&jw, cfg->dimensions[d].name);
        else
          jw_null(&jw);
      }
      jw_array_end(&jw);
    }
  }

  jw_object_end(&jw);

  if (jw_error(&jw))
    return -1;
  return write_file(path, buf, jw_length(&jw));
}

// --- Create / Destroy ---

struct zarr_sink*
zarr_sink_create(const struct zarr_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->array_name);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);

  struct zarr_sink* zs = (struct zarr_sink*)calloc(1, sizeof(*zs));
  CHECK(Fail, zs);

  zs->base.open = zarr_sink_open;
  zs->queue = io_queue_create();
  CHECK(Fail_alloc, zs->queue);
  zs->rank = cfg->rank;

  // Compute geometry
  zs->shard_inner_count = 1;
  for (int d = 0; d < cfg->rank; ++d) {
    zs->tile_count[d] = ceildiv(cfg->dimensions[d].size,
                                cfg->dimensions[d].tile_size);
    uint64_t tps = cfg->dimensions[d].tiles_per_shard;
    zs->tiles_per_shard[d] = (tps == 0) ? zs->tile_count[d] : tps;
    zs->shard_count[d] = ceildiv(zs->tile_count[d], zs->tiles_per_shard[d]);
    if (d > 0)
      zs->shard_inner_count *= zs->shard_count[d];
  }

  // Build array directory path
  snprintf(zs->array_dir, sizeof(zs->array_dir), "%s/%s",
           cfg->store_path, cfg->array_name);

  // Create directory tree: ensure all shard directories exist
  {
    uint64_t total_shards = 1;
    for (int d = 0; d < cfg->rank; ++d)
      total_shards *= zs->shard_count[d];

    for (uint64_t flat = 0; flat < total_shards; ++flat) {
      char path[4096];
      if (shard_path(path, sizeof(path), zs->array_dir, zs->rank,
                     zs->shard_count, flat) != 0)
        goto Fail_alloc;

      // Get the directory portion (everything up to last '/')
      char* last_slash = strrchr(path, '/');
      if (last_slash) {
        *last_slash = '\0';
        if (mkdirp(path) != 0) {
          log_error("zarr_sink: mkdirp(%s) failed", path);
          goto Fail_alloc;
        }
      }
    }
  }

  // Write metadata
  CHECK(Fail_alloc, write_root_metadata(cfg->store_path) == 0);
  CHECK(Fail_alloc,
        write_array_metadata(zs->array_dir, cfg, zs->tiles_per_shard) == 0);

  // Allocate writer pool
  zs->num_writers = zs->shard_inner_count;
  zs->writers = (struct zarr_shard_writer*)calloc(
    zs->num_writers, sizeof(struct zarr_shard_writer));
  CHECK(Fail_alloc, zs->writers);

  for (uint64_t i = 0; i < zs->num_writers; ++i) {
    zs->writers[i].base.write = zarr_shard_write;
    zs->writers[i].base.finalize = zarr_shard_finalize;
    zs->writers[i].fd = PLATFORM_FD_INVALID;
    zs->writers[i].queue = zs->queue;
  }

  return zs;

Fail_alloc:
  free(zs->writers);
  free(zs);
Fail:
  return NULL;
}

void
zarr_sink_flush(struct zarr_sink* s)
{
  if (!s || !s->queue)
    return;
  struct io_event ev = io_queue_record(s->queue);
  io_event_wait(s->queue, ev);
}

void
zarr_sink_destroy(struct zarr_sink* s)
{
  if (!s)
    return;

  zarr_sink_flush(s);

  if (s->writers) {
    for (uint64_t i = 0; i < s->num_writers; ++i) {
      if (s->writers[i].fd != PLATFORM_FD_INVALID)
        platform_close(s->writers[i].fd);
    }
    free(s->writers);
  }
  io_queue_destroy(s->queue);
  free(s);
}

struct shard_sink*
zarr_sink_as_shard_sink(struct zarr_sink* s)
{
  return &s->base;
}

// --- Multiscale (LOD) zarr sink ---

struct zarr_multiscale_sink
{
  struct zarr_sink** levels; // [num_levels]
  int num_levels;
};

static int
write_multiscale_attributes(const char* store_path,
                            const struct zarr_multiscale_config* cfg,
                            const struct dimension* const* level_dims)
{
  char path[4096];
  snprintf(path, sizeof(path), "%s/zarr.json", store_path);

  char buf[8192];
  struct json_writer jw;
  jw_init(&jw, buf, sizeof(buf));

  jw_object_begin(&jw);

  jw_key(&jw, "zarr_format");
  jw_int(&jw, 3);

  jw_key(&jw, "node_type");
  jw_string(&jw, "group");

  jw_key(&jw, "attributes");
  jw_object_begin(&jw);

  jw_key(&jw, "multiscales");
  jw_array_begin(&jw);
  jw_object_begin(&jw);

  jw_key(&jw, "version");
  jw_string(&jw, "0.4");

  // axes
  jw_key(&jw, "axes");
  jw_array_begin(&jw);
  for (int d = 0; d < cfg->rank; ++d) {
    jw_object_begin(&jw);
    jw_key(&jw, "name");
    if (cfg->dimensions[d].name)
      jw_string(&jw, cfg->dimensions[d].name);
    else {
      char dim_name[4];
      snprintf(dim_name, sizeof(dim_name), "d%d", d);
      jw_string(&jw, dim_name);
    }
    jw_key(&jw, "type");
    jw_string(&jw, "space");
    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  // datasets
  jw_key(&jw, "datasets");
  jw_array_begin(&jw);
  for (int lv = 0; lv < cfg->num_levels; ++lv) {
    jw_object_begin(&jw);

    jw_key(&jw, "path");
    char arr_name[16];
    snprintf(arr_name, sizeof(arr_name), "s%d", lv);
    jw_string(&jw, arr_name);

    jw_key(&jw, "coordinateTransformations");
    jw_array_begin(&jw);
    jw_object_begin(&jw);
    jw_key(&jw, "type");
    jw_string(&jw, "scale");
    jw_key(&jw, "scale");
    jw_array_begin(&jw);
    for (int d = 0; d < cfg->rank; ++d) {
      // Scale = level_0_size / level_lv_size (or equivalently, 2^lv for
      // downsampled dims). Compute from actual dimension sizes.
      double scale = 1.0;
      if (cfg->dimensions[d].downsample && lv > 0 && level_dims[lv]) {
        scale = (double)cfg->dimensions[d].size /
                (double)level_dims[lv][d].size;
      }
      jw_float(&jw, scale);
    }
    jw_array_end(&jw);
    jw_object_end(&jw);
    jw_array_end(&jw);

    jw_object_end(&jw);
  }
  jw_array_end(&jw);

  jw_key(&jw, "type");
  jw_string(&jw, "mean");

  jw_object_end(&jw); // multiscales[0]
  jw_array_end(&jw);  // multiscales
  jw_object_end(&jw); // attributes
  jw_object_end(&jw); // root

  if (jw_error(&jw))
    return -1;
  return write_file(path, buf, jw_length(&jw));
}

struct zarr_multiscale_sink*
zarr_multiscale_sink_create(const struct zarr_multiscale_config* cfg)
{
  CHECK(Fail, cfg);
  CHECK(Fail, cfg->store_path);
  CHECK(Fail, cfg->rank > 0 && cfg->rank <= MAX_ZARR_RANK);
  CHECK(Fail, cfg->dimensions);
  CHECK(Fail, cfg->num_levels > 0);

  struct zarr_multiscale_sink* ms =
    (struct zarr_multiscale_sink*)calloc(1, sizeof(*ms));
  CHECK(Fail, ms);
  ms->num_levels = cfg->num_levels;

  ms->levels =
    (struct zarr_sink**)calloc((size_t)cfg->num_levels, sizeof(struct zarr_sink*));
  CHECK(Fail_ms, ms->levels);

  // Build dimensions for each level
  struct dimension* all_dims =
    (struct dimension*)calloc(
      (size_t)cfg->num_levels * cfg->rank, sizeof(struct dimension));
  CHECK(Fail_levels, all_dims);

  // Pointer array for attributes writing
  const struct dimension** dim_ptrs =
    (const struct dimension**)calloc((size_t)cfg->num_levels,
                                     sizeof(const struct dimension*));
  CHECK(Fail_dims, dim_ptrs);

  // Level 0: copy dimensions as-is
  for (int d = 0; d < cfg->rank; ++d)
    all_dims[d] = cfg->dimensions[d];
  dim_ptrs[0] = all_dims;

  // Compute dimensions for each subsequent level
  for (int lv = 1; lv < cfg->num_levels; ++lv) {
    struct dimension* prev = all_dims + (lv - 1) * cfg->rank;
    struct dimension* cur = all_dims + lv * cfg->rank;
    for (int d = 0; d < cfg->rank; ++d) {
      cur[d] = prev[d];
      if (prev[d].downsample)
        cur[d].size = ceildiv(prev[d].size, 2);
    }
    dim_ptrs[lv] = cur;
  }

  // Create zarr_sink for each level
  for (int lv = 0; lv < cfg->num_levels; ++lv) {
    char arr_name[16];
    snprintf(arr_name, sizeof(arr_name), "s%d", lv);

    struct zarr_config zcfg = {
      .store_path = cfg->store_path,
      .array_name = arr_name,
      .data_type = cfg->data_type,
      .fill_value = cfg->fill_value,
      .rank = cfg->rank,
      .dimensions = dim_ptrs[lv],
    };
    ms->levels[lv] = zarr_sink_create(&zcfg);
    CHECK(Fail_sinks, ms->levels[lv]);
  }

  // Write OME-NGFF multiscale attributes
  CHECK(Fail_sinks,
        write_multiscale_attributes(cfg->store_path, cfg, dim_ptrs) == 0);

  free(dim_ptrs);
  free(all_dims);
  return ms;

Fail_sinks:
  for (int lv = 0; lv < cfg->num_levels; ++lv) {
    if (ms->levels[lv])
      zarr_sink_destroy(ms->levels[lv]);
  }
Fail_dims:
  free(dim_ptrs);
Fail_levels:
  free(all_dims);
  free(ms->levels);
Fail_ms:
  free(ms);
Fail:
  return NULL;
}

void
zarr_multiscale_sink_destroy(struct zarr_multiscale_sink* s)
{
  if (!s)
    return;

  if (s->levels) {
    for (int lv = 0; lv < s->num_levels; ++lv)
      zarr_sink_destroy(s->levels[lv]);
    free(s->levels);
  }
  free(s);
}

void
zarr_multiscale_sink_flush(struct zarr_multiscale_sink* s)
{
  if (!s)
    return;
  for (int lv = 0; lv < s->num_levels; ++lv)
    zarr_sink_flush(s->levels[lv]);
}

struct shard_sink*
zarr_multiscale_get_level_sink(struct zarr_multiscale_sink* s, int level)
{
  if (!s || level < 0 || level >= s->num_levels)
    return NULL;
  return zarr_sink_as_shard_sink(s->levels[level]);
}
