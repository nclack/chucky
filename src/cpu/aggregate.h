#pragma once

#include "types.aggregate.h"
#include <stddef.h>

// Aggregate compressed chunks into shard order on CPU.
//
// compressed: input buffer, chunk i at offset i * max_comp_chunk_bytes.
// comp_sizes[M]: actual compressed size of each chunk.
// layout: pre-computed aggregate layout (from aggregate_layout_compute).
// result: output — caller must free result->data, result->offsets,
//         result->chunk_sizes via aggregate_cpu_result_free.
//
// Returns 0 on success.
int aggregate_cpu(const void* compressed,
                  const size_t* comp_sizes,
                  const struct aggregate_layout* layout,
                  struct aggregate_result* result);

void aggregate_cpu_result_free(struct aggregate_result* result);
