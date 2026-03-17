#pragma once

#include "types.stream.h"

// Validate config and compute all CPU-only layout math.
// Returns 0 on success.
int
compute_stream_layouts(const struct tile_stream_configuration* config,
                       struct computed_stream_layouts* out);

// Free resources owned by computed_stream_layouts.
void
computed_stream_layouts_free(struct computed_stream_layouts* cl);

// Create a named stream_metric with best_ms initialized to a large value.
struct stream_metric
mk_stream_metric(const char* name);
