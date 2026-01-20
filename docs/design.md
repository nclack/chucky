# Design Documentation

## Architecture Overview

### Tiled Tensor Transformation Pipeline

The core pipeline transforms D-dimensional tensors through tiling and
transposition for efficient parallel compression.

#### Mathematical Foundation

Given:
- Input tensor shape: $(s_{D-1}, \ldots, s_0)$
- Tile shape: $(n_{D-1}, \ldots, n_0)$
- Number of tiles per dimension: $t_d = \lceil s_d / n_d \rceil$
- Total tiles: $T = \prod_{d=0}^{D-1} t_d$

Transformation sequence:
1. **Lift**: Treat input as shape $(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$
2. **Transpose**: Reorganize to $(t_{D-1}, \ldots, t_0, n_{D-1}, \ldots, n_0)$
3. **Compress**: Apply zstd to each of the $T$ tiles in parallel

#### Streaming Architecture

```
Input stream (row-major bytes)
    ↓
Writer interface (std::span<const std::byte>)
    ↓
Double-buffered host pinned memory (~1GB each)
    ↓ (async H2D transfer)
GPU staging buffer
    ↓ (CUDA kernel)
Tiled/transposed GPU memory
    ↓ (nvcompBatchedZstdCompressAsync)
Compressed tiles in GPU memory
    ↓ (async D2H transfer)
Host output buffer
    ↓
Disk I/O (eventual zarr store)
```

**Performance considerations:**
- Double buffering: ~1GB buffers @ 2-10 GB/s input → ~200ms fill time
- Overlap CPU→GPU transfer with kernel execution
- Batch compress all tiles in parallel using nvcomp

### File Organization

```
Public Interface:    lib.hh, writer.hh          (declarations only)
Private Details:     lib.priv.hh, writer.priv.hh (impl details if needed)
Implementation:      lib.cc, writer.cc           (C++ code)
CUDA Implementation: *.cu                        (CUDA kernels)
```

## Key Data Structures

### Dimension Metadata

```cpp
struct Dimension {
    uint32_t size_px;       // Size of this dimension
    uint32_t tile_size_px;  // Tile size for this dimension
};
```

### Layout

Describes the tensor layout and tiling scheme. Maximum 64 dimensions supported (natural limit: if all dimensions have size 2, max elements = $2^{64}$). Dimensions with size ≤1 are elided.

## Data Types

Supported element types:
- `uint8_t` (u8)
- `uint16_t` (u16) - most common for target application
- `uint32_t` (u32)
- `float` (f32)
- `double` (f64)

## Writer Interface

Writers process incoming byte streams:

```cpp
struct WriteResult {
    std::span<const std::byte> unconsumed;
    bool all_consumed() const;
};

// Writer accepts bytes, copies what fits, returns unconsumed portion
WriteResult write(std::span<const std::byte> data) noexcept;
```

**Semantics:**
- Accepts span of input bytes (begin/end pointers)
- Copies as much as fits into available buffer space immediately
- Returns span covering any uncopied bytes
- Empty span indicates all bytes consumed
- Caller retries unconsumed bytes later

## Error Handling

- Use **return codes** (not exceptions)
- Use **spdlog** for error logging and diagnostics
- Noexcept functions where possible for performance

## CUDA Integration

**Compiler constraints:**
- nvcc 12.8 supports up to **C++20** (not C++23)
- Static linking: `libcudart_static.a`, `libnvcomp_static.a`
- Separate compilation: `.cu` files compiled with nvcc, `.cc` with clang++

**Memory management:**
- Pinned (page-locked) host memory for fast transfers
- Async H2D/D2H transfers overlapped with computation
- GPU kernels handle boundary zero-padding

## Compression

Using `nvcompBatchedZstdCompressAsync` from nvcomp to compress all $T$ tiles in parallel on GPU.

## Future Considerations

- Sharding layer (mentioned but not yet designed)
- Zarr store backend integration
- Multi-GPU support
- Dynamic buffer sizing based on input rate
