# Chucky

A high-performance streaming library for tiled transformation and compression of
large multidimensional arrays (tensors) using CUDA.

## Overview

Chucky enables efficient processing of large $D$-dimensional tensors by:

1. **Tiling**: Partitioning the input tensor into smaller tiles
2. **Transpose**: Reorganizing data so tiles are stored contiguously in memory
3. **Compression**: Using nvcomp's zstd compression on each tile in parallel
4. **Streaming**: Processing data in chunks using double-buffered pinned memory

This approach is designed as a proof-of-principle pipeline for performant
streaming writes to compressed, sharded Zarr stores.

## How It Works

Given an input tensor of shape $(s_{D-1}, \ldots, s_0)$ and tile shape
$(n_{D-1}, \ldots, n_0)$:

- The number of tiles along dimension $d$ is $t_d = \lceil s_d / n_d \rceil$
- The tensor is conceptually "lifted" to shape $(t_{D-1}, n_{D-1}, \ldots, t_0, n_0)$
- Data is transposed to $(t_{D-1}, \ldots, t_0, n_{D-1}, \ldots, n_0)$ for
  tile-contiguous layout
- Each of the $T = \prod_{d=0}^{D-1} t_d$ tiles is compressed in parallel using nvcomp
- Tiles at boundaries are zero-padded as needed

Input data arrives in row-major order as spans of contiguous bytes. A streaming
pipeline processes these spans, buffering them for GPU transfer and processing.

## Dependencies

- **CUDA Toolkit** (12.8 or later) - provides CUDA runtime and nvcc compiler
- **nvcomp** (5.x) - NVIDIA compression library for GPU
- **CMake** - build system

## Status

Early development - building out the core streaming pipeline and tiling
transpose functionality.
