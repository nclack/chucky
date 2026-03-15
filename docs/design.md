# Design

## The problem

Scientific imaging instruments produce sustained, high-bandwidth streams of
multidimensional data. A light-sheet microscope, for example, generates
2–10 GB/s of 16-bit pixel values organized across time, channel, z, y, and x
dimensions. These streams may run indefinitely — an instrument can acquire for
hours or days, appending along one dimension with no predetermined end.

The storage system must handle this with a fixed resource footprint: bounded
memory, bounded open file handles, and the ability to write completed regions
incrementally without revisiting earlier data. It must also support random
access to rectangular sub-regions for downstream analysis.

[Zarr v3][zarr-v3] addresses this by partitioning the array into independent,
individually addressable **chunks** that can each be compressed and read in
isolation. (We use the term **tile** for chunk throughout this document to avoid
overloading "chunk," which is used in other contexts.) Zarr's
[sharding codec][zarr-shard] groups tiles into **shards** — single storage
objects containing multiple tiles plus an index — reducing the number of files
and amortizing I/O overhead. Tile sizes balance compression ratio against
random-access granularity; shard sizes balance file count against write
amplification.

[zarr-v3]: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html
[zarr-shard]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html

Because the stream is potentially unbounded, the pipeline cannot buffer the
full array before writing. It must tile and compress data incrementally. The
key observation is that as data arrives in row-major order, only a bounded set
of tiles are active at any time — those sharing the same position along the
outermost (slowest-varying) dimension, the **append dimension**. We call this
set of simultaneously live tiles an **epoch**. The pipeline processes one epoch
at a time, flushes the completed tiles, and reuses the memory. This bounds
the working set regardless of how long the stream runs. (The formal analysis
is in [streaming.md](streaming.md); the mathematical details appear in the
[Approach](#approach) section below.)

During acquisition, scientists need to visualize incoming data in real time —
zooming and panning across a dataset that may already be hundreds of gigabytes.
This requires a **multiscale pyramid**: progressively downsampled copies of the
array at half the resolution along selected dimensions. Viewers like
Neuroglancer read coarse levels for overview and load finer levels on demand,
even while acquisition is still running. The [OME-NGFF][ome-ngff]
specification standardizes how these pyramids are stored alongside zarr arrays
for bioimaging, and is the target output format.

[ome-ngff]: https://ngff.openmicroscopy.org/0.5/

## Approach

We model the data as a **multidimensional array** of rank $D$ with shape
$(s_0, s_1, \ldots, s_{D-1})$, where $s_d$ is the extent along dimension $d$.
Dimensions are ordered slowest-to-fastest: $d = 0$ varies slowest and
$d = D{-}1$ varies fastest. The array is stored in **row-major order** —
elements are laid out in memory such that the last index changes fastest, and
each element is identified by a flat index $i \in [0, \prod_d s_d)$.

### Mixed-radix representation

The pipeline performs several index-space transformations — tiling, storage
reordering, shard aggregation — that look different but share a common
mechanism. Mixed-radix arithmetic gives us a uniform language for all of them.

The coordinates of an array with shape $(s_0, \ldots, s_{D-1})$ form a bounded
integer lattice — the set of all $(r_0, \ldots, r_{D-1})$ with $0 \le r_d <
s_d$. Interpreting coordinates as **mixed-radix** numbers — where the shape
serves as the **radix vector** — gives each point in the lattice a natural
ordering (the row-major order) and a flat index. In a mixed-radix system with
radix vector $(b_0, \ldots, b_{D-1})$, every non-negative integer $i <
\prod_d b_d$ has a unique coordinate vector $(r_0, \ldots, r_{D-1})$ where
$0 \le r_d < b_d$. The two conversions are:

- **Unravel.** Recover coordinates from a flat index by repeated division
  against the radix vector: $r_{D-1} = i \bmod b_{D-1}$, then
  $r_{D-2} = \lfloor i / b_{D-1} \rfloor \bmod b_{D-2}$, and so on.

- **Ravel.** Recover a flat index from coordinates using **place values**.
  Each position $d$ has a place value — the product of all faster-varying
  bases: $\sigma_d = \prod_{k=d+1}^{D-1} b_k$. Then
  $i = \sum_d r_d \cdot \sigma_d$.

These place values are more commonly called **strides**. The strides derived
from the array shape — $\sigma_d = \prod_{k=d+1}^{D-1} s_k$ — are the
**natural strides**, and raveling with them recovers the original row-major
index.

The key property: raveling the same coordinates with a different stride vector
produces a different flat index — placing the element at a different memory
location. If that stride vector is itself the natural strides of some target
radix vector, the mapping is an isomorphism: every input position maps to
exactly one output position and vice versa. This is a transposition.

The pipeline's transformations all reduce to unraveling with one radix vector
and raveling with the strides of another. The scatter kernel, shard
aggregation, and LOD-to-tiles scatter differ only in which radix and stride
tables they use.

### Lifted coordinates and scatter

The central operation is reorganizing a flat row-major stream into tiles. We do
this by **lifting** — replacing the original radix vector with a finer one that
separates tile identity from position within a tile.

Given the array shape $(s_0, \ldots, s_{D-1})$ and tile shape $(n_0, \ldots,
n_{D-1})$, define the tile count $t_d = \lceil s_d / n_d \rceil$ for each
dimension. The original rank-$D$ radix vector is replaced by a rank-$2D$
**lifted radix vector**:

$$(t_0, n_0, \; t_1, n_1, \; \ldots, \; t_{D-1}, n_{D-1})$$

Unraveling a flat index against this radix vector produces coordinates where
$t_d$ identifies the tile along dimension $d$ and $n_d$ is the position within
that tile. To assemble tiles, we ravel these same coordinates with the strides
of the **tile-major** radix vector:

$$(t_0, \ldots, t_{D-1}, \; n_0, \ldots, n_{D-1})$$

The first $D$ coordinates identify the tile; the last $D$ identify the position
within it. This is the unravel/ravel pattern from the previous section — the
two radix vectors share the same elements, just in a different order, so the
mapping is an isomorphism.

A GPU **scatter kernel** implements this: each thread unravels its input index
against the lifted radix vector, then ravels the coordinates with the
tile-major strides, writing the element directly to its tile slot.

**Storage order.** The tile-major strides encode the desired dimension ordering
in the on-disk layout. Changing the storage order (e.g., from `tzcyx` to
`tczyx`) is a different permutation of the same radix vector, producing
different strides. The scatter kernel itself is unchanged — only the stride
table differs.

### Parallel compression with nvcomp

Once tiles are assembled in GPU memory, they are compressed in a single
batched call using NVIDIA's nvcomp library (`nvcompBatchedZstdCompressAsync`
or the lz4 equivalent). nvcomp compresses all tiles in the batch simultaneously,
one tile per GPU thread block. This is efficient when the batch contains many
tiles (1000+), which motivates batching multiple epochs together.

### Batching and the tile pool

A single epoch may produce relatively few tiles (e.g., 576 for a typical
camera configuration). Compressing a small batch underutilizes the GPU. To
address this, the pipeline accumulates $K$ consecutive epochs into a **batch**
before triggering the compress → aggregate → transfer sequence. $K$ is chosen
so the total tile count across all levels is large enough for good GPU
occupancy.

The **tile pool** is a contiguous GPU buffer with $K \times M$ slots (times the
number of LOD levels). Two pools are allocated: while one receives scatter
writes from the current batch, the other drains through compression and
transfer. This double-buffering ensures the scatter and compress stages overlap
completely.

### Shard aggregation

After compression, tiles sit in the tile pool in tile-major order — the order
they were scattered into. But shards group tiles by spatial locality, which is a
different order. A GPU **aggregation** kernel reorders compressed tiles into
shard-major order using a three-pass algorithm:

1. **Permute sizes.** Compute the shard-major destination for each tile and
   write its compressed size to that position.
2. **Prefix sum.** Exclusive scan over the permuted sizes to compute byte
   offsets.
3. **Gather.** Copy each tile's compressed bytes to its destination offset.

The result is a single contiguous buffer where all tiles belonging to the same
shard are adjacent. This means each shard can be written to disk with one I/O
call. When direct I/O is configured, padding is inserted between shards to
align to page boundaries.

The permutation is another instance of the unravel/ravel pattern. Each tile
coordinate $t_d$ is unraveled into a shard index $s_d$ and within-shard
position $w_d$ (using radix $p_d$), then the full coordinate vector is raveled
with shard-major strides. See [sharding.md](sharding.md) for the derivation.

### Multiscale via compacted morton order

The multiscale pyramid requires reducing over $2 \times \ldots \times 2$ blocks
at each level. A naive approach would iterate over blocks explicitly, but this
maps poorly to GPU execution. Instead, we reorder elements into **compacted
morton order** — a bit-interleaved indexing that places each $2^D$-element block
in a contiguous run.

The **morton index** of a coordinate $(r_0, \ldots, r_{D-1})$ is formed by
interleaving the bits of each coordinate: if $r_d(k)$ denotes the $k$-th bit of
$r_d$, then

$$\text{morton}(r) = \ldots\, r_0(k)\, r_1(k) \cdots r_{D-1}(k) \;\ldots\; r_0(0)\, r_1(0) \cdots r_{D-1}(0)$$

In this order, every consecutive run of $2^D$ elements forms a
$2 \times \ldots \times 2$ block. Reducing each run produces the next coarser
level, and the process repeats. The result is a pyramid of levels computed by
successive reduction of contiguous runs — ideal for GPU parallelism.

The complication is that the array shape is not a power of two. A
$2^p$-sized bounding box would contain many out-of-bounds indices. The
**compacted** morton order skips these, producing a dense sequence that covers
only in-bounds elements. Boundary elements are handled by replicate padding:
edge elements are averaged with copies of themselves rather than with zeros, so
there is no darkening artifact at array boundaries.

Not all dimensions participate in downsampling (e.g., a channel dimension
should not be reduced). The morton interleaving is restricted to the
downsampled dimensions; non-downsampled dimensions are treated as batch
dimensions.

### Separable fold on the append dimension

The append dimension ($d = 0$) requires special treatment. The spatial
dimensions within an epoch are fully available and can be reduced immediately,
but $d = 0$ extends over multiple epochs. Computing a $2\times$ reduction along
$d = 0$ requires data from two consecutive epochs.

Rather than buffering $2^L$ epochs (where $L$ is the number of LOD levels —
potentially 32+ epochs), the pipeline uses a **separable fold**: a temporal
accumulator that maintains partial reductions across epochs. Each epoch
contributes its data to the accumulator, and when $2^l$ epochs have been
accumulated for level $l$, the level emits its reduced tiles and resets. This
keeps the memory cost proportional to one epoch regardless of pyramid depth.

The fold is separable in the sense that the spatial reduction (within an epoch)
and the temporal reduction (across epochs along $d = 0$) are independent and
can use different reduction operators.

## Pipeline

Data flows through four CUDA streams. Double-buffered pools and event-based
synchronization overlap every stage.

```
                          ┌─────────────────────────────────────────────────┐
 Host input               │                    GPU                          │
 (row-major bytes)        │                                                 │
        │                 │                                                 │
        ▼                 │                                                 │
 ┌──────────────┐         │                                                 │
 │ Staging      │ ── H2D ─┤►  d_staging                                     │
 │ (pinned, 2×) │  stream │       │                                         │
 └──────────────┘         │       ▼                                         │
                          │  ┌─────────┐   ┌──────────────────────────────┐ │
                          │  │ Scatter │   │ LOD (if multiscale)          │ │
                 compute  │  │ kernel  │   │                              │ │
                  stream  │  │         │──►│  gather → reduce → dim0 fold │ │
                          │  │         │   │   → morton-to-tiles scatter  │ │
                          │  └────┬────┘   └──────────────┬───────────────┘ │
                          │       │                       │                 │
                          │       ▼                       ▼                 │
                          │  ┌──────────────────────────────┐               │
                          │  │ Tile pool (2× batched)       │               │
                          │  │ L0 tiles + L1..Ln LOD tiles  │               │
                          │  └──────────────┬───────────────┘               │
                          │                 │                               │
                 compress │                 ▼                               │
                  stream  │  ┌──────────────────────────┐                   │
                          │  │ Batch compress           │                   │
                          │  │ (nvcomp lz4/zstd)        │                   │
                          │  └─────────────┬────────────┘                   │
                          │                │                                │
                          │                ▼                                │
                          │  ┌──────────────────────────┐                   │
                          │  │ Aggregate by shard       │                   │
                          │  │ (permute, scan, gather)  │                   │
                          │  └─────────────┬────────────┘                   │
                          │                │                                │
                     d2h  │                ▼                                │
                  stream  │  ┌──────────────────────────┐                   │
                          │  │ D2H transfer             │                   │
                          │  │ (offsets, then data)     │                   │
                          │  └─────────────┬────────────┘                   │
                          │                │                                │
                          └────────────────┼────────────────────────────────┘
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │ Shard delivery│
                                    │ + index build │
                                    └──────┬────────┘
                                           │
                                           ▼
                                    ┌───────────────┐
                                    │ Zarr v3 store │
                                    │ (direct I/O)  │
                                    └───────────────┘
```

### Stage details

**Ingest (H2D stream).** Host data is copied into one of two pinned staging
buffers. When a buffer is full (or the data chunk ends), it is transferred to
the GPU asynchronously. The H2D stream waits on the prior scatter to finish
before overwriting the staging area on the device.

**Scatter (compute stream).** Each thread unravels its input index against the
lifted radix vector and ravels with tile-major strides, writing the element to
its tile pool slot. When multiscale is enabled, the raw data is instead copied
linearly for the LOD pipeline, and L0 scatter happens as part of the LOD stage.

**LOD (compute stream).** If multiscale is enabled, each epoch triggers:
1. *Gather* — reorder elements into compacted morton order
2. *Reduce* — apply the reduction operator across $2 \times \ldots \times 2$
   blocks for each level
3. *Dim0 fold* — accumulate into temporal reduction buffers
4. *Morton-to-tiles* — unravel morton indices and ravel with per-level
   tile-major strides to scatter reduced data into tile regions

Each LOD level produces its own set of tiles, interleaved in the same tile pool
as L0.

**Compress (compress stream).** All tiles in the batch (across all levels) are
compressed in a single nvcomp batch call.

**Aggregate (compress stream).** Compressed tiles are reordered from tile-major
to shard-major order using the three-pass algorithm described above.

**D2H (D2H stream).** Two-phase transfer: first the offsets array (small), then
the compressed data (sized by the actual compressed output, not the worst-case
bound).

**Shard delivery (host).** The host iterates over tiles in shard-major order,
dispatching contiguous runs to per-shard writers and building the shard index.
When a shard's tiles are complete, the index is serialized with a CRC32C
checksum and emitted.

### Double buffering

| Resource          | Count | Purpose                                    |
|-------------------|-------|--------------------------------------------|
| Staging buffers   | 2     | Overlap host memcpy with H2D transfer      |
| Tile pools        | 2     | Overlap scatter with compress+aggregate    |
| Compressed pools  | 2     | Overlap compress with D2H                  |
| Aggregate buffers | 2     | Overlap aggregate with D2H                 |

Event-based synchronization (no stream-wide barriers) ensures each stage waits
only on its actual data dependency.

## Related documents

- [streaming.md](streaming.md) — tile lifetime analysis, FIFO proof, epoch
  derivation
- [sharding.md](sharding.md) — tile-to-shard lifting, aggregation kernel, zarr
  shard binary format
