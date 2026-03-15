# Design

## The problem

Scientific imaging instruments produce sustained, high-bandwidth streams of
multidimensional data. A light-sheet microscope, for example, generates
2–10 GB/s of 16-bit pixel values organized across time, channel, z, y, and x
dimensions.

A **multidimensional array** of rank $D$ has a shape $(s_0, s_1, \ldots,
s_{D-1})$, where $s_d$ is the extent along dimension $d$. We order dimensions
slowest-to-fastest: $d = 0$ varies slowest and $d = D{-}1$ varies fastest. The
array is stored in **row-major order** — elements are laid out in memory such
that the last index changes fastest. Each element is identified by a flat index
$i \in [0, \prod_d s_d)$. The mapping between flat indices and coordinate
vectors is described in [Mixed-radix representation](#mixed-radix-representation)
below.

These arrays are too large to hold in memory or store as flat files. They
require a format that supports random access to rectangular sub-regions without
reading the entire dataset. **Zarr v3** is a chunked array format designed for
this: it partitions the array into independent, individually addressable
**chunks** (called **tiles** here) that can each be compressed and read in
isolation.

### Tiles and shards

A **tile** is a fixed-size rectangular sub-block of the array. Given a tile
shape $(n_0, n_1, \ldots, n_{D-1})$, the array is partitioned into
$\prod_d \lceil s_d / n_d \rceil$ tiles. Each tile is compressed independently
and stored as a unit. Tile sizes are chosen to balance compression ratio (larger
tiles compress better) against random access granularity (smaller tiles allow
more targeted reads).

A **shard** groups multiple tiles into a single storage object. Shards reduce
the number of files or objects in the store and amortize I/O overhead. Each
shard contains the compressed bytes of its constituent tiles followed by an
**index** — an array of `(offset, nbytes)` pairs that locates each tile within
the shard. The number of tiles per shard is specified per dimension by a
parameter $p_d$ (tiles per shard along dimension $d$), so each shard covers a
rectangular region of $\prod_d p_d$ tiles.

### Streaming and epochs

The data arrives as a stream of bytes in row-major order. The full array may
never exist in memory — and one dimension (the **append dimension**, always
$d = 0$) may be unbounded, growing indefinitely as the instrument runs. The
pipeline must accept data incrementally and produce complete, compressed shards
without ever requiring the full extent of $s_0$.

To tile this stream with bounded memory, we observe that the number of tiles
simultaneously "live" — receiving elements from the input stream — is limited.
Define the **tile count** along each dimension as $t_d = \lceil s_d / n_d
\rceil$. As data arrives in row-major order, all tiles sharing the same
outermost tile index $t_0$ are active at the same time. This set of tiles is an
**epoch**. An epoch contains

$$M = \prod_{d=1}^{D-1} t_d$$

tiles — the product of tile counts along every dimension except $d = 0$.
Within an epoch, tiles activate and retire in FIFO order: the first tile to
receive data is the first to complete. This means a fixed-size pool of $M$ tile
buffers suffices to hold an entire epoch, and completed tiles can be flushed in
order without any reordering.

Since $d = 0$ is potentially unbounded, the pipeline processes one epoch at a
time (or a small batch of epochs), flushes the resulting tiles, and reuses the
memory.

### Multiscale visualization

During acquisition, scientists need to visualize the incoming data in real
time — zooming and panning across a dataset that may already be hundreds of
gigabytes. Rendering the full-resolution data at every zoom level is
impractical, so the store provides a **multiscale pyramid**: a hierarchy of
progressively downsampled copies of the array, each at half the resolution of
the previous level along selected dimensions.

Because the pipeline already touches every element during the tiling scatter, it
can compute the pyramid levels at negligible additional cost — the data is
already on the GPU and in registers. Each level applies a **reduction** (mean,
min, max, median, or suppressed extremum) over $2 \times \ldots \times 2$
blocks to produce the next coarser level. The reduced data is tiled, compressed,
and written alongside the full-resolution output.

This means visualization tools can read coarse levels for overview and
progressively load finer levels on demand, even while the acquisition is still
running.

## Approach

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
this by **lifting** the index space into a higher-rank mixed-radix system that
separates tile identity from position within a tile.

Given the array shape $(s_0, \ldots, s_{D-1})$ and tile shape $(n_0, \ldots,
n_{D-1})$, define the tile count $t_d = \lceil s_d / n_d \rceil$ for each
dimension. The original radix vector $(s_0, \ldots, s_{D-1})$ is replaced by a
finer one — the **lifted shape**:

$$(t_0, n_0, \; t_1, n_1, \; \ldots, \; t_{D-1}, n_{D-1})$$

Unraveling a flat index $i$ against this lifted shape produces coordinates
where $t_d$ identifies the tile along dimension $d$ and $n_d$ is the position
within that tile. To assemble tiles, we ravel these coordinates with a
different set of strides — those corresponding to the **tile-major** ordering:

$$(t_0, \ldots, t_{D-1}, \; n_0, \ldots, n_{D-1})$$

The first $D$ coordinates identify the tile; the last $D$ identify the position
within it. A GPU **scatter kernel** implements this: each thread unravels its
input index against the lifted shape, then ravels the coordinates with the
tile-major output strides, writing the element directly to its tile slot.

**Storage order.** The output strides encode the desired dimension ordering in
the on-disk layout. Changing the storage order (e.g., from `tzcyx` to `tczyx`)
is just a different set of output strides — the scatter kernel is unchanged.
Transposition to any storage order is zero additional cost.

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

The permutation itself is another instance of the lifted-stride technique: tile
coordinates are decomposed into shard index and within-shard position, then
raveled with shard-major strides. See [sharding.md](sharding.md) for the
derivation.

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

**Scatter (compute stream).** Each thread decomposes its input index into
lifted coordinates and writes the element to the appropriate tile pool slot.
When multiscale is enabled, the raw data is instead copied linearly for the LOD
pipeline, and L0 scatter happens as part of the LOD stage.

**LOD (compute stream).** If multiscale is enabled, each epoch triggers:
1. *Gather* — reorder elements into compacted morton order
2. *Reduce* — apply the reduction operator across $2 \times \ldots \times 2$
   blocks for each level
3. *Dim0 fold* — accumulate into temporal reduction buffers
4. *Morton-to-tiles* — scatter reduced data into per-level tile regions

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
