# dev log

## 2026-02-16

Adding multiscale. Starting with mean.

Also pulling out common error handling macros and trimming some of the old
unused kernels/tests that I added to when I was just trying to establish the
repo.

Downsampling kernel turned out pretty complicated. Seems like there's got to
be a way to simplify.

Each epoch we yield chunks across potentially several scales. It would be nice
if these could be handled uniformly to downstream compress/aggregation steps.
At that point, we're mostly doing the same things to the tiles, we just need
some more complicated shard addressing.


## 2026-02-15

Testing on 5090 (oreb). New data distribution hits pretty hard :(

```
=== test_bench_zarr ===
  output: .\build\test_output.zarr
  shape:       (4264, 2048, 2048, 3)  tiles: (32, 128, 128, 1)
  total:       99.94 GiB (53653536768 elements, 134 epochs)
  tile:        524288 elements = 1024 KiB  (stride=524288)
  epoch:       768 slots, 768 MiB pool
  compress:    max_chunk=1048636 comp_pool=768 MiB

  --- Benchmark Results ---
  Input:        99.94 GiB (53653536768 elements)
  Compressed:   40.86 GiB (ratio: 0.407)
  Tiles:        102912 (768/epoch x 134 epochs)

  Stage        avg GB/s best GB/s     avg ms    best ms
  Source           2.18    22.00      28.70       2.84
  H2D             34.66    53.90       1.81       1.16
  Scatter         75.03   119.36       0.83       0.52
  Compress         3.74     8.19     200.53      91.57
  Aggregate       41.38    50.66      18.13      14.81
  D2H             45.15    53.45      16.61      14.03
  Sink             3.28  2976.18       5.44       0.01

  Wall time:     46.107 s
  Throughput:    2.17 GiB/s
```

Trying to see if I can view the zarr in neuroglancer.

It's not quite as simple as

```powershell
uv run python -m http.server 8080
```

The server also needs to set CORS headers, and use the `RangeHTTPServer`.

After that, you can open [neuroglancer] and set source to  `zarr://http://
localhost:8080/test_output.zarr/0`. You have to point it at an array.

I added a "visual" option to `test_bench` that outputs something a little
easier to look at, and confirmed it works in napari.

Next is probably to start thinking about multiscale. 

I'm still a little annoyed by the performance. The compression is pretty heavily
the bottleneck.

Found another unneccessary sync in the compression pipeline. Removing that 
and putting compress+downstream on another stream got it to 2.9 GB/s (without
the write to disk).

[neuroglancer]: https://neuroglancer-demo.appspot.com/

## 2026-02-14

### Profiling

Trying to look at nsight on the 5090 (oreb).

I noticed I was doing 2-byte loads for the transpose. Changed to 4-bytes.
Can do larger, but that just hits more cache lines and uses more registers.

```
           GB/s   avg     best   
Scatter         78.09   115.40 -- Before
Scatter         78.28   115.94 -- Before
Scatter         79.50   118.56 -- After
Scatter         79.69   119.00 -- After
Scatter         69.62   118.87 -- After
```

Throughput by wall time is 12 GB/s.

So faster but not a lot faster. nsight predictably complains about non-coalesced
stores.

### Sharding

I'll want to write a cuda kernel to aggregate the compressed chunks by shard. My
guess is that will look like a segmented scan. This will necessrily generate the
list of offsets for each chunk. The chunks will get reordered to group them by
shard. That reorder is conceptually the same kind of lifting/transpose that was
done to scatter the input  into the chunks. This time though, we're transposing
the logical tiles into something tiled by the `tiles_by_shard`.

I'm thinking that array of offsets + the aggregated bytes are enough to transmit
back to the host. The host side will have to do some index operations to map
each chunk to a shard and figure out how to update the chunk index for that
shard.

We'll probably want to keep the index for each shard in memory. That'll help
with the crc32 checksum (which just checksums the index). Shards will complete
in layers/epochs, just like tiles, so we can limit the amount of memory
required.

Added the kernel for aggregating the compressed chunks into shards. It's a
three step process where we permute the compressed sizes, do a scan to compute
offsets and then a gather to copy the compressed chunks into shard order.

The aggregated compressed bytes and the offsets buffer then get copied back
to host. The host side figures out how to dispatch those ranges to shards, and
maintain the shard indices. I abstract the shard writer for now.

Wrote a test that uses a coordinate encoding to:
- verify the right chunks are getting copied into the right places (and values)
- verify the chunk index for the shard looks right

Now I think it's time to do the zarr writing. This is tricky because it needs a
few things all at once.

1. Need to write json
2. Need to create the directory structure, probably epoch by epoch. That
   should use a background thread, a job queue, and some barrier.
3. Need to implement the shard store.

Ok, most of that is done. Need to review in the morning, but it looks like
things are working.

Changed the input distribution on the benchmark to be:
- 1/3 12-bit gaussian
- 1/3 uniformly random
- 1/3 constant

On auk (laptop, 5070)

```
Stage        avg GB/s best GB/s     avg ms    best ms
Source           1.78    20.69      35.08       3.02
H2D              9.86    13.47       6.35       4.64
Scatter         26.08    48.16       2.40       1.30
Compress         2.59     3.40     290.05     220.35
Aggregate       11.98    12.96      62.58      57.86
D2H             12.46    13.32      60.21      56.30
Sink             3.97  5867.83       3.24       0.00

Wall time:     56.510 s
Throughput:    1.77 GiB/s  
```

### Thoughts

It occurred to me that if the tile shape has 1's in the larger dimensions we
can yield epochs more quickly - should check on that.

Are we waiting until the first epoch needs to flush to alloc the codec ctx.

## 2026-02-13

Thinking about aggregating by shard.

Could scatter to shard, but shards can be sparse and shouldn't be zero padded.
So I don't think this is the right approach.

In the end, we'd like to have 1 write of a contiguous set of bytes to update
a shard, and then we'd like to do 1 write to update that shards index, which
uses an implicitly indexed `(offset, length)` format with a checksum over the
indices.

Logically, sharding is another lifting from an array shape, though because
shards are sparse, the shape we're talking about can be much larger than the
shape of the input array - it must cover though.

If we specify

```c
struct dimension
{
  uint64_t size;
  uint64_t tile_size;
  uint64_t tiles_per_shard; // new
};
```

The the size of the covering array is `tile_size*tiles_per_shard` along a
dimension. Let's call that $e_d$. $e_d$ can be factored into $s_d * t_d * n_d$
where $s_d$ is the shard index, $t_d$ is the tile index, and $n_d$ is the
pixel index. Coordinates in the covering array are lifted:

$r_d = (s_{d-1},t_{d-1},n_{d-1},...,s_0,t_0,n_0)$

and (virtually) transposed

$r'_d = (s_{d-1},...,s_0,t_{d-1},...,t_0,n_{d-1}...n_0)$

Focusing on the output side, I just need to prepare the data for updating
shards following the [zarr sharding codec].

[zarr sharding codec]: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html 

## 2026-02-12

Not a lot of memory on auk (8GB). Kind of forces me to thin chunks so the
epochs aren't too thick.

Trying to make sure the timings make sense....

Overall wall tie looking better - 6.3 GB/s on auk.

Could think about file io. Probably need to do coalescing across chunks first.
Should think about sharding. Might be a kv abstraction there.

There's also the question of multiscale. Getting file io first might clarify.

I think my todo list at this point is:

0. clean up
1. try to get async compression working again
2. aggregate by shard? compaction.
3. ...

Got through a fair amount of clean up.

Looks like I can remove the excessive sync point now. Still has to sync before
the d2h, of course.

## 2026-02-11

Looked back over streaming code and marked some todo's etc. One thing is that
I haven't really thought cleanly about draining data out and how that affects
the calling thread.

Also reviewing the benchmark...
Changing the chunk size and front side buffer. Front side buffer was small.

Need to keep adjusting and double check things make sense. Still not quite
happy with the chunk size - would be nice to get more epochs through for
stats on timings.

## 2026-02-09

Getting the build to work on windows.

On my 5090:

```
--- Benchmark Results ---
Input:        49.97 GiB (26826768384 elements)
Compressed:   16.07 GiB (ratio: 0.322)
Tiles:        51168 (96/epoch x 533 epochs)

GPU step     avg GB/s best GB/s     avg ms    best ms
H2D             33.69        -       2.78          -
scatter        834.68  1202.66       0.11       0.08
compress         6.42    15.39      14.61       6.09
D2H             47.64    53.36       1.97       1.76

Source fill:   3.31 GB/s  (15099.5 ms total)
Sink:          11319.2 ms total  (11 epochs verified)

Wall time:     37.555 s
Throughput:    1.33 GiB/s
```

## 2026-02-08

Refactor to rely on the writer interface more. Decide on some compositional
boundaries. Might be some useful writers for testing. Think about getting the
compression in. I'd like to refactor the platform specific code (timers) as
well.

Added double buffering on the host input. Needed to also setup the wait for
the host side buffer to clear.

Added compression. Compressed tiles are not compacted at the moment.

I think I'd need to instrument before deciding if compaction is worth it.

The "tile_writer" interface is awkward. It looks like a writev, but doesn't
follow the same symmantics as the "writer interface"; it has to consume all
input.

I don't like the way the different code paths for w/wo compression are handled.
Would be nice to find a more composable thing. Need to clean up the code.

Have to do a hard sync on the compute stream to properly finish the compression
pass - doesn't seem right - need to look into that more.

At least on my laptop (auk) throughput is decent through the kernels. The
wall tije is bad though - 1 GB/s end to end. Table below:


```
  Benchmark Results (50 GiB, mixed data)

  GPU step     avg GB/s best GB/s     avg ms    best ms
  H2D             12.18        -       7.70          -
  scatter        341.81   464.37       0.27       0.20
  compress         3.71     8.91      25.29      10.52
  D2H             13.08    13.31       7.17       7.04
```

## 2026-02-07

I'm trying to get the simplest end-to-end pipeline I can for a streaming
transpose of multidimensional arrays/tensors. I think for streaming, I'll have
to introduce the concept of tiling.

The reason for that is that I need to define when to stream things back to the
host. Tile's make sense, and that's what I want to work toward anyway. Input
data comes in, gets scattered to different tiles, and then when a tile is filled
up, we should transfer it back. It'll probably be the case that many tiles get
filled up at once.

When we push some buffer to the transpose it represents some [beg,end) range of
input indices. These map to an ordered set of coordinates r(beg)..r(end) where
r=(t_{d-1},n_{d-1},...,t_0,n_0). If we just consider the tile indices, we can
write r_t=(t_{d-1},...,t_1,t_0). The sequence r_t(beg)...r_t(end) corresponds
to an ordered set of tiles - with tile indices corresponding to the tile-only
stride projection in the row-major input space.

The input sequence beg..end will map to a corresponding sequence of tile
indices beg_t..end_t. The sequence of tile indices is not strictly increasing.
That's because a carry in one of the n_i dimensions, will carry into t_i and
that in turn might induce a carry (to n_i+1), causing t_i to reset to 0
(decreasing the tile index).

But we should be able to track the min and max tile indices that might be hit
by [beg,end). Let's say u_t is the index of the first element of the input to
hit tile t, and v_t is the last. If beg>v_t, then we know t won't be hit again
and can flush the tile.

The max t for which v_t < beg should be strictly increasing. Similarly the min
t for which u_t < end should be strictly increasing. I'm thinking we'll need to
maintain a ring-buffer of tiles. Tiles finish in the same order they start, so
we can think about a fifo.

The input partitions in to non-overlapping epochs - these are "layers" of tiles
formed from all the dimensions except $$t_{D-1}$$. This gives the size of
fifo. It needs to hold that layer of tiles.

Actually, we need to keep the layer around anyway for compression, which means
we don't really need to think of this as a ring-buffer type of fifo. We just
need to make sure the append() call enforcese the split across layers. Will
need to rezero layers for padding tiles.


## 2026-01-25

Realized I've been making all of this too complicated. The vadd() algorithm
is nice off of the gpu, or when you have a big buffer of indices you write to.
It ends up collapsing the complexity of the loop over dimensions into what
would normally be the loop over elements, so you end up with something O(n)
rather than O(n x d).

But, we're not doing that on the gpu, so the simple unravel-and-accumulate
strides algorithm works. It's O(d) per thread.

Anyway, next step is to actually do the transpose without thinking too much
about memory coherence.

## 2026-01-24

Ok, it turned out to be very easy. I just need to stick to the mixed-radix
addition algorithm.

Added avx2 versions of `vadd` and `vadd2` to make sure I understood how
the vectorization works. The approach in `vadd2` is probably better forev
vectorization and it's simpler to understand, so I used this for both
of the vectorized implementations. They're not very fast - the non-vectorized
`vadd` is very fast. But that's not the point here. 

## 2026-01-24

Trying to see if I can get non-unit step sizes to work. The idea is that the
step should translate into a $\delta r$ in the input array. Each step we're
adding $\delta r$ in the input array space with carries. To compute the output
indices we just keep track of the effect of those carries, just like we did for
the unit-step case.

At the end of the day, I feel like I'm close. I'm stuck on how to handle
carries. I think this probably ends up being another scan.

## 2026-01-23

Working out the output index computation that needs to happen in the kernel.
I'm doing this by prototyping in `tests/index.op.c`.

The easy way to compute the output index is to just take the input index,
unravel it into a coordinate vector, and then project that vector with the
output strides.

However, in the kernel each thread would need O(rank) space and O(rank) time.

The idea is to use a prefix sum approach instead. If we track the change in the
output index for each step along the input index, then we just have to watch
for when the input index reaches a boundary in the array shape. When those
boundaries are hit, we can look up the effect in the output strides and correct
the steps. To get the final output index we just do a cumulative sum over the
deltas. It's possible to do this without storing a coordinate array.

I've tested it for unit step sizes of the input index. I think there's a way
to get that approach to work for larger steps. That's important because we
have to know the output offsets where each warp starts.

## 2026-01-20

Got a simple kernel running in test with a basic cmake-based build. The
kernel just fills a buffer and then it gets streamed back. D2H transfers
happen at around 13 GB/s. This machine has a 4x PCIe Gen4 link to the GPU.
I believe that's a 16 GB/s theoretical max (~83%).

Next is to actually try to write the transpose. I should just try the basic
thing without trying to optimally fill output lines.

What do I want to do first?

The input is going to be a 1GB buffers that contain samples starting at some
offset relative to the beginning of a contiguous array. 

We'll process the data in 32x32 warps. Each warp is responsible for loading
a contiguous 128 bytes (1024 bits) of data: 32 lanes/warp * 4 bytes per lane.
Depending on the sample type, there will be from 32-128 samples per warp.
Let's call that number $B$: the number of samples handled by a warp.

Note that the data we load from warp-to-warp doesn't have to be contiguous.
To keep things simple, we won't worry about that right now: warp $i$ should
load elements from $i*B$ to $(i+1)*B-1$.

Note that at this point we haven't used any information about the shape of
the array.

Then we need to compute where there should go.
