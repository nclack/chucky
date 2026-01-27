# dev log

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
