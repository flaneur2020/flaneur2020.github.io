## WGMMA Instructions

H100 有一波新的指令是 “warp group matrix multiply accumulate” (`wgmma.mma_async` in PTX, or `HGMMA`/`IGMMA`/`QGMMA`/`BGMMA` in SASS)。

过去的 GPU 上使用 tensor core 主要通过 `wmma.mma.sync` 和 `mma.sync` 指令。

在这些指令下，SM 中的一个 quadrant 的 32 个线程的 warp 会同步地喂一把数据到 tensor core，等待结果。

`wgmma.mma_async` 指令的做法是，开 128 个线程，在 SM 的所有 quadrant 上，协作式地同步，直接针对 shared memory 异步地发起 matmul。

作者发现要完整发挥 H100 的性能，这部分指令时必不可少的。不用它，GPU 也就 63% 的利用率。

> we suspect this is because the tensor cores want a deep hardware pipeline to keep them fed, even from local resources.

不幸的是，这部分指令需要的 memory layout 非常复杂。

> The unswizzled shared memory layouts suffer from very poor coalescing, and so they require substantial additional bandwidth from L2. The swizzled memory layouts are flat-out incorrectly documented, which took considerable time for us to figure out.

（gpt-4o: "Unsizzled" 意味着数据没有经过 swizzling 处理。也就是说，数据在内存中的布局是线性的，按照自然顺序存储。）

> They’re also brittle, in that they appear to only work for specific matrix shapes and do not play well with other parts of the wgmma.mma_async instructions. For example, the hardware can transpose sub-matrices on its way to the tensor cores -- but only if the layout is not swizzled.

针对特定 shape 优化的内存布局，对另一个 shape 就不好使了。

## Shared memory

shared memory 大约一次 30 个 cycle。

> In previous work (like Flash Attention), we’ve focused more on the HBM-SRAM bottleneck. And indeed: this really used to be the bottleneck! But as HBM has gotten faster and the tensor cores continue to grow out of proportion with the rest of the chip, even relatively small latencies like those from shared memory have also become important to either remove or hide.

随着 H100 的算力提升，SRAM 的 cycle 也成问题了。

> Shared memory can be tricky to work with because it is “banked” into 32 separate stores of memory.

shared memory 将内存拆分为 bank。

不小心就会发生 bank conflict。

> The solution is to rearrange shared memory with various “swizzling” patterns so as to avoid these conflicts, but it is an important detail to get right.

解决就是做好 swizzling 模式。

> we have found it very valuable to avoid movement between registers and shared memory when possible, and otherwise to use the built-in hardware (wgmma and TMA instructions) to do data movement asynchronously when possible.

最终发现尽量禁烧 shared memory 与 register 之间的内存搬运，而是尽量使用硬件内置的 wgmma 和 TMA 指令来异步地搬运数据。

## Address Generation

H100 的 tensor core 和 memory 都很快，当涉及到复杂的内存访问模式（如交织或混排模式）时，生成内存地址的过程变得更加复杂。

英伟达搞了 TMA （Tensor Memory Accelerator）。TMA 可以指定一个 Tensor layout 告诉它拿 tensor 的 subtile，在 global 和 shared memory 之间移动，在执行结束后 barrier 同步一下。

> nvidia docs: TMA allows applications to transfer 1D and up to 5D tensors between global memory and shared memory, in both directions, as well as between the shared memory regions of different SMs in the same cluster

> This saves all of the address generation costs, and additionally makes it much easier to construct pipelines.

> We have found TMA to be, like wgmma.mma_async, completely indispensable in achieving the full potential of the H100.

类似 `wgmma.mma_async` 没有 TMA，不可能利用 H100 的性能。

> It saves register resources and instruction dispatches, and also has useful features such as the ability to perform reductions onto global memory asynchronously, too

## ThunderKittens

> It is meant to be as simple as possible, and contains four templated types:
>
> - Register tiles -- 2D tensors on the register file.
> - Register vectors -- 1D tensors on the register file.
> - Shared tiles -- 2D tensors in shared memory.
> - Shared vectors -- 1D tensors in shared memory.
>
> We also give operations to manipulate them, either at the warp level or at the level of a collaborative group of warps. Examples include:
>
> - Initializers -- zero out a shared vector, for example.
> - Unary ops, like exp
> - Binary ops, like mul
> - Row / column ops, like a row_sum

（看起来有点像 xtensor，做了一个 tensor 的抽象）

## Tiles Seem Like a Good Idea

> But ThunderKittens has good abstractions -- small tiles -- that match where both AI and hardware are going. ThunderKittens doesn’t support any dimension less than 16. But in our view, this doesn’t really matter, since the hardware doesn’t particularly want to, either. And we ask: if your matrix multiply is smaller than 16x16, are you sure what you’re doing is AI?

small tile 还是个好主意。

thunderkitten 的每个维度都必须大于 16。

> From a philosophical point of view, we think a frame shift is in order. A “register” certainly shouldn’t be a 32-bit word like on the CPUs of old. And a 1024-bit wide vector register, as CUDA uses, is certainly a step in the right direction.

小的 32 位寄存器应该不行了，1024 bit 宽的寄存器按说是更好的方向。

> But to us a “register” is a 16x16 tile of data. We think AI wants this -- after all this time, it’s still just matrix multiplies, reductions, and reshapes.

