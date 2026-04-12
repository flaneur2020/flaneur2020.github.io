  
At the highest level, a GPU performs two essential tasks:

1. Move and store data (the memory system)
2. Do useful work with the data (the compute pipelines)

TMA 可以用来传输 global memory 和 shared memory 之间异步传输数据。

它也支持通过 swizzling 来减少 bank conflicts。

## Compute

H100 总共有 132 个 Stream Processor。每个 GPC 包含 18 个 SM。

每个 SM 也包括：

- Tensor Cores：特化的、面向小 tile （64x16 @ 16x256）执行 matrix 乘法的单元；大的矩阵乘法，会拆分成上面这些小的 tile 操作，使用这部分指令对性能至关重要；
- CUDA Cores：CUDA Core 是一个 marketing 术语，主要用于计算标准的浮点运算，比如 FMA；
- SFU（Special Function Units）：用于处理一些常用的函数，比如 sin、cos、exp、log、乃至 sqrt 等等；
- Load/Store Units：用于装载、保存指令，包括 TMA；
- Warp scheduler：每个 SM 有四个 Warp scheduler；每个 SM 会发射 32 个线程组成的 warp；每个 cycle 执行一个 warp instruction。

## Speed of light and power throttling

最高的性能约等于 maximum clock frequency × number of tensor cores × FLOPs per tensor core per cycle.

## CUDA programming model

抽象主要包括：

1. thread
2. warp（32 个 thread）
3. thread block
4. thread block cluster
5. grid（of thread blocks or clusters）

Every thread is "aware" of its position in the CUDA hierarchy through variables such as `gridDim`, `blockIdx`, `blockDim`, and `threadIdx`.

Connecting the CUDA model back to the hardware, one fact should now be clear: **a thread block should contain at least 4 warps (i.e., 128 threads).**

- 一个 thread block 会活在一个单独的 SM 上；
- 每个 SM 有 4 个 warp scheduler，如果不满 4 个 warp，硬件就闲置了；

Hopper 中，warp group（4 个 warp）是 WGMMA （matmul）tensor core 指令的基本单位。

## GMEM Model

So when people say “GMEM coalescing is very important”, this is what they mean: threads should access contiguous memory locations to minimize the number of DRAM rows touched.

## SMEM Model

Shared memory (SMEM) has **very** different properties from GMEM. It is built from SRAM cells rather than DRAM, which gives it fundamentally different speed and capacity trade-offs.

SMEM is organized into 32 banks, each bank 32 bits wide (4 bytes).

SMEM can serve data from all 32 banks (128B) in a single cycle — but only if one rule is respected:

**Threads in a warp must not access different addresses within the same bank. Otherwise, those requests are serialized across multiple cycles.**

warp 中的不同线程，不能访问同一个 bank 的不同地址（可以访问同一个地址）。

Importantly: if multiple threads in a warp access the same address within a bank, SMEM can broadcast (or multicast) that value to all of them.

## L1 Model

L1 和 SMEM 使用的同样的物理元件。

