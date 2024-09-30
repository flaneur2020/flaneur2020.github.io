https://github.com/hayden-donnelly/notes/blob/main/cuda_optimization.md


---

**Q: Latency Hiding 是怎样 work 的？**

warp 是调度的基本单位。派发 LD/ST 指令时，当前 warp 可能进入阻塞，SM 可以执行别的 warp。

**Q: 什么是 Occupancy ？**

如果线程在等待访问 global memory 卡着，而且一时半会没有可以调度的 warp，occupancy 就会降低

occupancy = 当前活跃的 warp / 当前 GPU 最大的 active warp 数

**Q：除了访问全局之外内存，还可能有哪些原因？**

- 一个 thread block 使用的寄存器较多，会限制每个 SM 同时跑的 block 数量；
- 共享内存：每个 SM 上的共享内存是有限的，如果需要大量共享内存，也会减少每个 SM 上跑的 block 数量
- 控制流不均匀：if else 等控制流，可能使一部分线程进入闲置
- 同步操作：`__sync_threads()` 同步期间，会有线程进入等待


---

## General Architecture

- SM 是 Streaming Multiprocessor 的缩写
- Kepler GPUs 每个 SM 有 192 个 SP units，这就是俗称的 "cores"，SP Unit 用来计算单精度的浮点计算
- Kepler GPU 每个 SM 有 64 个 DP，用来做双精度的浮点计算
- 每个 SM 有 warp schedulers，来发射指令
- 同一代的 GPU 的区别就在于 SM 的个数和内存的大小
- Pascal、Volta GPU 可以在两倍的 SP rate 上计算 F16（半精度）
- Compute capability 7.0 引进了 Tensor Cores，用来加速矩阵相乘
- SP、DP、Tensor Core Unit 可以同时使用

## Execution Model

- CUDA 程序面向由 thread 组成的 block，最后组成一个 grid
- 在硬件层面，线程由 scalar processor 执行，thread block 由 multiprocessors 执行，concurrent thread block 跑在一个单独的 SM 中，一个 kernel 按一个 grid 来执行
- Thread -> Thread Block -> Grid
- Scalar Processor -> Multi Processor (SM), Device
- Thread Block 终生都位于同一个 SM
- 每个 SM 由内置的 shared memory，Thread Block 中的线程都可以访问同一波 shared memory

## Warps

- 一个 warp 包含 32 个线程
- 一个 thread block 会拆分为 1 个或者多个 warp
- GPU 按 warp 发射指令，warp 中的每个 thread 在物理上是并行的

## Launch Configuration

- 一个线程，在指令的 operand 尚未 ready 时，会 stall
- 通过 switching threads，可以起到隐藏 latency 的效果
- 访问 全局内存的延时（Global Memory Latency）可能是最大的 latency 来源
- 要分派多少个 threads/ thread blocks 来执行呢？最好是多到能够帮助 hide latency 的程度

## GPU Latency Hiding

```
int idx = threadIdx.x + blockDim.x * blockIdx.x;
c[idx] = a[idx] * b[idx];
```

```
I0: LD R0, a[idx];      // Load a[idx] into R0
I1: LD R1, b[idx];      // Load b[idx] into R1
I2: MPY R2,R0,R1        // Multiply R0 and R1, and store the result in R2
```

- Objective is to keep the machine busy. Idle time means decreased performance.
- I0 和 I1 都是只读指令，因此可以同时发射给 warp
- I2 必须等待 I0 和 I1 执行完之后，才能执行
- 一个减少延时的策略是，在第一个 warp 中同时发射 I0 和 I1，然后给下一个 warp 发射 I0 和 I1，直到第一个 warp 就绪时，再向第一个 warp 发射 I2
- warp scheduler 能够在每个周期中，通过向非阻塞的 warp 发射指令来隐藏延时
- 使用更多的 warp、threads，可以更好地 hide latency
- 这部分 latency hiding 不需要使用 cache，但是 cache 也有主语减少从内存取数据的 latency
- Dual issue warp scheduler 能够在同一时刻 issue 两个独立的指令，来 hide latency
- 线程的数量和潜在的 global memory throughput 有一定相关性
- 使用更多的线程，有助于提高 global memory throughput
- 每个线程处理更大块的数据，也可能有主语最大化 global memory throughput
- 超量地最大化内存吞吐很重要，因为 global memory access 很慢

## What is Occupancy?

- Occupancy 是衡量 SM 中线程的实际负载 vs 理论最高负载的比值
- Occupancy 通常受寄存器数量、threads per thread block、shared memory usage 等因素的影响

## Summary

- 应当使用尽可能多的线程来 hide latency、最大化性能
- Lantency 不是唯一限制性能的因素
- Thread Block 应当是 warp size 的复数大小
- 指令面向整个 warp 发射的，每个 warp 有 32 个线程
- 如果 thread block 不是复数的 warp size，那么 warp 可能 occupied 但是有 inactive thread、闲置的 LD/ST Units
- 如果 thread block 是复数的 warp size，那么可以保证每个线程都用起来
- SM 可以至少并发地运行 16 个 thread blocks
- 每个 thread block 的线程数最大为 1024，要利用更多的线程，可以开更多的 thread blocks
- 特别大与特别小的 thead blocks 都不利于好的 occupancy
- 