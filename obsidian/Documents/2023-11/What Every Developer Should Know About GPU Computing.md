
![[Pasted image 20231118225834.png]]
## GPU Compute Architecture

GPU 由多个 Streaming Multiprocessor 组成。

每个 SM 包含多个 Stream Processors 或者核心。比如 H100 有 132 个 SM，每个 SM 有 64 个核心，总共 8448 个核心。

每个 SM 只有很小的 on-chip memory，作为 shared memory 为多个核心共享。

每个 SM 也有多种功能组件，或其他加速单元，比如 tensor cores、ray tracing units。

## GPU Memory Architecture

![[Pasted image 20231119101352.png]]

## Understanding the Execution Model of the GPU

要想执行 kernel ，需要 launch 一组线程，这称作 grid。

一个 grid 会包括多个 blocks，然后一个 block 中包含多个 threads。

写一个 kernel 需要两部分，一部分是 CPU 上执行的 host part，另一部分是 GPU 的 device part。

## Steps Behind Execution of a Kernel on the GPU

### 将 Thread blocks 调度到 SM

block 中的所有的线程都是在 SM 中同时处理的；

实践中，每个 SM 一般会被分配多个 Block；

因为较大的 kernel 需要非常多的 blocks，并不能做到每个 block 都得到同时处理，GPU 会维护一个 waitlist 做调度；
### Single Instruction Multiple Threads (SIMT) and Warps

block 中的所有 thread 都会分配给同一个 SM。

<mark>block 中的这些 thread 会进一步拆分为 32 的 sub group，称作 warp</mark>。

The SM executes all the threads within a warp together by fetching and issuing the same instruction to all of them. These threads then execute that instruction simultaneously, but on different parts of the data.

### Warp Scheduling and Latency Tolerance

some instructions take longer to complete, causing a warp to wait for the result. In such cases, the SM puts that waiting warp to sleep and starts executing another warp that doesn't need to wait for anything. This enables the GPUs to maximally utilize all the available compute and deliver high throughput.

**Zero-overhead Scheduling:** As each thread in each warp has its own set of registers, there is no overhead for the SM to switch from executing one warp to another.

## Resource Partitioning and the Concept of Occupancy

GPU 里有一个指标是 『_occupancy_』，表示一个 SM 被分配的 warp 达到它能处理的上限的比例。

为了达到最高的效率，我们一般希望 100% 的 occupancy，但是在实际中经常因为种种原因不容易实现。

SM 有固定的计算资源，包括寄存器、shared memory、thread block slot、thread slot 等。这些资源会在现成之间动态地分配。比如 H100，每个 SM 可以处理 32 个 block，64 个 warp，每个 block 最高 1024 个线程。

Now, let's look at an example to see how resource allocation can affect the occupancy of an SM. If we use a block size of 32 threads and need a total of 2048 threads, we'll have 64 of these blocks. However, each SM can only handle 32 blocks at once. So, even though the SM can run 2048 threads, it will only be running 1024 threads at a time, resulting in a 50% occupancy rate.

Similarly, each SM has 65536 registers. To execute 2048 threads simultaneously, each thread can have a maximum of 32 registers (65536/2048 = 32). If a kernel needs 64 registers per thread, we can only run 1024 threads per SM, again resulting in 50% occupancy.

## Summary

- A GPU consists of several streaming multiprocessors (SM), where each SM has several processing cores.
- There is an off chip global memory, which is a HBM or DRAM. It is far from the SMs on the chip and has high latency.
- There is an off chip L2 cache and an on chip L1 cache. These L1 and L2 caches operate similarly to how L1/L2 caches operate in CPUs.
- There is a small amount of configurable shared memory on each SM. This is shared between the cores. Typically, threads within a thread block load a piece of data into the shared memory and then reuse it instead of loading it again from global memory.
- Each SM has a large number of registers, which are partitioned between threads depending on their requirement. The Nvidia H100 has 65,536 registers per SM.
- To execute a kernel on the GPU, we launch a grid of threads. A grid consists of one or more thread blocks and each thread block consists of one or more threads.
- The GPU assigns one or more blocks for execution on an SM depending on resource availability. All threads of one block are assigned and executed on the same SM. This is for leveraging data locality and for synchronization between threads.
- The threads assigned to an SM are further grouped into sizes of 32, which is called a warp. All the threads within a warp execute the same instruction at the same time, but on different parts of the data (SIMT). (Although newer generations of GPUs also support independent thread scheduling.)
- The GPU performs dynamic resource partitioning between the threads based on each threads requirements and the limits of the SM. The programmer needs to carefully optimize code to ensure the highest level of SM occupancy during execution.

