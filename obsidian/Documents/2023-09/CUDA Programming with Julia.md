## The CUDA Programming Model

### Threads

- 可以将 thread 看做是一个分配了一个 sub-task 的 worker；
- thread 可以远多于 GPU 物理的 core 数量；
- CUDA 允许开发者不关注 parallel 执行的细节，只需要关注怎样将问题拆解；
### Kernels

- 普通的函数被调用时，只会执行一次，而在 CUDA 调用一个 Kernel 时，会通过 N 个 thread 调用 N 次；
- SIMT 模型：指定一组指令；每个线程针对自己的 sub-task 执行这组指令；

### Thread Hierarchy
#### Blocks

- thread 往往需要协同操作；
- 在 CUDA  中，往往不会直接 “launch 1,048,576 threads”；而是将这些 thread 组织为 blocks，比如“launch 4,096 blocks, with 256 threads each”；
- 在同一个 block 中的线程，拥有“shared memory”，可以相互之间传递信息；
- 不同的 block 完全独立执行，不能感知其他 block 的执行顺序；
	- 这允许 GPU 能够自由调度 Block 的执行，不用担心产生副作用；
	- 允许开发者只关心 block 内部的并发协同；
- 同一个  block 中的线程会使用同一个 SM（Streaming Multiprocessor，一组 64 个 CUDA core）；
	- 这形成了一个约束：每个 block 不要超过 1024 个 threads；

#### Grid

- The grid refers to all the blocks scheduled to run for a problem.
## Writing a Kernel


![[Pasted image 20230929213623.png]]