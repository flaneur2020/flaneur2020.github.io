https://github.com/hayden-donnelly/notes/blob/main/cuda_optimization.md

## Memory Hierarchy Review

- 每个线程有自己的 local storage，也就是寄存器
- 寄存器通常是编译器管理的，很快
- 每个线程块内的线程可以访问同一组 shared memory
- L1 cache 和 shared memory 都位于 GPU Chip 里面，latency 都很低
- L2 cache 是 device 级的资源，所有对 global memory 的访问都先经过 L2
- global memory 可以为每个线程访问，也可以被 host 访问
- global memory 的 latency 比较高，需要数百个 cycle
- global device memory（DRAM）并不位于 GPU chip 上，L1 和 L2 cache 在 GPU chip 上
## GMEM (Global Memory) Operations

- 在 load 数据时，会先查下 L1 cache，再找找 L2 cache
- 从 L1/L2 cache 装载数据的最小单位是 128 byte 的 cache line
- Non-caching loads skip the L1 cache and go straight to L2, the granularity of this type of load is 32-bytes.
- Non-caching loads 会 invalidate L1 的 cache
- 有些时候 Non-caching load 反而更快
## Load Operation

- Memory 操作按 warp 进行批量发起（每次 32 threads）
- A "line" is a subdivision of a cache.
- 在 CUDA GPU 中，内存的 “unit of transaction” 是按 32 bytes 分段的
- 这意味着，访问内存时，读写的是 line/segements，不是独立的字节

## Caching Load

- gmem load 是按 32 bytes 分段的，意味着，如果一次访存的字节数小于 32 bytes，则 memory bus 的利用率就到不了 100%
- 如果一次访问 48 bytes，这样没对齐，就会导致两次访存，浪费了 16 bytes 的带宽
- 如果一次访问 32 个对齐的 4-bytes word，匹配一个 cache line，4 个 segment，这被称作「**perfect coalescing**」
- 设计将内存组织为 line 或者 segment 的做法，叫做「**coalescing**」
- A silver lining of this is that other warps may be able to use the previously unused data, and if this happens then the access will be much quicker because the data is already in cache.

## Non-Caching Load Revisited

- 从 gmem 做 non-caching loads 有时可以提高 bus utilization，因为不需要担心利用 128 bytes 的 cache line，只需要关心 32 bytes 的 segments

## GMEM Optimization Guidelines

- 追求 perfect coalescing，把 starting address 对齐到 segment boundaries，让 warp 去访问连续空间
- 安排足够的并发内存访问，加够线程数量
- 把所有的 cache 用起来（L1 和 L2 cache 是用户不能直接管理的，其他 cache 比如 constant cache、read-only cache 可以）

## Shared Memory

- 不是独立的 DRAM，就在 GPU 的 chip 中存储
- 每个 SM 内部的资源
- 可以用于跨线程通信
- 可以用于优化 global memory 的访问 pattern
- 可以把它看作一个 2 维数组
- 每行是 32 个 bank 组成，每列对应一个 4 bytes 的 bank
- 每个 cycle 可以从一个 bank 里面取一个 item
- shared memory 的访存也是按 warp 发的批量读取
- 从不同的 bank 中读取 4 个 byte 是并行的，但是从同一个 bank 读取 4 个不同的 byte，是顺序的
- 多个线程可以在同一个 bank 并行读取同一个 word，这称作「multicast」
- 为了最好的性能，最好避免**出现 bank conflict （多个线程读、写同一个 bank 中的不同 word）**
- **访问 gmem 时高效的 coalescing，到 smem 这里可能就成了 bank conflict**
- When there are at most 2 threads accessing the same bank, this is called a 2-way bank conflict and it runs 2x slower than the best case.
- When there are at most 16 threads accessing the same bank, this is called a 16-way bank conflict and it runs 16x slower than the best case.
- padding 可以有助于避免产生 bank conflict，把不同的列拆到不同的 bank 中