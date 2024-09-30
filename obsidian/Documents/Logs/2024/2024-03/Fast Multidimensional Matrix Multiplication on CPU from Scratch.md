

这是非常高的性能，这做到了 18FLOPS / core / cycle，每个 cycle 大约三分之一 ns。

numpy 内部使用了经过高度优化的 BLAS 实现，使用裸的 C++ 能否复现这个性能表现？
## Calculating total FLOPs

Matmul 总的计算量有： N(=rows) * N(=columns) * N(=dot product) * 2(mul + add) = 2N³ FLOPs.
### How can a single core do 18 FLOPs in a cycle?

numpy 使用了 intel 的 MKL 实现的 BLAS。

在代码中，有多种不同的计算核，比如 sgemm_kernel_HASWELL、sgemm_kernel_SANDYBRIDGE。

在运行时，blas 库会根据 cpuid 来选择出对应的 kernel。

在 sgemm_kernel_HASWELL 中，性能来自 vectorized FMA 指令，也就是 VFMADD 指令。

它能操作 3 个 256 bit 的 YMM 寄存器，(YMM1 * YMM2) + YMM3 并将结果保存到 YMM3。

这允许 CPU 能够在一个指令周期中，跑 16 个单精度 FLOPS。

VFMADD 可以最小只消耗 0.5 cycle。因此理论上可以同时跑 2 个 VFMADD，做到 32FLOPS/ cycle。

在内存访问延迟在 5 cycle 时，需要找出来 10 * 16 个 FLOPS 运算。

这需要我们将独立的 CPU 操作调度分组，来利用 CPU 的指令级并行能力。（Grouping enough independent operations such that the CPU can schedule all of them at once, fully exploiting its instruction-level parallelism (ILP) capabilities.）

intel 的 BLAS 可以做到 18 FLOPS/core/cycle，其实理论上还跑的更高，因为通过 numpy 调用它，还需要考虑到 python 的开销和 OpenMP 线程池的开销。

> Note how even though the matrices aren’t that big, we’re strongly compute bound already. Loading 8MB from RAM takes maybe 200μs, assuming a memory bandwidth of 40GB/sFor multi-threaded, SIMD memory accessing. Numbers from the excellent [napkin math](https://github.com/sirupsen/napkin-math) project.. If the matrices get bigger, we become more compute-bound, since we’re performing 2n³ FLOPs for 2n² loads.
## Trying to recreate this performance from scratch

作者最终做出来一个 9 FLOPS / core / cycle 的实现。

> I’m running this on a quadcore Intel i7-6700 CPU @ 3.40GHz, on a dedicated server. It has 32KiB per-core L1d cache, 256KiB of per-core L2 cache, and a shared 8MB L3 cache.

作者的机器是一个 4 c 的 i7-6700，有 3.40 GHZ，有 32KiB 的 L1d cache、256 KiB 的 L2 cache、8MB 的 L3 Cache

几个 benchmark 的输出：

|                                                             |        |
| ----------------------------------------------------------- | ------ |
| Naive Implementation (RCI)                                  | `4481` |
| Naive Implementation (RCI) + compiler flags                 | 1621   |
| Naive Implementation (RCI) + flags + register accumulate    | 1512   |
| Cache-aware loop reorder (RIC)                              | 89     |
| Loop reorder (RIC) + L1 tiling on I                         | 70     |
| Loop reorder (RIC) + L1 tiling on I + multithreading on R&C | 16     |
| Numpy (MKL)                                                 | 8      |
### Naive implementation

```
template <int rows, int columns, int inners>
inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }
```

在这里作者用模板硬编码了 matrix 的维度数，更易于编译器做优化。

和 BLAS 相比并不公平，因为 BLAS 支持任意的维度数。

在现实中，BLAS 通常会对不同的 size range 的 matrix 相乘有特化的实现。

此外，编译时开启 -O3 和  `-march=native` 可以使编译器尽可能多地利用本地指令，牺牲可移植性换取性能。

```
template <int rows, int columns, int inners>
inline void matmulImplNaiveRegisterAcc(const float *left, const float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        acc += left[row * columns + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
} } }
```

一个比较小的优化是使用一个寄存器来保存累加的中间结果。
### Cache-aware implementation

![[Pasted image 20240314091700.png]]

```
template <int rows, int columns, int inners>
inline void matmulImplLoopOrder(const float *left, const float *right,
                                float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }
```

调整一下 inner loop 的顺序，对这个 cache 友好大大提升，优化到了 89ms。

![[cache-aware-dot-prod-reorder-loops.png]]

看下编译出来的代码，会发现它已经被 编译器优化打开了向量化。

```assembly
; In the loop setup, load a single fp32 from the current A row
; and broadcast it to all 8 entries of the ymm0 register
; vbroadcastss ymm0, dword ptr [rsi + 4*r8]

; In each instruction, load 8 entries from 
; the current row of B into a ymm register
vmovups ymm1, ymmword ptr [rbx + 4*rbp - 96]
vmovups ymm2, ymmword ptr [rbx + 4*rbp - 64]
vmovups ymm3, ymmw
vmovups ymm4, ymmword ptr [rbx + 4*rbp]
; In each instruction, multipy the current entry of A (ymm0) times 
; the entries of C (ymm1-4) and add partial results from C (memory load) 
vfmadd213ps ymm1, ymm0, ymmword ptr [rcx + 4*rbp - 96] ; ymm1 = (ymm0 * ymm1) + mem
vfmadd213ps ymm2, ymm0, ymmword ptr [rcx + 4*rbp - 64] ; ymm2 = (ymm0 * ymm2) + mem
vfmadd213ps ymm3, ymm0, ymmword ptr [rcx + 4*rbp - 32] ; ymm3 = (ymm0 * ymm3) + mem
vfmadd213ps ymm4, ymm0, ymmword ptr [rcx + 4*rbp] ; ymm4 = (ymm0 * ymm4) + mem
; Store the partial results back to C's memory
vmovups ymmword ptr [rcx + 4*rbp - 96], ymm1
vmovups ymmwor
vmovups ymmword ptr [rcx + 4*rbp - 32], ymm3
vmovups ymmword ptr [rcx + 4*rbp], ymm4
```

### Tiling

![[Pasted image 20240315114325.png]]

```
template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right,
                             float *result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
} } } } }
```

在现实中，找出来一个合适的 tile size 并不那么容易，因为 cache 很少 fully associative。

此外，操作系统做 context switching 可能污染 cache。

在理论上，中间的循环包括：

- 1024 sliced rows of A: 1024 * `TILESIZE` * 4B
- `TILESIZE` rows of B: `TILESIZE` * 1024 * 4B
- 1 row of C: 1024 * 4B

L1d cache 有 32kb，理论上 <mark>最好的 tile size 是 ～3.5</mark>。

在 tile size 为 16 时有最好的表现，时间降到了 70ms。

> The optimal tile size will also be influenced by the loop overhead, and by how well the prefetcher can predict our memory accesses. Searching through many different combinations to find one that fits the microarchitecture well is common practice.

一般最佳实战是，针对不同的微架构做不同的尝试。
### Tiling on multiple dimensions

除了针对 inner dimensions，也可以针对 rows 做 tiling。

![[Pasted image 20240315114832.png]]
在这个小 matrix 相乘的 case 种，这个 tiling 方法收益不大，但是针对更大的 matrix 有更高的收益。

### Multithreaded matrix multiplication

最后一步，通过 OpenMP 开启多线程。

![[Pasted image 20240315114924.png]]

最终版本可以降到 16ms；

