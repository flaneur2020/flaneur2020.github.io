https://marek.ai/matrix-multiplication-on-cpu.html

## Machine Config

- CPU: Intel® Core™ i7-7800X @ 3.50 GHz
- RAM: 32 GB DDR4 @ 2.4 GHz (G-Skill Ripjaws F4-2400C15-16GVR)
- Motherboard: ASUS PRIME X299-A motherboard
- OS: Ubuntu 16.04.6 LTS (Xenial Xerus)
- C++ Compiler: GCC 5.4.0 (also others for comparison, if explicitly stated)
- Compiler flags: `-O3 -march=native -funroll-loops` (unless specified otherwise)

## Naive iterations

```C++
void naiveIterativeMatmul(
        float* const A,
        float* const B,
        float* const C,
        const int M,
        const int N,
        const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C[m * M  + n] += A[m * M + k] * B[k * K + n];
            }
        }
    }
}
```

上面的代码特别慢，开 O3 和 -march=native 也得 1601ms。好的实现能够将 1024x1024 的时间控制在 40ms 以下。

## A few words about compiler optimization flags

> To make libraries highly optimized yet portable, we can generate several shared library files, one for each architecture, and then `dlopen` the one that matches the runtime CPUID information.

## Optimizing the naive iterative algorithm

```C++
void iterativeMatmulRegisterSum(
        float* const A,
        float* const B,
        float* const C,
        const int M,
        const int N,
        const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * M + k] * B[k * K + n];
            }
            C[m * M  + n] = sum;
        }
    }
}
```

相比于使用 `+=` 来更新 `C[m * M + n]`，改成一个 float 变量，这样编译器更容易将它看作一个 register。

这个版本有 1070ms，相比于上一个版本 67% 的时间。

> You might ask the question - why didn't the compiler perform this optimization itself? It turns out that it's an illegal optimization to make, considering that the compiler can't understand the semantics of the algorithm. All it knows is that it had to preserve the program semantics from the perspective of the memory model.

> However, when the compiler generates the object file, it has no idea of who the users of this function would be. The data pointers are provided as arguments to this function, not allocated by the function itself. Therefore, it's conceivable that other threads might want to access the memory from the buffer represented by matrix C� pointer's at the same time. While this doesn't make sense for this particular algorithm, there are other algorithms which multi-thread data reads and writes, and in such cases the lack of RAM visibility of the intermediate result would be a problem.

编译器为什么不自动做这个优化？是因为编译器不懂算法的语义，只能保留内存的访问模式不变。

对于 row-major 的语言比如 C++，把 k 放到最内部的循环会有一个问题，就是对于 A 矩阵，k 代表列，对 B 矩阵，k 代表行。

对于大的矩阵来讲，我们有可能对每次访问 B 矩阵时都 cache miss。

对于（K x N） B 矩阵来讲，n 对应 B 的列。让 n 下标更频繁地在内侧循环，有助于缓存。

```C++
void iterativeMatmulLoopReorder(
        float* const A,
        float* const B,
        float* const C,
        const int M,
        const int N,
        const int K) {

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * M  + n] += A[m * M + k] * B[k * K + n];
            }
        }
    }
}
```

这个版本只需要 63ms！

这个技术称作 loop reordering 或者 loop interchange。

## Static dimensions

```C++
template <int M, int N, int K>
void iterativeMatmulLoopReorderTempl(
        float* const A,
        float* const B,
        float* const C) {

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * M  + n] += A[m * M + k] * B[k * K + n];
            }
        }
    }
}
```

使用 M、N、K 变成静态，这个函数只需要 53ms。

然而 numpy 跑同样的循环只需要 23ms。因为它内部使用了 intel 的 MKL 库。

## Cache behavior depending on matrix sizes

假设 cache 有 M bytes，而每个 cache line 有 B bytes，可知总共有 $\frac{M}{B}$ 个 cache line。

对于 matrix size，可以假设 M=N=K，统称为 N。

假设数据的精度为 P，比如单精度对应 4 个 bytes，这时 P = 4。
### case 1: $N > \frac{M}{BP}$

在遍历 B 矩阵的 k 下标时，每次都会遭遇 cache miss。

即使 A 完整地在 cache 中，所有的计算都需要等待从 B 中读取数据。

因为有 $O(N^3)$ 次乘法计算，cache miss 的数量也是 $O(N^3)$。

### case 2: $\frac{\sqrt{M}}{P} < N < c \frac{M}{BP} \space for 1 < c < 0$

在这个 case 下，只有在 matrix 下标每访问 BP 个 bytes 时才会有一次 cache miss。

B << M，这意味着 cache miss 的数量是 $O(\frac{N^3}{BP})$
### case 3: $\frac{\sqrt{M}}{P} < N$

在这个 case 下，所有操作都在 cache 中，N * N 小于 M。

A 和 B 都完整地在 cache 中，理想的情况是，A 和 B 两个 matrix 分别各占 cache 一半，这时有最优的表现。

## More on hardware caches

|Device|Access Time (ns)|CPU cycles|
|---|---|---|
|register|0.4|1|
|L1 cache|0.9|2.25|
|L2 cache|2.8|7|
|L3 cache|28|70|
|DRAM|100|250|
|NVMe SSD|25,000|62,500|
|SATA SSD|100,000|250,000|

cache 在现实世界中并不完全理想，它有一些 heuristic 来做 eviction，也并不是全相连。

但是假设它是完美的，可以简化程序的分析。

一个理想的 cache：

- 是全相连的
- 有一个全知全能的 eviction policy

The full associativity assumption is also true up to a point, e.g. L1 caches in recent Intel Core i7 CPUs are 8-way set associative, etc.

此外，我们也会做一个 tall cache 的假设，是说每个 cache line 的 bytes 远小于 cache line 的数量。

> Ideally, we'd want the cache line to be small, since then we could store small chunks from various locations in memory. However, the cache line has to be big enough to make contiguous reads efficient, by filling up the memory bus when the read is done. We will see more of that in the case of GPUs.

> However, the cache line has to be big enough to make contiguous reads efficient, by filling up the memory bus when the read is done

> Given all the above, let's assume either an omniscient or LRU cache, as convenient, that is fully or nearly fully associative, with a tall cache assumption.

基于上述分析，可以假设它是一个全相连的 LRU cache、是一个 tall cache。
## Iterative Algorithm with Tiling
