作者写了 84 个 kernel，在 prompt eval 性能上能超过 llama.cpp 很多。

在 skylake 架构上，llamafile 的用户的性能有 llama.cpp 的两倍性能。

作者只写了 q8_0, f16, q4_1, q4_0, f32 的 kernel。

> That's because my new kernels change the rules. They're doing such a good job fixing the memory bandwidth quants always solved, that quantization could become the bigger bottleck.

在经过量化之后，量化的计算过程本身成为了更大的瓶颈？


[https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp](https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp)

## Technical Details

> llama.cpp had the important insight that less is more when it comes to linear algebra. The alpha and beta parameters are never used, so they're always set to to 1 and 0

此外，LLM 的 op graph 中的 A matrix 往往是 transpose 过的，而 B matrix 从不 transpose。

这意味着 inner dimension 的 dot product 总是可以面向连续内存来向量化处理。

m/k dimension 往往都可以整除 64。

> BLAS libraries usually hurt more than they help for matrix-vector multiplication, because it's so computationally simple by comparison.

Blas 库在这个时候往往更吃亏。

> Matrix vector multiplication is an operation where latency (not throughput) is the bottleneck, and the bloat of fancy libraries has a measurable impact

GEMV 操作更关注延时而不是吞吐。

llama.cpp 做了类似下面的这个计算核，能跑到 233 gigaflops（反观 BLAS 跑到 47 gigaflops）。

```C++
template <typename T>
void LLMM(int m, int n, int k,
          const T *A, int lda,
          const T *B, int ldb,
          T *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            T d = 0;
            for (int l = 0; l < k; ++l)
                d += A[lda * i + l] * B[ldb * j + l];
            C[ldc * j + i] = d;
        }
}
```

llama.cpp 的 token 生成 speed 已经比较优化了，然而它的 prompt 处理速度不是特别优。

prompt 处理，就得需要 matrix-matrix 的相乘。在文本总结和 Llava 图片处理等地方，这是非常关键的。

这时最快的是 Intel 的 MKL 库能跑到 384 gigaflops。

但是 MKL 是不开源的，而且线程模型也不匹配，因此我们需要找出来它是怎样工作的，并自己复现出来。

作者认为 CPU math kernel 的关键就是利用指令集并行，并减少内存访问次数。

如果编译 `-O3 -ffast-math -march=native` 会有类似的输出：

```C++
void SLLMM(int m, int n, int k,
           const float *A, int lda,
           const float *B, int ldb,
           float *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            __m256 c = _mm256_setzero_ps();
            for (int l = 0; l < k; l += 8)
                c = _mm256_fmadd_ps(_mm256_loadu_ps(A + lda * i + l),
                                    _mm256_loadu_ps(B + ldb * j + l), c);
            C[ldc * j + i] = hsum(c);
        }
}
```

所以 llama.cpp 通行的做法是，当它想要优化时，就展开一下内部的循环：

```C++
void SLLMM2(int m, int n, int k,
           const float *A, int lda,
           const float *B, int ldb,
           float *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            for (int l = 0; l < k; l += 16) {
                c0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + lda * i + l + 0),
                                     _mm256_loadu_ps(B + ldb * j + l + 0), c0);
                c1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + lda * i + l + 8),
                                     _mm256_loadu_ps(B + ldb * j + l + 8), c1);
            }
            C[ldc * j + i] = hsum(c0) + hsum(c1);
        }
}
```

这个操作对优化树值稳定性有帮助，但是对性能帮助不大，因为现代 CPu 能够直接预测到后续的迭代。

其实更需要做的事 unroll 外部的循环，关键在于，把 a0 放到一个寄存器里。

```C++
void SLLMM4(int m, int n, int k,
            const float *A, int lda,
            const float *B, int ldb,
            float *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; j += 4) {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            for (int l = 0; l < k; l += 8) {
                __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l); // <-- 这里
                __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
                __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
                __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
                __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
                c0 = _mm256_fmadd_ps(a0, k0, c0);
                c1 = _mm256_fmadd_ps(a0, k1, c1);
                c2 = _mm256_fmadd_ps(a0, k2, c2);
                c3 = _mm256_fmadd_ps(a0, k3, c3);
            }
            C[ldc * (j + 0) + (i + 0)] = hsum(c0);
            C[ldc * (j + 1) + (i + 0)] = hsum(c1);
            C[ldc * (j + 2) + (i + 0)] = hsum(c2);
            C[ldc * (j + 3) + (i + 0)] = hsum(c3);
        }
}
```


（）

如果 unroll 两层循环，效果可以翻倍：

```C++
void SLLMM3X4(int m, int n, int k,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc) {
#pragma omp parallel for collapse(2) if (m * n * k > 300000)
    for (int i = 0; i < m; i += 3)
        for (int j = 0; j < n; j += 4) {
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c13 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();
            __m256 c21 = _mm256_setzero_ps();
            __m256 c22 = _mm256_setzero_ps();
            __m256 c23 = _mm256_setzero_ps();
            for (int l = 0; l < k; l += 8) {
                __m256 k0 = _mm256_loadu_ps(B + ldb * (j + 0) + l);
                __m256 k1 = _mm256_loadu_ps(B + ldb * (j + 1) + l);
                __m256 k2 = _mm256_loadu_ps(B + ldb * (j + 2) + l);
                __m256 k3 = _mm256_loadu_ps(B + ldb * (j + 3) + l);
                __m256 a0 = _mm256_loadu_ps(A + lda * (i + 0) + l);
                c00 = _mm256_fmadd_ps(a0, k0, c00);
                c01 = _mm256_fmadd_ps(a0, k1, c01);
                c02 = _mm256_fmadd_ps(a0, k2, c02);
                c03 = _mm256_fmadd_ps(a0, k3, c03);
                __m256 a1 = _mm256_loadu_ps(A + lda * (i + 1) + l);
                c10 = _mm256_fmadd_ps(a1, k0, c10);
                c11 = _mm256_fmadd_ps(a1, k1, c11);
                c12 = _mm256_fmadd_ps(a1, k2, c12);
                c13 = _mm256_fmadd_ps(a1, k3, c13);
                __m256 a2 = _mm256_loadu_ps(A + lda * (i + 2) + l);
                c20 = _mm256_fmadd_ps(a2, k0, c20);
                c21 = _mm256_fmadd_ps(a2, k1, c21);
                c22 = _mm256_fmadd_ps(a2, k2, c22);
                c23 = _mm256_fmadd_ps(a2, k3, c23);
            }
            C[ldc * (j + 0) + (i + 0)] = hsum(c00);
            C[ldc * (j + 1) + (i + 0)] = hsum(c01);
            C[ldc * (j + 2) + (i + 0)] = hsum(c02);
            C[ldc * (j + 3) + (i + 0)] = hsum(c03);
            C[ldc * (j + 0) + (i + 1)] = hsum(c10);
            C[ldc * (j + 1) + (i + 1)] = hsum(c11);
            C[ldc * (j + 2) + (i + 1)] = hsum(c12);
            C[ldc * (j + 3) + (i + 1)] = hsum(c13);
            C[ldc * (j + 0) + (i + 2)] = hsum(c20);
            C[ldc * (j + 1) + (i + 2)] = hsum(c21);
            C[ldc * (j + 2) + (i + 2)] = hsum(c22);
            C[ldc * (j + 3) + (i + 2)] = hsum(c23);
        }
}
```

> But does the C function above generalize to all matrix sizes? Nope. If I bump the complexity up from 512 to 1024, then I'm pretty much back at square one, not doing much better than a naive kernel, and MKL wins once more. I personally don't view this as too problematic, since llama.cpp by default processes prompts in modestly sized batches, and a kernel should only need to be good for its intended size. It's also only a matter of time until I unriddle the tricks needed for optimal tiling and cache locality that can make my kernels scale.

## How I Got Multiple Threads to Work

> Ops in the model graph are processed one by one. A thread is spawned for each core. Threads are restrained by a spinlock barrier and then set loose to compute different parts of an output matrix in parallel as soon as the next op is ready for execution. The id of each thread is called `ith` and the number of threads is called `nth`. There are no futexes or semaphores, because kernel scheduling would greatly reduce tokens/sec.