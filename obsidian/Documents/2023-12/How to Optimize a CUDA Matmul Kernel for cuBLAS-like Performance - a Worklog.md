https://siboehm.com/articles/22/CUDA-MMM

MatMul 代表了 deep learning 领域里几乎所有的 FLOPS 消耗，作者最终实现了 95% 的 cuBLAS 的性能。

## Kernel 1: Naive Implementation

最 naive 的实现是将每个线程对应一个 $C$ 矩阵的输出：

```C++
// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 thread per block
dim3 blockDim(32, 32, 1);
// launch the asynchronous execution of the kernel on the device
// The function call returns immediately on the host
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

```C++
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```


## Kernel 2: Global Memory Coalescing

有相邻的 threadID 的 thread 更容易被组织到同一个 warp 中。

利用这一个特性，warp 中的同一次内存顺序访问，可以为 warp 中的多个 thread 所公用。这被称作 **global memory coalescing** 。

如果每个 thread 装载相邻的 f32，那么 warp scheduler 有可能可以将这组内存访问 coalesce 到一起。

![[Pasted image 20231218224736.png]]

在这个 case 里，相比于 naive kernel，这一实现同时处理多个 $B$ 矩阵的列，使访问 $B$ 矩阵的内存能够  coalescing 起来，C 矩阵的访问也可以 coalescing 起来。

（周末写的 vectorized 实现中也天然满足 coalescing 这个性质）。

反观 naive 算法中，它的每个线程访问 $A$ 矩阵和 $C$ 矩阵的内存时无法利用 coalescing 的优势。

## Kernel 3: Shared Memory Cache-Blocking

相对于 Kernel 2，多考虑到了 SMEM 的特点。会将 $A$ 矩阵和 $B$ 矩阵从 global memory 转载到 shared memory。

和 naive kernel 一样，会使 $C$ 矩阵的每个元素对应一个 thread。


```C++
// advance pointers to the starting positions
A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
B += cCol * BLOCKSIZE;                        // row=0, col=cCol
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

float tmp = 0.0;
// the outer loop advances A along the columns and B along
// the rows until we have fully calculated the result in C.
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
  // Have each thread load one of the elements in A & B from
  // global memory into shared memory.
  // Make the threadCol (=threadIdx.x) the consecutive index
  // to allow global memory access coalescing
  As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
  Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

  // block threads in this block until cache is fully populated
  __syncthreads();

  // advance pointers onto next chunk
  A += BLOCKSIZE;
  B += BLOCKSIZE * N;

  // execute the dotproduct on the currently cached block
  for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
    tmp += As[threadRow * BLOCKSIZE + dotIdx] *
            Bs[dotIdx * BLOCKSIZE + threadCol];
  }
  // need to sync again at the end, to avoid faster threads
  // fetching the next block into the cache before slower threads are done
  __syncthreads();
}
C[threadRow * N + threadCol] =
    alpha * tmp + beta * C[threadRow * N + threadCol];
```

`As` 和 `Bs` 可以看做是一个缓存。

线程内的循环是一个分块（Block）一次迭代，比较有意思的地方是**每个线程分别负责拷贝其中的一个元素**。

`__syncthreads()` 后针对这个 block 做矩阵相乘，最终**每个 thread 将该 thread 对应的结果写回 $C$ 矩阵**。

这个 kernel 相比前一个有 50% 的性能提升，能跑到 2200GFLOPS，但是距离这个 GPU 的 ~30TFLOPS 性能上限仍比较大。

如果每个分块是横竖 32 大小，那么需要的 shared memory space 为 8kb，实际上 A6000 GPU 有 48kb 的 share memory。这是很充裕的。

在 CUDA 的实践中，每个 Block 占用的 SMEM 如果过多，会降低 occupancy。
### Occupancy Calculation for Kernel 3


> We’re not using special math instructions, nor dynamic branches, so it’s clear that we’re stalling waiting for our SMEM accesses to return. So how do we make our kernel issue less SMEM instructions? One way is to have each thread compute more than one output element, which allows us to perform more of the work in registers and relying less on SMEM.

好像说这个 Kernel 会卡在 SMEM 到 shared memory 之间的 IO 上有些阻塞。

一个 thread 挪一个 cache，做一个计算。

## Kernel 4: 1D Blocktiling for Calculating Multiple Results per Thread

