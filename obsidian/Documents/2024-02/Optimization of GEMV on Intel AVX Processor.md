#gemm

该论文比较了传统的 SSE 指令和 AVX 指令的加速效果。（年代好像久远些）

该论文使用 AVX 指令加速超过 GotoBlas、ATLAS 等 BLAS 实现 5% ～ 10% 左右。

## Analysis of GEMV Subroutine

### 2.1 BLAS Level 2 Subroutines

- Level 1 包括 Vector - Vector 相乘，$y = \alpha x + \beta y$，以及其他一些比如 scalar dot product、vector norms
- Level 2 包括 Matrix Vector 相乘，如 $y=\alpha A x + \beta y$
- Level 3 包括 Matrix Matrix 相乘，如 $C=\alpha A B + \beta C$

### 2.2. Analysis of GEMV Implementation

> To sum up, the GEMV m blocking is useful for the small matrix and GEMV mn blocking is suitable for the big matrix.

GEMV m blocking:


![[Screenshot 2024-02-20 at 22.24.59.png]]![[Screenshot 2024-02-20 at 22.25.15.png]]

针对更大的 Matrix，mn blocking 更合适。

## 3. Optimization

The AVX supports for 256-bit wide vectors.

### 3.2. Memory Access Optimization

> Since there is only tiny difference of speed between the L1 and L2 cache and the larger space of L2 cache, the data can be filled into the L2 cache as a chunk.

> In the experimental system, the L2 cache on Intel Sandy Bridge is 256Kbyte, in other words, the L2 cache can store 215 elements of double float type.

> The best way to take advantage of the L2 cache is to access the data chuck in one required dimension of matrix A, rather than the two dimensions.

> The main idea of cache hierarchy optimization is to put the whole matrix A into pieces of smaller matrix, the size of which are comparable to the cache, so the machine can access the data in a more efficient way.

### 3.3. SIMD Optimization for GEMV Kernel

![[Screenshot 2024-02-20 at 23.02.55.png]]