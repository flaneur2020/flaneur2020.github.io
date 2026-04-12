最近好像不大流行写技术笔记文了，毕竟大模型都有，而且写的更好。

不过有段时间没学习也不大对劲，最近想重新学习一下 GPU 编程的部分。

## Tiled GEMM



```cuda
// Tiled GEMM Kernel
__global__ void tiled_gemm_kernel(float* A, float* B, float* C, 
                                   int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C this thread computes
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Accumulator
    float sum = 0.0f;

    // Number of tiles
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A
        int aCol = t * BLOCK_SIZE + tx;
        if (row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B
        int bRow = t * BLOCK_SIZE + ty;
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[bRow * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```