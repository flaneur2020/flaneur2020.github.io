https://ianfinlayson.net/class/cpsc425/notes/21-cuda

- GPU 最初用于图形加速，包括几个常见的操作：Matmul、Cross Product、Dot Product
- GPU 与 CPU 的不同：有很多很多核心；每个核心的 Clock Speed 并不高；每个核心相对不那么独立；

## CUDA Programming Model
- 跑在 CPU 的代码称作 Host
- 跑在 GPU 的代码称作 Device
- ![[Pasted image 20230922144441.png]]
- 编写 CUDA 代码时必须显式地从 GPU 和 CPU 之间交换数据

## Memory Management
- CUDA 内存和 CPU 内存完全不同；
- 可以通过 cudaMalloc 从 GPU 中分配内存 `cudaMalloc(void** pointer, int bytes)`；
- 该方法取一个指针地址和分配的内存大小；
- 这段内存并不能直接访问，要写入这段内存，必须使用 `cudaMemcpy(destination, source, bytes, cudaMemcpyKind)` 方法；
	- kind 包括 cudaMemcpyHostToHost、HostToDevice、DeviceToHost、DeviceToDevice 四种

## Blocks
- 传输数据到 GPU 的 block 中，block 是一个有多个维度的 array；
- CUDA 能够自动将这些 block 分配到不同的 GPU Core 中执行；
- 可以通过 blockIdx.x 得到当前 block 的 1 维的维度号，相应的 2 维号和 3 维号是 blockIdx.y、blockIdx.z；

## __global__ Functions
- CUDA 的函数都必须包一个 `__global__`；
- CUDA 代码可以是任意的 C/C++ 代码，不过没有 RTTI、没有异常处理、没有虚函数；

## CUDA Sum Example

```C
__global__ void sum(int* partials) { 
  /* find our start and end points */
  int start = ((END - START) / CORES) * blockIdx.x;
  int end = start + ((END - START) / CORES) - 1;
  /* the last core must go all the way to the end */
  if (blockIdx.x == (CORES - 1)) {
    end = END;
  }
  /* calculate our part into the array of partial sums */
  partials[blockIdx.x] = 0;
  int i; for (i = start; i <= end; i++) {
    partials[blockIdx.x] += i; 
  }
}
```

## __device__ Functions

- CUDA 的 `__global__` 函数不能调用普通的 CPU 函数
- 可以声明一些函数为 `__device__` 类型，这样可以在 `__global__` 函数里调用

## Multi-Dimensional Kernels

- 