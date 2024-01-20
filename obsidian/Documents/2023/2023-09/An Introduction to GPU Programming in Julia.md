https://nextjournal.com/sdanisch/julia-gpu-programming

GPU 算力的难点：

- 从 CPU 转移内存到 GPU 比较花时间；此外，launch 一个 kernel 在 GPU 中也需要 10us 级别的时延；
- 在上层没有高级 wrapper 时，配置 kernel 会迅速变得复杂；
- 默认是低精度的计算，高精度的计算会迅速消耗掉性能上的收益；
- GPU 的函数天然并发，写 GPU 函数（kernel）的难度与 CPU 写并行代码是等同的；
- 因此，很多算法并没有自然地移植到 GPU 平台上；
- Kernel 通常是通过  C/C++ dialect 来编写的；
- CUDA 和 OpenCL 有一个割裂；CUDA 只支持 Nvdia 硬件，OpenCL 支持所有的但是差点火候；

## GPUArrays

GPUArrays.jl 在 julia 中提供了对 array 的抽象。

CLArray 可以直接生成 opencl 的 C 代码。

## Writing GPU Kernels

使用 julia 内置的 ndarray 方法，很多时候不需要自己写 kernel；

`gpu_call(kernel, A::GPUArray, args)` 

