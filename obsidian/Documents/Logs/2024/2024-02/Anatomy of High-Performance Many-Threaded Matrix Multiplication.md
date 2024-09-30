#gemm

TLDR:

- GotoBLAS 有三层 for 循环，BLIS 多加了两层循环，用来增加 kernel 的外部并行性和内部并行性，到最里面是一个只放在寄存器里的 microkernel。


> The primary innovation is the insight that the inner kernel—the smallest unit of computation within the GotoBLAS GEMM implementation—can be further simplified into two loops around a micro-kernel.

> This means that the library developer needs only implement and optimize a routine that implements the computation of C := AB+C where C is a small submatrix that fits in the registers of a target architecture.

> It is also shown that when many threads are employed it is necessary to parallelize in multiple dimensions.