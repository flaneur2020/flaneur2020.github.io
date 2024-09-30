方便开发者写计算核。

现有的优化库的体积都很感人：OneDNN 有 51mb、MKL 有 14mb、cuBLAS 有 150mb。

> The Modular implementation of matrix multiplication is typically less than **100kb** of machine code, which makes it practical to use in many different use cases, including mobile, web, and IoT.

## Performance portability

目前的 kernel 库的平台都很受限制。

## Dynamism

有些系统通过 JIT 和 AOT 来做 AI compiler，比如 google 的 XLA、Apache TVM、OneDNN Graph。

根据不同的 matrix size 动态的生成特化的代码。MKL 是内置了特化的实现。

这在遇到大模型的时候有问题，需要针对特定 size 的输入。

> The challenge is that the system only knows the input size at inference time, not at model training or compilation time.

有些系统在尝试使用 JIT，但是有一些长尾 latency 的问题。

> The Modular approach completely eliminates these problems by fully supporting dynamism. Modular’s matrix multiplication and other kernels are fully dynamic shape friendly (without JIT or AOT specialization) and support other forms of dynamism (e.g., irregular control flow, unusual data types, etc.) that many existing systems struggle with. This delivers a much simpler and more predictable system overall.


## Composability

Matmul 很少单独使用，但是很多时候需要 fusing。

> Unfortunately, they force you to choose from a small fixed operator set without extensibility.

> The Modular approach provides both benefits – it supports generalized fusions with a wide range of operators without having to manually write and maintain variants.