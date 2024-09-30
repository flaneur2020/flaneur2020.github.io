https://github.com/wangsiping97/FastGEMV/blob/main/method_and_result.md

## Optimization Strategies

### P threads per dot product

并非给每个线程计算一个 result vector 中的元素，而为每个 dot product 分配 `p` 个线程。

针对每行，开 `p` 个线程来计算一段 dot product 保存在中间结果 `result[p]` 中，最后相加一下。

随后优化问题可以拆分为：

1. 每个线程计算局部 dot product 的性能
2. 中间结果相加汇总

### Optimize partial result aggregation: reduction sum

使用 reduce sum 模式来相加。

### Optimize memory access time: vectorization

每个 gpu thread 负责 `K / p` 个元素。
### GPU thread layout

![[Pasted image 20240217110014.png]]

## ## Exploration of alternate strategies

### Leveraging shared memory

> One intuitive optimization technique involves employing shared memory for storing data that is reused multiple times within the same block. Given that we have `m` rows per block, elements fetched from the vector are used `m` times. To avoid redundant global memory access, we could load the values from the vector into shared memory for subsequent reuse. Despite the appeal of this strategy, experimental results did not show a performance improvement. One possible reason might be the synchronization overhead between the threads.

实验表示 shared memory 没有预期的效果。
### Precomputed result table for quantized integers