https://arxiv.org/pdf/2312.11514.pdf

## TLDR

- ReLU 的激活函数往往有 90% 比例都是 sparsity 现象，下一层需要装载到内存里的参数就大大减少了，因为大部分都是零值。
- 一般理解上每个 FFN 的第一层仍是需要全部放到内存中的，作者基于 《Deja vu: Contextual sparsity for efficient llms at inference time》 的思路做了改进；
- 然后有一个 window 策略，每 N 个 token 组成 N 个窗口，每个 token 对应一组 ffn 神经元的区块列表，在 DRAM 里的 FFN 参数就是这 N 个 token 中活跃的神经元区块列表的总集，会大大小于总的 FFN 存储量，论文里说能降到 2% 这个水平

---

作者构造了一套 inference cost model，优化目标包括：

- 减少 flash 的 data transfer；
- 通过更大、更连续的方式读数据；

引入了两个技术：

1. windowing 策略，通过重用旧的激活神经元来减少传输；
2. row column bundling：优化 flash 内容读取的连续性；

> These methods collectively enable running models up to twice the size of the available DRAM, with a 4-5x and 20-25x increase in inference speed compared to naive loading approaches in CPU and GPU, respectively.


能够跑起来超过两倍 DRAM 大小的模型，而且比常规的 CPU 推理快 4~5 倍，比常规的 GPU 推理速度快 20~25 倍。

（怎么做到的？）

## 1 Introduction

目前大多是将所有参数装载到内存中，比如一个半精度的 7B 模型需要 14GB。

> Our methodology is built on the top of recent works that have shown LLMs exhibit a high degree of sparsity in the FeedForward Network (FFN) layers

