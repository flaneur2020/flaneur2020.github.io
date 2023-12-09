https://arxiv.org/pdf/2311.00502.pdf

（感觉啥也没说，调了一个包？）

## Introduction

前人的工作：

- INT8 量化是目前最常见的办法，在推理性能和准确性之间有一个不错的权衡；
- 不过，INT8 在应用中大家发现 **outlier 的 activation 值对性能影响比较大**，因此开发了 **FP8 类型**，不过 FP8 仍相对硬件支持少一些，导致推广有限；
- 与此同时，**weight-only quantization** 也逐渐变得更受欢迎，它只将 weight 参数用低精度进行表示，同时为 activation 值保留较高的精度（16 bits），这在开源的推理引擎 llama.cpp, starcoder.cpp 上应用比较多。

本文介绍了一个 automatic INT4 quantization flow 和一个高效的 LLM 运行时的实现，能做到：

- accuracy loss within <1% from FP32 baseline
- covering 3B to 20B and demonstrate the promising per-token generation latency from 20ms to 80ms, much faster than the average human reading speed about 200ms per token.

## Approach

![[Screenshot 2023-11-19 at 14.07.14.png]]

好像说是有一个循环，在量化期间可以循环着量化，衡量效果后再微调量化的参数。

### 2.1 Automatic INT4 Quantization Flow

INT4 量化 flow 是通过 Intel Neural Compressor 实现的，这是 intel 一个专门的量化库，支持 GPTQ, SignRound, AWQ, TEQ, RTN 这几种。


