
TLDR

- 不算值得细看，主要做 kernel fusion 就是把 softmax、rmsnorm 等操作 fuse 成 kernel
- segment kv cache 区分了一下 prompt 和 response 的 kv cache，然后每 16 个 time step 分配一个 buffer，把 buffer 重用一下
- 不如 paged attention 细致

------
## Introduction

本文的主要 contribution：

1. 做了一个 Intel GPU 的 inference solution：Intel Extension for Pytorch
2. 简化 LLM 的 decoder layer 结构，减少 data movement overhead，做了一个 deep fusion 策略，来尽可能多地 fuse GEMM 和各种 Elementwise 操作
3. 为了提高吞吐，做了一个 segment KV cache 策略
4. 做了一个高性能的 SDPA 模块来 fuse 的 kernel

## Related Works

> Kernel fusion is a general approach to reduce such overhead by combining several successive operations together and compute them in a single kernel to avoid frequent memory access to slow GPU high bandwidth memory (HBM)


> Some previous works [23-26] focused on element-wise operations fusion to reduce kernel launch and memory access overhead.


##  Proposed Method

### 3.1 Model Structure Simplification

> The decoder layer usually has two basic modules Multi Head Attention (MHA) and Feed-Forward (FF), which are connected by some normalization operations.


> we respectively fuse the multiple operations in RMSNorm, RoPE and SDPA modules (labeled in pink in Figure 1) to single kernels.


### 3.2 Segment KV Cache

> In standard KV cache implementation, the prompt and response key/value will be concatenated together to create contiguous KV cache

> Moreover, since the KV Cache buffers grow larger at each time step in decoding phase, new bigger buffers will be allocated while may not reuse the previous smaller KV Cache buffers, resulting in many memory fragments.

memory fragment 可能会导致 gpu 的内存占用较多。

> holding the prompt and response key/value in different buffers and regularly empty cache the fragments at each time step

