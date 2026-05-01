![[Pasted image 20260426145144.png]]

## Model Key Features & New Capabilities

- Hybrid sparse-attention: 每个 layer 配合 sliding window attention 和两种压缩机制之一（4:1 topk 或者 128:1 dense）
- mHC：a generalization of standard residual connections that improves gradient flow and representation quality
- FP4 expert weight：原生的 FP4 MoE 的 expert 权重；

## Designs, Features and Performance Optimizations

### ShadowRadix: Native Prefix Caching for Hybrid Attention

> Every layer of DeepSeek-V4 combines **SWA** (sliding window attention over the last 128 raw tokens) with **either C4** (top-512 sparse over 4:1-compressed KV) **or C128** (dense over 128:1-compressed KV).

每一层都是组合 SWA （128 个token）配合两种压缩方式之一：

- C4: sparse 的 top 512 配合 4:1 KV 压缩
- C128: dense 的 128:1 的 KV 压缩

SWA 和 C4、C128 是两拨 KV Cache。SWA 有 128 项不压缩的 KV cache，C4 和 C128 都是单独的 KV Cache。

![[Pasted image 20260426164807.png]]

>also, for maintaining the inflight compressing KV slots, each compression layer has a state pool that stores the in-progress compression state. This complex mechanism breaks traditional prefix caching assumption: three heterogeneous KV pools and two compression-state pools must stay coherent across prefill, decode, and speculative decoding.

思路是区分开逻辑上的 token 和物理存储。

radix tree 的每个 slot 代表一个 token 有一个唯一坐标（Slot ID）。所有层共享这个 ID。

一个逻辑上的 token 并不直接保存在哪里，而是通过投影映射到不同的物理池中。

SWA、C4、C128 对应三种池子。

SWA 向后移动时，旧的 token 就不需要高精度表示了。这时系统会对 SWA 槽位做 tombstone 乃至释放物理内存。

不过，C4、C128 的压缩后的 KV 仍然保存在物理池中。

<mark>C4、C128 的 token 似乎来自于 SWA 翻过去的数据，还在 SWA 里的 token 不会进入到 C4、C128 里面。</mark>

系统为 radix tree 每个节点弄了两个独立的计数器。

- `full_lock_ref` 只要这个计数器 >0 ，该节点在 Radix Tree 中就存在。它保护的是源数据以及对应的C4/C128 的影子；
- `swa_lock_ref` 跟踪某个节点是否在某个请求的 SWA 内；

### Speculative Decoding

ds4 引入了一个单层的 MTP head。它是一个独立训练的解码层，输入是上一步的隐藏状态，和下一个 token 的 emebdding。

因为它用了一个比较复杂的内存管理机制（hybrid attention），导致 cpu 在告诉 gpu 数据在哪里时，开销比较大。

为了解决这个问题，把“寻找数据位置”的计算工作放在了 GPU 内不，用 CUDA Graph 把流程打包。

### HiSparse: Turbocharging Sparse Attention with Hierarchical Memory

HiSparse 是将不活跃的 kvcache 给 offload 到 CPU 内存的办法。

HiSparse 与 C4 层天然契合。top-k 仅接触及小部分的压缩位置，这意味着 C4 KV cache 大部分时间都是非活跃状态，可以挪到 CPU 去。

因为 C128 是稠密的，且 SWA 规模已经很小（128），这两个就不大能从 offload 技术中获得收益。

通过使用 CPU 内存池来仅仅针对 C4 的 KV 缓存池，将长文本服务的 token 容量和吞吐提升了 3 倍。

### Fast Kernel Integrations

- 支持 Hybrid Attention 的新的 FlashMLA 执行路径：使 MLA 和额外的注意力可以在单个 fused kernel 上运行。这个算子同时接受 k_cache 和 extra_k_cache 及其各自的索引。
- 基于 FlashInfer 的 TRTLLM-Gen 融合 MoE （mxfp8、mxfp4）：dsv4 使用了原生的 fp4 专家权重，使小 batch 的 MoE 解码对专家权重的带宽非常敏感。依赖 blackwell 的 fp4 tensor core，以及 tiled/persistent execution 机制。
- 支持 split k 的 tile lang 的 mHC 算子；
- DeepGEMM Mega MoE 集成：将 EP 分发、第一个 FP8xFP4 专家 GEMM、SwiGLU、第二个 FP8xFP4 专家 GEMM 以及 EP 合并全部融合进一个非对称内存的 mega kernel 里，实现了 nvlink 通信和 tensor core 计算的重叠。

### Various Kernel Optimizations

#### Flash Compressor: IO-aware Exact Compression

压缩注意力（Compressed Attention）通过对每个 Token 的得分进行 Softmax 加权平均，将一组 Token 压缩为一个 KV 对。朴素（Naive）的实现方式需要访问五次 HBM（显存），且需要在非连续维度上运行 Softmax，因此压缩过程本身的开销往往超过了它所服务的压缩注意力计算开销。

对于 C4，softmax 保持在 warp 级别局部化，对于 C128，则采用单 CTA 级别的归约。