https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c

> Increasing batch size will change the behavior from compute memory bound to memory compute bounded


> kernel fushion: reduce memory access ops, because we fuse multiple operations into one

## 1.2 - Transformer inference basics

prefilling 阶段偏 compute bound，后续的 inference 阶段会偏 memory bound。

更大的 batch 会有助于靠近 compute bound，但是我们不能无限增加 batch，因为显存大小是有限的。

![[Pasted image 20240111115529.png]]

要按 BF16 跑一个 13B 的模型，只有 10Gb 内存可用于 kv cache。

这意味着要么不能太大 batch，要么不能太长的 sequence。

## 2 - MLSys: Flash attention and vLLM

Paged Attention 给了一套 GPU 下做内存管理的办法。

Flash Attention 有效地减少了 IO。

> FlashAttention is a must for every single practioners. Please simply memorize the full algorithm

![[Pasted image 20240111115835.png]]

> Key idea:
>
> - Instead of storing the full attention matrix in the HBM, do blockwise computation of the dot product, such that all the computation is performed in the L2 cache
>
> Key advantage:
> 
> - significantly reduced memory usage, such that **you can put in 100k context length using brutal force — yes, there is no fancy algorithm for 100k, just brutal force**
    - in the original paper the authors only test up to 16k, but 100k is totally doable
> - significantly improved throughput, particularly for small models where a large portion of the flop is in the dot-product operation

## 2.3 Flash Decoding

> key idea: instead of using one query to scan the full kv cache, duplicate the query such that different chunks of the kv cache can be scanned in parallel


## 3.1.2 - Quantization

> Quantization is nowadays a must to deploy a large model. The good news is that it does not really harm performance


Yi-34B chat 经过 4 bit 量化后只需要 17G 内存。

## 3.1.3 - Multi-query attention and group-query attention

> In genereal, multi-query attention significantly speed up training and inference by simultanously reducing memory and compute.
> 
> Multi-query attention is also a great example where differences in small models do not exist in large models: for small models like 7B, multi-query attettion is worse than full attention, but when models become as large as 70B, multi-query attention has basically the same performance as full attention. LLaMA2 7B uses full attention, and 70B uses GQA.
> 
> Current state of art large models by default use multi-query attention

小模型 MQA 比 MHA 差，但是超过 70B 就都一样了。

LLAMA2 7B 用的 Full attention，70B 用的 GQA。

## 3.2 - Advanced techniques

### 3.2.1 - Mixture of experts

> Mistral MoE shows a possibility of achieving the performance of a much large model while reducing the cost to a much smaller model

### 3.2.3 - Blockwise decoding

典型就是 **Speculative decoding**。