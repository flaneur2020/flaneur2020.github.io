https://pytorch.org/blog/flash-decoding/

TLDR

- flash attention 对推理的加速没大用处，因为 flash attention 是基于 batch 和 query 维度来拆分并发任务
- 推理的 query 维度只有 1，唯一的并发维度 batch 要远小于 GPU 的 SM 数量，无法并行起来
- 思路很简单，就是把 kv cache 作为拆分并发任务的维度
- 还提供了一个例子：https://github.com/facebookresearch/xformers/tree/main/examples/llama_inference

---

> When scaling on the batch size dimension, the attention can also become a bottleneck even with relatively small contexts. This is because the amount of memory to read scales with the batch dimension, whereas it only depends on the model size for the rest of the model.

## MULTI-HEAD ATTENTION FOR DECODING

在 decoding 阶段，每个 new token 需要关注前面的所有 token，需要计算 `softmax(queries @ keys.transpose) @ values`

这个操作在 flash attention 中做了优化，因为它的瓶颈在于读取中间结果（ `Q @ K^T`）

然而这个优化对 inference 阶段帮助不大，因为瓶颈不同。

在训练阶段，FlashAttention 会根据 batch size 和 query length 两个维度 dimension 来拆分并发任务。

但是在推理阶段，query length 的长度一般都是 1: 这意味着 batch size 往往小于 GPU 的 SM 数量。这个操作只能利用少量的 GPU 算力。

尤其在长 context 中更为明显，因为内存限制了 GPU 使用更小的 batch size。

batch size 如果是 1，GPU 算力的利用率只能到 1%。

![[Pasted image 20240308230741.png]]

_FlashAttention parallelizes across blocks of queries and batch size only, and does not manage to occupy the entire GPU during decoding_

## A FASTER ATTENTION FOR DECODING: FLASH-DECODING

![[Pasted image 20240308230829.png]]

FlashDecoding 的思路很简单，还是基于 FlashAttention 的思想来拆分 softmax 计算。

但是拆分工作的维度改为了面向 k v cache。

> Flash-Decoding works in 3 steps:

> 1. First, we split the keys/values in smaller chunks.
> 2. We compute the attention of the query with each of these splits in parallel using FlashAttention. We also write 1 extra scalar per row and per split: the log-sum-exp of the attention values.
> 3. Finally, we compute the actual output by reducing over all the splits, using the log-sum-exp to scale the contribution of each split.

## Benchmarks

![[Pasted image 20240308231016.png]]