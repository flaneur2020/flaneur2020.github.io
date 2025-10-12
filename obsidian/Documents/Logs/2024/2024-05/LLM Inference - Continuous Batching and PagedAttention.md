https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/

在 OSDI'22 的 Orca 发了两个 novel 的技术：**continuous batching** （也叫做 iteration scheduling）和 **selective batching**。

## Continuous Batching

在 continuous batching 之前，一般的做法是 static batching，发一组请求攒成一个 batch，然后等它们结束。

但是每个请求响应的长度不一，导致 GPU 闲置。

![[Pasted image 20240513213226.png]]

continuous batching 的思想很简单，在一个请求的响应结束之后，它持续地调度一个新的请求到 batch 中。

> With iteration-level scheduling, a next request will be scheduled as soon as a request finishes its iteration.

![[Pasted image 20240513213324.png]]

## Selective Batching

Orca 不会将 prefill stage 拆分到多个 iteration 中，而且，orca 完全不会对 attention 计算进行 batch 处理。

> This is because, for batch process, [`torch.bmm`](https://pytorch.org/docs/main/generated/torch.bmm.html#torch.bmm) is typically used, however, two input tensors must have a strict shape: `[b, n, m]` and `[b, m, p]` so that output can be a `[b, n, p]` shaped tensor.

在 attention 计算中，bmm 涉及的各个 tensor 的 size 各不相同。

但是，orca 的作者发现，除了 attention 计算之外的所有算子，对应的 tensor 的 shape 都是相同的。

因此可以对非 attention 的各种操作进行 batch 处理，而对 attention 计算仍顺序计算。

> thus they selectively batch those operations, while sequencially execute attentions after splitting the input:

PagedAttention 可以针对一个 batch 的 attenion 进行 fuse 处理，将整个 attention 放在 `paged_attention_kernel` 这一个 kernel 中。

> Use [FlashAteention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention) with HuggingFace transformers to boost performance. [A fused flash attention](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/opt/modeling_opt.py#L433-L450) is called. For more information, visit [flash attention repository](https://github.com/Dao-AILab/flash-attention) and read [this paper](https://arxiv.org/abs/2307.08691).

## PagedAttention

Paged Attention 主要用于解决 KV Cache 内存管理的低效问题。

Orca 主要的缺憾是它们为 KV Cache 完全固定预分配的内存。

如果一组请求的 KV Cache 长短不一，内存会有浪费。

Paged Attention 会将 KV Cache 拆分为 KV Blocks，每填满一个 Block吗，则申请一个新的。

### Preemption with Page Miss

> As the entire cache size is fixed, it can run out of available blocks to store the newly genereated KV cache. In deciding which blocks should be evicted, PagedAttention interestingly implements all-of-nothing eviction policy, evicting all caches to either CPU (swapping) or discarding them (and later recompute it).

完整的 KV Cache 的 size 仍是固定的。如果吃光了 KV Cache，需要选择 evict 哪些 block，PagedAttention 的 evict 策略是 all-of-nothing：清除所有的 cache 给 CPU （swapping），或者 discarding them（重新计算）。

> The decision was because that all blocks of a sequence are accessed together, so it was not reasonable to remain some cache blocks for a sequence

这个决定是因为，一个 sequence 要用的话就是整个 sequence 都用，只保留一部分是没有意义的。

> After that, if the sequence was in progress, evicted blocks will be brought back to the GPU memory and finish execution.

这种抢占发生在 `scheduling requests` 阶段，在 scheduling requests 之前，它会为所有的 running requests 分配一个 page slot。

如果分配不到，则进行抢占。

Paged Attention 也实现了 Orca 的 iteration level scheduling。

它会在每轮迭代调用 `vllm.core.Scheduler._schedule()`。

### Prompt Handling

`forward()` 会检查 input 是否为 prompt。

> Because query, key, and value arguments include a batched input, all inputs should be either prompt or decode, and cannot be coalesced.

Batch 中的所有的输入要么是 prompt 要么是 decode，不能混合。

> They might use several iterations, however, to finish all pending prompts before resuming decoding. Prompts can be grouped with padding or separated and executed in different iterations.