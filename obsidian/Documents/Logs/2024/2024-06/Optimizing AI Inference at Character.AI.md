https://research.character.ai/optimizing-inference/

cai 每秒有 20000 个推理请求；

> We have reduced serving costs by a factor of 33 compared to when we began in late 2022. Today, if we were to serve our traffic using leading commercial APIs, it would cost at least 13.5X more than with our systems.
## Memory-efficient Architecture Design

LLM 吞吐最大的瓶颈是 attention 中的 KV Cache，它不止限制了 GPU 中最大的 batch size，也是 attention layer 的 IO 开销的来源；

作者使用了一系列的技术，在未损失质量的前提下使 KV Cache 的 size 降低二十倍；

**1. Multi Query Attention**，相比 GQA 减少八倍

**2. Hybrid Attention Horizons**，混合局部注意力 ([Beltagy et al., 2020](https://arxiv.org/abs/2004.05150v2?ref=research.character.ai)) 和全局注意力, 局部注意力使用 sliding window attention 进行训练，复杂度从 O(length ^ 2) 降低到 O(length)；将大多数局部注意力范围限制在 1024 并没有显著影响，包括 find a needle in haystack 测试；每六个层中，只有一个使用全局注意力；

**3. Cross Layer KV-Sharing**：重用相邻层的 KV Cache，能够减少二到三倍；对于 global attention layer，重用了所有层的 KV Cache；

![[Pasted image 20240623114117.png]]

## Stateful Caching

作者提到他们做的一个主要的创新点是，在多轮 chat 中间，将 kv cache 缓存在宿主机上。

在 character.ai 中，多数 chat 都是长对话，每个消息平均有 180 条消息历史。

对于 prefilled prefix 和 generated message，将 KV cache 缓存在宿主机内存上。类似 Radix Attention 的做法，将这些 KV cache 保存在一个 LRU cache 的 tree structure。

cached kv 值按 prefix token 的一个 rolling hash 进行索引。对每个新的 query，根据 context 计算一个 rolling hash，寻找最长的匹配。

在集群层面，使用 sticky session 来路由 query 到原本的服务器。由于 kv cache 很小，每个服务器可以 cache 几千条对话。这个系统可以实现 95% 的 cache rate，进一步减少了 inferencing cost。

![[Pasted image 20240623121050.png]]
## Quantization for Training and Serving

使用 int8 作为 model weight、activation 和 kv cache 的量化。

实现了自己的 matmul 和 attention 的 int8 kernel。

与通常意义上用的 “post-training quantization” 技术不同，作者自己从零按 int8 训练的模型，减少 training/serving 的 mismatch 情况，也显著提升了 training 的效率。