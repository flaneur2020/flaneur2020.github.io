### The Two Stages of Token Generation: Prefill and Decode

- prefill 中的所有 token 都可以并行执行，而 decode 阶段，每个请求不可能有并发

### Metrics

- prefill 和 decode 两个阶段有两个不同的指标：ttft 通常由 prefill 决定；tpot（time per output token）通常由单个 decode step 决定；
- prefill 阶段的输出也是一个单独的 token，但是耗时会比单个 decode step 长得多；
- 另一方面，prefill 的 tok/s 的吞吐，会比 decode 高得多；（这是为什么商业的 LLM 对 input token 的收费比 output token 的收费低得多）
- 通常之间，都会面临一个总体吞吐与延迟之间的 trade off；

![[Pasted image 20251129214921.png]]


### Resource Utilization

- prefill 阶段很消耗 GPU 算力资源，而 decode 消耗的算力资源很少，而更受内存带宽限制；
- 在 prefill 阶段，在 GPU 算力利用率饱和之前，可以一直提高 token 的吞吐；
- 在 decode 阶段，GPU 利用率只能通过增加 batch 的请求数量来提高；
	- 在低并发时，吞吐受内存带宽限制；
- prefill 中，更短的 prompt 意味着更低的 gpu 利用率；这时在更高的请求速率时才会发生饱和；
	- 也就是说，短 prompt 的话，需要更多的请求才能跑满 GPU；

```
吞吐量
  ↑
  │     线性增长区域          平台期
  │    (内存备限制)          (计算饱和)
  │    /─────────────────────
  │   /
  │  /
  └──────────────────────────→ 并发请求数
```

### Concurrent Processing

- contingous batching 中，需要做的一个 trade off 是，**如何在同一个GPU上平衡新请求（prefill）和已有请求（decode）的时间和资源？**
- Prefill-First 策略：新请求来的时候，立即做完整的 prefill
	- 优势：降低 Time To First Token
	- 劣势：其他用户的 Decode 响应停顿
- Chunked Prefill 策略：将新请求的 Prefill 任务拆分成小块
	- Prefill Chunk 1 → Decode 1个token每个request → Prefill Chunk 2 → Decode 1个token...
	- 牺牲一点新请求的 ttft
	- chunk 值越小，tpot 更小，但是牺牲新请求的 ttft；
	- vLLM 默认的 chunk 值是 512，后来调大了