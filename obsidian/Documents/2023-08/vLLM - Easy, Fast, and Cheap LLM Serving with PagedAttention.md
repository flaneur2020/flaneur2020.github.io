https://vllm.ai/

- 作者引进了 PagedAttention，一套新的高效地管理 attention key & value 的 attention 算法，据说可以 24x 加速 HuggingFace 的 transformer 的库

## The Secret Sauce: PagedAttention
- LLM Serving 的瓶颈在于内存；
- 在 decoding 的 auto regressive 过程中，LLM 需要记忆住之前所有的 input tokens 才能预测下一个 token；
- 这组被记住的 key 和 value 向量被称作 KV cache；
	- kv cache 很大，在 LLama13B 中一个 sequence 就能吃 1.7GB 内存；
	- dynamic：它的体积取决于 sequence length，这是难以预测的，因此高效地管理 KV Cache 很有挑战；作者观察在一个系统重往往有 60%~80% 的内存是浪费的，因为碎片化和 over-reservation；
- 为了应对这些问题，作者引进了 Paged Attention。它受传统的虚拟内存和分页机制的启发。
- 与传统的 Attention 算法不同，Paged Attention 允许在非连续的内存空间中存储连续的 KV。
- PagedAttention 将 kv cache 分成 Block