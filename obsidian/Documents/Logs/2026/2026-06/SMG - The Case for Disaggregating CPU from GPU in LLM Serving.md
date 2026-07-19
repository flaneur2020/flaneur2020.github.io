> figure out if cache-aware load balancing could improve routing across inference replicas.

在 sglang 和 vllm 中，tokenization 和 detokenization 甚至成为了瓶颈。

尽管两个引擎都适用了 rust/C++ 的 tokenizer，但是调用都来自 python，意味着 GIL。