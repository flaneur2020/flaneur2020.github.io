TLDR：

- huggingface 大约是 10MiB 一个 chunk；
- hf_transfer 是一个 rust 的加速包，做的事情就是：
	- 按 range: 0-0 请求上游得到一个文件完整的 size；
	- 按 10MiB 拆分 chunk；
	- 按顺序循环 chunks，每个 chunk 开一个 task 跑下载，用一个 semaphore 限制并发；
