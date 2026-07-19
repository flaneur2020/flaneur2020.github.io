tldr：

- runai streamer 看起来不错，组成组之后，多个 Pod 可以并发只下载自己的一小部分；

## 看代码

- run:ai 有一个 distributed 模式，可以只装载 1/N 的 rank 到 GPU 显存，然后通过 NCCL broadcast 发给同组其他 rank；
- 相当于一个节点做种子，用 broadcast 发权重，没有重复；
- （能不能让一个组内的不同 rank 并发下载不同的权重？）
- 默认同一个物理节点上的所有 rank 分类为一组（走 女link 高速广播），RUNAI_STREAMER_DIST_GLOBAL=1 可以改成全局组，走 nccl 跨节点广播；
- partition：整批 chunk 分给 N 个 rank，使每个 rank 读取总量大致相等；
- 流水线式 broadcast：
	- 读和广播是分批流水线；
	- 会分两个 GPU 显存缓冲区（data_buffer 给读取 rank 用，received_buffer 给接收 rank 用）
	- prefill 阶段：
		- 流式拿到 CPU buffers，逐个 copy 进 data_buffer，上限是 256 chunks；
	- broadcast 阶段：
		- 先广播 metadata tensor（让所有 rank 知道这批有哪些 chunk）；
		- 再广播 data_buffer 这一整块；只有划分到该 chunk 的那个读取 rank 会真的读取数据，其他在 received_buffer 里收到广播后，按 metadata 索引挑出来属于自己的那部分；
		- 累计处理完 self.total_chunks_to_read 个 chunk 后结束；
- 开启条件：
	- is_distributed=True、且 torch.distributed 已初始化、group_size > 1；
	- CUDA 空闲显存 >= 2 x max_chunks，否则回退并告警；
	- RUNAI_STREAMER_DIST=0|1|auto
	

---
- nvidia run:ai Model streamer：是一个 python sdk 用于并发从 storage 读取 weight，并直接 stream 到 GPU 内存中；
- 作者为它做 benchmark，与 vLLM 默认的 HF safetensor loader 和 coreweaver tokensorizer 对 SSD 和 S3 做了 benchmark 比较；

## How is a model loaded to a GPU for inference?

1. 从 storage 中读取权重到 CPU memory；
2. 将 model 移动到 GPU；

从 s3 等 cloud 存储上装载权重，往往还有多余的一步，就是先下载到 local disk 上，再挪到 CPU、GPU memory。

历史上，这些流程都是顺序的，使得 model loading 成为了一个比较明显的 bottleneck。

### How does the Model Streamer work?

- Model Streamer 是一个高性能的 C++ 后端，从各种存储后端比如本地盘、NFS、s3 等地方，使用多线程来下载 tensor 到 CPU memory；
- 每个 tensor 有一个唯一的 identifier，能够并发地读取、transfer，比如，一部分 tensor 还在下载中的同时，完成下载的 tensor 可以并发地挪到 GPU；
- 这个库利用了这个特性：GPU 可以直接读 CPU 内存，不需要 CPU 参与，从而实现<mark>下载和 transfer 到显存的 overlap</mark>；

### How does the HF Safetensors Loader work?

- 它使用一个 memory mapped 的 filesystem 来减少 data copying；
- 在 CPU 上，tensor 直接映射给内存；
- GPU 上，它会通过 pytorch 创建一个 empty tensor，然后使用 cudaMemcpy 来读取内存；

### How does the CoreWeave Tensorizer work?

- Instead of loading an entire model into RAM before moving it to the GPU, Tensorizer streams the model data tensor by tensor from an HTTP/HTTPS or S3 source.

### Where loading meets inference engines: Loading weights with vLLM

![[Pasted image 20260627194517.png]]