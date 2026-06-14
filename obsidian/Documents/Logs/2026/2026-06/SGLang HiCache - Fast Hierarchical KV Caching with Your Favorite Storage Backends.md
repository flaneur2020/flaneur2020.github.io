
- Novita AI 的数据：对接 HiCache 到 3FS KVStore 用于存储历史 kv cache，平均 ttft 掉了 56%，inference 吞吐 double 了，cache hit 从 40% 到 80%；（对接 tiered 存储u之后，提高了 cache 命中率）
- Ant Group 的数据：集成 HiCache 到 Mooncake service，面向 DeepSeek-R1-671B 使用 PD 分离部署，在一个 QA 场景上，跑到 cache hit 时，可以实现一个 84% 的 TTFT 减少；（这个只比较了 cache hit 和未 hit 的 ttft，这不是废话么。。。）
- 除了 mooncake 和 3fs，也支持 NIXL 作为一个 local file 的 backend；

## Why Hierarchical KV Caching Matters

一开始 RadixAttention 只面向单机的 GPU 内存，但是因为空间有限，注定会被 evict 掉。

> To address this challenge, we present SGLang HiCache, which extends RadixAttention with a HiRadixTree that acts as a page table for referencing KV caches residing locally in GPU and CPU memory.

一个 cache controller 自动管理跨层次的 loading/backing up KV cache。比如 CPU 内存、GPU 内存，乃至 disk、远端内存。

![[Pasted image 20260607132904.png]]

## Design of SGLang HiCache

### Optimized data plane

- 主要的瓶颈点：从更慢的 tier 移动数据的延迟；
- 在 cudaMemcpyAsync 之外，作者开发了一组 GPU-assisted IO kernels，能够提供 3x 的 CPU 到 GPU 的 transfer；
- 作者解耦了 host memory pool 与 GPU pool 的 layout；
- GPU pool 偏 “layer first”，出于计算的需要；
- host pool 采用了 “page-first” 的布局，更易于传输；

![[Pasted image 20260607140420.png]]

### Versatile control plane

- 根据数据所在层级（GPU显存、CPU内存、外部存储）的不同延迟和带宽特性，采用不同的加载和预取策略；
	- GFU 缓存未命中，但命中 CPU 内存：使用层间 overlap，GPU 在执行第 N 层计算时，同时加载 N+1 层所需要的 kvcache 数据；
	- 涉及外部存储时（如 SSD、远端存储）：尽可能预取，但不保证；延迟优先：如果某个请求被调度执行，则停止预取操作；吞吐量优先：更积极地预取，从而尽量提高整体吞吐；

### Pick your favorite storage backend or bring your own!

- 接入一个新的存储后端，只需要实现三个函数：get、exist、set；
- 所有复杂的、与存储后端无关的控制面工作，都在中央的 cache controller 负责；