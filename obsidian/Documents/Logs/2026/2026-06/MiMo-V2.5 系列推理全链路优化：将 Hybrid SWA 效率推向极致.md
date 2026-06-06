- 使用 Hybrid Sliding Window Attention 将 KVCache 存储降低到 Full Attention 的 1/7；将局部窗口注意力（Sliding Window Attention, SWA）与全局注意力（Full Attention）之间进行分层混合；
- 多模态 Encoder 支持视觉、音频、视频等跨模态理解

## 一、Hybrid SWA 架构的推理效率优势

- 一共 70 层，其中 10 层为 Full Attention，其余 60 层为 SWA，SWA 的大小为 128；
- Hybrid SWA 架构的计算量约为 Full Attention 的 1/7；该差距约等于 Prefill 成本的理论缩减；
- Decode 阶段的延迟正比于模型参数加 KVCache 的读取量；<mark>在长序列下，KVCache 的体积可能远超模型参数</mark>，因此 KVCache 存储的减少几乎直接等价于长序列场景下的 decode 成本的降低；

## 二、KVCache 系统重构

### 2.1 SWA KVCache 管理

- 一开始使用 SGLang v0.5.5 作为后端，但是之前 SGLang 的 Hicache 不支持 SWA；
- SWA KVCache 双池
	- Full Attention 需要保留全序列，而 SWA 层只需要维护滑动窗口内的 KV；
	- 在传统单一 KV Pool 的设计下，系统必须按照 O(N) 为所有层统一分配显存，使得 SWA 层也分了个 O(N) 的 KVCache；
	- 为此，将 KVCache 拆分为了 Full Attention 和 SWA 两个独立的池；
	- 物理层面：SWA Pool 按窗口大小配置一个固定的容量，并且支持独立 eviction；
	- 逻辑层面：与上层仍暴露单一序列视图，由 Full Attention 索引作为权威索引；
- KV Cache 异步拉取
	- SWA 层只需 prefetch 极少量 KVCache，这使得从 Host prefetch KVCache 到 Device 的过程通过 layerwise 粒度调度，就能实现完美的 overlap；
- SWA-aware 前缀缓存树
	- 传统 RadixAttention 有一个朴素的假设：token 序列相等 -> KV 也相等；
	- 节点需要同时维护两套索引：Full Attention 段索引和 SWA 段索引；淘汰也要分别管理；
- KVCache 命中率提升优化
	- 改为 SWA-aware 后，device、host、storage backend 几端会各自维护一套“哪些位置有有效 SWA” 的状态；

### 2.2 GCache：高性能分布式缓存基础设施

支持文件和 KV 语义、支持内存/磁盘/远程的多级缓存、支持 shm 内存持久化和全链路零拷贝，支持高并发非阻塞 IO 和 RDMA 通信等特性；

![[Pasted image 20260606103005.png]]

- gcache 架构设计
	1. 去中心化的架构：master 只管理发现，不参与数据路径；
	2. 服务端同时支持内存和磁盘缓存：内存的冷数据会淘汰到磁盘；内存支持持久化到 shm，重启服务不丢缓存；支持平滑扩容或者缩容；
	3. 提供多语言 SDK，启动专属线程，将用户请求进行切片和派发；
- 网络优化
	- 目前主流的 GPU 机型，都配置了 8 张 400G 高性能网卡；然而即使考虑了 PD 分离，<mark>现阶段的推理框架还是很难跑满网络带宽</mark>，以至于业内出现了要给网卡减配的声音；
	- GCache 优先使用 GPU 网卡，而不是前端网卡进行通信，并在通信模块上做了大量优化，如 NUMA 绑定、同轨亲和等；
	- <mark>使用 1MB 大小的 IO，单进程的 RDMA 读吞吐可以达到 170GiB/s</mark>，而延迟只有 280us；
	- <mark>GDR 场景下，由于 HBM 的带宽更高，单进程可以跑到 350GiB/s</mark>
- 存储成本优化
	- GCache 优先采用 GPU 机器上混部的方式，接管了 Prefill 和 Decode 节点的部分内存和机器自带的 NVMe SSD，没有额外的机器；
- 稳定性保障
	- 基本上每天都会遇到 server 所在的机器故障；
	- 结合底层平台提供的硬件检测功能，提前发现故障，使用自动化流程进行数据迁移；此外设置一个较低的 SDK 超时时间，丢了就重算；
	- <mark>GCache 得以在混布状态下一直保持单副本存储</mark>
- 针对缓存命中率的讨论
	- 更低的存储占用、辅以更稳定的大容量 GCache 作为 L3 存储，得以显著延长 Cache 的 TTL，从而大幅提升 KV Cache 的缓存命中率；
	- <mark>KVCache 的淘汰本质上源于存储容量约束</mark>；
	- 模型上线以来，在主流优质 harness 框架下，服务端 KV Cache 命中率平均可达 **93%**；对于高强度、长周期使用的个人用户，该指标更可攀升至 **95%** 以上乃至更高。

## 三、调度优化

