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

早期 SGLang 对 router 还不大成熟，多个实例之间没有数据共享。小米做了一个无状态的 LLM-router，用 redis 作为中心存储避免单服务故障后的 kvcache 调度回退现象。

### 3.1 KVCache 与负载亲和调度

Router 中通过将分发过的请求维护在 Radix 前缀树中，实现了 KVCache 亲和调度。

在多个 Prefill 实例中，选择已经缓存当前前缀的节点。

该策略上线后，L2 缓存命中率提升了 25%。单机输入吞吐提升了约 30%。

```
# 选择 score 最大的 worker，含义：缓存命中率高 + 负载低 = 得分高 = 优先选择
score(worker) = matchWeight × prefix_match_percentage − normalized_load
```

### 3.2 TTFT 优化

> 当模型服务出现排队现象时，传统的 First Come First Serve 策略没有考虑较高命中率和较低命中率请求的优先级关系，使得命中 Cache 更多但真实计算 token 更少的请求有可能等待 Cache 命中率更低的请求推理结束后才能开始推理。整体服务的 TTFT P99 会变得异常的长，拖慢整体吞吐的平均性能。
>
> 为了解决这个问题，在 waiting 队列选择优先执行 prefill 服务时，Router 侧优先调度真实计算 token 数更少的请求，避免这些原本计算时间很短的请求被阻塞而导致 P99 过度劣化的问题。同时，这种策略会导致某些请求长期得不到调度的饥饿现象，所以我们还增加了等待时间的惩罚机制来平衡这一现象。结果表明，该策略对于较短的请求并不会降低服务质量，而对于较长的请求，该策略最高可以将 TTFT 的 P90 指标降低 **30%**。


## 四、Prefill 优化

### 4.1 分布式配置

- 理论上，Prefill 阶段 EP 越小，性能与吞吐越优：跨机器更少，通信开销更低；DP 数更少；每台机器承载的 Expert 数量更多，MoE 均衡性更好；
	- EP 越小，表示每个 EP 组内的卡越少，比如 EP=2，表示 EP 组里有两张卡；
	- EP小 → 在满足显存约束的前提下，可以用更少的总卡数完成同样的工作；
- 但是 EP 大小受显存约束，需要满足模型参数与 KVCache 的显存占用；
	- EP 大时，单卡显存压力小，但需要更多卡，DP组被迫增多 → 负载不均问题突出;
	- EP 小时，单卡显存压力大（需要存更多专家），但如果能通过 KVCache 优化解决显存问题，就可以用更少的总卡数 → DP组数减少 → 负载更均衡
- 将 EP 缩减至原先的 1/2，端到端性能提升约 40%

### 4.2 长度分桶策略

- 采用**三级长度分桶策略**（0–64K / 64K–256K / 256K–1M），将负载特征相近的请求聚合至同一桶内做计算；

### 4.3 MoE 负载均衡

> 由于预训练阶段引入了负载均衡的训练目标、且训练较为稳定，模型在训练时已学习到较为均匀的专家分配策略。推理阶段，在未启用任何专家负载均衡策略的条件下，各层平均专家负载度（一层中所有 rank 的平均 token 数与该层 rank 最大 token 数之比）约为 0.85，已处于较优分布水平。因此，我们<mark>目前并未引入任何专家负载均衡策略</mark>。


## 五、Decode 优化

tbd