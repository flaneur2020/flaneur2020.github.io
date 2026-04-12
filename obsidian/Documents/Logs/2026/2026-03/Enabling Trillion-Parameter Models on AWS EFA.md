TP 和 DP 可以直接使用集合通信进行实现，而 EP 需要在稀疏的 P2P 通信中路由。

> A dispatch kernel must split the set of incoming tokens and dispatch them to the ranks hosting the experts, while its dual problem, the combine kernel, must receive their processed counterparts and compute their weighted average.

这些传输并非和集合通信的原语匹配，要降低延迟，需要自定义的 kernel。

## Inference over InfiniBand

TP 和 EP 在单个节点内可以很高效地扩展，因为 Nvlink 有高速的吞吐（900GB/s）和 us 级的延迟。

aws 的 p5en 机型最多 8 个 H200 GPU，它的 HBM 上限为 1120GB。一些模型仍需要拆分到多台机器上。

infiniband 已经很快了，但是吞吐只能在 50GB/s 这个水平，并且会增加几百 us 的延迟。

幸运的是，这部分延迟可以通过 shared expert、micro-batching、computation-communication overlapping 来隐藏掉。

deepseek 在 deepseek-r1 的部署上，使用 H800 GPU、nvidia connectX-7 
infiniband adapter 首先应用了这套技术。

但是在 aws 上，情况会有不同，因为需要使用他们的 EFA。

> Even though these deliver peak 400Gbps throughput on the collectives typically used in training workloads, they fall slightly short of ConnectX-7 on the message sizes exchanged during MoE dispatch and combine.

此外，EFA 并不支持 GPUDirect Async。需要一个 CPU proxy thread 来 bridge 起来 GPU 和 NIC 来初始化传输。

> Besides the added complexity, the additional PCIe transactions add microseconds worth of overhead to all transfers.

nvshmem 这样的框架允许提供设备无关的实现，但是这些通用的 API 存在开销，在 cost 和 latency 意义上都有。

## NVSHMEM-based _pplx-kernels_

作者之前的 kernel 通过 nvshmem 抽象掉了底层传输的实现。

它们依赖 `nvshmemx_putmem_signal_nbi_warp` 来按 token 传输，使用 atomic counter 实现 peer 之间的协调。

这在 ConnectX-7 NICs with IBGDA 上的性能尚可。

the proxy-based ConnectX-7 IBRC implementation was significantly slower, 而 EFA implementation 无法将延迟降低到 1ms 以下。

作者设计了基于 EFA 的新内核，对外保持相同的接口，但在底层优化了设备内核与主机代理之间的交互。

## MoE all-to-all over EFA

