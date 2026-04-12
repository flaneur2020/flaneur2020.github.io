tldr;

- 做了一个 kv messenger 的抽象；
- kv cache 的传输并不等待前向传播完成，而是一旦模型完成对每一层 KV 缓存条目的追加，系统便会立即启动 KV 页的复制操作；
- 传输模块维护一个专用线程，持续轮询一个计数器，该计数器在注意力机制的输出完成后递增；
- D 节点并不会收到显式的通知，而是它通过计数器跟踪已完成的操作数量；
- 好像 KV cache 在 TP 时有关于 shard 的考虑，这部分没仔细看；
- 通过将 P 与 D 分离后，每个请求的 TTFT 增加了约 100 毫秒的延迟，但单个 P 节点能够持续为三个 D 节点维持稳定的批处理规模，实现超过 90 TPS 的吞吐量，同时每个解码节点仅需处理约 1 QPS 的负载。

----

## KV Messenger

- 在 perplexity 内部，会通过一个 kv messenger 来协调 LLM engine 通过网络来传输 KV cache。
- 在 P 节点上，messenger 会从 D 节点接收请求，handing them over to the batch scheduler and keeping track of the forward pass execution to dispatch KV caches with as little latency as possible。
- 在 D 节点上，messenger 会 after un-evictable pages are allocated, the messenger blocks the request from being scheduled for decode until it is notified of the completion of the KV cache and decoder context transfers.

![[Pasted image 20260315142639.png]]

- 传输需要高吞吐、低延迟的网络，因此作者的实现围绕着 RDMA、同时支持 EFA 和 ConnectX NIC。
- KV Messenger 基于 libfabric，使用作者自己的 fabric-lib 在 RDMA 上提供高级的低延迟的抽象，允许实现高速的数据和元信息的传输；
- 在后台，fabric-lib 协调 GPU 和它直接连接的 NIC 来拷贝数据。
- <mark>在收到请求时，P 节点会分配一组 KV 页，使用自己本地的 engine 调度 prefill 请求；为最小化延迟，KV 传输不等待前向传播完成：相反，一旦模型完成对每一层 KV 缓存条目的追加，系统便会立即启动 KV 页的复制操作；</mark>
- 由于预填充请求支持分块（chunked），批调度器会在执行前通知 KV 传输模块当前待调度的分块信息。
- 为在支持 CUDA 图（CUDA Graphs）的同时仍能追踪各层的进度，传输模块维持一个专用线程，持续轮询一个计数器，该计数器在注意力机制的输出投影完成后递增；
- 该计数器仅在分片环境中的主节点（lead node）上维护：尽管 KV 缓存条目在追加后、注意力计算前即已有效，但输出投影操作会在多个设备（rank）间进行归约（reduction），从而隐式地实现了同步。
- 一旦检测到计数器发生变化，传输模块便会收到通知，并调用 fabric-lib 启动对应层的 KV 缓存传输。

![[Pasted image 20260315143604.png]]

- 当最后一个分块的传输完成后，任何额外的元数据（如推测解码或 MTP 所需的 logits 和隐藏状态）也会被一并复制到 D 节点。
- 这些复制操作同样通过 RDMA 完成，源与目标均为预先分配的缓冲区。
- 当最后一个分块的所有待处理传输完成后，P 节点会释放对应的 KV 页；
- D 节点并不会收到显式的通知，而是，它通过计数器跟踪已完成的操作数量；
- 一旦已知数量的页和上下文复制全部完成，fabric-lib 会调用 KV 传输模块，通知其某个请求已就绪，可进入解码阶段。

## Sharded KV Cache Transfers

- 如果 P 喝 D 节点依赖 TP 且 shard、replicate 这些 KV cache，则一个单独的 transfer engin 协同多个设备，来发送、接收所有的 replica 的页面。
- 若源端与目标端的分片结构完全一致，则传输极为简单，因为源与目标设备及其对应的 KV 页之间存在一一映射关系。在此情况下，分片机制本身有助于降低传输延迟：通过使用更多 GPU，可同时启用更多关联的网卡，从而更接近网络带宽的极限利用率。
- 然而，若源端与目标端的分片结构不匹配（如分片数量或粒度不同），传输引擎必须根据源与目标切片的比例，对 KV 页进行拆分或重组，以完成正确的数据映射。

## Disaggregated Deployments

### DeepSeek-R1

- 对于 DeepSeek 模型，会同时考虑 TP 和 DP 部署。
- DeepSeek 依赖 Multi Head Latent Attention，会对 KV Cache 进行压缩，因为所有的 KV head 都被压缩到了单独的 latent vector 中，TP 不能够 shard 这个 kv cache，而是需要将它 replicate 到所有的 rank 上；在解压缩之后，再进行 sharding，每个 rank 从不同的 head 中解压出自己关心的部分。
- 在跨 TP 的跨节点部署上，prefiller 和 decoder 是同样地 shard 的；
- 在 DP 部署中，TP 的 rank size 比 DP 的 rank 要小。
- 在混合 P 与 D 的部署模式下，我们的 R1 系统因频繁出现数百毫秒量级的预填充中断，难以稳定突破 50 TPS。
- 相比之下，<mark>通过将 P 与 D 分离后，每个请求的 TTFT 增加了约 100 毫秒的延迟，但单个 P 节点能够持续为三个 D 节点维持稳定的批处理规模，实现超过 90 TPS 的吞吐量，同时每个解码节点仅需处理约 1 QPS 的负载。</mark>
- 在采用数据并行（data-parallel）部署的场景中，TPS 略低，约为 50，但每个实例的每个计算单元（rank）可独立处理 1 QPS 的负载，且单个节点可容纳 8 个 rank，整体系统仍具备良好的扩展能力。