使用 source-side 的 CPU engine replica 和 P2P RDMA Transfer，将传输 1T 参数的 kimi-k2 7 倍提升（53s -> 7.2s），代价是每个 training rank 一个额外的 inference engine replica（32G）在 CPU 内存上；

## Background

- NCCL 针对 all-gather、broadcast 等原语做了优化，自动探测硬件的拓扑，按 ring 或者 tree 来协调控制流；
- 不过，这依赖集合通信的语义，需要每个 rank 按同一个 shape 执行同一个操作；
- 这个设计，在动态的环境中不那么高效：NCCL 在 lock step 中操作，意味着有任何一个 slow start 的 receiver 都会将整个 group 拖慢；
- RDMA 允许机器访问远端内存，同时 bypass 远端的 CPU 和内核网络栈；
- 与 NCCL 有全局的同步不同，RDMA 允许任意两个 endpoint 独立并发通信；
- RL Weight Transfer Problem
	- 在大规模的 RL 训练中，从 trainer 到 inference engine 的 weight transfer 是一个 critical path；
	- 在 weight transfer 中，RL training 是 halt 的；
	- 随着模型 size 的增长，transfer 必须能够跨机器、rack 增长，都需要碰到有限的带宽；
	- 目前开源解决方案基于 NCCL-based workflow 的都依赖一个 broadcast 来自一个 single rank，快速地成为了瓶颈；![](https://www.lmsys.org/images/blog/p2p-update/blog-1.png)

## Challenges with Existing NCCL Broadcast

- 目前 NCCL broadcast 的方案有如下挑战：
	- 同一份数据会被传输很多次；
	- Inactivity：多数 trainer rank 会在 transfer 期间保持 idle；
	- rigidity：在定义之后，NCCL communication group 会被固定，无法动态扩展；


## Design

作者的设计，从中心化的 broadcast，切到了去中心化的 RDMA 的 P2P mapping；

- Source-side engine replicas：会在 cpu 内存中，创建 model 的 replica；
- P2P Mapping Heuristics：在 rank 之间，构造一个 p2p 的 mapping，相对于少数 rank 广播 everything，每个 trainer rank 都会参与传输一部分 shard；
- zero-copy transfer：使用 TransferEngine，在启动时将内存都注册进来，减少昂贵的 CUDA IPC 序列话、kernel side copies；
