### 2.2 RoCE and Collective Communication

- RDMA 会实现一组 “verb” API，比如 read 和 write
- TCP/IP 中，一个包需要经过内核处理最后拷贝到内存，而 RDMA 能够在发送和接收两端，都 bypass 掉内核，直接写入到对端的内存 
- RDMA 的封包、解包等工作，都是 NIC 硬件完成的
- **集合通信**是 workload 与 NIC 之间的接口抽象，它将 `AllReduce` 等 "collective" 翻译成逻辑上的拓扑实现（比如 Ring 或者 Tree），最后拆解为点对点的、GPU 与 verb 之间的 data transaction 的调度；
- 这些 transaction 需要 GPU-to-RDMA 的 NIC 支持才能有最优的性能
- collective library 通过 Queue-Pairs 之间调度 verb
- 比如，NCCL 将所有的 collective 算法和点对点通信的语义，通过 RDMA write 的操作进行实现
- 每个 GPU to GPU 的 pairwise transaction 可以有多个 channel，每个 NIC-to-NIC pairwise transaction 可以有多个 queue pairs；

collective 的网络负载的特征：

- 首先，"collective" 由 parallel 策略决定，比如 Distributedd Data Parallel 使用 AllReduce、Fully Sharded Data Parallel 使用 `AllGather` 和 `ReduceScatter`
- 其次，collective 可以产生几种不同的网络 traffic 特征，比如 AllToAllv 会产生一个全连接的流量特征，可能导致临时的 congestion，However, its high number of active flows simplifies routing, reducing persistent congestion risks with hashing schemes
- Third, the choice of logical topology from collective operations impacts network congestion and data exchange between GPUs. 比如，AllReduce 按 Ring 和按 Tree 来实现，有不同的 congestion 和哈希冲突的后果。

### 3 HARDWARE

