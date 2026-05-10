> 所以这里提出的安全复用策略就是：禁用 RDMA Write，RDMA Read 成功后向远端确认 Buffer 还未修改。

| 本地                             | 远端                                                  |
| ------------------------------ | --------------------------------------------------- |
| 1. 发起 RPC                      |                                                     |
|                                | 2. 收到 RPC，准备好 Buffer，返回 Buffer 地址                   |
| 3. 发起 RDMA Read，等待完成           |                                                     |
| 4. 发起 RPC，确认刚才的 Buffer 没有被修改   |                                                     |
|                                | 5. 收到 RPC，检查 Buffer 是否已经被复用，返回结果，同时可以安全的释放 Buffer 了 |
| 6. 收到检查结果。如果已经被复用，则本轮通信失败，可以重试 |                                                     |
> 本质上，这套方案是在 One-sided RDMA + Two-sided RPC 的组合里，用 RPC 来弥补 RDMA 在 buffer 生命周期和错误语义上的不足。

> 笔者坚持不使用 RDMA Write 的另一个原因是，远端并发的 RDMA Write 会很容易触发网络上的拥塞。只有 RDMA Read 的话可以很方便地在应用层做 Receiver Driven 的流量控制，避免网络层拥塞。