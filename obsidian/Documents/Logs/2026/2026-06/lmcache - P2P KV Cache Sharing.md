TLDR

- 看起来是一个类似去借其他节点的 kv cache 的机制，不会主动写，看起来也不会将 kv cache 打散；
- 在 cache miss 时，requester 按 zmq rpc 会广播一个 lookup request 到 peer node 上；

---

- P2P kv cache sharing：在本地没有特定 prefix 的 cache 时，可以从一个 peer 节点上读取这个 prefix KV cache，使用 RDMA 网络；
- 这个 transfer 是一个 one-sided RDMA read，读取到自己的 L1 buffer；
- 在 RDMA 网络中，这个操作显著地快于重新计算，或者访问对象存储；

## How it works

- Coordinator：一个小的 http service（每个 deployment 一个），跟踪每个 lmcache server 的存活性；每个 server 会找它定期注册 heartbeat；只用于管理 membership；
- LMCache Server：对每个 live peer，开一个连接，用于读取 KV；
- transfer channel：真正做 remote memory read 的 RDMA layer；每个 server 会注册自己的 L1 buffer 进来，从而 peer 可以读取到它；

在 cache miss 时，一个 node 会询问拥有这个 prefix 的 peer 执行 `lock` 和 `locate`，得到 remote address，执行 RDMA read 到自己的 L1，并用这个 L1 来处理请求。

## Requirements

- A coordinator
- An RDMA capable network
- A single, contiguous L1 region：P2P 和 GDS L1 tier 或 Device DAX L1 tier 不兼容；

## Transfer engine backends

- 默认是 nixl


