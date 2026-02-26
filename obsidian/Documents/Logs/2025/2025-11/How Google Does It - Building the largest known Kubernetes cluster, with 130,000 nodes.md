
- 这个规模的 scaling 不只是 node 数量的增加，也需要扩展其他关键的维度，比如 Pod creation 和 scheding throuhput。
- 作者在测试中，验证了 1000 个 Pod 的创建。
### The rise of the mega cluster

- 一张 nvidia gb200 的 GPU 就需要 2700W 的电。
- 对于超过 10w 节点的 AI 平台，必须拥有一个稳健的跨集群解决方案，能够协同分布式 training 和跨集群的 RL。
- 作者有积极地在一些 MultiKueue 之类的工具上投资。
- 同时，也推进高性能的 RDMA 网络方案 DRANET，从而提高拓扑感知能力，最大化 AI 网络的性能。
## Key architectural innovations

### Optimized read scalability

- 首先需要的就是一个高性能的、强一致的、snapshottable 的 API Server watch cache。
- 13w 个节点之下，到 api server 的 read request 的数量是压倒性的。
- 优化首先是 Consistent Read from Cache 特性（KEP-2340），它能够允许 API server 强一致地从它的 in-memory cache 读取数据；这显著地减少了访问 object storage db 的压力，比如 filtered list 请求（特定 node 上的所有 pod）；大约的实现原理大约是，等待 cache 同步到后端存储的点位；
- 在这套基础之上，Snapshottable API Server Cache 特性（KEP 4988）进一步提高了 API Server 响应 LIST 请求能力，允许从一个特定的版本号读取分页的数据；通过生成一个 BTree 的 cache 的 “snapshot”，API Server 能够高效地响应 List 请求，减少对后端数据库的访问；
- 这两套机制减少了读放大的问题，保证 api server 总是能够快速响应；

### An optimized distributed storage backend

- 存储改到了 google 的 spanner 上面；
- 130k 节点的规模下，需要 13000 qps 来更新 lease 对象，从而保证关键的集群操作比如 node health check 不会成为瓶颈；

### Kueue for advanced job queueing

- 默认的 k8s scheduler 用于调度单独的 Pod，但是复杂的 AI/ML 环境需要更精细的、Job 粒度的管理。
- Kueue 是一个 job queuing controller 为 k8s 添加 batch 能力；它能够决定何时一个 job 能够在集群中 admitted，参考 fair sharing policy、priority、resource quota、all-or-nothing scheduling 等机制；

### Future of scheduling: Enhanced workload awareness

- 目标是将 Pod centric 转移向 workload-centric 的调度；
- 主要是 KEP-4671: Gang Scheduling；

### GCS FUSE for data access

- Cloud Storage FUSE 支持 parallel downloads 和 caching，配合 region 内的 Anywhere Cache，允许访问模型数据就像访问一个本地文件系统一样，减少 70% latency。
- 此外，也有 Google Cloud Managed Lustre，一个全托管的 persitent zonal storage solution，能够存储 pb 级别的容量，TB/s 的带宽、sub-ms 的延迟；

## Benchmarking GKE for large-scale, dynamic AI workloads

### Demonstrating GKE’s scalability across dimensions

### Intelligent workload management with Kueue

### Low pod startup latency

- P99 的 pod startup 时间大约 10s；