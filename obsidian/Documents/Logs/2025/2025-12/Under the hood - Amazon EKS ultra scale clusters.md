EKS 声明支持了 10w 节点，一个集群里跑了 160w 个 aws trainium 芯片或者 80w 个 nvidia gpu

### Next generation data store

- **Consensus offloaded**：将 raft 的 consensus 机制切换到了 journal，一个 aws 内部跑了 10 年的基础设施；将 consensus offload 给 journal，允许自由地 scale etcd 实例，不必受 quorum 数量约束；
- **In-memory database**：作者将 etcd 后面的 boltdb 换成了纯的 in-memory store；这使得 latency 变得 predictable；也将存储的上限推广到了 20gb；
- **Partitioned key-space**：k8s 原生就支持按 k8s 中的资源类型去拆分 etcd 集群；之前没有在 etcd 这一层实现；aws 做的支持下，允许配置静态规则，来使特定资源路由到特定的 etcd 集群上；实践来看，静态的规则已经足够好了；

### Extreme throughput API servers

- 写入负载的可扩展性在 etcd 这一层已经得到了解决，API Server 这一侧关注的主要是读的能力的扩展了；
- **API server and webhook tuning:**  We achieved optimal performance by carefully tuning various configurations such as request timeouts, retry strategies, work parallelism, throttling rules, http connection lifetime, garbage collection, and etcd client settings.
- **Consistent reads from cache:** k8s v1.31 提供了<mark>强一致的 cache read；过去，需要 label 或者 field based filtering 需要 API Server 去 list 出来完整的集合，然后执行过滤后返回给客户端；新的机制下，会跟进 etcd 的数据新鲜度，如果是最新，则直接从 API Server cache 读取并返回</mark>；这个优化减少了服务端的 CPU 利用率 30%，对特定的 LIST 查询的性能提升了三倍；
- **Reading large collections efficiently**：大集群往往有大量的 collection，k8s v1.33 中加入的 streaming list 响应对内存的利用率、list request 的并发度有大幅提升（大约八倍）；（似乎就是从全量传输改为了分页传输）
- **Binary encoding for custom resources**：Kubernetes v1.32 引入了 Concise Binary Object Representation（CBOR）编码机制（在 alpha 阶段），This also benefits high-throughput high-cardinality CR clients, such as node daemons commonly used by AI/ML customers.

### Lightning fast cluster controllers

- **Minimizing lock contention**: 在 worker 在 shared informer 上执行 large list 操作时，有大量读写锁的 contention，导致 incoming events 处理延迟，乃至更多二阶影响比如 piled up queues、高内存占用乃至最终崩溃；
- **Scheduling optimizations**：we achieved consistently a high throughput of 500 pods/second even at the 100K node scale by carefully tailoring scheduler plugins based on the workload and optimizing node filtering/scoring parameters.


### Scaling the cluster network

- EKS 支持 native 的 VPC 网络，不必承担封包的开销；此外，也支持高级的网络特性比如 custom subnet、security groups、elastic fabric adapter 等；
- **Moving from IP assignments to warm prefixes:** 随着集群的增长，你不得不关注 Network Address Usage（NAU）这种指标；VPC CNI 默认位每个 pod 分配一个 IP，使用 prefix 模式之后改为每个 node 分配一个 prefix，然后 prefix 内给 pod 分配 IP；
- **Maximizing network performance:** VPC CNI 默认会选择一个网卡给 ENI；With accelerated computing instance types supporting multiple network cards, we enabled plugin support to create pod ENIs on additional network cards.相当于一个 pod 可以配多个 ENI 网卡；

### Rapid container image pulls

- We observed that ultra scale AI/ML workloads tend to use large container images such as PyTorch training, TensorFlow notebooks, and SageMaker distribution, often exceeding 5 GB
- 做了一个 Seekable OCI fast pull，支持并发下载、解压；Our testing demonstrates up to a 2x reduction in overall image download and unpack compared to the default
- 此外，专门开了一个 Amazon S3 VPC endpoint 能保证每个 AZ 100 GB/s 的带宽；