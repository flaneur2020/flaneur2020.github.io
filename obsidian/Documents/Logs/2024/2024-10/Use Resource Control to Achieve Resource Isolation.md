tidb 有两种资源控制能力：1. tidb 层的 flow control capability；2. tikv 层的 priority scheduling capability；

能够允许 tidb 基于 quota 控制用户读写的 flow。

tidb flow control 用了一个<mark>令牌桶</mark>算法。如果没有令牌，且没有 BURSTABLE 选项，这个请求就会等待令牌桶中出新的令牌，并受 timeout 限制。

tikv scheduling：可以设置一个绝对的优先级。不同的资源按优先级进行调度。每个 resource group 会跟踪一个 RU_PER_SEC 的值。

tiflash 的 flow control：可以控制每个 query 的 CPU 消费，转换为 Request Units，然后再基于令牌桶。

tiflash 的 scheduling：当系统资源不足时，tiflash 在多个 resource group 之间根据优先级来调度 task。

## Scenarios for resource control

1. 可以组合多个 small 和 medium sized 应用到一个单独的 tidb 集群中；多个业务可以混用一个集群；
2. 可以考虑将多个测试环境融合为一个 tidb 集群，或者将 batch task 或者大查询到一个单独的 resource group 中；
3. 如果有混合的 workload，可以将不同的 workload 拆分到不同的 resource group；
4. 当集群遇到性能问题时，可以通过控制 resource group 的配置，临时调整资源的限制；

