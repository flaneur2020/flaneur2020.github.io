tldr

- 有一个 <mark>cost 预估的模型</mark>，结合一个 informer 抽象来收集 model server 的状态，来预估一个 request 的 cost，以及目标的 model server 是否处于饱和
- 通过 EPP 获取 model server 的 endpoint 的 candidate
- 设计了一套类似 k8s scheduler 的接口，可以插拔，可以挂过滤逻辑，和排序 score 逻辑

---

inference gateway 会结合请求的 cost 与模型的动态负载能力，争取提高优先级和更可预测的 latency。

这里包括：

1. prompt length
2. output length
3. 当前的流量分布
4. 后端的 kv cache 余量
5. workload 的 latency 目标
6. anticipated savings from prefix cache aware routing
7. 异构加速卡的性能指标
8. 后端的拓扑（比如 pd 分离、模型微调 a/b）

在设计 scheduler 时，会希望它有比较好的可扩展性。

> the reference endpoint picker implementation should support build time plugins, through a clearly defined interface, or fully delegating scheduling decisions per pool to an alternative **replacement scheduler**

## Non-Goals

- Dynamic reconfiguration of the reference scheduler algorithms at runtime
- Being a general scheduler framework for any type of load balancing besides inference
- Determining the characteristics of the underlying model servers and hardware
## Proposal

### Definitions

- **Scheduler**：会放在请求队列之后。是一个算法，结合不同的优化维度、model server 信息，选出来 endpoint，能够最好的满足当前的 workload，并最大化地提高利用率。
- **Saturation**：一个 model server 会攒 batch 的方式来处理请求。 每个 batch cycle 的延迟如果递增，每个 request 的延迟都会增加，不过这时吞吐往往也会增长。Saturation 用于描述一个点位，latency/throughput 不再成立。在 inference gateway 看来，有两个 saturation 定义：
	- hard：model server 彻底进入超载状态，request 已经不能再接收了；
	- soft：根据 latency sensitivity 间接得出的一个 saturation limit；
- **Request Cost**：一个 inference request 的 cost 指一个 request 消耗的资源量。在这里，指 GPU 内存和 GPU 计算时间，通常根据 saturation 的维度来衡量。

### Personas

- OSS Algorithm Researcher
- Production Algorithm Contributor
- Inference Platform Admin
- Inference Workload Owner
### Requirements

- 允许 model server 更可预测地到达 saturation
- 使用户可见的请求 latency 更可预测
- 允许在 model server 到达 saturation 之前，实现不同 workload 的隔离（好像意思是如果一个 model server 到达 saturation 的话，会调度流量到空闲的 model server 去）
- 结合优先级与公平性，在 saturation 之后，调度 workload 到其他的 model server

### Design

- 理解 incoming request 的 cost，以及安置到目标 model server 的影响
- 跟踪之前的请求的 cost，避免使目标 server 超载
- 允许集成未来的 cost 预估特性到一个整体的 cost model 中，比如 prefix cache routing
- 允许异构的计算资源，有不同的容量、latency、memory 和 features

### reference scheduler

- reference scheduler 会从 EPP 中拿到一组 candidate endpoints；
- reference scheduler 通过 infromer 获取当前 model server 的状态（目前的 informer 是通过一个 fast-polling loop 按 model server protocol 去获取 model server 的 metrics）
- reference scheduler 会有一组 predicate 和 filters，移除不匹配的 match；能够打 score，排优先级；（有点参考 k8s 的 scheduler 的感觉）
- 一旦选择到一个 endpoint，会假设这个 endpoint 在运行状态。