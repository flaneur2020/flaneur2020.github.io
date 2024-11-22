https://netflixtechblog.com/netflixs-distributed-counter-abstraction-8d0c45eb66b2

Distributed Counter 建立在 Distributed TimeSeries 的基础上，而且这些都建立在 Data Gateway Control Plane 来控制 shard、配置、部署；

这个 counter service 会接近精确，而且低延时。

## Use Cases and Requirements

counter service 会用来跟踪用户的交互，监控特定 feature 或者交互展示给用户的次数，在 A/B Test 中各种计数；

这些 use case 可以拆分为：

1. Best-effort：这些场景不需要绝对的精确，但是希望尽量实时的响应；
2. Eventual Consistent：需要尽量准确，但是稍微有一定延迟和更高的 infra 成本；

## Distributed Counter Abstraction

接口和 AtomicInteger 类似：

AddCount / AddAndGetCount:

```
{  
"namespace": "my_dataset",  
"counter_name": "counter123",  
"delta": 2,  
"idempotency_token": {  
"token": "some_event_id",  
"generation_time": "2024-10-05T14:48:00Z"  
}  
```

## Types of Counters

三种：Best-Effort、Eventually Consistent、

### Best Effort Regional Counter

Best-Effort 基于 EVCache，适用于 ABTest；

没有 cross-region replication、没有 consistency guarantee、不能幂等，不能 retry；

### Eventually Consistent Global Counter

**Approach 1: Storing a Single Row per Counter**

**Approach 2: Per Instance Aggregation**

每个节点自己根据日志进行聚合，并根据 interval 写入到持久的存储中；

风险是重启的话容易丢失数据、不能方便地重置 counter、没有幂等性；

**Approach 3: Using Durable Queues**

加了一个 kafka；

弄了一个 kafka stream 或者 flink 做 windowed aggregation；

挑战是 Potiential Delays、Rebalancing Partitions；

**Approach 4: Event Log of Individual Increments**

加一个 event_time 和 event_id 作为幂等 key；

## Netflix’s Approach

log 每个计数操作作为事件，在队列中后台持续按 sliding window 聚合；

使用 netflix 内部的 TimeSeries Data Abstraction 作为 event store；

TimeSeries Data Abstraction 底下用的 Cassandra；

![[Pasted image 20241117115656.png]]

### Aggregating Count Events

设置了一个水位，使 aggregation 总是在一个 immutable window 中

![[Pasted image 20241117120745.png]]

每次聚合，取 `lastRollupTs` 和 `now() - acceptLimit()` 之间的 event；

### Rollup Store

![[Pasted image 20241117120838.png]]

聚合后的结果会写入到 rollup store 表中。每个 counter 保存一个 lastRollupTs 和 lastWriteTs。

lastWriteTs 是个会随着 event 的进入而更新的字段，用于发现需要 rollup 的 counter；

### Rollup Cache

给 rollup 会加一个 EVCache。