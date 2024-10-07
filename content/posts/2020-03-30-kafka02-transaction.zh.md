---
date: "2020-03-30T00:00:00Z"
title: 'Kafka 笔记 02: 事务'
---

Kafka 过去一直保障到 At Least Once 语义，用户生产的消息不会丢失，但是可能发生重复。比如，broker 接收 Producer 消息后崩溃，但是没来得及向 Producer 返回 ack，而使 Producer 重试而导致消息重复。为了避免 Producer 到 Broker 端的消息重复，Kafka 引进了 Idempotent Producer 特性，使每条消息携带 Producer ID 和递增的 sequence 实现去重，从而能够实现生产侧的 Exactly Once。

但是仅靠生产消息的 Exactly Once 语义仍不能满足 Kafka Stream 这种流式处理场景的需要。在流式处理场景中，大部分操作都属于从一个或者多个 Topic Partition 消费数据，经处理后写入另外的多个 Topic Partition，即 Consume-Transform-Produce 过程。这一过程涉及多轮读写操作，而系统随时可能崩溃，崩溃重试时不希望看到消息产生重复消费而导致处理结果不确定，会更希望有事务原子性，做到要么全成功，要么全没发生。

简而言之，Kafka 事务能保证向多个 Topic Partition 的原子写入。

至于 Consumer Group 的消费操作，在 Kafka 也是基于向内部 __consumer_offsets 这个 Topic 写入点位实现的 commit。因此，事务中的消费操作也可以视作向多个 Topic Partition 的原子写入之一。

## 事务设计

按《Transactional Messaging in Kafka》中的介绍，在 Kafka 事务的实现方面存在一些设计约束：

* 要使同一 Topic 做到同时支持事务性消息和普通的非事务性消息，并不影响非事务性消息的性能；
* 要使事务性消息的性能尽可能接近普通非事务性消息的性能；
* 尽量不增加客户端的复杂性；

事务的实现大致上是一个两阶段提交过程，一是增加 Transaction ID，二是增加事务控制消息，三是增加一个 Transaction Coordinator 组件。

![](/images/kafka-transaction-summary.png)

### 事务控制消息

Kafka 事务在设计上要求同一 Topic 中既包含事务性消息，也包含非事务性消息。相比非事务性消息，事务性消息需要携带额外的事务信息，此外，还需要通过事务控制消息（Control Message）用于在两阶段提交中标记事务的 Commit 或者 Abort。

### Transaction ID 与 Epoch

事务处理应用需要在遭遇重启时，能够偶从中断的地方进行事务恢复，将未完成的事务回滚或者继续完成。为此需要一个跨 session 的 ID 标识，即 Transaction ID。该 ID 由用户在 producer 的参数中指定，是个类似 Consumer Group ID 的字符串。

每个 Transaction ID 也对应着一个递增的 Epoch 号，用于防御 Zombie Writer 的写入，确保同一 Transaction ID 在同一时刻只允许一个 Producer 执行。

### Transaction Coordinator

Transaction Coordinator 会将事务状态日志记录到 __transaction_control 这个内部 topic 内，这单与 Group Coordinator 有点类似，两者均依赖 Kafka 本身的日志持久性。Transaction Coordinator 的日志中只有事务的状态变化，不包含任何事务消息的数据内容，单纯从写入方面似乎不大容易成为瓶颈。

__transaction_control 这一 Topic 内部默认有 50 个 Partition，会根据 Transaction ID 作为分区 Key，并将目标分区 Leader 所在的 Broker 作为 Transaction Coordinator。

### LSO（Last Stable Offset）

未提交的事务会落在 LSO 之后，类似 HW，如果 Consumer 配置了 Read Committed 隔离级别，那么只有当事务提交即收到 Commit 控制消息之后，才能读到事务内的消息。如果事务被 Abort，则忽略该事务相关的所有消息。

## 事务接口

事务的接口大致上长这样：

```scala
kafkaProducer.initTransactions()
kafkaProducer.beginTransaction()
ConsumerRecords<String, String> messages = consumer.poll(100)
try {
    messages.forEach { message ->
        val order = objectMapper.readValue<Order>(message.value())
        if (order.meal == "Pizza") {
            kafkaProducer.send(ProducerRecord(pizzaIncomeIncreased, order.value))
        }
    }

    val consumedOffsets = getConsumedOffsets(kafkaConsumer)
    kafkaProducer.sendOffsetsToTransaction(consumedOffsets, "kafka-transactions-group")
    kafkaProducer.commitTransaction()
} catch (ProducerFencedException | OutOfOrderSequenceException | AuthorizationException e) {
    kafkaProducer.abortTransaction()
}
```

## 事务过程

从《How do transactions work in Apache Kafka?》这篇文章里面盗两个图：

![](/images/kafka-transaction-flow.png)

![](/images/kafka-transaction-state.png)

流程上大约是：

### 寻找 Transaction Coordinator

producer 向任意 broker 发送 FindCoordinatorRequest，得到 Coordinator 的地址；

### 向 Coordinator 发送 InitPidRequest，得到 PID

1. 将 transaction ID 与 PID 绑定，允许重启后仍能得到同样的 PID，从而做到跨 Session 的 Exactly Once 保证；
2. 递增 PID 对应的 Epoch，避免前一代卡死掉的 Producer 继续搞事情；
3. 继续或回滚该 Producer 未完成的事务；

### 开始事务

Producer 调用 beginTransaction 方法表示开启新事务，这里并没有对 Coordinator 的请求，只是将 Producer 标记为 IN_TRANSACTION 状态；

### Consume-Transform-Produce 循环

1. Producer 向 Coordinator 发送 AddPartitionsToTxnRequest 消息，Coordinator 将事务相关的 TopicPartition 列表记录下来，用于两阶段提交中为这些 TopicPartition 发送 Commit 或者 Abort。
2. 生产消息，在消息批次中记录 PID、Epoch 和 sequence 号等元信息；
3. AddOffsetCommitsToTxnRequest：类似 AddPartitionsToTxnRequest 消息，将提交 __commit_offsets 点位对应的 TopicPartition 记录到 Coordinator，用于两阶段提交中的 Commit 或者 Abort；
4. TxnOffsetCommitRequest：向 Consumer Coordinator 发送点位提交；

### 事务 Commit 或者 Abort：

commitTransaction() 与 abortTransaction 均为向 Coordinator 发送 EndTxnRequest 消息。

Commit 流程的话，Coordinator 首先向事务日志写入 PREPARE_COMMIT，随后通过 TxnMarkerRequest 向每个 TopicPartition 发送 COMMIT Marker，最后向事务日志写入 COMITTED。Abort 流程与之类似。

## 总结

* 幂等 + At Least Once == Exactly Once：PID 与 Sequence 号允许 Kafka 的 Producer 实现 Exactly Once 语义。
* Kafka 事务只保证 Kafka 系统内部读写操作的原子性；
* 利用提交点位是向 __commit_offset 这个内部 Topic 写入的性质， 可以将 Consume-Transform-Produce 循环视为原子批量写入。
* 消费侧通过事务控制消息即 Commit/Abort Marker 来决定消费事务消息还是忽略事务消息，开启 Read Committed 后，Consumer 最多读取到 LSO（Last Stable Offset）位置的消息，LSO < HW < LEO。

## References

* [Kafka Exactly Once语义与事务机制原理](http://www.jasongj.com/kafka/transaction/)
* [KIP-98 - Exactly Once Delivery and Transactional Messaging](https://cwiki.apache.org/confluence/display/KAFKA/KIP-98+-+Exactly+Once+Delivery+and+Transactional+Messaging) 
* [Transactional Messaging in Kafka](https://cwiki.apache.org/confluence/display/KAFKA/Transactional+Messaging+in+Kafka)
* [How do transactions work in Apache Kafka?](https://chrzaszcz.dev/2019/12/kafka-transactions/)
* [Exactly-once Semantics is Possible: Here's How Apache Kafka Does it](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
* <https://bravenewgeek.com/you-cannot-have-exactly-once-delivery/>
