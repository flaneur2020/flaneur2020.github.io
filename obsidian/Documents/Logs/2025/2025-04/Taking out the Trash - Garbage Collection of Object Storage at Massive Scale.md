
tldr：

- 异步清理队列：使用元数据中心的事务性，来跟踪删除的文件，通过这个队列进行异步删除
- 异步 reconciliation loop：LIST 所有文件，与 meta store 做对比，类似 mark-sweep 的方式删除文件，是 warpstream 一开始用的策略；
- 通过 meta store 跟踪的方式是比较浪费的，作者希望有更轻量的办法；
- “optimistic deletion queue”：在每次 compaction 之后，agent 知道哪些文件要被删除，它会把它记录在 go 的 channel 中，进行异步清理；
- warpstream 的存储类似 Log Structured Merge Tree，因此所有的旧文件待清理的信息，都可以通过Compactor 来获取到；
- 又了这套架构之后，作者发现大部分文件都可以在 reconciliation loop 之前发现并清理掉。

---
![[Pasted image 20250426170151.png]]

---

warp stream 在几种情况下会遇到需要删除文件的时候：

1. topic 中有设置 TTL，超过 ttl 的 message 会认为 expired
2. 整个 topic 被删除
3. input 文件在 compact 之后，可以删除

作者的思路：

1. delayed queue
2. async reconciliation

相比之下，bucket policy 和 sync deletion 工作的没有预期的那么好。

- **延迟队列（Delayed Queue）：** 将逻辑删除的文件先加入一个延迟队列，等待一段时间后再物理删除。为了避免引入孤儿文件，可以将延迟队列集成到元数据存储中，利用其事务性保证原子操作。这种方法的问题在于难以维护无 bug 的实现。
- **异步协调（Asynchronous Reconciliation）：** 将元数据存储作为真相的来源，定期扫描对象存储，识别并删除元数据存储中不再跟踪的文件。这类似于“标记-清除”算法。这种方法更容易保证正确性，但成本较高且难以调优，因为列出和检查对象存储中的文件是慢且昂贵的操作。

warp stream 采用了综合的方式：

- 在 compaction 时收集所有可以删除的文件列表（可以把文件列表落到文件里？）
