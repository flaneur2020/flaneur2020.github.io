CRAQ: 强一致的写，同时面向高的读吞吐优化。

## Chain Replication

所有的写操作都从 client 发到 chain 的 head。

然后再发送给 chain 的下一个节点，直到发送给 tail 节点；

tail 节点负责所有的读请求。写操作在复制给所有的 replica 之前，是到不了 tail 的。一旦抵达 tail，那么这个写入，被视为 **committed**。

![[Pasted image 20260517202532.png]]

这意味着，read 操作总是只返回 committed 的值。

### Chain replication achieves strong consistency

因为所有的读请求都打到 tail，而 tail 保存着 committed 的写操作，tail 可以针对所有写入操作实现 total ordering。

client 一定不会读到 stale 的数据。

### Write operations are cheaper

另一个优势是，写操作的开销在所有节点上均摊了。

与 primary-backup replication 不同，pimary-backup replication 需要 primary 节点将所有数据复制到所有节点上。

chain replication 中的每个节点，只需要传输给它的后继节点。

在模拟的测试中，paper 的作者展示了 chain replication 相比 primary/backup replication 能够实现更高的写入吞吐量。

### The tail is a bottleneck

chain replication 唯一的缺点就是，读请求都必须打到 tail 节点上。

这使得读的负载能力，并不能随着 chain 的扩展而增长。

如果强行读取中间的节点的值，那么 client 可能读到中间的值，不同的读请求，会读到不同的结果。

而 CRAQ 主要就是为了解决这个缺陷而生。

