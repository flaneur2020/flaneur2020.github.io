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

## CRAQ

craq 是对传统 chain based replication 的增强，允许读取 chain 的任意节点的数据从而扩展读能力，同时仍保持强一致。

CRAQ 的工作过程：

1. 每个节点保存单个对象的多个版本：对每一次写入，保存一个 clean 版本、一个 dirty 版本；
2. 对于写操作：
	1. client 将 write 发送给 head；
	2. write 每传递给一个 replica，该 replica 会创建一个新的 dirty 版本；
	3. tail 节点会在收到 write 后，创建一个该对象的 clean version，并向前驱节点发送一个 ack；
	4. 当一个节点收到特定 object 的版本的 ack 时，它将最新的 object version 标记为 clean，删除所有旧版本；
3. 对于非 tail 节点的读操作：
	1. 如果该对象的 latest version 是 clean 的，就返回该版本的数据；
	2. 否则，<mark>找 tail 问最新的 last committed version，返回该版本的数据（这被称作 version query）</mark>；

CRAQ 相比传统的 chain replication，对两个场景的读吞吐都做到了优化：

1. read-mostly workloads：多数读操作都会是 clean read，因此读请求的吞吐可以随着 Chain size 的增长而扩展；
2. write-heavy workloads：会产生一些针对 tail 节点的 version query，但是 version query 的请求比较轻量，

### CRAQ needs a configuration manager

和 raft 不同，CRAQ 协议自己并不能阻止 split brain。

它只关注 replication 本身，并不能管理 leader election 等事情。

CRAQ 必须搭配一个 zookeeper 这样的 configuration manager 才能工作。

当 head 失败时，它的直接后继可以成为 head，同理，当 tail 失败时，tail 的前驱节点成为 tail；中间节点也可以被其他节点替代掉。

configuration manager 管理着 chain 中的节点，并管理着谁是 head，谁是 tail。

### One slow node can weaken the chain

CRAQ 的主要缺陷是，写入任何数据，都需要每个节点都参与。这与 raft、zk 不同，它们只需要半数节点参与。

因此，CRAQ 相比 raft 会显得 fault tolerant 差一些，有一个节点变慢，就会影响整个写入。