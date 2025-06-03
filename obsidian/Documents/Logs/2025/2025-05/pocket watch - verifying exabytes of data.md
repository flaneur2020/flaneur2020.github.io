![[Pasted image 20250523102029.png]]
> In this post we’ll discuss the mechanisms we use to ensure that Magic Pocket constantly maintains its <mark>extremely high level of durability</mark>.

## Table-stakes: Replication

Magic Pocket 使用 RC 编码做复制。

如果只考虑随机的磁盘损坏，有 replication 的话理论上有 27 个9️⃣的 durability。

除了随机的磁盘岁坏，还有很多导致错误的风险：自然灾害、软件 bug、运维错误、错误的变更等等。

## How correct is correct?

> The most important question we ask ourselves every day is **“is this system correct?”** This is an easy question to ask, but a surprisingly difficult question to answer authoritatively. <mark>Many systems are “basically correct”, but the word “basically” can contain a lot of assumptions</mark>. Is there a hidden bug that hasn’t been detected yet? Is there a disk corruption that the system hasn’t stumbled across? Is there bad data in there from years and years ago that is used as a scapegoat whenever the system exhibits unexpected behavior? A durability-centric engineering culture requires rooting out any potential issues like this and establishing an obsessive focus on correctness.

> more than 50% of the workload on our disks and databases is actually our own internal verification traffic.

一个 extent 是一个 storage node 中的一个 data volume 的表示，每个 storage node 有几千个 gb 的 extent。

## Verification Systems

|                     |                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Cross-zone Verifier | Application-level walker that verifies all data is in the storage system and in the appropriate storage regions. |
| Storage Watcher     | Sampled black-box check that retrieves blocks after a minute, hour, day, week, etc.                              |
| Metadata Scanner    | Verifies all data in the Block Index is on the correct storage nodes.                                            |
| Extent Referee      | Verifies that all deleted extents were justified according to system logs.                                       |
| Trash Inspector     | Verifies all deleted extents contain only blocks that are deleted or have been moved to other storage nodes.     |
| Disk  <br>Scrubber  | Verifies data on disk is readable and conforms to checksums.                                                     |
|                     |                                                                                                                  |
## Disk Scrubber

> When we detect a bad disk in MP we quickly re-replicate the data to ensure the volumes aren’t vulnerable to a subsequent disk failures, usually in less than an hour. If a failure were to go undetected however, the _window of vulnerability_ would expand from hours to potentially months, exposing the system to data loss.

定期轮询磁盘上的内容，检查 checksum。如果发现有损坏的，则自动修复一下 replication。

## Trash Inspector

> If Magic Pocket needs to move a volume between storage nodes, or rewrite a volume after garbage collecting it, then it writes the volume to a new set of storage nodes before deleting it from the old nodes. This is an obviously-dangerous transition: what if a software bug caused us to erroneously delete an extent that hadn’t yet been written stably to a new location?

迁移 volume 是一个危险操作，如果删除老节点上的数据的时候，新的节点还没有写进去（如果因为什么 bug），怎么办？

会有一个保护机制就是，删除操作，会先放到垃圾箱中。

![[Pasted image 20250523111516.png]]

Trash Inspector 会定期扫描垃圾箱中的文件，检查它是否在其他节点上已经 replicate 了。

## Extent Referee

> The Trash Inspector does a good job of ensuring that we only delete data as intended, but what if a bad script or rogue process attempts to delete an extent before it has passed inspection? This is where the Extent Referee comes in.

它会监控文件系统的通知，保证任何一次 move 和 unlink 操作，都在 trash inspection 通过并有来自 master 的相关指令之后。

## Metadata Scanner

> One advantage of storing our Block Index in a database like MySQL is that it’s really easy to run a table scan to validate that this data is correct. The Metadata Scanner does exactly this, iterating over the Magic Pocket Block Index at around a million blocks per second (seriously!), determining which storage nodes should hold each block, and then querying these storage nodes to make sure the blocks are actually there.

> Our goal is to perform a full scan over our metadata approximately once per week to give us confidence that data is entirely correct in one storage zone before we advance our code release process and deploy new code in the next zone.

遍历 Magic Pocket 的所有元信息，确认数据是否存在那里。

每秒遍历 100w 个对象听起来很多，然而有几千亿的对象。作者的目标是能够一周这么确认一次，这样有充足的自信。

## Storage Watcher

> We sample 1% of all blocks written to Magic Pocket and record their corresponding storage keys (hashes) in queues in [Kafka](http://kafka.apache.org/). The Watcher then iterates over these queues and makes sure it can correctly fetch these blocks from Magic Pocket after one minute, one hour, one day, one week and one month.

相当于一个黑盒的 proptest。

## Cross-zone Verifier

> While the other verifiers confirm that MP is correctly storing the blocks that we know we should have, the Cross-zone Verifier ensures that there’s agreement between what MP is storing and what our clients (the rest of Dropbox) think we should be storing.

## Verification as testing

> This is because a comprehensive verification stack doesn’t just ensure that the system is correct in production, it also provides a highly-valuable testing mechanism for new code.

也是对新代码的验证。

![[Pasted image 20250523113654.png]]