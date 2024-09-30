https://jack-vanlightly.com/blog/2024/8/7/table-format-comparisons-how-do-the-table-formats-represent-the-canonical-set-of-files

所有的 table formats 都保存着一组 data 和 deleted files list，以及 metadata 文件。

不同的表格式组织起来不同，不过大致可以分为两类：

1. the log of deltas approach: Hudi and DeltaLake
2. the log of snapshots approach: Iceberg and Paimon
## The log of deltas approach

将新的修改记录为 log entries，其中记录修改了什么，比如新增、删除了哪些文件，对 schema 的修改等等；

- deltalake 将它们称作 delta logs，每个记录叫做 delta entry；
- apache hudi 将 log 称作 Timeline，每个 entry 称作 Instant；
### Delta Lake

每个 delta entry 对应一个特定的 action，包括：Change Metadata、Add Remove Files、Add CDC files、and mores。

![[Pasted image 20240810115244.png]]
![[Pasted image 20240810115311.png]]

### Apache Hudi

文件的列表通过两个组件表示：

1. the timeline
2. the metadata table with among other things, act as a file index for hudi tables

### File groups, file slices, base and log files

Hudi table 会拆分为多个 file group，可以理解为 sharding。

一个 Primary key 会对应到一个 shard 上，这个 mapping 会保存在一个 index 中。

file groups 既可以固定数字，也可以动态增长。

每个 file group 包含一个或者多个 file slices，一个 file slice 包含一个 base file（parquet 格式）和一组 log files（也是 parquet）。

在 MoR Tables 中，新的写入会落到 Logs 文件中，包含着 delta（new rows 和 deletion vectors）。

在 CoW Tables 中，只有一个 Parquet 文件。

在 Hudi 中，timestamp 扮演了一个关键角色。
### The timeline

每个 write 操作对应一组写入到 timeline 的 instant，每个 Action 有 Request、Inflight、Completed 三种状态； 

Command 有 Commit、Delta Commit、Compaction、Clean 等这些种类。

每个 Instant 会按 `(timestamp, action, action state)` 来写入到 Timeline；

![[Pasted image 20240810131920.png]]



File slices (and even log files within file slices) are filtered out based on timestamps, instead of using an explicit logical delete mechanism.

在 hudi 的 timeline 中，不包含任何逻辑上删除的文件，在 table scan 中只通过 timestamp 来区分读取哪个 base 和 log files。
### Finally, how Hudi clients learn of the canonical set of files

1. 如果 client 希望知道最新版本的 file slices，只需要读取 hudi 的 metadata table，里面包含有所有的 committed file slices，只需要读取每个 file group 中版本最大的 file slice；
2. 如果 client 希望得到更早版本的，也是同样读取 hudi 的 metadata table，不过会过滤一下版本

The timeline is not the source of the canonical set of files for the latest table version, but it is required for filtering in time-travel queries.
## The log of snapshots approach

> With the _log of deltas_ approach, a new commit only adds a delta and a reader must roll up the log of deltas to make a logical snapshot. With the _log of snapshots_ approach, this roll-up process into snapshots is already done during the writing phase.

![[Pasted image 20240810132639.png]]

![[Pasted image 20240810132759.png]]

当前的 metadata 文件会保存一组历史的 snapshot 的 log。

### Apache Paimon

paimon 和 iceberg 相似，但是没有依赖一个 catalog 来指向最新的 metadata 文件。

它采用的方法是类似 deltalake 的 numbering 方法：

![[Pasted image 20240810133017.png]]

paimon 有两个 manifest list 文件：

1. Base manifest list：表示表的基本内容
2. Delta manifest list：插入与逻辑删除的内容

![[Pasted image 20240810133729.png]]

普通的写入操作不会在逻辑上删除文件，只有当 compaction 任务执行时才会删除。

