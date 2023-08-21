https://clickhouse.com/blog/clickhouse-cloud-boosts-performance-with-sharedmergetree-and-lightweight-updates?utm_medium=social&utm_source=twitter&utm_campaign=smt

## Challenges with running ReplicatedMergeTree in ClickHouse Cloud

不再需要显式的 replication；

sharding 也不需要了；

zero-copy replication：在 shared object storage 上跑 ReplicationMergeTree 的表；好像相当于在一个 Clickhouse Keeper 中更新日志，其他 server 追这个元信息日志的更新，并在内存中重放。数据完全写入 Object Store。

![[Pasted image 20230821172420.png]]

这一架构仍存在问题：

- Metadata 在服务器之间产生复制；
- zero-copy replication 的可靠性受限于 3 个组件：1. object storage；2. Keeper；3. Local Storage；
- 这一设计仍然是针对少数服务器所优化的，如果服务器较多，对于 Replication Log 仍会产生比较多的争用；