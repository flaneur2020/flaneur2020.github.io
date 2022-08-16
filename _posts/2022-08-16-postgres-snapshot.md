---
title: Snapshot in Postgres
layout: post
---

最近想学习下 postgres 生态的东西，之前看它的 MVCC 机制没大明白，捞回来尝试重新理解一下。这里先忽略一把 MVCC 的并发控制与清理的部分，先只看 Snapshot 的部分。

## Tuple

Postgres 没有 MySQL 那种 UNDO log，多版本数据（Tuple）会直接存放于表空间，并附带有区分版本的元信息，这里先只看 xmin 和 xmax 两个字段：

- xmin：表示该 Tuple 被插入时的 xid（事务 ID）
- xmax：表示该 Tuple 被删除时的 xid

比如有一个插入并 commit 的 tuple：

``` plain text
| xmin | xmax | band  | fans  |
| 023  | 0    | tfboy | 9000w |
```

在一个新的事务中删除后：

``` plain text
| xmin | xmax | band  | fans  |
| 023  | 024  | tfboy | 9000w |
```

可见设置了 xmax 为新事务的 xid。

如果在一个新的事务中更新这个 Tuple 呢？Postgres 这里会将更新操作看作删除 + 插入：

``` plain text
| xmin | xmax | band  | fans   |
| 023  | 024  | tfboy | 9000w  |
| 024  | 0    | tfboy | 10000w |
```

这里有个比较反直觉的地方是，事务 COMMIT 还是 ROLLBACK，表空间的 Tuple 是没有立即变化的，事务的提交状态取决于 XACT 结构体的记录。

XACT 可以视为 clog （Commit Log）的近义词，它由一组 8kb 的页面组成，页面中为每个事务 ID 对应两个 bit，表示这个事务是 Ingress、Committed 还是 Aborted。clog 会持续追加，每 256kb 轮换一次，不过它并不会无限制增长，vacuum 能够清理无用的 clog 文件。

所以在查询表数据的时候，往往需要二分查找一把 XACT（clog）来获取这行数据的提交状态，多查询一次 XACT 有一定开销，因此 postgres 还在 Tuple 中有两个 hint bit，分别指代 committed 或者 rollbacked，如果在读时发现 Tuple 被 committed/rollback 的话，则设置一把 hint bit，这样下次就不需要再来访问 XACT 了。比较像一个 Read Repair 的过程。

为什么这样设计呢？贴下《MVCC in PostgreSQL-3. Row Versions》的原话：

> Why does not the transaction that performs the insert set these bits? When an insert is being performed, the transaction is yet unaware of whether it will be completed successfully. And at the commit time it's already unclear which rows and in which pages were changed. There can be a lot of such pages, and it is impractical to keep track of them. Besides, some of the pages can be evicted to disk from the buffer cache; to read them again in order to change the bits would mean a considerable slowdown of the commit.

这么看下来 XACT 的设置有点像 percolator 中 Commit Point 的意思，一步原子操作决定 N 个事务参与者的提交状态。

## Snapshot

在 Rocksdb 这种存储层没有未提交数据的存储中，Snapshot 只需要一个 sequence 序号即可。不过 Postgres 中需要多一些信息：

- xmin：当前事务启动时仍活跃的最早的 XID，所有创建时小于 xmin 的数据都应可见（除了 rollback 的数据）
- xmax：当前最近一个 Commit 的事务的 XID + 1，所有大于 xmax 的数据都不可见
- xip[]：当前活跃的事务 XID 列表，所有活跃事务的相关数据应当不可见

参考《How Postgres Makes Transactions Atomic》的图画一个类似的：

![](/images/2022-08-16-postgres-snapshot/Screen_Shot_2022-08-16_at_22.48.18.png)

在这个 Snapshot 里，满足 100 ≤ XID < 105 的事务有 100 和 102 两个，它俩 Commit 出来的数据是可见的，XID = 99 与 104 的事务因为被 Rollback 所以不可见，XID = 101 与 XID = 103 事务还在进行中，也是不可见的。判断事务可见性与否似乎主要看事务的状态，而 xmin 和 xmax 范围能够起到剪枝的作用。

## References

- [1] [https://brandur.org/postgres-atomicity](https://brandur.org/postgres-atomicity#commit)
- [2] [https://habr.com/en/company/postgrespro/blog/477648/](https://habr.com/en/company/postgrespro/blog/477648/)
- [3] [https://philipmcclarence.com/what-is-the-pg_clog-and-the-clog/](https://philipmcclarence.com/what-is-the-pg_clog-and-the-clog/)