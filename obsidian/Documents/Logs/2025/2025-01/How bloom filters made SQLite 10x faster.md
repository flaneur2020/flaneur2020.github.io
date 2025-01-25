在 SSB 这个 benchmark 下，数据主要由一个大的 fact table 和一些小的 dimension table 组成。

Buffalo University 在 2015 年的一个研究发现，duckdb 在这组 benchmark 上比 sqlite 块 30~50 倍。

## Cause

sqlite 有一个内置的 vdbe_profile 能力，能够跟踪各个指令的 cpu cycles。

![[Pasted image 20250104215814.png]]

其中 SeekRowID、Column 这两项是最耗时的：

1. SeekRowID 指给定义一个 RowID，probe BTree 中的一个行；
2. Column：从特定 record 中取出来一个 Column

## Database Joins

SQLite 默认是一个 Nested Loop Join。

（虽说是 Nested Loop Join，但是实际上好像也并不是真正的两层 for 循环，大致上还是一个 BTree 的查询，理论上还是 O(logN) 的）

这样似乎会有很多 BTree probe 产生。

## Join Order

![[Pasted image 20250104220644.png]]

## Optimization

不过 sqlite 不大想用 Hash Join 是因为，不想增加内存占用。

而且，增加一个 join 算法，会增加 query planner 的复杂性。

这个作者的优化思路是，先根据 Join 条件构造一个 bloom filter，只有当 probe 存在时，才执行 BTree Probe。

实现上，增加了一个 Filter 和 FilterAdd 指令，来维护 BloomFilter。

> The result? SQLite became <mark>7x-10x faster</mark>!

这个修改加入到 v3.38.0 版本中。