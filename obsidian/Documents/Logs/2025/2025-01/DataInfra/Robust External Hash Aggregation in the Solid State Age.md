OLAP 系统通常有比较大的中间结果吃内存。

对内存占用比较高的时候，OLAP 系统通常选择拒绝执行 Query，或者退化到磁盘的路径上，产生一个 “performance cliff”。

> In this paper, we go beyond Cooperative Memory Management and take a unified approach to memory management for persistent and temporary.

论文中比较核心的贡献：

1. 可变 size 的统一内存页管理；
2. 一套用于管理中间结果的 page layout，可以不经过 serialize 直接落盘；
3. 能够在内存不足时，平滑降低 hash aggregation 性能；

## II. TEMPORARY QUERY INTERMEDIATES

### Blocking Operators

一些关系计算算子，如 Aggregation，是 Blocking 的，在所有 input data 都到位之前，不能输出结果。

这意味着这些中间结果需要被 Materialized，并临时存储在 operator 中。

如果中间结果持续增长，这些中间结果会往下移动到更低的 storage hirearchy 中。

如果中间结果超过内存的极限，就需要 “spill” 出去。

传统的数据库往往采用 spillable 的数据结构，比如 BTree 来应对这个情况。（传统时代的内存过于稀缺）这时性能会大幅下降。

Hash Table 相比 Btree 不是一个容易 spill 的数据结构。
### Robustness

> Operator implementations that adapt to larger-than-memory intermediates at runtime provide more robust query runtimes [16]. Hybrid algorithms such as hybrid hash join [17] realize this with a single, efficient algorithm that works regardless of whether intermediates fit in memory. Hybrid algorithms are limited to an input size less than “the square of memory” [17], which is enough for many use cases.

(Hybrid algorithms 指什么？为什么会受内存限制？)

### Memory Management.

Common wisdon 是使用一个 fixed size 的 page size 和一个 fixed size pool。

使用多个页来管理 hash table 会比较复杂。（为什么？）

> Therefore, temporary query intermediates are usually allocated differently

经常会有这样的情况是：临时 query 中间结果往往单独分配。

> Going from byte-addressable to blockaddressable requires changes to operator algorithm design: blocks must be loaded into memory before the data can be randomly accessed, e.g., by a hash table.

## III. UNIFIED MEMORY MANAGEMENT

### Persisted Data

没有给这个 pool 安排固定的大小。因为 duckdb 是一个嵌入式引擎，内存没事就及时回收。

duckdb 使用 256kb 的页大小。

### Temporary Data

区分了三种类型：

1. Non-paged allocations
2. Paged-fixed size allocations：最常用，
3. Paged variable size allocations：单独对应一个存储上的文件

## V. ROBUST EXTERNAL HASH AGGREGATION