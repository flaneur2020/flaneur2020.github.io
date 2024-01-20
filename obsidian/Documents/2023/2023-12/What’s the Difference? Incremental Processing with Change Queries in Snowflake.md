https://dl.acm.org/doi/pdf/10.1145/3589776

## TLDR

- snowflake 的 CHANGES 相当于 mysql 的 binlog，比 mysql 强大在于它还能支持 view 的 CHANGES；
- STREAM 相对而言是一个比较简单的机制，它就是记住目标表、View 的 CHANGES 的点位；
- STREAM 比较有意思在于它可以在事务中使用，只有在事务 COMMIT 时才移动点位；
- CHANGES 的实现不是基于 REDO Log，而是**找出时间范围内新增、删除的 micro-partition 文件，根据这些文件，计算出去重后的 delta**；
	- 这个去重会是一个相对昂贵的计算操作，但是多数操作都是 insert or delete 而已；

---

- Incremental 算法是 stream processing 的核心；
- 在本文中，作者提出了 CHANGES 和 Stream 对象，作为 snowflake 用来检索和消费 table objects 的 incremental changes 的原语。

## Introduction

- Stream 可以看做是 insert-only 的 table；
- 通过 MERGE，可以实现 Stream 到 Table 的 DML 转换；
- 能够<mark>允许 SQL 对 changes 进行 query</mark>，也是一项很重要的能力：
	- Event Queuing
	- Notifications
	- Incremental View Maintenance
	- Flexible change transformation
## Semantics

![[Screenshot 2023-12-18 at 13.04.39.png]]

CHANGES Query 可以用来观察一个 table object 在两个时间点之间的差异。

相当于 CDC logs。

还支持

### 2.3 Stream Objects

Stream 是一个 schema level 的 catalog object。它可以关联到一个 table object （table 或者 view）。

Stream 会在自己的状态中保存一个点位，表示消费的位置。

如果在一个事务中，消费的 Stream 点位会在事务提交时更新，反之如果 abort，则回归原位。

Stream 有一个 feature 是 SHOW_INITIAL_ROWS，意思大约是指创建 Stream 时，将现有的数据也加入 Stream 中。

Stream 的存储开销极小，可以针对一个 Table 创建很多 Streams。

## Implementations

snowflake 早期就已经有了内置的 CHANGES 能力。作者做的主要工作内容是：

1. 在 storage 层面增加元信息来按行的精度来记录 changes；
2. 记录 query differentiation framework 来重写这部分 query 来产生 changes；
3. 使 Stream 和 transaction 集成；

### 3.1 Table Metadata

snowflake 的 table 保存在一组 micro partitions 文件中。

在任意时刻，table 的状态称作一个 table version，其中包括：

- 一个 system timestamp 对应版本号；
- 该版本号对应的 micro partitions 的集合；
- partition-level 的 statistics 信息，包括 minmax 索引、null count 等信息。

这个架构很容易实现 time travel，但是 CHANGES query 需要做更多工作。


### 3.2 Query Differentiation

> given that we want the CHANGES of a query $Q$ over an interval $I$, we say we differentiate $Q$ to obtain the derivative of $Q$, $Δ_I Q$, which varies over $I$.

![[Screenshot 2023-12-18 at 13.42.09.png]]

没大看明白上面这个 simple procedure 是啥。

#### 3.2.2 Change Tracking

Query differentiation 最终会把 derivative operator 下推到 query plan 的 scan operator 中。

需要一个办法来计算一个 persistent table 的 changes。在 snowflake 中，这个信息已经记录在了 per-row 的元信息 change tracking 列中。

change tracking 列有助于得到两个重要信息：

1. 这个行数据的 unique identifier，在更改后仍然不变，可以用于去重；
2. 判断这行数据是插入到当前 micro-partition 的，还是从其他拷贝来的，用于产生 append-only 的 changes；

在第一次插入表时，该行的 change tracking 是 NULL。在经过 Copy-on-Write 操作时，每行数据的旧 location 会写入到新的 micro-partition 中。

INSERT DML 中，change tracking 没有开销，UPDATE 和 DELETE 操作有一定开销，必须存储对应的值，但是好处是比较易于压缩。

#### 3.2.3 Change Formats

snowflake 的 CHANGE query 有两种 change 格式：min-delta 和 append-only。

min-delta 模式下，最核心的挑战是去重。

> It works by iterating over the table versions during the change interval to find all micro-partitions that were added to or removed from a table during the change interval.


找出来时间范围内，新增、删除的 micro-partitions。

这些 micro-partitions 中肯定会有相当多重复性的数据。

> Delta minimization is a costly operation that requires repartitioning its input.

> Delta minimization is a costly operation that requires repartitioning its input. But the majority of change queries only have inserts or deletes during their change interval, in which case no minimization is necessary.


去重是一个昂贵的操作，但是多数差异都只是 insert 或者 delete 而已，不大需要去重。

> the MINIMIZE shape is the default, and the ADDED_ONLY and REMOVED_ONLY shapes omit delta minimization and one branch of the union all for each base table

提供了 ADDED_ONLY 和 REMOVED_ONLY 选项，允许走 fast path。

![[Screenshot 2023-12-18 at 14.54.58.png]]

Metadata$IsUpdate 是根据行数据的 id 计算得出的。

#### 3.2.4 Metadata Columns

- $ACTION 包括 INSERT 或者 DELETE。
- $ISUPDATE 由 min-delta 计算出来。
- $ROW_ID: 是一个哈希值，只用于去重；

## 4 USAGE AND PERFORMANCE ANALYSIS

在文章发表时候，Stream 和 Changes 已经上线了三年。

48% 的 Stream 都是 append-only 的，52% 是 min-delta 的。

append-only 的 Stream 有更好的性能，应用的更多。

1/3 的 append-only 的 Stream 应用于 Merge 语句。

2/3 的 min-delta 语句应用于 Merge 语句。

Stream 发起的 DML 操作往往来自 Task，它要么来自 CRON-like 的调度，要么来自另一个 Task 的依赖调度。文章写作时，Task 的最低频率为 1 分钟。
