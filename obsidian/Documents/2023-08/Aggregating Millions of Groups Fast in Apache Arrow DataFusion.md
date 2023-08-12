
## Introduction to high cardinality grouping

- high cardinality grouping 的定义：group by 中有大量的唯一 key，超过 10000；
- 比如 hits 的 Query 17: `SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase" ORDER BY COUNT(*) DESC LIMIT 10;
- hits 数据集中有 17,630,976 个不同的 user，6,019,103 个不同的 search phrases;
- 要回答这个 query，就得记住 2400w 个不同的 group；

## Two phase parallel partitioned grouping

datafusion 27 和 28 使用了 two phase parallel partitioned grouping，类似 duckdb 的 parallel grouped aggregates；

![[Pasted image 20230812095415.png]]

两阶段的设计对于使 cpu core 保持忙碌至关重要。第一个 phase 在数据产生时，尽早地进行聚合；第二步，将多个 core 产生的 group 进行 shuffle，使同一个 group 都落到同一个 core 来处理。

Datafusion 在实现上还需要考虑一些地方：
- 何时 emit 第一阶段 hash table 的中间聚合数据（因为数据是 partially sorted）
- 处理特定的过滤逻辑（比如 FILTER）
- 数据类型的处理
- 在超过内存限制时的操作

## Hash grouping in 27.0.0

Datafusion 27.0 中，会将 Group 的数据保存在 GroupState 中来跟踪每个 group 的状态。每个 group 包括状态有：

1. group column 的值（Arrow format）；
2. In-progress accumulations
3. Scratch space for tracking which rows match each aggregate in each batch.

![[Pasted image 20230812101704.png]]

在计算聚合时:

1. 通过向量化的代码来计算哈希值，为每种 datatype 做了特化
2. 通过 hash table 确定每行的 group index；
3. 更新每个 group 对应的 accumulator

Datafusion 也在表中保存了哈希值，避免重复的哈希计算。

然而这个架构对高基数的 grouping 不大适用：

- 每个 group 会分配多次内存；
- 非向量化的更新操作

## Hash grouping in 28.0.0

28.0.0 版本中作者重写了 grouping 的逻辑，应用常用的系统优化手段：fewer allocations, type specialization, and aggressive vectorization。

![[Pasted image 20230812103517.png]]

28.0.0 使用了同样的 RawTable，也是同样的 Group Index。主要的不同在于 Group Value 可能保存在：

1. Inline 的 RawTable 中（for single columns of primitive types），这时转换成 Row 格式大于收益；
2. 在单独的 Rows 结构体中，连续保存着所有的 Group Value，而非每个 Group Value 对应一次内存分配。

优势：

1. 减少了内存分配，不再为每个 group 产生内存分配；
2. 连续的 native accumulator states：类型特化的 accumulators，将所有的 group 放在连续的 Rust 的 native type 的 Vec<T> 内存中；
3. 向量化的状态更新：更新操作体现在对于原生的类型特化的 Vec 更新操作上；