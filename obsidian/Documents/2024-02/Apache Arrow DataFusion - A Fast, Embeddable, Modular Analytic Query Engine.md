## 5 Datafusion Features

**5.1 Catalog**

Datafusion 每个 table 需要一个 catalog，来提供表的列、类型、统计信息、存储信息等元信息。

Datafusion 内置了一个 in-memory 的 catalog，和一个类似 hive 的 partitioned 的 file/directory 的 catalog。

Datafusion 针对 Catalog 做了抽象，可以提供自己的实现。

**5.2 Datasources**

提供了五种内置的 TableProvider：Parquet、Avro、JSON、CSV、Arrow IPC File；

parquet reader 利用了 arrow 的 rust 实现的 predicate pushdown、late materialization、bloom filter、nested types 等能力；


**5.3.1 Data Types**

直接使用 Arrow 的 type system，包括变长的定点数、变长的字符串、binary、date、times、interval、nested structs、lists 等。

执行时，operator 将数据按 array 或者 scalar value 进行处理。

**5.3.2 Planner**

使用 sqlparser-rs 解析 sql 生成 LogicalPlan。

然后生成结合了数据执行信息的 Execution Plan。

![[Screenshot 2024-02-14 at 11.09.18.png]]


## 5.4 Plan representations and Rewrites

### 6.1 Query Rewrites

> LogicalPlan rewrites include projection pushdown, filter pushdown, limit pushdown, expression simplification, common subexpression elimination, join predicate extraction, correlated subquery flattening, and outer to inner join conversion.

> ExecutionPlan rewrites include eliminating unnecessary sorts, maximizing parallel execution, and determining specific algorithms such as Hash or Merge joins.
### 6.2 Sorting

> Most commercial analytic systems include heavily optimized multi-column sorting implementations, and DataFusion is no exception. Broadly based on the techniques described in [35], it incorporates a tree-of-losers, a RowFormat (Section 6.6), the ability to spill to temporary disk files when memory is exhausted, and specialized implementations for LIMIT (aka "Top K").
### 6.3 Grouping and Aggregation

> Datafusion contains a two phase parallel partitioned hash grouping implementation, featuring vectorized execution, the ability to spill to disk when memory is exhausted, and special handling for no group keys, partially ordered and fully ordered group keys.

### 6.4 Joins

> The in memory hash join is implemented using vectorized hashing and collision checking used in MonetDB.

### 6.6 Normalized Sort Keys / Row Formats


针对 multi-column soring 和 multi-column 相等性检查等操作，DataFusion 内置了一个 RowFormat，允许按字节进行相等性检查，能够提供 predictable memory access patterns。

Row Format 是 densely packed，每个类型有特化的编码方式，可以参考 ASC 还是 DESC、NULL 等 sort options 进行不同的编码。

比如 unsigned 和 signed integer 通过 big endian 进行编码，而浮点数会转换为 signed integer。

### 6.7 Leveraging Sort Order

Datafusion 的 Optimizer 会尽力利用所有的 input、intermediate result 的任何顺序。

> 1. Physical Clustering: Secondary indexes are often too expensive to build and maintain at high ingest rates, and thus the sort order of primary storage is the only available physical optimization to cluster data. 

> 2. Memory Usage and Streaming Execution: The sort order defines how the data that flows through Streams is partitioned in time, defining where values may change and thus where intermediate results can be emitted.

### 6.8 Pushdown and Late Materialization

> DataFusions’s Parquet reader uses pushed down predicates to 1) prune (skip) entire Row Groups and Data Pages based on metadata and Bloom filters, and 2) apply predicates after decoding only a subset of column values, a form of late materialization






