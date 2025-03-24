假设下面的 query：

```sql
SELECT count(*) FROM store_sales
JOIN date_dim ON store_sales.ss_sold_date_sk = date_dim.d_date_sk
WHERE d_following_holiday=’Y’ AND d_year = 2000;
```

如果没有 dynamic filtering，trino 会 push predicate 到 dimension table 中 `date_dim` 的 table scan，然后 scan fact table 中所有的数据，因为这个 sql 中并没有针对 fact table `store_sales` 的任何过滤条件。

如果开启 dynamic filtering，trino 会搜集 join 中 <mark>dimension table 侧可能的 value 集合</mark>。

在 broadcast join 中，这个集合中生成的 collection 会 push 到 join 左侧的 local table scan 中。

此外，这些 runtime predicate 可以通过网络与 coordinator 通信，因此 dynamic filtering 可以在寻找 table scan splits 期间在 coordinator 上执行。

比如，在 hive connector 中，dynamic filter 被用于跳过一部分不匹配 join 条件的 partition。这被称作 dynamic partition pruning。

在构造完 dynamic filter 中的集合之后，coordinator 可以将它们分散到 worker node。这允许在 worker node 上下推 dyanmic filter。

dynamic filter 是默认开启的。

## Analysis and confirmation

dynamic filtering 需要一系列因素：

- 支持特定的 join 操作符，比如 `=`, `<`, `<=`, `>=`, `>` 或 `IS NOT DISTINCT FROM` join 条件，以及 semi join 中 `IN`
- Connector 需要支持利用 dynamic filter 来下推 table scan。比如，hive connector 可以下推 dynamic filter 到 orc 和 parquet reader 来实现 stripe、row group pruning；
- connector 需要支持 dynamic filter 在 split enumeration 阶段；
- join 右侧（build）的 size；

## Dynamic filter collection thresholds

要让 dynamic filtering 工作，需要选择更小的一侧的 dimension table 做为 join 的 build side。

CBO 可以利用 connector 提供的表统计信息自动完成这一选择。

从 build 侧收集 join 键值可能增加 CPU 开销，为限制这种开销，Trino 设置了 dynamic filter 收集的大小阈值，对于大型的 build 侧的 join，可通过`enable-large-dynamic-filters`配置属性启用。

## Dimension tables layout

> Dynamic filtering works best for dimension tables where table keys are correlated with columns.

在 join 的 key 和 where 中的 build 侧的过滤条件列有相关性，这样是最好的。