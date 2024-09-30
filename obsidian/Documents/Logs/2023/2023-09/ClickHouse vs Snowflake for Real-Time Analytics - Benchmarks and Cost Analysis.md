https://clickhouse.com/blog/clickhouse-vs-snowflake-for-real-time-analytics-benchmarks-cost-analysis

## Introduction

场景：统计 python 包的下载量。数据集来自 PyPI，包含 6 亿行数据。

结论：

- **ClickHouse Cloud is 3-5x more cost-effective than Snowflake in production.**
- **ClickHouse Cloud querying speeds are over 2x faster compared to Snowflake.**
- **ClickHouse Cloud results in 38% better data compression than Snowflake.**

数据和 benchmark 脚本可见于：[https://github.com/ClickHouse/clickhouse_vs_snowflake](https://github.com/ClickHouse/clickhouse_vs_snowflake).

## Benchmarks

在 GCE 的 us-central-1 区。

在 Clickhouse Cloud 上使用 177, 240 和 256 vCPUs 来测试。

在 Snowflake 上，使用 2X-Large 和 4x-Large 来测试，分别被认为是 256 和 512 个 vCPU。

相比于 Snowflake 的 cpu:memory 为 1:2 的配置，Clickhouse Cloud 有更好的 cpu:memory 比例 1:4。

开启 cluster 优化的话，导入数据到 Snowflake 需要花费 $1100。而在 Clickhouse Cloud 上只需要 $40。

### Application & dataset

目前这份数据集位于 BigQuery。每行数据对应 Python 包的一次下载。有 parquet 格式。

Snowflake 建议每个文件大小约为 100MiB~150MiB。作者将 8.7TiB 的数据拆为了 70608 个平均为 129MiB 的文件。

#### Proposed application

输入包名字，取回一组有意思的趋势信息，最好可以渲染为图表。这可能包括：

1. 时间的下载趋势，按 line chart 进行绘制
2. 每个 Python version 的下载趋势，绘制为 multi-series line chart
3. 每个系统比如 Linux 的下载趋势，绘制为 multi-series line chart
4. 每个项目的 distribution 类型（比如 egg、wheel 等），绘制为饼图

除此之外，也需要用户来展示：

1. 相关的 sub-project 的总下载量，比如 clickhouse-connect
2. 针对特定操作系统的 top 项目列表

![[Pasted image 20230912154754.png]]

### Assumptions & limitations

- 没有评估 Snowflake 的 multi-cluster warehouse 特性。
- snowflake 的 persistent cache 在 warehouse restart 后仍然适用，有助于提高 cache hit 比率。Clickhouse Cloud 目前有一个 query cache，但是 per-node 的，暂时没有分布式。
- 没有评估 Snowflake 的 Query Acceleration Service。这个性能加速不稳定而且费用高昂，影响压测复现。

## Results

### Summary

- Clickhouse 38% 更好的压缩率
- Snowflake 中执行 Cluster 之后可以优化 45% 的压缩。
- Clickhouse 在装载数据上比 Snowflake 快 2x，这建立在 Snowflake 建议的文件大小上，文件小一些的话 Snowflake 会更差一些。
- 在 workload 中利用 clustering/ordering key 是一个实时分析中常见的模式。Clickhouse 在这个场景下能够比 Snowflake 快 2~3 倍。
- cold query 差异相对小一些，但是 Clickhouse 仍比 Snowflake 快 1.5x ~ 2x。
- Snowflake 的物化视图和 Clickhouse 的 Projection 相似，对于特定的 query 可以跑的比 Clickhouse 快，但平均性能仍 1.5x 慢于 Clickhouse。
- Snowflake 在无法利用 Ordering/Clustering Key 的特定 GROUP BY 查询上比 Clickhouse 快 30%，作者认为这建立在 vCPU 更多的优势上；
- 对于全表扫描的查询，比如 LIKE，Snowflake 可以提供更低的 95/99th percentiles，但是更高的平均执行时间；
- Clickhouse 的二级索引机制可以提供 Snowflake 的 Search Optimization Service 类似的机制，有更好的 hot query 性能，但是 cold query 性能略差；但是这个机制在 Clickhouse 中没有增加特别多开销，但在 Snowflake 中就特别贵；

