目前有两种常见的架构：1. lambda 架构；2. 通过 lake house 做到的 streaming-batch storage based processing 架构。

这两种架构都有些问题：

1. flink sql 有两种执行模式，streaming 和 batch。
2. 如果有需要 bulk refresh 特定的历史 partition 的需要，streaming job 的逻辑不能直接用起来；
3. 需要手工的创建、管理任务；

snowflake 有一个 dyanmic tables 的概念，来简化 data pipeline 的管理。

databricks 有一个生命式的 ETL 框架 delta live tables。"This also indicates that simplifying data processing pipelines and unifying streaming and batch are current technological trends in the data warehouse field."

> starting from the concept of unified streaming and batch processing (One SQL & Unified Table),  we propose a new table type based on the Dynamic Tables concept[1], called Materialized Table, to simplify streaming and batch ETL pipeline cost-effectively. This allows users to no longer be confused by various complex concepts such as streaming and batch processing, as well as underlying technical details, and to focus solely on business logic and data freshness.

基于 dyanmic tables 的概念，新建了 Materialized Table，来简化 streaming 和 batching 的 ETL pipelien。允许用户不再被复杂的 streaming 和 batching 处理的概念所困扰，而只关注在 business logic 和 data freshness 方面。

## User Story

假设 mysql 数据库中有三个 business table：orders、oders_pay、products。

这三张表最初通过 flink CDC 导入到 paimon。随后希望对三张表做打平。

![[Screenshot 2024-04-30 at 18.25.00.png]]

```
ALTER MATERIALIZED TABLE dwd_orders SUSPEND;
ALTER MATERIALIZED TABLE dwd_orders RESUME;

-- 手工刷新：
ALTER MATERIALIZED TABLE dwd_orders REFRESH PARTITION(ds='20231023')

-- 修改 freshness：
ALTER MATERIALIZED TABLE dwd_orders SET FRESHNESS = INTERVAL '1' DAY
```

freshness 有两种模式：1. Continuous Refresh Mode，会持续启动一个 flink streaming job 来刷新；2. Full Refresh Mode： 定时执行一个 flink batch 任务，来刷新全量。

