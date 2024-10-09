snowflake 客户平均每年在平台上花费 $300000。

> It turns out that [Snowflake](https://github.com/resource-disaggregation/snowset) and [Redshift](https://github.com/amazon-science/redset) have both published representative samples of what real user queries look like on their systems, and they are filled with insights about how data warehouses are used in the real world.

## Data warehouses are ETL Tools

snowflake 和 redshift 都没有展示用户执行的具体的 query 内容，但是提供了足够多的信息，给我们理解不同的 query 类型。

![[Pasted image 20240924113900.png]]

可以这样理解数据处理的生命周期：

1. Ingest 将新数据摄入到系统，并将这些数据与系统现有的数据进行合并
2. Transformation 将 "raw" data 转换为简单的、易于处理的视图
3. Read 是常规的 BI dashboard 和 data science
4. Export 是将数据从 data warehouse 导出到其他系统
5. Other 指 system maintaince functions；

## Most queries are small

 我们经常看到 vendor 发布 100tb 级数据的 performance benchmark。真实的 query 的规模在什么水平呢？

![[Pasted image 20240924114210.png]]
中位数的 query scan 在 100mb 这个水平。99.9 分位的 query scan 在 300gb 水平。

Snowflake 和 Redshift 往往作为 “MPP” 系统而为人熟知，而实际上 99.9% 的真实世界 query 都可以在一个 single large node 上完成。

[small data hypothesis](https://www.smalldatasf.com/) 是成立的。

## Do we need massively parallel processing?

But suppose you want to do something that can't be expressed as a SQL query, for example if you want to fit a logistic regression model using scikit-learn:

```
  
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
```

You will have to run models like this on a large single node outside the elegant MPP world.

This solution works, but the simplicity of the MPP system is lost, and you pay a meaningful cost to simply export the data out of your data warehouse.

## Data lakes will change everything

Datalake 架构显式地区分开了 storage layer 使之与查询引擎无关。

![[Pasted image 20240924121041.png]]

这个范式下，允许 customer 针对不同的 workloads 使用特定的 execution engine。

而不是使所有的 workload 都限定在 MPP 中。这已经发生了：

1. Fivetran datalake writer 是一个专门的 ingest service
2. Microsoft PowerBI 包含一个特定的 BI engine

针对特定的 workload，你可以选择特定的 engine。比如 fivetran 开发了 datalake writer service，使得 ingestion 变得免费。

另外，local compute 在 datalake 生态中会变得更重要。small query 在真实世界的 workload 中分布比例相当高，如果你的 query scan 规模小于 100mb，可能更好的办法是把数据下载下来，再在本地执行。