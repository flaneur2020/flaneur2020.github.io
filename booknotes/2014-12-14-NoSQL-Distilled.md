---
layout: default
title: NoSQL Distilled
---

# 读书笔记: NoSQL Distilled

<https://book.douban.com/people/fleure/annotation/7952514/>
## Migration

<原文开始>Databases with strong schemas, such as relational databases, can be migrated by saving each schema change, plus its data migration, in a version-controlled sequence.</原文结束>

版本号最好存在数据库本身。

<原文开始>Schemaless databases can also read data in a way that's tolerant implicit schema and use incremental migration to update data. </原文结束>

Incremental Migration: 改一个 schema 时保持对旧版本 schema 的兼容，向新的字段写数据同时也向旧的字段写数据，确保稳定之后再删旧字段。
## Column-Family Stores

<原文开始>Cassandra is not great for early prototypes or initial tech spikes: During the early stages, we are not sure how the query patterns may change, and as the query patterns change, we have to change the column family design.... RDBMS impose high cost on schema change, which is traded off for a low cost of query change; in Cassandra, the cost may be higher for query change as compared to schema change.</原文结束>

Cassandra 不适合做早期原型。日常读写的 query 与 schema 高度相关，如 RDBMS 调整 schema 困难但 query 灵活，而 cassandra 调整起 Query 便已足够困难了。

schema 的解范式，某种意义上也是以 query 的灵活性换取性能，也最好在业务经过沉淀之后再做比较好，不适合在项目试错阶段就上。

## Consistency

<原文开始>Replication comes with some alluring benefits, but it also comes with an inevitable dark side—inconsistency.</原文结束>

<原文开始>Concurrent programming involves a fundamental tradeoff between safety (avoiding errors such as update conflicts) and liveness (responding quickly to clients).</原文结束>

<原文开始>Maintaining session consistency with sticky sessions and master-slave replication can be awkward if you want to read from the slaves to improve read performance but still need to write to the master. One way of handling this is for writes to be sent the slave, who then takes responsibility for forwarding them to the master while maintaining session consistency for its client... Another approach is to switch the session to the master temporarily when doing a write, just long enough that reads are done from the master until the slaves have caught up with the update.</原文结束>

处理读写分离的不一致

<原文开始>many application builders need to interact with remote systems that can’t be properly included within a transaction boundary, so updating outside of transactions is a quite common occurrence for enterprise applications.

At the other end of the scale, some very large websites, such as eBay [Pritchett], have had to forgo transactions in order to perform acceptably—this is particularly true when you need to introduce sharding.
</原文结束>

<原文开始>If you’re sufficiently confident in bringing the master back online rapidly, this is a reason not to auto-failover to a slave.</原文结束>

如果期望 master 快速重启的话，就不要使用 auto-failover。
## Why NoSQL?

<原文开始>Clustered relational databases, such as the Oracle RAC or Microsoft SQL Server, work on the concept of a shared disk subsystem</原文结束>

<原文开始>As the 2000s drew on, both companies produced brief but highly influential papers about their efforts: BigTable from Google and Dynamo from Amazon.</原文结束>

<原文开始>The most important result of the rise of NoSQL is Polyglot Persistence.</原文结束>

<原文开始>we put into four categories widely used in the NoSQL ecosystem: key-value, document, column-family, and graph.</原文结束>

图数据库还接触不到场景；key-value 要么 redis 这种 in-memory store，要么就是就是 CDN 这种 blob 存储了；document 系看不出明显的好处，单看扩展性目测不如 column-family 系？

