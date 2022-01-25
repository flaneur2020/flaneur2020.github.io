---
title: ReplacingMergeTree 和 CollapsingMergeTree
layout: post
---

前面了解到 ck 中数据是不支持更新的，只支持 insert 追加。同一个主键对应的数据有更新的话，体现在表中是两行数据。

这时候就有 MergeTree 家族的 ReplacingMergeTree 出现了，它可以在跑 compaction 时做一些计算，将重复的数据清理掉，留下主键中最新的数据。

看起来没毛病，但是在流式地插入数据期间，是不能保证没有主键没有重复数据的。

（T+1 的数据导入场景的话，好像够用？导入一波进来，然后 optimize table 跑一把 compaction，就没有重复了。或者 SELECT ... FINAL 在每次查询时触发 compaction，对于不常变化且不大的维度表的话，听起来够用？）

总的来说不支持 UPDATE，似乎是一个难受的限制。

然而这样想：OLAP 数据库不是做聚合的吗，所谓同一个主键的数据有重复，用聚合操作去做去重如何？

ck 中有一个聚合函数 argMax 可以用在这个场景：

``` c++
argMax(col, val)
```

在一个 Group By 的聚合中，按最大的 val（时间戳）找出来对应的 col 字段值。

用 argMax 和 Group By 建一个 View，查这个 View 就长得跟仿佛支持 Update 操作的表差不多了，网上找个例子：

``` sql
CREATE VIEW PowerConsumption_view ON CLUSTER default_cluster AS
  SELECT    User_ID,
            max(Record_Time) AS R_Time,
            District_Code,
            Address,
            argMax(Power, Record_Time) AS Power,
            argMax(Deleted, Record_Time) AS Deleted
  FROM default.PowerConsumption
  GROUP BY User_ID, Address, District_Code
  HAVING Deleted = 0;
```

基于聚合得到的一个不小的中间结果，再进一步聚合分析，增加一步存中间结果的代价。直观上来看，这个做法还是只适合小表。

## CollapsingMergeTree

如果希望查询能反应出主键最新的结果又不想跑 Compaction，可以用 CollapsingMergeTree。不过需要上层改写 SQL。

在 CollapsingMergeTree 中会指定一个 sign 字段，取值是 1 或者 -1，1 代表插入的一行，-1 代表删除的一行：

``` c++
┌──────────────PageID─┬─PageViews─┬─Duration─┬─Sign─┐
│ 4324182021466249494 │         5 │      146 │    1 │
│ 4324182021466249494 │         5 │      146 │   -1 │
└─────────────────────┴───────────┴──────────┴──────┘
```

有点像 mysql binlog 里删除一行数据的 binlog 中带有原始的一行完整数据。

在查询时，将 sign 字段乘一下相应的列，算 sum() 聚合就等价于主键去过重的表的结果：

``` sql
SELECT
    PageID,
    sum(PageViews * Sign) AS PageViews,
    sum(Duration * Sign) AS Duration
FROM UAct
GROUP BY PageID
HAVING sum(Sign) > 0
```

因为 5 * 1 + 5 * -1 = 0 等于相互抵消掉了。

这样算 PV UV 之类的求和、求平均，一步到位拿到聚合结果。但是思维转过来这个弯还挺难的（真的有数据分析师愿意这么写嘛），猜这个思路该是 Clickhouse 独家？

CollapsingMergeTree 在 compact 阶段所做的事情，就是做到 compact 前后，这种风格的聚合查询能够结果一致，也就是将主键相同的 sign +1 -1 的两行正负消除掉。

总的来看，ck 中的 MergeTree 跑 compact 不是 leveldb 那种只管清理旧数据，更像是做预聚合的一个时机，不同的 MergeTree 满足不同的聚合操作，compact 前后聚合的结果不变。

## References

- [https://xie.infoq.cn/article/7ab73847b53646acd13093e3e](https://xie.infoq.cn/article/7ab73847b53646acd13093e3e)
- [https://guides.tinybird.co/guide/deduplication-strategies](https://guides.tinybird.co/guide/deduplication-strategies)
