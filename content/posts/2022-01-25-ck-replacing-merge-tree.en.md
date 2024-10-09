---
date: "2022-01-25T00:00:00Z"
title: ReplacingMergeTree and CollapsingMergeTree
---

We learned earlier that data in ClickHouse doesn't support updates, only insertions. If the same primary key has updates, it results in two rows in the table.

That's where ReplacingMergeTree from the MergeTree family comes in. It can do some merging during compaction to clean up duplicate data, leaving only the latest data for the primary key.

Sounds good, right? But during contiguous data insertion, you can't guarantee there won't be duplicate primary keys.

(For T+1 data import scenarios, maybe it's enough? Import a batch, then run `OPTIMIZE TABLE` to compact, and no more duplicates. Or use `SELECT ... FINAL` to trigger compaction on each query. For small, rarely changing dimension tables, sounds doable?)

Overall, not supporting UPDATE seems like a painful limitation.

But think about it: OLAP databases are for aggregation, right? If the same primary key has duplicates, why not use aggregation to deduplicate?

ClickHouse has an aggregation function `argMax` for this scenario:

``` c++
argMax(col, val)
```

In a Group By aggregation, it finds the `col` field value corresponding to the maximum `val` (timestamp).

Create a View with `argMax` and Group By, and querying this View looks like a table that supports UPDATE operations. Here's an example:

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

With an aggregated intermediate result, further aggregation analysis adds the cost of storing intermediate results. Intuitively, this approach is only suitable for small tables.

## CollapsingMergeTree

If you want the latest primary key results without running Compaction, use CollapsingMergeTree. But you need to rewrite SQL at the upper layer.

In CollapsingMergeTree, you specify a `sign` field, which can be 1 or -1. 1 represents an inserted row, and -1 represents a deleted row:

``` c++
┌──────────────PageID─┬─PageViews─┬─Duration─┬─Sign─┐
│ 4324182021466249494 │         5 │      146 │    1 │
│ 4324182021466249494 │         5 │      146 │   -1 │
└─────────────────────┴───────────┴──────────┴──────┘
```

Kinda like how MySQL binlog includes the original full row data when deleting a row.

When querying, multiply the `sign` field with the corresponding columns, and calculate `sum()` aggregation, which is equivalent to the result of a deduplicated table by primary key:

``` sql
SELECT
    PageID,
    sum(PageViews * Sign) AS PageViews,
    sum(Duration * Sign) AS Duration
FROM UAct
GROUP BY PageID
HAVING sum(Sign) > 0
```

Because 5 * 1 + 5 * -1 = 0, they cancel each other out.

This way, you get aggregated results like PV UV sums and averages in one go. But it's tough to wrap your head around (do real data analysts write like this?). Guess this idea is exclusive to ClickHouse?

What CollapsingMergeTree does during compaction is to ensure that the aggregation query results are consistent before and after compaction, by eliminating the positive and negative rows with the same primary key.

Overall, compaction in ClickHouse's MergeTree isn't just about cleaning up old data like in LevelDB. It's more like a chance for pre-aggregation, with different MergeTrees satisfying different aggregation operations, keeping the aggregation results consistent before and after compaction.

## References

- [https://xie.infoq.cn/article/7ab73847b53646acd13093e3e](https://xie.infoq.cn/article/7ab73847b53646acd13093e3e)
- [https://guides.tinybird.co/guide/deduplication-strategies](https://guides.tinybird.co/guide/deduplication-strategies)
