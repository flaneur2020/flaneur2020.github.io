之前的文章讲的是，怎样 prune 跳过 parquet 文件或者 row group。

这个文章讲怎样在 row group 的 scan 中跳过不相关的行。

## Why filter pushdown in Parquet?

```sql
SELECT val, location 
FROM sensor_data 
WHERE date_time > '2025-03-12' AND location = 'office';
```

![[Pasted image 20250321105134.png]]
> Parquet pruning skips irrelevant files/row_groups, while filter pushdown skips irrelevant rows.

在当前的默认实现中，会读出来 sensor_id, val, location 这几个列到 RecordBatch 中，然后对 location 做过滤。

更好的做法是 filter pushdown。首先对 filter 条件进行过滤，只解码出通过这些条件的行。

在实现上，它的做法通常是先对过滤的列计算，得到一个 boolean mask，然后用这个 mask 来过滤、解码其他列中相关的数据（sensor_id, val）。

它的思路很简单，但是不好的实现通常让性能变得更差。

## Why slower?

rust parquet reader 目前的做法：

1. 构造 filter mask
2. 对其他列应用 filter mask

在每一步操作中，都需要将 parquet 转为 arrow：

1. 解压缩 parquet 的 page
2. 解码到 arrow format
3. 对 arrow data 执行过滤

然而 <mark>location 这个列被 decompressed 且 decoded 了两次</mark>，第一次是在构造 filter mask 时，第二次是在 build output 时。

| Decompress | Decode | Apply filter | Others |
| ---------- | ------ | ------------ | ------ |
| 206 ms     | 117 ms | 22 ms        | 48 ms  |
可见 decompress 和 decode 是主要的耗时。

## Solution

We need a solution that:

1. Is simple to implement, i.e., doesn’t require thousands of lines of code.
2. Incurs minimal memory overhead.

> This section describes my [<700 LOC PR (with lots of comments and tests)](https://github.com/apache/arrow-rs/pull/6921#issuecomment-2718792433) that **reduces total ClickBench time by 15%, with up to 2x lower latency for some queries, no obvious regression on other queries, and caches at most 2 pages (~2MB) per column in memory**.

这个新的 pipeline 将两个阶段合并成了一个 pass：

1. decompress 过的 page 能够立即用于 build filter mask 和 output columns
2. 只缓存 decompressed page 一个很短的时间；一轮 pass 之后，cache memory 就可以释放；

只有 `location` 列的 page 会被 cache，`val` 的列不会。