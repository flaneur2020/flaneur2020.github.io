## Current State

**ListFilesCache**: 里面就是个哈希表 `HashMap<Path, Vec<ObjectMeta>>`

**FileStatisticsCache**: `HashMap<Path, (ObjectMeta, Statistics)>`，包含 row count、column statistics（min/max value），用于 query optimizer 来优化执行计划，比如 pruning。

**ParquetMetadataCache**：获取 parquet 的 metadata 可能会比较 costy（需要两次网络开销，先找到 footer offset，再读元信息；处理开销，有些表的 column 太多）

datafusion 提供了 ParquetFileReaderFactory 这个 trait，允许开发者实现自定义的 metadata 处理策略。

**ParquetFileRangeCache**：datafusion 允许用户实现 `AsyncFileReader` 这个 trait 来实现自己的 cache，这允许：

1. 在内存中 cache 住常访问的 data range；
2. 实现 tiered caching；
3. 压缩来减少内存使用；
4. 根据 access patterns 实现 cache evction 策略；

`AsyncFileReader` 里面有一个 `get_bytes_ranges` 方法：

```rust
pub trait AsyncFileReader: Send {
    ... // previously mentioned methods 

    fn get_byte_ranges(&mut self, ranges: Vec<Range<usize>>) -> BoxFuture<'_, Result<Vec<Bytes>>>;
}
```

开发者可以实现自己的 IO coalescing logic 来优化性能，比如：

1. 合并相邻的 range 来减少 request 数量；
2. 将小的 range 进行批量成一个大 request；
3. 实现 prefetching

## Caching Arrow

arrow 的 cache 位于 ParquetAccessPlan 和 Arrow RecordBatch 之间。

在 query 请求数据时：

1. 先查一下 RecordBatch 是不是在 cache 中；
2. 如果有，则不请求，在 AccessPlan 中 prune 掉这部分数据；
3. 如果没有，则照常来解析 Parquet 数据，然后将它插入到 Arrow RecordBatch 的 cache 中；

![[Pasted image 20241107131145.png]]

这个架构有一些挑战：

1. 这样将 Parquet 的 byte range 映射到 Arrow RecordBatch；
2. Granularity/shape of caching：按 column-level 还是 batch-level；
3. 怎样高效的测试 cached ranges；
4. 内存管理：怎样实现高效的 spill to disk；

目前作者在研究怎么做一套高效的 arrow caching 系统。

## Standalone caching service

目前的这套 cache 是面向单机的，也可以做成独立的 cache 服务，让多个 datafusion 实例都用起来。

### Caching interface

最简单的做法就是将 cache service 作为一个 object store 的透明代理。这个 service 可以实现 Datafusion 的 `AsyncFileReader` trait 来拦截 Parquet 文件读取，cache 这些 byte range。

也可以使用 arrow flight protocol 来实现这部分 cache。

在 query 需要数据时，它可以发送一个 ParquetExec 的 physical plan 给 cache service。service 可以执行：

1. 针对 cached data 执行 plan
2. 执行 cache 数据执行 filter 和 projection
3. 只返回必要的 record batch 给 querier

> This architecture provides several advantages: - Reduced network transfer by filtering data at the cache layer - Lower client-side CPU usage since filtering happens at the cache
>
> The tradeoff is increased complexity in both the client and cache service implementations compared to the simple byte-range caching approach.

（好像意思是，在 plan 层面 query 这个 cache，再把数据在 plan 层面拼回来）

（没觉得比 simple byte range caching 优势大？👀）

