# The logical model of Paimon

paimon 中有 catalog、databases、tables 的概念。作者这里只覆盖 primary key tables 这里。

paimon 将 table 拆分为两层：

1. metadata layer：管理 table 的 schema、哪些文件存在、各个文件的 statistics 信息等等；
2. data layer：data 和 index 文件，构成表中的具体的数据；这一层按一个或者多个 LSM tree 进行管理；与 Iceberg、Hudi、Delta 的最大差异就是，data 按 LSM tree 进行管理；
## The metadata layer

paimon 将元信息组织为一组文件，每次修改对应一个树根。

每次写入操作写入的文件包括：

1. 树根：snapshot 文件
2. 两个 manifest-list 文件（base 和 delta）
3. 一个 index manifest 文件
4. 一个或者多个 manifest 文件
5. 一个或者多个 data files

![[Pasted image 20240812111450.png]]

snapshot 文件会写入到 snapshots 目录中，有一个递增的数字后缀。

它看起来像是 deltalog 的设计，但是实际上是不同的，它更接近 iceberg 的 snapshot list 的作用。

paimon 中，每个 snapshot 文件是一个 single root of tree of files。

相比之下，Delta 和 hudi 会通过从 checkpoint 开始，重放 log 来复原一个逻辑的 snapshot。

也有一个 LATEST 文件，指向最新的 snapshot。允许 paimon 不用像 iceberg 这样，仍强依赖一个 catalog。

每个 snapshot 文件中，有如下重要的字段：

1. Version：snapshot 的 table version；
2. SchemaId：指向一个 schema file，包含有哪些 fileds、primary keys、partitioning columns 等信息；
3. baseManifestList：包含前一个 snapshot 中所有的 manifest 文件的 manifest list；
4. deltaManifestList：包含写入到 snapshot 的新的 manifest 文件
5. indexManifest：包含当前的 index 文件，包括 bloom filter、bucket mapping 和 deletion vectors；
6. commitKind：commit 包括 Append、Compact、Overwrite 几种，本文作者主要关注：
	1. Append commits，包括 append insert、update、delete row 等操作；
	2. Compact commits，包括重写 data file，形成 LSM tree 层面更小、更大的文件；

![[Pasted image 20240812112700.png]]

每个 manifest 文件可以指向一个或者多个 data 文件，并包含这些 data file 的 statistics 信息。query engine 可以借助这些信息来 prune 扫描的范围。

这种 base + delta 两个 manifest 文件的设计，<mark>允许 streaming readers 简单地读取到最新的 change 内容</mark>。

batch-based query 需要将两个 manifest list file 合并，形成一个 table 的 global view。

每次产生新的 snapshot 文件，它会合并一把前一个 snapshot 的 base + delta manifest list，从而将 manifest files 合并成更少、更大的一组文件。

这波 compaction 与 data file 的 compaction 并不相同，可以看作是 commit 期间的 house cleaning 操作。
## The data layer

每个 table 均保存为一组 parquet 或者 orc。

这些文件会根据 schema 中指定的 partition column 保存为一个或者多个 partition，每个 partition 再拆分为一个或多个 bucket，bucket 可以是固定的一组数字，也可以动态随着数据写入而增长。

一般 partition 是基于时间的，query 在查询中，如果带有时间字段，那么可以避免 load 很多相关的元信息进来。

![[Pasted image 20240815133734.png]]

每个 partition 中，按 bucket key （或者 primary key）来对文件做进一步的分组，文件内部根据 primary key 来排序。

每个 bucket 对应一个自己的 LSM Tree。

对于固定数量的 bucket，数据根据 bucket key 的哈希进行路由。

对于动态增长的 bucket，会有一个单独的 bucket key 到 bucket 的全局索引。

Partition 和 Bucket 的目的不同，Partition 有助于 compute engine 在读取阶段更高效地 prune data files。

Bucket 有助于提高读、写的并行性。

![[Pasted image 20240815134311.png]]

在文件布局上，不同的 bucket、partition 和元信息都在独立的目录中。

## LSM tree storage (one LSM tree per bucket)

### Paimon’s LSM tree approach

相比于 rocksdb，更容易看到的是 clickhouse 的影子：

- 数据暂存在内存中，flush 到 level 0 的 sorted runs 中，没有 WAL
- compaction 与时间对齐，只有连续时间相关的 sorted runs 会被 compaction
- data files 可能会被 tag 上 levels，但是这些 levels 并不会影响 read path，read path 不会跨层来寻找最新值，而是会根据 query keys 去找出所有的文件，并执行一个 merge 策略来去重；
- 支持不同的 merge engines，默认是 deduplicate merge engine（等价于 clickhouse 的 Replacing Merge Tree），也支持等价于 AggregatingMergeTree 的策略

![[Pasted image 20240815135128.png]]

每个 datafile 中包含一组 row，每个 row 有一个 row kind，表示是 Insert、Update、Delete；

也有一个 partial update merge engine，允许 writer 只更新行中的一部分列。
### Merge-on-read

数据有一个 sequence number，每个 writer 会维护一个 counter 用于生成 sequence number。也可以直接使用表中的一个 timestamp 类型的字段。

Paimon writer 可以在每次写入 sorted run 时执行一次 full compaction，也可以在每 N 个 sorted run 时 compact 一次。

文档比较推荐每个 bucket 200mb～1gb 左右。
### Deletion vectors

Deletion Vectors 有助于提高读取的性能，并对写入性能不怎么影响。

![[Pasted image 20240815135707.png]]

没有 deletion vectors，readers 的数量会受到 bucket 数量限制，一个 bucket 只能一个 reader 扫，因为它要把所有的 sorted runs 都读到内存里才能去重。除非读完所有的 sorted runs，一个 reader 并不能得知一行是否是 invalid 的。

Deletion Vectors 允许读取 bucket 时候，开多个 reader 来并行扫。读取一个 positional delete 的 bitmap 即可。

这个做法在 Iceberg、Delta 和 Hudi 中都有体现。

> It is also a more efficient merge operation, as it is not a multi-way merge between data files, just one file at a time with data skipping based on the deletion vectors.
### Deletion vector maintenance

Deletion Vector 并非在写入操作时更新，而是在 compaction 中维护的。

Level 0 是没有 Deletion Vector 的。

在开启 Deletion Vector 时，reader 需要跳过 level 0，不然就需要 apply 一把 merge 操作，反而损失了省略 merge 的优势。

每个 bucket 有一个唯一的 Deletion Vector 文件。

![[Pasted image 20240815142117.png]]

### Support for Copy-on-write (COW) and merge-on-read (MOR)

paimon 可以看作是一个 MoR 设计。

当然，也可以通过每次写入都执行一次 Full Compaction，来模拟 CoW 的效果。