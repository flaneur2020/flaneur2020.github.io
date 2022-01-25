---
title: ck 的 MergeTree: Compact
layout: post
---

对 MergeTree 这个名词先入为主地以为是 leveldb 的那种 LSM，然而在 ck 的 MergeTree 里一概没找着那些熟悉的东西，看蒙了一段时间。

首先 leveldb 的 LSM 有 WAL + MemTable + SSTable，读取数据时候每层 level 一个 iterator 组成一个 MergeIterator，可以反应出最新的更新。跑 compaction 主要是清理旧数据，留下新数据，主要是为 Update 服务的：在不可变文件的基础上，实现可变的 KV 语义，通过 compaction，减少查询需要扫描的文件数，从而提高查询的性能。

然而 ck 中 MergeTree 对 Update 不那么需要，表中的数据是一系列的 events，可以认为只支持追加（ALTER TABLE ... UPDATE .. 相当于整表重写，听起来不像是给人建议没事写着玩的）。

所以 ck 中没有 MemTable 与 WAL，也不需要，因为攒批次这件事情认为是用户的责任，每次写入操作落盘成一个 part。

一次写入操作产生一波文件，这在 TP 场景下是不能想象的。在 AP 就还好，主要是要求用户攒批次，不过出 1w 个 part 扫起来也会费劲一些，数据只在 part 内部有顺序，扫描时候要做不少归并排序。

对 ck 来讲，part 越少，有序的数据就越相邻，可以更高效地在扫描时做剪枝，毕竟需要剪的枝少了嘛。compaction 操作在这点上是与 leveldb 是一致的：通过 compaction，减少扫描相关的文件数，以及减少需要归并排序的工作量。

ck 跑完 compaction 最终会把表的所有 part 合并成一个 part。有配置 partition 的话，就是将 partition 中的所有 part 合并成一个 part。直观上，如果遇到大表的话按时间维度来 partition 该挺好的。

在跑完 compaction 终态下一个 partition 对应一个 part，但是 part 并不是 partition 的缩写，个人感觉可以这样理解：part 对应一个 partition 的一次写入操作，没有指定 partition key 的话，相当于整个表是一个 partition。

leveldb 在跑完合并之后会依据 manifest 文件来体现 compaction 前后的元信息变化，不过 ck 好像没有 manifest 同等的文件，是通过 part 目录的文件名的元信息变化，part 目录名字的格式是这样：

``` sql
{partition}_{minBlockNum}_{maxBlockNum}_{level}
```

（这个 min max 好像在 MESA 里有个长得差不多的东西）

每个新的 part 都会递增一次数据块编号，新插入的 part 的 minBlockNum 和 maxBlockNum 是相同的，比如连续 5 次 insert，得到的 part 会是：

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- 202102_4_4_0/
- 202102_5_5_0/

如果只对 202102_1_1_0、202102_2_2_0 和 2021_3_3_0 跑 compaction，新的目录的是：

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- __202102_1_3_1/__
- 202102_4_4_0/
- 202102_5_5_0/

202102_1_3_1 文件名中的的 _1_3 表示这个 part 中包含 1_1 到 3_3 中三个 part 中的完整数据，因此 202102_1_1_0、2021_2_2_0、2021_3_3_0 三个 part 理论上就可以随时清理掉了。但它们不会被立即删除，会等待后台任务慢慢 gc 掉。

末尾的 _1 是 level，表示 compaction 的次数。

再对 202102_1_3_1 和 2021_4_4_0 跑 compaction，新的目录会是：

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- 202102_1_3_1**/**202102_1_3_1**/**
- 202102_4_4_0/
- __202102_1_4_2__/__202102_1_4_2__/
- 202102_5_5_0/

画个图：

![](/images/2022-01-24-ck-merge-tree-compact/v2-bf074298e93c6cc9eddceb4c084e9ed5_1440w.jpg)

一个 block number 对应一块原始的 part 数据，因为有 compaction，因此在同一时刻，这份数据可能会同时存在于多个 part 之中。对于 ck 来说，读取时的策略好像可以总结为一句：如果有多个 part 包含同一块数据，取 level 最高的 part 为准。

总结一把：

- ck 的 MergeTree 中没有 leveldb 那种 WAL 和 MemTable，这是用户的责任
- 每次写入默认会新生成一个 part 目录，数据在 part 中有序
- compaction 的单位是 partition，会最终合并成一个 part
- ck 中不需要 leveldb 那种 manifest 文件来存放文件的元信息，通过 part 文件名的命名规范来找出当前活跃的 part 文件