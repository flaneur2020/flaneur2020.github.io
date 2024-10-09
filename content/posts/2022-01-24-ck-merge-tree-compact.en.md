---
date: "2022-01-24T00:00:00Z"
title: "MergeTree in ClickHouse: Compaction"
---

First off, I thought MergeTree was just like leveldb's LSM, but after digging into ClickHouse's MergeTree, I couldn't find any of the familiar stuff, and I was confused for a while.

First, leveldb's LSM has WAL + MemTable + SSTable. When reading data, each level has an iterator that forms a MergeIterator, reflecting the latest updates. Running compaction mainly cleans up old data and keeps the new data, mainly serving updates: on the basis of immutable files, it implements mutable KV semantics. By compacting, it reduces the number of files that need to be scanned for queries, thus improving query performance.

However, ClickHouse's MergeTree doesn't need updates that much. The rows in the table is a series of events, which can be considered as append-only (ALTER TABLE ... UPDATE .. is equivalent to rewriting the entire table, which doesn't sound like something you'd do for fun).

So, ClickHouse doesn't have MemTable and WAL, and it doesn't need them because batching is the user's responsibility. Each write operation is written to disk as a *part*.

One single write operation may generates a bunch of files, which is unimaginable in OLTP scenarios. In OLAP it's okay, it asks users to batch rows, but if you end up with 10,000 parts, scanning can be a bit tough. Data is only ordered within parts, and a lot of merging is needed during scanning.

For ClickHouse, the fewer parts, the closer the ordered data, allowing for more efficient pruning during scanning, since there are fewer parts to prune. The compaction operation here is consistent with leveldb: by compacting, it reduces the number of files related to scanning and the amount of work needed for merging.

After running compaction, ClickHouse will eventually merge all parts of the table into one part. If there's a partition configuration, it will merge all parts within the partition into one part. Intuitively, if you have a large table, partitioning by time dimension should make a lot of sense.

After running compaction, one partition corresponds to one part in the final state, but *part* is actually not an abbreviation for *partition*. Personally, I think it can be understood like this: a part corresponds to one write operation of a partition. If no partition key is specified, the entire table is considered one partition.

In Leveldb, the changes before & after compaction is reflected in the changes of the manifest files. However, ClickHouse doesn't seem to have a file equivalent to manifest. It changes the metadata of the part directory name, and the format of the part directory name is like this:

``` sql
{partition}_{minBlockNum}_{maxBlockNum}_{level}
```

(This min max thing looks similar to something in MESA)

Each new part increments the data block number, and the minBlockNum and maxBlockNum of the newly inserted part are the same. For example, after 5 consecutive inserts, the parts will be:

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- 202102_4_4_0/
- 202102_5_5_0/

If you only run compaction on 202102_1_1_0, 202102_2_2_0, and 2021_3_3_0, the new directory will be:

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- __202102_1_3_1/__
- 202102_4_4_0/
- 202102_5_5_0/

The _1_3 in the 202102_1_3_1 filename indicates that this part contains the complete data from the three parts 1_1 to 3_3, so the three parts 202102_1_1_0, 2021_2_2_0, and 2021_3_3_0 can theoretically be cleaned up at any time. But they won't be deleted immediately; they'll wait for the background task to slowly gc them.

The _1 at the end is the level, indicating the number of compactions.

If you run compaction again on 202102_1_3_1 and 2021_4_4_0, the new directory will be:

- 202102_1_1_0/
- 202102_2_2_0/
- 202102_3_3_0/
- **202102_1_3_1**/
- 202102_4_4_0/
- __202102_1_4_2__/
- 202102_5_5_0/

Here's a diagram:

![](/images/2022-01-24-ck-merge-tree-compact/v2-bf074298e93c6cc9eddceb4c084e9ed5_1440w.jpg)

A block number corresponds to a raw part of data. Due to compaction, this data might exist in multiple parts at the same time. For ClickHouse, the read strategy seems to be: if multiple parts contain the same block, use the part with the highest level.

Summary:

- In ClickHouse's MergeTree, there's no WAL or MemTable like in LevelDB. That's the user's responsibility.
- Each write defaults to creating a new part directory, with data ordered within the part.
- Compaction is done at the partition level, eventually merging into one part.
- ClickHouse doesn't need a manifest file like LevelDB to store metadata. It identifies the active part files through the naming convention of part filenames.
