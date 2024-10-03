iceberg 本身并没有原生的 CDC 支持，compute engine 必须支持自己的 infer-CDC-on-read 实现。

## Some relevant internals background

### Iceberg snapshot types

iceberg 有多种 snapshot 类型，需要关注的有：

1. Append：只增加 data file 的 transaction；
2. Overwrite：用于 row 级别的 insert、update、deletes，可以包含增加、删除数据文件；
3. Delete: 只包含 delete file
4. Replace：一个 compaction job，能够将一组数据和 delete files 替换为一组新的文件，逻辑上这些数据是没有变化的，所以计算 delta 的 compute engine 不需要关注这部分修改；

### Incremental table scans

iceberg core 模块为 compute engines 提供了三种计算 incremental table scan 的方法：

1. Data file iterators：
	1. Table.IncrementalAppendScan: 针对 Append snapshots 可以得到一个 incremental scan，可以用来读取 inserted rows；
	2. Table.IncrementalChangelogScan: 针对 Append, Overwrite, Delete snapshots 来计算，适用于 CoW 的 tables
2. Snapshot iteration：
	1. 可以使用 SnapshotUtil 遍历 snapshots 的 log，发现每个 snapshot 中新增/删除的 data/delete files

### Infer-CDC-on-read with copy-on-write vs merge-on-read

在一个 CoW 的 table 中，update 和 delete 会在应用修改时使 data file 重新写入。

这个新的数据文件可能包括：

1. 之前的 data file 中未包含的新的 row
2. 未经修改的 row
3. 之前的 data file 中有这些 row，经过了修改
4. 之前的 data file 中有这些 row，新的 data file 中移除了

UPDATE、DELETE、MERGE 操作能够创建多对 deleted 和 added 的 data files，有多个 added 和 deleted files 时，就无法容易地根据 add/deleted pair 来计算差异了，只能通过所有的 added/deleted files 来计算差异。

![[Pasted image 20240930184733.png]]
### Merge-on-read (MOR)

在 MoR table 中，data file 并未在逻辑上删除或者 rewrite，而是：

- 新的、updated 的 rows 写入到新的 data files；
- 旧的 location 中，会被 delete file entry 进行标记；

<mark>针对 MoR 表计算 changes 的代价相对低一些</mark>：

1. 将所有的 added 的数据文件的行认为是 inserts 或者 updates；
2. 如果没有新增 delete file，那么所有数据文件中的 行 都是 insert；
3. 如果新增了一个 delete file，则可以告知 compute engine，哪些行属于 updates，哪些行属于 deletes
