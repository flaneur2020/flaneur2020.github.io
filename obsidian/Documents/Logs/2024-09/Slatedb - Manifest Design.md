### Q：怎么避免 zombie writer 问题？

zombie writer 问题好像是说，新的 writer 进程启动了，但是旧的 writer 进程还没有退出，也产生了写入，从而产生脏数据。

增加了一个 `writer_epoch`。

每个 writer client 在启动时，会递增这个 `writer_epoch`。

`writer_epoch` 在 manifest 和 WAL 中都有存在。起到一个占坑防止老进程写入的作用。

老进程一看最新的 manifest 和 WAL 都大于了自己的 `writer_epoch`，会主动退出。
### Q：writer 和 compactor 怎么同时执行？

好像 compactor 也有一个单独的 epoch，用于防止 compactor 的 zombie writer。
## Still not clear...
### Q：reader 会更新 manifest 吗？

看起来会？reader 会创建 snapshots，而 snapshot 会记录在 manifest 中。
### Q：为什么 WAL 也需要 writer epoch？

感觉只要 manifest 有 writer epoch 是不是就够了？

---

> We propose persisting SlateDB's `DbState` in a manifest file to solve problem (1). Such a design is quite common in [log-structured merge-trees](https://en.wikipedia.org/wiki/Log-structured_merge-tree) (LSMs). In particular, RocksDB [follows this pattern](https://github.com/facebook/rocksdb/wiki/MANIFEST). In SlateDB, we will persist `DbState` in a manifest file, which all clients will load on startup. Writer, compactor, and readers will all update the manifest in various situations. All updates to the manifest will be done using [compare-and-swap](https://en.wikipedia.org/wiki/Compare-and-swap) (CAS).
> 
> To prevent zombie writers, we propose using CAS to ensure each SST is written exactly one time. We introduce the concept of writer epochs to determine when the current writer is a zombie and halt the process. The full fencing protocol is described later in the _Writer Protocol_ section

每个 SST 写入时，都走一个 CAS。

引入了 writer epochs 的概念。

### CAS on S3

### Two Phase CAS Protocol

在如下操作中，需要考虑 CAS：

1. Writes：一个进程写入文件到对象存储，但是这个文件已经存在了
2. Deletes：一个进程删除了文件；
3. Recovery：一个进程尝试恢复失败的写入或删除
#### Writes

假设一个 writer 尝试写入到 manifest/00000000000000000000.manifest，它要执行的步骤是：

1. 写到一个临时位置：`manifest/00000000000000000000.manifest.[UUID].[checksum]`
2. 写入到一个 transactional store，比如 dynamodb，包含下述信息：
	1. source：`manifest/00000000000000000000.manifest.[UUID].[checksum]`
	2. destination：`manifest/00000000000000000000.manifest`
	3. committed: `false`
3. 从 `manifest/00000000000000000000.manifest.[UUID].[checksum]` 拷贝到 `manifest/00000000000000000000.manifest`
4. 将 dynamodb 中的 committed 修改为 `true`
5. 删除旧的 `manifest/00000000000000000000.manifest.[UUID].[checksum]` 文件

> _NOTE: This design is inefficient on S3. A single SST write includes two S3 `PUT`s, two writes to DynamoDB (or equivalent), and one S3 `DELETE`. See [here](https://github.com/slatedb/slatedb/pull/43#issuecomment-2105368258) for napkin math on API cost. Latency should be minimally impacted since the write is considered durable after one S3 write and one DynamoDB write. Nevertheless, we are gambling that S3 will soon support pure-CAS like every other object store. In the long run, we expect to switch S3 clients to use CAS and drop support for two-phase CAS._
##### Recovery

写入的程序可能会在任何位置失败，客户端进程可以执行恢复：

1. 列出来 dynamodb 中所有 `committed` 为 false 的项
2. 对每个项：
	1. 拷贝 `source` 到 `destination` 位置
	2. 设置 `committed` 为 `true`
	3. 删除 source object

### File Structure

```
some-bucket/
├─ manifest/
│  ├─ 00000000000000000000.manifest
│  ├─ 00000000000000000001.manifest
│  ├─ 00000000000000000002.manifest
├─ wal/
│  ├─ 00000000000000000000.sst
│  ├─ 00000000000000000001.sst
│  ├─ ...
├─ levels/
│  ├─ ...
```

WAL 相当于 `L0`？

```
message Manifest {
  // Manifest format version to allow schema evolution.
  uint16 manifest_format_version = 1;

  // The current writer's epoch.
  uint64 writer_epoch = 2;

  // The current compactor's epoch.
  uint64 compactor_epoch = 3;

  // The most recent SST in the WAL that's been compacted.
  uint64 wal_id_last_compacted = 4;

  // The most recent SST in the WAL at the time manifest was updated.
  uint64 wal_id_last_seen = 5;

  // A list of the SST table info that are valid to read in the `levels` folder.
  repeated SstInfo leveled_ssts = 6;

  // A list of read snapshots that are currently open.
  repeated Snapshot snapshots = 7;
}
```
#### `wal/00000000000000000000.sst`

slatedb 中 wal 的格式和 SST 相同。

每个 sst 有一个 `writer_epoch` 的 attribute。

### Manifest Updates

一个 read-modify-write 的 Manifest 更新包括如下步骤：

1. list 出最大 id 的 manifest；
2. 读取这个 manifest；
3. 内存中更新 manifest 文件；
4. 写入到一个新的 manifest 文件；

其中 (4) 是一个 CAS 操作。

有这些人可能修改 manifest：

1. Reader：reader 可能会修改 snapshots 来创建、修改、删除 Snapshot；Reader 也可能修改 `wal_id_last_seen`，在创建快照的同时更新自己知道的最新的 WAL id。
2. Writer 必须在每次启动时，更新 writer_epoch;
3. Compactor：compactor 必须在每次启动时更新一下 `compactor_epoch`。compactor 也必须在每次 compaction 之后更新 `wal_id_last_compacted`、`wal_id_last_seen` 和 `leveled_ssts`。
### Writers

`writer_epoch` 会在每个客户端启动时递增 1，用于避免来自 zombie writer 产生的 split brains。

> A zombie writer is a writer with an epoch that is less than the `writer_epoch` in the current manifest.

`writer_epoch` 会存在于 manifest 的一个字段、每个 WAL 的 sst 文件的元信息

#### Writer Protocol

每次启动时，一个 writer 必须递增 `writer_epoch`。

1. list manifest 文件，找出最大的 ID（比如 manifest/00000000000000000002.manifest）
2. 读出该 manifest 文件，在内存中递增它的 `writer_epoch`；
3. 在该 manifest 的内存中创建一个新的 snapshot；
4. 基于更新后的 `writer_epoch` 写入新的 manifest（manifest/00000000000000000003.manifest）

> The writer client must then fence all older clients

writer client 也会写入一个空的  WAL 的 sst 文件。

