leveldb 有一个 smallest_snapshot。

https://github.com/google/leveldb/blob/23e35d792b9154f922b8b575b12596a4d8664c65/db/db_impl.cc#L966

在 compact 时，如果遇到一个 kv 有多版本，会参考 smallest_snapshot 来清理旧版本：

```C++
      if (last_sequence_for_key <= compact->smallest_snapshot) {
        // Hidden by an newer entry for same user key
        drop = true;  // (A)
      } else if (ikey.type == kTypeDeletion &&
                 ikey.sequence <= compact->smallest_snapshot &&
                 compact->compaction->IsBaseLevelForKey(ikey.user_key)) {
        // For this user key:
        // (1) there is no data in higher levels
        // (2) data in lower levels will have larger sequence numbers
        // (3) data in layers that are being compacted here and have
        //     smaller sequence numbers will be dropped in the next
        //     few iterations of this loop (by rule (A) above).
        // Therefore this deletion marker is obsolete and can be dropped.
        drop = true;
      }
```

怎样跟踪 smallest_snapshot 的呢？

```C++
  if (snapshots_.empty()) {
    compact->smallest_snapshot = versions_->LastSequence();
  } else {
    compact->smallest_snapshot = snapshots_.oldest()->sequence_number();
  }
```

有两个逻辑，`snapshots_` 为空时，拿 `versions_` 中保存的 `LastSequence()`。这个 seq 值是从 manifest log 和 WAL 重放得到的，还比较容易理解。每次重启之后，都会有一个新的 manifest log，相当于在启动之前的所有的 seq num 都是可以清理的。

有 `snapshot_` 时，找到最古老的 `snapshot`，来当做 `smallest_snapshot`。这个也不难理解，找到最早的 snapshot，凡是活跃的 snapshot 访问的数据都不能清理。

每个 snapshot 都会串在一个 `SnapshotList` 的双向链表中。

## 问题：iterator 是怎样跟踪的？

不过 leveldb 似乎没有在跟踪活跃的 iterator，每个 iterator 也并没有在跟 Snapshot 绑定。iterator 内部只有一个 seqnum 表示一个逻辑上的 Snapshot。

这可能是因为，每个 Iterator 是和 Version 绑定的，Version 关联一组 sst 内存中的文件描述符，只要 Version 还在内存中，这些文件就不会被删除，这些 iterator 也就是安全的。

Snapshot 为什么不这样做，也和 Version 绑定？

问 claude 说法是 “Version binding would be too restrictive for long-lived snapshots” 确实也合理，Snapshot 可能设定上是生命周期比较长的，可能跑一两个小时。这样能够更及时地释放内存中的 `Version`？因为每次 flush Memtable，都会有新的 Version 产生。

## 问题：badger 是怎么做的？

badger 中每个 Iterator 关联了一个 `Txn`。似乎更直接一些，是同一套机制。