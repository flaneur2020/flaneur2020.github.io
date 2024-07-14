---
date: "2019-09-08T00:00:00Z"
title: leveldb 笔记：Version
---

leveldb 中的 version 这个名字有点奇怪，实际上 version 指代着 leveldb 的元信息：各个 Level 中有哪些 sstable 文件、有哪些 WAL 文件。每当因为 compaction 生成新的 sstable 时，这部分元信息就会随之改动，元信息的改动也必须落日志（MANIFEST 文件）用于崩溃恢复。leveldb 允许同一时刻能有其他人仍按较旧的元信息来访问数据库，同一时刻会存有多版本的元信息。可以大致这样理解：leveldb 的元信息是按照版本控制的方式来管理的，一套完整的元信息便是一个 version。

Version 相关的类主要是 VersionSet、Version、VersionEdit。其中 Version 表示一个不可变的元信息版本，所有活跃的 Version 都包含在 VersionSet 的双向链表之中。每个 Version 有引用计数，在生命周期结束之后，会将自己从这个双向链表中摘除：

![leveldb-versionset](/images/2019-09-08-leveldb-version/leveldb-versionset.png)

VersionSet 其实是 leveldb 的一个大入口，维护了不少信息。日常的读写操作和 compaction 都需要走 VersionSet 的 current 找到当前 version 中的元信息来找对应的 sstable。



```
  Env* const env_;
  const std::string dbname_;
  const Options* const options_;
  TableCache* const table_cache_;
  const InternalKeyComparator icmp_;
  uint64_t next_file_number_;  // 下一个 manifest 文件编号，写入到 manifest 日志里
  uint64_t manifest_file_number_;  // manifest 文件编号，每次重启后递增
  uint64_t last_sequence_;  // sequence 号，用于 snapshot，每次写入操作都会递增
  uint64_t log_number_;  // wal 日志文件编号
  uint64_t prev_log_number_;  // 0 or backing store for memtable being compacted

  // Opened lazily
  WritableFile* descriptor_file_;
  log::Writer* descriptor_log_;
  Version dummy_versions_;  // Head of circular doubly-linked list of versions.
  Version* current_;        // == dummy_versions_.prev_

  // Per-level key at which the next compaction at that level should start.
  // Either an empty string, or a valid InternalKey.
  std::string compact_pointer_[config::kNumLevels];
```

Version 的成员：

```
131   VersionSet* vset_;            // VersionSet to which this Version belongs
132   Version* next_;               // Next version in linked list
133   Version* prev_;               // Previous version in linked list
134   int refs_;                    // Number of live refs to this version
135
136   // List of files per level
137   std::vector<FileMetaData*> files_[config::kNumLevels];
138
139   // Next file to compact based on seek stats.
140   FileMetaData* file_to_compact_;
141   int file_to_compact_level_;
142
143   // Level that should be compacted next and its compaction score.
144   // Score < 1 means compaction is not strictly needed.  These fields
145   // are initialized by Finalize().
146   double compaction_score_;
147   int compaction_level_;
```

大致上可以分为三部分，首先是 VersionSet 双向链表与引用计数的运行时信息，然后是最主要的 FileMetaData 信息，用于发现各层次有哪些 SST 文件，最后是 compaction 辅助信息。compaction 后文件变化是确定性的，在 Version 变化的时刻能够结合 compaction_score_ 等信息钦点出下一任 compaction 的 level 和文件。

FileMetaData 的字段如下，主要的内容似乎是 smallest 和 largest 这两个键范围：

```
struct FileMetaData {
  int refs;
  int allowed_seeks;          // Seeks allowed until compaction
  uint64_t number; 
  uint64_t file_size;         // File size in bytes
  InternalKey smallest;       // Smallest internal key served by table
  InternalKey largest;        // Largest internal key served by table

  FileMetaData() : refs(0), allowed_seeks(1 << 30), file_size(0) { }
}
```

对 Version 的主要修改发生于 compaction 之后。compaction 分为 minor compaction 和 major compaction，其中 minor compaction 是落 memtable 到 L0，只会新增文件，而 major compaction 会跨 level 做合并，既新增文件也删除文件。每当这时，便会生成一个新的 VersionEdit 产生新的 Version，插入 VersionSet 链表的头部。

VersionEdit 的成员：

```
  typedef std::set< std::pair<int, uint64_t> > DeletedFileSet;
  std::string comparator_;
  uint64_t log_number_;  // WAL 日志文件的编号，不是 MANIFEST 日志文件的编号
  uint64_t prev_log_number_; 
  uint64_t next_file_number_; //  下一个 MANIFEST 文件的编号；

  SequenceNumber last_sequence_;
  bool has_comparator_;
  bool has_log_number_;
  bool has_prev_log_number_;
  bool has_next_file_number_;
  bool has_last_sequence_;

  std::vector< std::pair<int, InternalKey> > compact_pointers_;
  DeletedFileSet deleted_files_;
  std::vector< std::pair<int, FileMetaData> > new_files_;
}
```

每个 VersionEdit 会都落日志到 MANIFEST 文件中以备崩溃恢复，可以认为 MANIFEST 文件中的记录与 VersionEdit 总是一一对应的。但是崩溃恢复期间，每次应用 VersionEdit 都生成一份新的 Version 对象是没有太大意义的，VersionSet::Builder 的功能有点类似 StringBuilder，将连续操作合并，合并一系列操作产生最终结果，省去生成中间结果的开销：

![img](/images/2019-09-08-leveldb-version/wjxyapo.png)

VersionEdit 的序列化逻辑位于 EncodeTo，可见 VersionEdit 的序列化格式大致上是个 KV map，其中一部分字段（如 `log_number_`, `comparator_`, `next_file_number_`）用于表示 Version 的快照而非增量修改：

```
void VersionEdit::EncodeTo(std::string* dst) const {
    if (has_comparator_) {
	PutVarint32(dst, kComparator);
	PutLengthPrefixedSlice(dst, comparator_);
    }
    if (has_log_number_) {
	PutVarint32(dst, kLogNumber);
	PutVarint64(dst, log_number_);
    }
    if (has_prev_log_number_) {
	PutVarint32(dst, kPrevLogNumber);
	PutVarint64(dst, prev_log_number_);
    }
    if (has_next_file_number_) {
	PutVarint32(dst, kNextFileNumber);
	PutVarint64(dst, next_file_number_);
    }
    if (has_last_sequence_) {
	PutVarint32(dst, kLastSequence);
	PutVarint64(dst, last_sequence_);
    }
    for (size_t i = 0; i < compact_pointers_.size(); i++) {
	PutVarint32(dst, kCompactPointer);
	PutVarint32(dst, compact_pointers_[i].first);  // level
	PutLengthPrefixedSlice(dst, compact_pointers_[i].second.Encode());
    }
    for (DeletedFileSet::const_iterator iter = deleted_files_.begin();
    iter != deleted_files_.end();
	++iter) {
	PutVarint32(dst, kDeletedFile);
	PutVarint32(dst, iter->first);   // level
	PutVarint64(dst, iter->second);  // file number
    }

    for (size_t i = 0; i < new_files_.size(); i++) {
	const FileMetaData& f = new_files_[i].second;
	PutVarint32(dst, kNewFile);
	PutVarint32(dst, new_files_[i].first);  // level
	PutVarint64(dst, f.number);
	PutVarint64(dst, f.file_size);
	PutLengthPrefixedSlice(dst, f.smallest.Encode());
	PutLengthPrefixedSlice(dst, f.largest.Encode());
    }
}
```

MANIFEST 文件的格式与 Memtable 的 WAL log 的格式相同，log 逻辑上按 record 进行读写，而物理上按 block 进行组织，每个 record 有个小 header，保存着 checksum、长度和类型。单个 record 的长度可能大于单个 Block，为此对于这类 record 配备了 FIRST、MIDDLE、LAST 三种类型，表示横跨多个 block。每次重启恢复时，会轮转 MANIFEST 文件，使文件编号递增 1，轮转后，会在新 MANIFEST 文件的开头写一份当前 version 元信息的快照（VersionSet::WriteSnapshot）。

![img](/images/2019-09-08-leveldb-version/leveldb-log2.png)

## References

- <https://catkang.github.io/2017/02/03/leveldb-version.html>
- <https://github.com/google/leveldb/blob/master/db/version_set.h>
- <https://ayende.com/blog/161698/reviewing-leveldb-part-vii-the-version-is-where-the-levels-are>
- <https://sf-zhou.github.io/leveldb/leveldb_07_version.html>
