---
date: "2019-09-08T00:00:00Z"
title: "Notes on LevelDB: Version"
---

The term "version" in LevelDB is a bit weird. Actually, it refers to the metadata of LevelDB: which sstable files are in each Level, and which WAL files are there. Whenever a new sstable is generated due to compaction, this metadata changes accordingly. Changes to this metadata must be logged (in the MANIFEST file) for crash recovery. LevelDB allows others to access the database using older metadata at the same time, so multiple versions of metadata can exist simultaneously. You can roughly think of it this way: LevelDB's metadata is managed in a version control manner, and a complete set of metadata is a version.

The main classes related to Version are VersionSet, Version, and VersionEdit. Version represents an immutable version of metadata, and all active Versions are contained in a doubly linked list in VersionSet. Each Version has a reference count, and when its lifecycle ends, it removes itself from this doubly linked list:

![leveldb-versionset](/images/2019-09-08-leveldb-version/leveldb-versionset.png)

VersionSet is actually a big entry point for LevelDB, maintaining a lot of information. Daily read and write operations, as well as compaction, need to go through VersionSet's current to find the metadata in the current version to locate the corresponding sstable.

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

The members of Version:

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

It can be roughly divided into three parts. First, there's the runtime information of the VersionSet doubly linked list and reference count. Then, the main FileMetaData information, used to discover which SST files are in each level. Finally, there's auxiliary information for compaction. After compaction, file changes are deterministic, and at the moment of Version change, the next compaction level and files can be designated based on information like compaction_score_.

The fields of FileMetaData are as follows, with the main content being the key ranges of smallest and largest:

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

The main modifications to Version occur after compaction. Compaction is divided into minor compaction and major compaction. Minor compaction writes the memtable to L0 and only adds files, while major compaction merges across levels, adding and deleting files. Whenever this happens, a new VersionEdit generates a new Version and inserts it at the head of the VersionSet linked list.

The members of VersionEdit:

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

Each VersionEdit is logged to the MANIFEST file for crash recovery, and it can be considered that the records in the MANIFEST file always correspond one-to-one with VersionEdit. However, during crash recovery, generating a new Version object each time a VersionEdit is applied is not very meaningful. VersionSet::Builder functions somewhat like a StringBuilder, merging consecutive operations to produce the final result, saving the overhead of generating intermediate results:

![img](/images/2019-09-08-leveldb-version/wjxyapo.png)

The serialization logic of VersionEdit is in EncodeTo. It can be seen that the serialization format of VersionEdit is roughly a KV map, with some fields (like `log_number_`, `comparator_`, `next_file_number_`) representing snapshots of the Version rather than incremental modifications:

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

The format of the MANIFEST file is the same as the WAL log of the Memtable. The log is logically read and written by record, but physically organized by block. Each record has a small header, storing checksum, length, and type. The length of a single record may exceed a single Block, so for such records, there are three types: FIRST, MIDDLE, and LAST, indicating spanning multiple blocks. Each time the system restarts, the MANIFEST file is rotated, incrementing the file number by 1. After rotation, a snapshot of the current version's metadata (VersionSet::WriteSnapshot) is written at the beginning of the new MANIFEST file.

![img](/images/2019-09-08-leveldb-version/leveldb-log2.png)

## References

- <https://catkang.github.io/2017/02/03/leveldb-version.html>
- <https://github.com/google/leveldb/blob/master/db/version_set.h>
- <https://ayende.com/blog/161698/reviewing-leveldb-part-vii-the-version-is-where-the-levels-are>
- <https://sf-zhou.github.io/leveldb/leveldb_07_version.html>
