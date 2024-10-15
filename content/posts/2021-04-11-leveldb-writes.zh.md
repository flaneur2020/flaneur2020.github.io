---
date: "2019-09-08T00:00:00Z"
title: "Notes on LevelDB: Writes"
---

LevelDB 中的所有写入操作包括 Put 和 Delete，都会统一记录在 WriteBatch 结构体中，再经过 Write 函数作为唯一的写入操作的入口：

```
Status DB::Put(const WriteOptions& opt, const Slice& key, const Slice& value) {
  WriteBatch batch;
  batch.Put(key, value);
  return Write(opt, &batch);
}

Status DB::Delete(const WriteOptions& opt, const Slice& key) {
  WriteBatch batch;
  batch.Delete(key);
  return Write(opt, &batch);
}
```

Write 函数接受一个 WriteOptions 和 WriteBatch 作为参数：

```
Status DBImpl::Write(const WriteOptions& options, WriteBatch* updates) {
```

后面顺着段落来看 Write 函数的内容。

## 入队 writers_ 并等待

```
  Writer w(&mutex_);
  w.batch = updates;
  w.sync = options.sync;
  w.done = false;

  MutexLock l(&mutex_);
  writers_.push_back(&w);
  while (!w.done && &w != writers_.front()) {
    w.cv.Wait();
  }
  if (w.done) {
    return w.status;
  }
  // ...
```

`Write` 在函数的开头初始化了一个 `Writer` 对象，用于跟踪这次写入操作的生命周期，并在锁的保护下将 writer 加入到类型为 `deque<Writer*>` 的成员变量 `writers_` 的末尾。

随后判断如如果当前 Writer 并非队列头部，则 `w.cv.Wait()` 进入等待。等待结束后，如果 `w.done`，则直接返回 `w.status`。

这里的逻辑有点巧妙，同事说这里相当于一个选举，使第一个发起写入的线程作为唯一写者，负责发起 compaction、批量化处理来自其他线程的写入操作。

Writer 就在这里扮演了这个过程的协调工具，它的实现很简单：

```
// Information kept for every waiting writer
struct DBImpl::Writer {
  explicit Writer(port::Mutex* mu)
      : batch(nullptr), sync(false), done(false), cv(mu) {}

  Status status;
  WriteBatch* batch;
  bool sync;
  bool done;
  port::CondVar cv;
};
```

大致上分为三段：

1. 操作的原始参数（batch, sync）
2. 等待操作过程的协调原语（cv）
3. 操作结束的结果（done, status）

## 写入前的准备工作：MakeRoomForWrite 与 BuildBatchGroup

```
  // May temporarily unlock and wait.
  Status status = MakeRoomForWrite(updates == nullptr);
  uint64_t last_sequence = versions_->LastSequence();
  Writer* last_writer = &w;
  if (status.ok() && updates != nullptr) {  // nullptr batch is for compactions
    WriteBatch* write_batch = BuildBatchGroup(&last_writer);
    WriteBatchInternal::SetSequence(write_batch, last_sequence + 1);
    last_sequence += WriteBatchInternal::Count(write_batch);
```

到这里还没有进入写入，仍是写入前的准备阶段。

`MakeRoomForWrite` 方法会判断 Memtable 是不是满了、当前 L0 的 SSTable 是不是太多了，从而发起 Compaction 或者限速。这种耗时的操作期间会释放锁，这也正是其他线程进入 `writers_` 队列攒批量的契机。

`MakeRoomForWrite` 中还有一个关于 `bg_error_` 的判断，认为当前库是否有不可恢复的严重错误，如果有，则停止此库实例后续的所有写入操作。

`BuildBatchGroup` 方法会在 `writers_` 队列中 pop 一些 `Writer` 出来，合并它们的 `WriteBatch` 到 `tmp_batch_` 中，允许攒批次聚合写入。

因为同一时刻只有单一写者，因此可以在这里安全地复用同一个 `tmp_batch_`。如果 `writers_` 队列中只有一个 `Writer`，那么 `BuildBatchGroup` 方法就不会多做拷贝，直接返回此 Writer 内的 WriteBatch。

`BuildBatchGroup` 传入的 `&last_writer` 是一个指针的指针，它的值会被替换为 `BuildBatchGroup` 选择的批次中的最后一个 `Writer`。剩下的其他 `Writer` 以及新插入的 Writer，就正常留在 `writers_` 队列中，到下次写入再处理。在这次 Write 方法退出之后，会尝试唤醒 `writers_` 队列的首个 `Writer`，相当于使它选举成为新的唯一写入者。

在选择批次时 BuildBatchGroup 有一定的策略，首先批次并非越大越好，优化吞吐可能是以单次请求的时延为代价的，有可能有这种情况是，100 个小 Put 请求搭车了一个大 Put 请求，导致这 100 个小 Put 请求返回响应的时延都被这个大请求给拖高了。

体现在代码上是，首先有一个 max_size 的限制，如果当前的批次已经超过了 max_size，则停止继续攒批次。此外一个 heuristic 是如果首个 Writer 是个比较小的写入，则缩小 max_size 的上限，那么这种小写入可以更早地结束，可以在一定程度上避免一个小写入被批次中搭车了一个大写入，而使 tail latency 变高的情况。

其次，是否开启 sync 对一个批次的响应速度影响巨大，如果用户的一个 Put 操作没有开启 sync，那么他可能认为这个请求可以较快地返回，但是如果一个非 sync 操作被搭车到一个 sync 操作的批次中，会使这个非 sync 的操作的延时给拖高。

如果当前 Writer 并未开启 sync，BuildBatchGroup 当遇到开启 sync 的 Writer 时会停止攒批次。可以认为一个批次中的参数都与当前唯一写者对齐，如果一个批次都不需要 sync，那这轮批次处理起来就能比较快地结束。

## 写入日志与 MemTable

```
    // Add to log and apply to memtable.  We can release the lock
    // during this phase since &w is currently responsible for logging
    // and protects against concurrent loggers and concurrent writes
    // into mem_.
    {
      mutex_.Unlock();
      status = log_->AddRecord(WriteBatchInternal::Contents(write_batch));
      bool sync_error = false;
      if (status.ok() && options.sync) {
        status = logfile_->Sync();
        if (!status.ok()) {
          sync_error = true;
        }
      }
      if (status.ok()) {
        status = WriteBatchInternal::InsertInto(write_batch, mem_);
      }
      mutex_.Lock();
      if (sync_error) {
        // The state of the log file is indeterminate: the log record we
        // just added may or may not show up when the DB is re-opened.
        // So we force the DB into a mode where all future writes fail.
        RecordBackgroundError(status);
      }
    }
    if (write_batch == tmp_batch_) tmp_batch_->Clear();

    versions_->SetLastSequence(last_sequence);
  }
```

接下来就是持久性落盘，先将 `write_batch` 中的内容写入 `log_`，后写 MemTable，最后更新 `VersionSet` 中记录的 `last_sequence`。到这里已完成写入过程，剩下的就是协调通知了。

如果日志 `Sync()` 失败，那么会认为这个库的日志已不一致，不应该有任何新的写入，避免日志文件进一步乱掉。`RecordBackgroundError` 会将错误记录到 `bg_error_` 字段中，并唤醒 `background_work_finished_signal_` 用于向后台任务周知此错误。

```
void DBImpl::RecordBackgroundError(const Status& s) {
  mutex_.AssertHeld();
  if (bg_error_.ok()) {
    bg_error_ = s;
    background_work_finished_signal_.SignalAll();
  }
}
```

一旦设置 `bg_error_`，就认为这个库进入了不可恢复的错误状态中，停止一切后续的新写入操作。

## 唤醒 writers_

```
while (true) {
    Writer* ready = writers_.front();
    writers_.pop_front();
    if (ready != &w) {
      ready->status = status;
      ready->done = true;
      ready->cv.Signal();
    }
    if (ready == last_writer) break;
  }

  // Notify new head of write queue
  if (!writers_.empty()) {
    writers_.front()->cv.Signal();
  }

  return status;
}
```

完成写入之后，就是最后的协调通知环节：

1. 使当前写入批次中的 Writer 唤醒，它们的 `done` 字段为 true，会直接返回 Writer 中的 `status` 字段进行返回；
2. 选举 `writers_` 队列中剩余的队首作为唯一写者，它的 `done` 字段为 false，会发起下一轮的写入操作；

## 总结

1. LevelDB 在写入过程中定义了 `Writer` 结构，用于辅助写入过程的协调，这类基于原始的同步原语之上封装简单的同步工具类，继而基于同步工具类来组织起主要流程的做法，在之前看 grpc 与 etcd 代码中也常有遇到，很值得学习。
2. 唯一写者场景下，在 go 里面比较容易想到使用一个单独的 goroutine 来执行写操作，在 LevelDB 在这里的做法相当于在每轮写入周期中，选举其中的一个线程作为唯一写者。
3. 批次的取舍：批次中每个操作的延时取决于整个批次的延时，因此批次并非越大越好，最好有一定策略，能尽量把小操作和大操作分离在不同的批次中，尽量避免小操作被大操作拖慢 tail latency。