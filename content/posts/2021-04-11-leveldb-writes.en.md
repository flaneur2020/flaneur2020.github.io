---
date: "2019-09-08T00:00:00Z"
title: "Notes on LevelDB: Writes"
---

All write operations in LevelDB, including `Put` and `Delete` are unified and recorded in the `WriteBatch` structure, then passed through the Write function as the only entry point for write operations:

```C++
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

The `Write` function takes a WriteOptions and WriteBatch as parameters:

```C++
Status DBImpl::Write(const WriteOptions& options, WriteBatch* updates) {
```

Let's dive into the content of the `Write` function.

## Enqueue writers_ and Wait

```C++
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

`Write` initializes a `Writer` object at the beginning of the function to track the lifecycle of this write operation, and under the protection of a lock, adds the writer to the end of the member variable `writers_`, which is of type `deque<Writer*>`.

Then, if the current Writer is not at the head of the queue, `w.cv.Wait()` enters a wait state. After the wait, if `w.done`, it directly returns `w.status`.

This logic is a bit clever. A colleague said it's like an election, making the first thread that initiates the write the only writer, responsible for initiating compaction and batch processing writes from other threads.

The Writer here acts as the coordination tool for this process, and its implementation is simple:

```C++
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

It's roughly divided into three parts:

1. Original parameters of the operation (batch, sync)
2. Coordination primitives for the waiting process (cv)
3. Results after the operation is done (done, status)

## Pre-Write Preparation: MakeRoomForWrite and BuildBatchGroup

```C++
  // May temporarily unlock and wait.
  Status status = MakeRoomForWrite(updates == nullptr);
  uint64_t last_sequence = versions_->LastSequence();
  Writer* last_writer = &w;
  if (status.ok() && updates != nullptr) {  // nullptr batch is for compactions
    WriteBatch* write_batch = BuildBatchGroup(&last_writer);
    WriteBatchInternal::SetSequence(write_batch, last_sequence + 1);
    last_sequence += WriteBatchInternal::Count(write_batch);
```

We haven't started writing yet, still in the preparation phase before writing.

The `MakeRoomForWrite` method checks if the Memtable is full or if there are too many L0 SSTables, and initiates Compaction or throttling accordingly. During these time-consuming operations, the lock is released, which is the opportunity for other threads to enter the `writers_` queue and accumulate batches.

There's also a check for `bg_error_` in `MakeRoomForWrite`, which determines if there's an unrecoverable serious error in the current database. If so, all subsequent write operations for this database instance are stopped.

The `BuildBatchGroup` method pops some `Writer`s from the `writers_` queue, merges their `WriteBatch`es into `tmp_batch_`, allowing batch aggregation for writes.

Since there's only one writer at a time, it's safe to reuse the same `tmp_batch_` here. If there's only one `Writer` in the `writers_` queue, `BuildBatchGroup` won't do any extra copying and will directly return the WriteBatch in this Writer.

The `&last_writer` passed into `BuildBatchGroup` is a pointer to a pointer, and its value will be replaced with the last `Writer` in the batch chosen by `BuildBatchGroup`. The remaining `Writer`s and newly inserted ones stay in the `writers_` queue for processing in the next write. After this Write method exits, it tries to wake up the first `Writer` in the `writers_` queue, making it the new sole writer.

When choosing batches, `BuildBatchGroup` has some strategies. First, batches aren't always better the bigger they are. Optimizing throughput might come at the cost of latency for individual requests. There might be cases where 100 small Put requests hitch a ride with one large Put request, causing the latency for those 100 small requests to be dragged out by the large request.

In the code, there's a limit on max_size. If the current batch exceeds max_size, it stops accumulating more batches. Additionally, a heuristic is used: if the first Writer is a small write, the upper limit of max_size is reduced, so these small writes can finish earlier, avoiding the situation where a small write is delayed by a large write in the batch, thus increasing tail latency.

Secondly, whether sync is enabled has a huge impact on the response speed of a batch. If a user's Put operation doesn't enable sync, they might expect a quick return. However, if a non-sync operation is included in a batch with a sync operation, the latency of the non-sync operation will be increased.

If the current Writer doesn't enable sync, `BuildBatchGroup` stops accumulating batches when it encounters a sync-enabled Writer. It can be considered that the parameters in a batch align with the current sole writer. If no sync is needed in a batch, this batch can be processed relatively quickly.

## Writing to Log and MemTable

```C++
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

Next up is the persistence to disk. First, write the contents of `write_batch` to `log_`, then write to MemTable, and finally update the `last_sequence` recorded in `VersionSet`. At this point, the writing process is complete, and all that's left is coordination and notification.

If the log `Sync()` fails, it means the log for this database is inconsistent, and no new writes should be allowed to prevent further corruption of the log files. `RecordBackgroundError` will log the error into the `bg_error_` field and wake up `background_work_finished_signal_` to notify background tasks about this error.

```C++
void DBImpl::RecordBackgroundError(const Status& s) {
  mutex_.AssertHeld();
  if (bg_error_.ok()) {
    bg_error_ = s;
    background_work_finished_signal_.SignalAll();
  }
}
```

Once `bg_error_` is set, the database is considered in an unrecoverable error state, and all subsequent new write operations are halted.

## Waking Up Writers_

```C++
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

After completing the write, the final coordination and notification phase begins:

1. Wake up the Writers in the current batch. Their `done` field is set to true, and they will return directly with the `status` field from the Writer;
2. Elect the remaining head of the `writers_` queue as the sole writer. Its `done` field is false, and it will initiate the next round of write operations;

## Summary

1. LevelDB defines a `Writer` structure during the write process to assist with coordination. This approach of encapsulating simple synchronization tools on top of primitive synchronization primitives, and then organizing the main process based on these tools, is often encountered in grpc and etcd code, and is worth learning from.
2. In a single writer scenario, it's easy to think of using a separate goroutine to perform write operations in Go. In LevelDB, this approach is equivalent to electing one thread as the sole writer in each write cycle.
3. Trade-offs in batching: The latency of each operation in a batch depends on the latency of the entire batch. Therefore, batches are not always better when larger. It's best to have a strategy that separates small and large operations into different batches to avoid small operations being slowed down by large operations, thus minimizing tail latency.
