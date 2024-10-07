---
date: "2021-08-14T00:00:00Z"
title: "Transaction Internals: RocksDB"
---

RocksDB supports two concurrency control modes: `PessimisticTransactionDB` and `OptimisticTransactionDB`. Both seem to be external wrappers around the `DB` object, implementing concurrency control outside of storage. This allows applications to perform transactional KV read and write operations through `BEGIN`, `COMMIT`, and `ROLLBACK` APIs.

RocksDB inherently supports atomic write capabilities with `WriteBatch`. Transactions build on `WriteBatch`, with writes within a transaction temporarily stored in its own `WriteBatch`. During reads, it first reads from its `WriteBatch`, then from `MemTable`, L0, L1, etc. Other transactions can't see the contents of the `WriteBatch` until the transaction commits and the `WriteBatch` values are placed into MemTable and WAL, making them visible to other transactions.

Since the LSM Tree already has atomic write capabilities, what transactions do in RocksDB mainly involves concurrency control outside the LSM Tree, supporting both optimistic and pessimistic concurrency control:

- In optimistic concurrency control transactions, the keys read and written within the transaction are tracked. At `Commit()`, conflict detection checks if these keys have been modified by other transactions. If so, the WriteBatch is abandoned, meaning nothing happened.
- In pessimistic concurrency control transactions, locks are placed on keys written within the transaction. Once these keys are flushed to disk in `Commit()`, the locks are released, eliminating the conflict detection step.

The transaction API is used roughly like this:

```c++
TransactionDB* txn_db;
Status s = TransactionDB::Open(options, path, &txn_db);

Transaction* txn = txn_db->BeginTransaction(write_options, txn_options);
s = txn->Put("key", "value");
s = txn->Delete("key2");
s = txn->Merge("key3", "value");
s = txn->Commit();
delete txn;
```

## Optimistic Concurrency Control: Snapshots and Conflict Detection

Let’s look at optimistic concurrency control first. As mentioned earlier, it primarily involves conflict detection at the `Commit()` moment and relies on the Snapshot mechanism to achieve transaction isolation. The Snapshot in RocksDB is unchanged from LevelDB, incrementing the sequence number with each `WriteBatch` write. The content of a Snapshot is just a sequence number; queries filter based on the latest value of keys less than or equal to this sequence number.

Key tracking starts from the `OptimisticTransactionImpl::TryLock` calling the `TrackKey()` method:

```c++
void TransactionBaseImpl::TrackKey(uint32_t cfh_id, const std::string& key,
                                   SequenceNumber seq, bool read_only,
                                   bool exclusive) {
  PointLockRequest r;
  r.column_family_id = cfh_id;
  r.key = key;
  r.seq = seq;
  r.read_only = read_only;
  r.exclusive = exclusive;

  // Update map of all tracked keys for this transaction
  tracked_locks_->Track(r);

  if (save_points_ != nullptr && !save_points_->empty()) {
    // Update map of tracked keys in this SavePoint
    save_points_->top().new_locks_->Track(r);
  }
}
```

`tracked_locks_` is a unique_ptr to a LockTracker object:

```c++
// Tracks the lock requests.
// In PessimisticTransaction, it tracks the locks acquired through LockMgr;
// In OptimisticTransaction, since there is no LockMgr, it tracks the lock
// intention. Not thread-safe.
class LockTracker {
```

According to the `LockTracker` class description, the difference between optimistic and pessimistic concurrency control is that pessimistic uses `LockMgr` for locking and unlocking management, whereas optimistic does tracking only, without actual locking. `LockTracker` is essentially a convenient wrapper over a set, which we'll skip for now to look at how `Commit` uses its recorded information for conflict detection.

The main function for conflict detection is `OptimisticTransaction::CheckTransactionForConflicts()`, which further calls `TransactionUtil::CheckKeysForConflicts()` for conflict detection:

```c++
Status TransactionUtil::CheckKeysForConflicts(DBImpl* db_impl,
                                              const LockTracker& tracker,
                                              bool cache_only) {
  Status result;

  std::unique_ptr<LockTracker::ColumnFamilyIterator> cf_it(
      tracker.GetColumnFamilyIterator());
  assert(cf_it != nullptr);
  while (cf_it->HasNext()) {
    ColumnFamilyId cf = cf_it->Next();

    SuperVersion* sv = db_impl->GetAndRefSuperVersion(cf);
    if (sv == nullptr) {
      result = Status::InvalidArgument("Could not access column family " +
                                       ToString(cf));
      break;
    }

    SequenceNumber earliest_seq =
        db_impl->GetEarliestMemTableSequenceNumber(sv, true);

    // For each of the keys in this transaction, check to see if someone has
    // written to this key since the start of the transaction.
    std::unique_ptr<LockTracker::KeyIterator> key_it(
        tracker.GetKeyIterator(cf));
    assert(key_it != nullptr);
    while (key_it->HasNext()) {
      const std::string& key = key_it->Next();
      PointLockStatus status = tracker.GetPointLockStatus(cf, key);
      const SequenceNumber key_seq = status.seq;

      result = CheckKey(db_impl, sv, earliest_seq, key_seq, key, cache_only);
      if (!result.ok()) {
        break;
      }
    }

    db_impl->ReturnAndCleanupSuperVersion(cf, sv);

    if (!result.ok()) {
      break;
    }
  }

  return result;
}
```

RocksDB doesn't implement SSI and only tracks the write conflicts, making the process simpler than Badger's SSI. It only needs to check one thing: at commit time, whether any key in the db has a write with a sequence number greater than the transaction's starting sequence number.

The simplest approach would be to read each key from the db once to get its latest sequence number and compare it with the transaction's sequence number. If the db's sequence number is larger, then sorry, the transaction is conflicted.

RocksDB definitely wouldn't perform N IOs for each key in a transaction. One idea here is that transaction execution time isn't long, and for conflict checks, you don't need all historical data for a key, just recent data. The most recent data is in the MemTable, so for recent conflict detection, just reading MemTable data is sufficient, no IO needed.

However, this introduces a constraint: if the transaction start time is earlier than the oldest key in the MemTable, it can't be judged. In such cases, RocksDB simply says the transaction is expired and should be retried. Below is the relevant logic in `TransactionUtil::CheckKey`:

```c++
  // Since it would be too slow to check the SST files, we will only use
  // the memtables to check whether there have been any recent writes
  // to this key after it was accessed in this transaction.  But if the
  // Memtables do not contain a long enough history, we must fail the
  // transaction.
  if (earliest_seq == kMaxSequenceNumber) {
    // The age of this memtable is unknown.  Cannot rely on it to check
    // for recent writes.  This error shouldn't happen often in practice as
    // the Memtable should have a valid earliest sequence number except in some
    // corner cases (such as error cases during recovery).
    need_to_read_sst = true;

    if (cache_only) {
      result = Status::TryAgain(
          "Transaction could not check for conflicts as the MemTable does not "
          "contain a long enough history to check write at SequenceNumber: ",
          ToString(snap_seq));
    }
  } else if (snap_seq < earliest_seq || min_uncommitted <= earliest_seq) {
    // Use <= for min_uncommitted since earliest_seq is actually the largest sec
    // before this memtable was created
    need_to_read_sst = true;

    if (cache_only) {
      // The age of this memtable is too new to use to check for recent
      // writes.
      char msg[300];
      snprintf(msg, sizeof(msg),
               "Transaction could not check for conflicts for operation at "
               "SequenceNumber %" PRIu64
               " as the MemTable only contains changes newer than "
               "SequenceNumber %" PRIu64
               ".  Increasing the value of the "
               "max_write_buffer_size_to_maintain option could reduce the "
               "frequency "
               "of this error.",
               snap_seq, earliest_seq);
      result = Status::TryAgain(msg);
    }
  }
```

Using MemTable to implement conflict detection is clever but only suitable for SI scenarios that track writes. If SSI used the same mechanism, all read operations would need to become MemTable writes, which is not as effective as Badger's key collection tracking.

## Pessimistic Concurrency Control: Lock Management

Pessimistic concurrency control implements concurrency control by locking keys during read and write operations, causing other transactions attempting to acquire the locks to wait. However, this "locking" does not mean each key corresponds to a mutex at the implementation level but rather has its own row lock semantics implementation.

The main objects are `LockMap`, `LockMapStripe`, and `LockInfo`:

```c++
// Map of #num_stripes LockMapStripes
struct LockMap {
  // Number of separate LockMapStripes to create, each with their own Mutex
  const size_t num_stripes_;

  // Count of keys that are currently locked in this column family.
  // (Only maintained if PointLockManager::max_num_locks_ is positive.)
  std::atomic<int64_t> lock_cnt{0};

  std::vector<LockMapStripe*> lock_map_stripes_;
};
```

```c++
struct LockMapStripe {
  // Mutex must be held before modifying keys map
  std::shared_ptr<TransactionDBMutex> stripe_mutex;

  // Condition Variable per stripe for waiting on a lock
  std::shared_ptr<TransactionDBCondVar> stripe_cv;

  // Locked keys mapped to the info about the transactions that locked them.
  // TODO: Explore performance of other data structures.
  std::unordered_map<std::string, LockInfo> keys;
};
```

```c++
struct LockInfo {
  bool exclusive;
  autovector<TransactionID> txn_ids;

  // Transaction locks are not valid after this time in us
  uint64_t expiration_time;
}
```

`LockMap` is the entry point for all locks in a Column Family, divided into 16 `LockMapStripes` (stripes), with each `LockMapStripe` containing a map of keys to `LockInfo`.

In simple terms, `LockMap` is a mapping table from Key to `LockInfo`, internally divided into stripes to improve concurrency.

LockInfo has read-write lock semantics; `exclusive` indicates exclusivity. If not, multiple read operations can hold the lock simultaneously, and the list of transactions holding the lock is maintained in `txn_ids`.

If a lock acquisition enters waiting, it waits on `stripe_cv`. The lock here is a user-space read-write lock based on CondVar and mutex system primitives.

Let's look at the `TryLock()` implementation:

```c++
Status PointLockManager::TryLock(PessimisticTransaction* txn,
                                 ColumnFamilyId column_family_id,
                                 const std::string& key, Env* env,
                                 bool exclusive) {
  // Lookup lock map for this column family id
  std::shared_ptr<LockMap> lock_map_ptr = GetLockMap(column_family_id);
  LockMap* lock_map = lock_map_ptr.get();
  if (lock_map == nullptr) {
    char msg[255];
    snprintf(msg, sizeof(msg), "Column family id not found: %" PRIu32,
             column_family_id);

    return Status::InvalidArgument(msg);
  }

  // Need to lock the mutex for the stripe that this key hashes to
  size_t stripe_num = lock_map->GetStripe(key);
  assert(lock_map->lock_map_stripes_.size() > stripe_num);
  LockMapStripe* stripe = lock_map->lock_map_stripes_.at(stripe_num);

  LockInfo lock_info(txn->GetID(), txn->GetExpirationTime(), exclusive);
  int64_t timeout = txn->GetLockTimeout();

  return AcquireWithTimeout(txn, lock_map, stripe, column_family_id, key, env,
                            timeout, std::move(lock_info));
}
```

This part is fairly readable: it gets `LockMapStripe`, generates `LockInfo`, and finally calls `AcquireWithTimeout` to proceed with the lock acquisition:

```c++
// Helper function for TryLock().
Status PointLockManager::AcquireWithTimeout(
    PessimisticTransaction* txn, LockMap* lock_map, LockMapStripe* stripe,
    ColumnFamilyId column_family_id, const std::string& key, Env* env,
    int64_t timeout, LockInfo&& lock_info) {
  // ... Acquire the stripe mutex

  // Try to acquire the lock
  uint64_t expire_time_hint = 0;
  autovector<TransactionID> wait_ids;
  result = AcquireLocked(lock_map, stripe, key, env, std::move(lock_info),
                         &expire_time_hint, &wait_ids);

  if (!result.ok() && timeout != 0) {
    bool timed_out = false;
    do {
      // ... Calculate cv_end_time based on AcquireLocked()'s expire_time_hint, determining wait time
      assert(result.IsBusy() || wait_ids.size() != 0);

      // ... Based on wait_ids returned by AcquireLocked(), determine if the current transaction is dependent on other transactions' locks
      // ... Initiate deadlock detection

      // Enter waiting
      if (cv_end_time < 0) {
        // Wait indefinitely
        result = stripe->stripe_cv->Wait(stripe->stripe_mutex);
      } else {
        uint64_t now = env->NowMicros();
        if (static_cast<uint64_t>(cv_end_time) > now) {
          result = stripe->stripe_cv->WaitFor(stripe->stripe_mutex,
                                              cv_end_time - now);
        }
      }

      // ... Clean up deadlock detection context

      if (result.IsTimedOut()) {
          timed_out = true;
          // Even though we timed out, we will still make one more attempt to
          // acquire lock below (it is possible the lock expired and we
          // were never signaled).
      }

      // Retry acquiring the lock
      if (result.ok() || result.IsTimedOut()) {
        result = AcquireLocked(lock_map, stripe, key, env, std::move(lock_info),
                               &expire_time_hint, &wait_ids);
      }
    } while (!result.ok() && !timed_out);
  }

  stripe->stripe_mutex->UnLock();

  return result;
}
```

Ignoring the deadlock detection part, the logic here is fairly simple:

1. Attempt to acquire the lock; `AcquireLocked()` is non-blocking and returns failure if it can't acquire the lock.
2. If the lock acquisition fails, back off and wait for an event notification from `stripe_cv`.
3. Loop and retry acquiring the lock.

`AcquireLocked()` is a textbook read-write lock implementation:

1. If the lock for the key is not held, it generally succeeds, saving lock info in the stripe and setting `txn_ids` to the current transaction ID.
    1. The exception is if RocksDB's `max_num_locks_` configuration limits the total number of locks, exceeding the limit results in failure with `Status::Busy`.
2. If the lock for the key is held:
    1. If both the held lock and the requested lock are non-exclusive, multiple readers can hold the lock simultaneously, appending the current transaction ID to `txn_ids` in the stripe's lock_info, indicating successful read lock acquisition.
    2. If either the held lock or the requested lock has an exclusivity flag, the lock acquisition generally fails and returns `Status::TimedOut`, also returning `txn_ids` to assist in deadlock detection. However, there are two special cases:
        1. Recursive lock: If the lock holder is the current transaction, the lock's exclusivity flag is overwritten, and the lock acquisition succeeds.
        2. Lock expiration: If the held lock is expired, it can be seized.

The code for `AcquireLocked()` is as follows:

```c++
// Try to lock this key after we have acquired the mutex.
// Sets *expire_time to the expiration time in microseconds
//  or 0 if no expiration.
// REQUIRED:  Stripe mutex must be held.
Status PointLockManager::AcquireLocked(LockMap* lock_map, LockMapStripe* stripe,
                                       const std::string& key, Env* env,
                                       LockInfo&& txn_lock_info,
                                       uint64_t* expire_time,
                                       autovector<TransactionID>* txn_ids) {
  assert(txn_lock_info.txn_ids.size() == 1);

  Status result;
  // Check if this key is already locked
  auto stripe_iter = stripe->keys.find(key);
  if (stripe_iter != stripe->keys.end()) {
    // Lock already held
    LockInfo& lock_info = stripe_iter->second;
    assert(lock_info.txn_ids.size() == 1 || !lock_info.exclusive);

    if (lock_info.exclusive || txn_lock_info.exclusive) {
      if (lock_info.txn_ids.size() == 1 &&
          lock_info.txn_ids[0] == txn_lock_info.txn_ids[0]) {
        // The list contains one txn and we're it, so just take it.
        lock_info.exclusive = txn_lock_info.exclusive;
        lock_info.expiration_time = txn_lock_info.expiration_time;
      } else {
        // Check if it's expired. Skips over txn_lock_info.txn_ids[0] in case
        // it's there for a shared lock with multiple holders which was not
        // caught in the first case.
        if (IsLockExpired(txn_lock_info.txn_ids[0], lock_info, env,
                          expire_time)) {
          // lock is expired, can steal it
          lock_info.txn_ids = txn_lock_info.txn_ids;
          lock_info.exclusive = txn_lock_info.exclusive;
          lock_info.expiration_time = txn_lock_info.expiration_time;
          // lock_cnt does not change
        } else {
          result = Status::TimedOut(Status::SubCode::kLockTimeout);
          *txn_ids = lock_info.txn_ids;
        }
      }
    } else {
      // We are requesting shared access to a shared lock, so just grant it.
      lock_info.txn_ids.push_back(txn_lock_info.txn_ids[0]);
      // Using std::max means that expiration time never goes down even when
      // a transaction is removed from the list. The correct solution would be
      // to track expiry for every transaction, but this would also work for
      // now.
      lock_info.expiration_time =
          std::max(lock_info.expiration_time, txn_lock_info.expiration_time);
    }
  } else {  // Lock not held.
    // Check lock limit
    if (max_num_locks_ > 0 &&
        lock_map->lock_cnt.load(std::memory_order_acquire) >= max_num_locks_) {
      result = Status::Busy(Status::SubCode::kLockLimit);
    } else {
      // acquire lock
      stripe->keys.emplace(key, std::move(txn_lock_info));

      // Maintain lock count if there is a limit on the number of locks
      if (max_num_locks_) {
        lock_map->lock_cnt++;
      }
    }
  }

  return result;
}
```

Finally, let's look at the unlock part:

```c++
void PointLockManager::UnLock(PessimisticTransaction* txn,
                              ColumnFamilyId column_family_id,
                              const std::string& key, Env* env) {
  std::shared_ptr<LockMap> lock_map_ptr = GetLockMap(column_family_id);
  LockMap* lock_map = lock_map_ptr.get();
  if (lock_map == nullptr) {
    // Column Family must have been dropped.
    return;
  }

  // Lock the mutex for the stripe that this key hashes to
  size_t stripe_num = lock_map->GetStripe(key);
  assert(lock_map->lock_map_stripes_.size() > stripe_num);
  LockMapStripe* stripe = lock_map->lock_map_stripes_.at(stripe_num);

  stripe->stripe_mutex->Lock().PermitUncheckedError();
  UnLockKey(txn, key, stripe, lock_map, env);
  stripe->stripe_mutex->UnLock();

  // Signal waiting threads to retry locking
  stripe->stripe_cv->NotifyAll();
}
```

`UnLockKey()` is a wrapper for removing the current transaction ID from LockInfo in the stripe or deleting the entire LockInfo.

The lock waiting wake-up is a brute-force `stripe_cv->NotifyAll()`, waking up all contenders waiting to acquire the lock, but only one can successfully obtain the `stripe_mutex`.

## Pessimistic Concurrency Control: Deadlock Detection

Deadlock detection tracks lock dependencies between transactions, using BFS traversal to check for cycles, preventing circular lock operations. In RocksDB, deadlock detection is off by default; even if deadlocks occur, they can still be recovered from through lock timeouts.

Deadlock detection occurs in the AcquireWithTimeout function:

```c++
      // We are dependent on a transaction to finish, so perform deadlock
      // detection.
      if (wait_ids.size() != 0) {
        if (txn->IsDeadlockDetect()) {
          if (IncrementWaiters(txn, wait_ids, key, column_family_id,
                               lock_info.exclusive, env)) {
            result = Status::Busy(Status::SubCode::kDeadlock);
            stripe->stripe_mutex->UnLock();
            return result;
          }
        }
        txn->SetWaitingTxn(wait_ids, column_family_id, &key);
      }
```

A single call to AcquireLocked failing to acquire a lock returns a list of waiting locks `wait_ids`. This information is used to track the lock dependency graph.

Fields related to deadlock detection include:

```c++
  // Must be held when modifying wait_txn_map_ and rev_wait_txn_map_.
  std::mutex wait_txn_map_mutex_;

  // Maps from waitee -> number of waiters.
  HashMap<TransactionID, int> rev_wait_txn_map_;
  // Maps from waiter -> waitee.
  HashMap<TransactionID, TrackedTrxInfo> wait_txn_map_;
  DeadlockInfoBuffer dlock_buffer_
```

`rev_wait_txn_map_` seems to be used for pruning, tracking the number of waiters for each transaction ID. If the number of waiters is 0, there is certainly no deadlock dependency, so further graph traversal isn't necessary. Conversely, if the wait count > 1, a deadlock isn't guaranteed, and traversal is still needed to confirm.

`wait_txn_map_` is the target of the BFS traversal. In IncrementWaiters, traversal of `wait_txn_map_` begins from the current transaction's `wait_ids`. If the current transaction's ID is encountered, a cycle is detected.

The BFS logic is as follows:

```c++
  for (int tail = 0, head = 0; head < txn->GetDeadlockDetectDepth(); head++) {
    int i = 0;
    if (next_ids) {
      for (; i < static_cast<int>(next_ids->size()) &&
             tail + i < txn->GetDeadlockDetectDepth();
           i++) {
        queue_values[tail + i] = (*next_ids)[i];
        queue_parents[tail + i] = parent;
      }
      tail += i;
    }

    // No more items in the list, meaning no deadlock.
    if (tail == head) {
      return false;
    }

    auto next = queue_values[head];
    if (next == id) {
      // Deadlock detected, record path info from queue_parents into dlock_buffer_
      return true;
    } else if (!wait_txn_map_.Contains(next)) {
      next_ids = nullptr;
      continue;
    } else {
      parent = head;
      next_ids = &(wait_txn_map_.Get(next).m_neighbors);
    }
  }
```

Here, two arrays `queue_values[]` and `queue_parents[]`, both with a maximum length of `deadlock_detect_depth_`, are used as the BFS queue, with `tail` representing the end and `head` the front. Traversal ends when `head` catches up to `tail`. When a deadlock dependency is found, the path forming the deadlock is recorded in `dlock_buffer_` using the path information from `queue_parents[]`.

## Summary

- In OptimisticTransaction, RocksDB directly uses MemTable to get the latest sequence number of keys for conflict detection at commit, determining if keys were written by other transactions during the transaction.
- In PessimisticTransactionDB, RocksDB uses row lock semantics implemented in LockManager with CondVar and mutex. Each key corresponds to a row lock, and if a lock conflict exists when acquiring a lock, it waits for CondVar notification.
- Deadlock detection involves a BFS of the lock dependency graph to check for cycles. If a cycle exists, it indicates a potential deadlock and prevents locking. RocksDB does not enable deadlock detection by default; if deadlocks occur, they can still be resolved through lock timeouts.

## References

- [https://zhuanlan.zhihu.com/p/31255678](https://zhuanlan.zhihu.com/p/31255678)
- [https://kernelmaker.github.io/Rocksdb_transaction_1](https://kernelmaker.github.io/Rocksdb_transaction_1)
- [https://github.com/facebook/rocksdb/wiki/Transactions](https://github.com/facebook/rocksdb/wiki/Transactions)