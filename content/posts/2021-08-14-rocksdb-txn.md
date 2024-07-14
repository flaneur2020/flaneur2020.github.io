---
date: "2021-08-14T00:00:00Z"
title: rocksdb 事务笔记
---

rocksdb 支持 PessimisticTransactionDB 和 OptimisticTransactionDB 两种并发控制模式的支持，两者似乎都是对 DB 对象的外部包装，在存储外面做并发控制，允许应用程序按 BEGIN、COMMIT、ROLLBACK 几个 API 做到事务性的 KV 读写能力。

rocksdb 原本就拥有原子性写入 WriteBatch 的能力，事务就正好在 WriteBatch 的基础上做事情，事务内的写入会暂存在自己的 WriteBatch 中，事务内读取时会先读取自己的 WriteBatch，再读取 MemTable、L0、L1 等。期间其他事务看不到 WriteBatch 中的内容，到事务提交时，WriteBatch 的值会落入 MemTable 和 WAL，才可以使其他事务看到。

既然 LSM Tree 原本就有了原子写入能力，rocksdb 中事务所作的事情，主要就在于 LSM Tree 外部的并发控制层面，而且对乐观并发控制和悲观并发控制都做了支持：

- 乐观并发控制事务中，会跟踪事务内 Key 的读写，到 Commit() 时进行冲突检测，检查这些 Key 是否被其他事务修改过了，如果有，则放弃写入 WriteBatch，就等于什么都没有发生；
- 悲观并发控制事务中，会对事务内写入的 Key 进行上锁，当 Commit() 中当这些 Key 落盘之后，释放锁，就没有冲突检测环节了；

事务 API 使用起来，大致上这样：

``` c++
TransactionDB* txn_db;
Status s = TransactionDB::Open(options, path, &txn_db);

Transaction* txn = txn_db->BeginTransaction(write_options, txn_options);
s = txn->Put(“key”, “value”);
s = txn->Delete(“key2”);
s = txn->Merge(“key3”, “value”);
s = txn->Commit();
delete txn;
```

## 乐观并发控制：快照与冲突检测

先看一下乐观并发控制。前面提到乐观并发控制主要在于 Commit() 时刻的冲突检测，除此之外，也依赖着 Snapshot 机制实现事务的隔离性。rocksdb 的 Snapshot 相比 leveldb 没有变化，就是每次写入 WriteBatch 时会递增 sequence 号，Snapshot 的内容就是这个 sequence 号，查询时过滤按小于等于 sequence 号的 Key 的最新值为准。

跟踪 Key 的入口来自 OptimisticTransactionImpl::TryLock 调用的 TrackKey() 方法：

``` c++
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

其中 tracked_locks_ 是个 LockTracker 对象的 unique_ptr：

``` c++
// Tracks the lock requests.
// In PessimisticTransaction, it tracks the locks acquired through LockMgr;
// In OptimisticTransaction, since there is no LockMgr, it tracks the lock
// intention. Not thread-safe.
class LockTracker {
```

按 LockTracker 类的介绍，乐观并发控制与悲观并发控制的不同在于悲观并发控制会使用 LockMgr 对上锁解锁进行管理，而乐观并发控制中没有 LockMgr，只做跟踪，并不真正上锁。LockTracker 的实现大致上是对集合的便捷封装，这里先略过，先看一下 Commit 中是怎样基于它里面记录的信息进行冲突检测的。

冲突检测的主要函数入口是 OptimisticTransaction::CheckTransactionForConflicts()，它会进一步调用 TransactionUtil::CheckKeysForConflicts() 执行冲突检测：

``` c++
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

rocksdb 没有实现 SSI，只有对写操作的冲突跟踪，过程会比 badger 的 SSI 要简单一些，只需要检查一件事情，就是在提交时，每个 Key 在 db 中有没有比当前事务开始的 sequence 号之后更新的写入存在。

最笨的办法就是把每个 Key 读一次 db，获得它最近的 sequence 号，与事务的 sequence 号做比较。如果 db 中的 sequence 号更大，那么对不起，事务冲突了。

rocksdb 肯定不会做对事务中的每个 Key 都读 N 次 IO 这样的事情。这里有一个思路是，事务的执行时间并不长，在冲突检查这个场景下，并不需要找到 Key 的所有历史数据来做判断，而只需要近期的数据就可以了。而最近期的数据就是 MemTable，所以做近期的冲突检测，只需要读 MemTable 的数据就足够了，不需要执行 IO。

不过这一来就多出来一个约束条件，就是事务的开始时间如果比 MemTable 中最老的 Key 还早，就无法判断了，这时 rocksdb 的处理也比较暴力，就直接说这个事务过期了，重试吧。下面是 TransactionUtil::CheckKey 中相关的逻辑：

``` c++
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

使用 MemTable 实现冲突检测判定的思路还是很巧妙的，但只适用于只需要跟踪写入操作的 SI 场景，若 SSI 也用同一机制的话，就得使所有读操作都变成 MemTable 写入了，是远不如 badger 那样跟踪 key 集合好使的。

## 悲观并发控制：锁管理

悲观并发控制是通过在对键读写时上锁，使其他尝试取得锁的事务进入等待从而实现的并发控制。不过这里“上锁”在实现层面并非每个键真的对应一个 mutex，而是有一套自己的行锁语义实现。

主要的对象有 LockMap、LockMapStripe、LockInfo 几个：

``` c++
// Map of #num_stripes LockMapStripes
struct LockMap {
  // Number of sepearate LockMapStripes to create, each with their own Mutex
  const size_t num_stripes_;

  // Count of keys that are currently locked in this column family.
  // (Only maintained if PointLockManager::max_num_locks_ is positive.)
  std::atomic<int64_t> lock_cnt{0};

  std::vector<LockMapStripe*> lock_map_stripes_;
};
```

``` c++
struct LockMapStripe {
  // Mutex must be held before modifying keys map
  std::shared_ptr<TransactionDBMutex> stripe_mutex;

  // Condition Variable per stripe for waiting on a lock
  std::shared_ptr<TransactionDBCondVar> stripe_cv;

  // Locked keys mapped to the info about the transactions that locked them.
  // TODO(agiardullo): Explore performance of other data structures.
  std::unordered_map<std::string, LockInfo> keys;
};
```

``` c++
struct LockInfo {
  bool exclusive;
  autovector<TransactionID> txn_ids;

  // Transaction locks are not valid after this time in us
  uint64_t expiration_time;
}
```

其中 LockMap 是一个 Column Family 中所有锁的入口，每个 LockMap 会分为 16 个 LockMapStripe（条带），LockMapStripe 里面有一个键到 LockInfo 之间的映射。

简单来讲，LockMap 就是一个 Key 到 LockInfo 的映射表，内部分一下条带 LockMapStripe 来提高自己的并发度。

LockInfo 有读写锁的语义，exclusive 表示是否独占，如果不是，则可以允许多个读操作同时持有该锁，持有锁的事务列表会维护在 txn_ids 中。

如果获取锁时进入等待，会等待在 stripe_cv 上面。可见这里的锁是基于 CondVar 和 mutex 等系统原语实现的用户态读写锁。

回来看一下 TryLock() 的实现：

``` c++
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

这部分比较易读，就是获取 LockMapStripe，生成 LockInfo，最后调用 AcquireWithTimeout 走到获取锁的过程：

``` c++
// Helper function for TryLock().
Status PointLockManager::AcquireWithTimeout(
    PessimisticTransaction* txn, LockMap* lock_map, LockMapStripe* stripe,
    ColumnFamilyId column_family_id, const std::string& key, Env* env,
    int64_t timeout, LockInfo&& lock_info) {
  // ... 获取 stripe 的 mutex

  // 尝试获取锁
  uint64_t expire_time_hint = 0;
  autovector<TransactionID> wait_ids;
  result = AcquireLocked(lock_map, stripe, key, env, std::move(lock_info),
                         &expire_time_hint, &wait_ids);

  if (!result.ok() && timeout != 0) {
    bool timed_out = false;
    do {
      // ... 根据 AcquireLocked() 返回的 expire_time_hint，计算 cv_end_time，即超时等待时间
      assert(result.IsBusy() || wait_ids.size() != 0);

      // ... 根据 AcquireLocked() 返回的 wait_ids，判断得知当前事务在依赖其他事务所持有的锁
      // ... 发起死锁探测

      // 进入等待
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

      // ... 清理死锁检测上下文

      if (result.IsTimedOut()) {
          timed_out = true;
          // Even though we timed out, we will still make one more attempt to
          // acquire lock below (it is possible the lock expired and we
          // were never signaled).
      }

      // 重新尝试获取锁
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

抛开死锁探测部分，这里的逻辑还比较简单：

1. 尝试获取锁 ，AcquireLocked() 是非阻塞的，拿不到就返回失败
1. 如果获取锁未成功，则退避等待 stripe_cv 的事件通知
1. 循环重新尝试获取锁

AcquireLocked() 里面就是一个教科书的读写锁实现了：

1. 如果 Key 对应的锁没有被占用，如无意外可上锁成功，会在 stripe 中保存锁信息，将 txn_ids 配置为当前事务 ID
    1. 这里的意外是指 rocksdb 中有一个配置 max_num_locks_ 限制锁的总数上限，若超过上限则上锁失败返回 Status::Busy
1. 如果 Key 对应的锁已被占用
    1. 如果被占用的锁和待获取的锁都没有互斥标记，则允许多位读者共同持有该锁，在 stripe 中的 lock_info 的 txn_ids 中追加当前事务 ID，表示获取读锁成功
    1. 如果被占用的锁和待获取的锁，任一存在着互斥标记，如无意外应该会获取锁失败返回 Status::TimedOut，同时也会返回 txn_ids 向调用者提示持有着锁的事务列表用于辅助死锁探测，不过有两个特例：
        1. 可递归锁：持有着该锁的正好是当前事务，则将锁的互斥标记覆盖，获取锁成功
        1. 锁超时：如果被占用的锁恰好超时了，可以抢到这把锁

AcquireLocked() 的代码如下：

``` c++
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

最后看一下解锁部分：

``` c++
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

其中 UnLockKey() 是对 stripe 中 LockInfo 中移除当前事务 ID 或者删除整个 LockInfo 的封装。

唤醒锁等待这里是一个很暴力的 stripe_cv→NotifyAll()，惊群的形式唤醒所有等待获取锁的选手，但只有一位选手可以成功得到 stripe_mutex。

## 悲观并发控制：死锁探测

死锁探测做的事情是跟踪事务之间的锁依赖关系，通过 BFS 遍历判断里面是否有环，提前阻止这类成环的上锁操作。rocksdb 中 deadlock_detect 默认是关闭的状态，关闭主动的死锁探测时，仍可以通过锁超时机制从死锁中恢复回来。

死锁探测发生在 AcquireWithTimeout 函数中：

``` c++
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

单次调用 AcquireLocked 上锁失败时会返回等待锁的 wait_ids 列表，wait_ids 这部分信息作为跟踪锁依赖图的依据。

与死锁探测相关的字段有：

``` c++
  // Must be held when modifying wait_txn_map_ and rev_wait_txn_map_.
  std::mutex wait_txn_map_mutex_;

  // Maps from waitee -> number of waiters.
  HashMap<TransactionID, int> rev_wait_txn_map_;
  // Maps from waiter -> waitee.
  HashMap<TransactionID, TrackedTrxInfo> wait_txn_map_;
  DeadlockInfoBuffer dlock_buffer_
```

其中 rev_wait_txn_map_ 似乎用于剪枝，跟踪每个事务 ID 的等待者的个数，如果等待者数为 0，那么一定是不会存在死锁依赖的，就不必走后面的图遍历了。反过来如果等待数 > 1，也不一定有死锁依赖，仍需要遍历一遍才知道。

wait_txn_map_ 字段是 BFS 遍历的目标对象，在 IncrementWaiters 中，会从当前事务等待的 wait_ids 开始对 wait_txn_map_ 进行遍历，如果遍历到当前事务的 ID，则认为有环存在。

BFS 部分的逻辑如下：

``` c++
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
      // 有死锁存在，结合 queue_parents 中的路径信息，记录到 dlock_buffer_ 中
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

这里使用了两个最大长度为 deadlock_detect_depth_ 数组 queue_values[] 以及 head 和 tail 两个下标来作为 BFS 队列，tail 代表队列的尾部，head 代表头部，当 head 追上 tail 时遍历结束。当找到死锁依赖时，结合根据 queue_parents[] 中的信息，记录形成死锁的路径到 dlock_buffer_ 中用于辅助诊断。

## 总结

- 在 OptimisticTransaction 中，rocksdb 直接使用 MemTable 来获取 Key 的最新 Sequence 号，用于 Commit 时的冲突探测，判断事务期间 Key 有没有被其他事务写入过
- 在 PessimisticTransactionDB 中，rocksdb 在 LockManager 中基于 CondVar 和 mutex 实现的行锁语义，每个 Key 对应一把行锁，获取行锁时若存在锁冲突，则等待 CondVar 通知
- 死锁冲突是一个锁依赖关系图的 BFS，寻找锁依赖之间是否存在环，如果存在，则认为会存在死锁，则提前阻止上锁，rocksdb 默认没有开启死锁探测，如有发生死锁仍可以通过锁超时恢复

## References

- [https://zhuanlan.zhihu.com/p/31255678](https://zhuanlan.zhihu.com/p/31255678)
- [https://kernelmaker.github.io/Rocksdb_transaction_1](https://kernelmaker.github.io/Rocksdb_transaction_1)
- [https://github.com/facebook/rocksdb/wiki/Transactions](https://github.com/facebook/rocksdb/wiki/Transactions)