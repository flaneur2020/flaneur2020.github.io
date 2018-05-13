---
layout: default
title: Designing Data-Intensive Applications
---

# 读书笔记: Designing Data-Intensive Applications


## Chapter 7: Transactions - Serializable Snapshot Isolation (SSI)

> SSI is fairly new: it was first described in 2008 [40] and is the subject of Michael Cahill's PhD thesis. ... it has the possibility of being fast enough to become the enw default in the future.
>



> Detecting writes that affect prior reads (the write occurs after the read).
>

SI 只跟踪活跃的写集合，而 SSI 需要跟踪活跃的读集合。commit 时冲突探测，判断事务的写操作是否影响到其他事务的读集合，如果影响到，那些事务的 premise 就认为已经不再成立了，那些事务如果只读便没问题，但是如果有写操作就会 abort。只是这时你仍不知道那些事务是不是只读的。

> In the context of two-phased locking we discussed index-range locks. .. Wee can use a similar technique here, except that SSI don't block other transactions.
>
> When a transaction writes to the database, it must look in the indexes for any other transactions that have recently read the affected data. This process is similar to acquiring a write lock on the affected key range, but rather than blocking until the readers have committed, the lock acts s a tripwire: it simply notifies the transactions that the data they read may no longer be up to date.
>

冲突探测的原理与 next-key lock 相似。


## Chapter 7: Transactions - Two Phase Locking

> Preducate locks
>
> It works similarity to the shared/exclusive lock described earlier, but rather than belonging to a particular object  (.e.g one row in), it belongs to all objects that match some search condition, such as:
>
> SELECT * FROM bookings WHERE room_id = 123 AND end_time > '2018-01-01 12:00' AND start_time < '2018-01-01 13:00'
>
> ..
>
> If transaction A wants to insert, update,or delete any object, it must check whether either the old or the new value matches any existing predicate lock.
>

事务 A 想插入、更新、删除对象时，必须遍历所有活跃 predicate lock 检查是不是匹配这些 predicate，如果存在，则等待对应的锁。


遍历所有活跃 predicate lock 并做检查的开销巨大。应该没人真的用它。

> Index-range locks
>
> Unfortunately, predicate locks do not perform well: if there are many locks by active transaction, checking for matching locks becoms time-consuming. For that reason, most databases with 2PL actually implement index-range locking (also known as next-key locking), which is a simplified approximation of predicate locking.
>
> an approxinmation of the search condition is attached to one of the indexes. Now, if anohter transaction wants to insert, update, or delete a booking for the same room and/or an overlapping time period, it will have to update the same part of the index. In the process of doing so, it will encounter the shared lock, and it will be forced to wait until the lock is released.
>

之前一直没弄明白 next-key lock 为什么上在索引上，如果换一个索引访问数据不就没有保护到了吗？这里的 point 是，在更新、插入、删除复合查询条件的数据时，必须要更新这条数据 *全部* 的索引，要更新索引就得拿到索引上的 next-key lock。next-key lock 保护一个查询条件在事务中不发生 phantom。


next key lock 上锁到索引上的坑是：SELECT .. FOR UPDATE / UPDATE / DELETE 要是写出来一个全表扫描的查询条件，会怎么办？





## Chapter 7: Transactions

> Lost update: Two clients concurrently perform a read-modify-write cycle. One overwrites the other's write without incorporating its changes, so data is lost. Some implementations of snapshot isolation prevent this anomaly automatically, while others require a manual lock(SELECT FOR UPDATE).
>
> Write skew: A transaction reads something, makes a decision based on the value it saw, and writes the decision to the database. However, by the time the write is made, the premise of the decision is no longer true. Only Serializable isolation prevents this anomaly.
>




## Chapter 5: Replication

> The major difference between a thing that might go wrong and a thing that cannot possiblly go wrong is that when a thing that cannot possibly go wrong goes wrong it ususally turns out to be impossible to get at or repair. - Douglas Adams, Mostly Harmless(1992)
>
