---
layout: default
title: Understanding MySQL Internals
---

# 读书笔记: Understanding MySQL Internals


## Mutually Exclusive Locks (Mutexes)

<原文开始>The frequent calls to pthread_mutex_lock( ) and pthread_mutex_unlock() would cause a performance degradation, and the program- mer would be very likely to make a mistake in the order of calls and cause a deadlock.</原文结束>

<原文开始>The solution is in some balanced grouping of the global variables and in having a mutex for each group.</原文结束>


## innodb_lock_wait_timeout

<原文开始>While it is possible to use an algorithm that avoids potential deadlocks, such a strategy can very easily cause severe performance degradation. InnoDB approaches the problem from a different angle. Normally deadlocks are very rare, especially if the application was written with some awareness of the problem. Therefore, instead of preventing them, InnoDB just lets them happen, but it periodically runs a lock detection monitor that frees the deadlock “prisoner” threads and allows them to return and report to the client that they have been aborted because they’ve been waiting for their lock longer than the value of this option.

Note that the deadlock monitoring thread does not actually examine the sequence of the locks each thread is holding to figure out the existence of a logical database dead- lock. Rather, it assumes that if a certain thread has exceeded the time limit in wait- ing for a lock that it requested, it is probably logically deadlocked. Even if it is not, it doesn’t matter—the user would appreciate getting an error message rather than wait- ing indefinitely while nothing productive is being accomplished.
</原文结束>

innodb 并没有上严格的死锁避免策略，而是探测长时间等待在锁上的线程，使它的事务 abort。