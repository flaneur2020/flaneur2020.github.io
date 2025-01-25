## _Writing Takes Commitment_

> The really interesting thing start with the `COMMIT`. We need to do three things to commit this transaction:
>
> 1. Check whether our isolation rules will allow it to be committed (or, alternatively, it attempts to conflict with other concurrent transactions and must be aborted).
> 2. Make the results of the transaction durable (the _D_ in ACID) atomically (the _A_ in ACID).
> 3. Replicate the transaction to all AZs and regions where it needs to be visible to future transactions.

在 DSQL 中，冲突检查也是 disaggregated 的设计。

实现在一个叫做 _adjudicator_ 的服务中。它可以横向扩展。

如果一个事务写入的 row 跨越多个 _adjudicator_，则需要跑一个 _cross-adjudicator_ 的 coordination protocol。

一旦 adjudicator 同意可以 commit 事务了，则先将它写入 journal 用于 replication。

journal 也是一个 AWS 内部的服务，为跨 host、AZ 乃至 region 的 ordered data replication 而优化。

到这里，事务就已经 durable，而且原子地提交了。然后就可以告知客户端已经提交。

在向客户端返回成功的同时，可以 apply 这个事务的操作到左右相关的 replica 上。

![[Pasted image 20250122232220.png]]

## _Optimism_

DSQL 采用 OCC 的策略，这是个 big win，因为全程只需要在 COMMIT 中进行一次 coordination。

> In fact, I believe that Aurora DSQL does the provably minimal amount of coordination needed to achieve strong consistency and snapshot isolation, which reduces latency.

> That’s not the only reason we chose OCC. We’ve learned from building and operating large-scale systems for nearly two decades that coordination and locking get in the way of scalability, latency, and reliability for systems of all sizes. In fact, avoiding unnecessary coordination is the [fundamental enabler for scaling in distributed systems](https://brooker.co.za/blog/2021/01/22/cloud-scale.html). No locks also means that clients can’t hold locks when they shouldn’t (e.g. take a lock then do a GC, or take a lock then go out to lunch).

作者在构造大规模系统的经验告诉他，减少 Coordination 是分布式系统的可扩展性的根本。

没有锁，也意味着客户端不会在不该有锁的时候意外地持有锁。（比如持有着锁赶上一个 GC ）

> OCC does have some side-effects for the developer, mostly because you’ll need to retry `COMMIT`s that abort due to detected conflicts. In a well-designed schema without hot write keys or ranges, these conflicts should be rare. Having many of them is a good sign that your application’s schema or access patterns are getting in the way of it’s own scalability.

冲突的问题如果多，是你程序写的不对...

## _What Goes on the Journal_

在事务写入 journal 成功持久的时候，就认为已经提交了。

## _Consistency_

adjudicators 好像还需要做好授时工作。

好像类似原子钟啥的。