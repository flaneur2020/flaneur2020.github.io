> deleting tombstones based on repair execution, not a strict duration (known as `gc_grace_seconds`)

## Background: Tombstones in ScyllaDB and Other CQL-Compatible Databases

scylladb 和其他数据库都在使用 sstable，里面有 tombstone 来作为删除标记。

在 compaction 时，这些 tombstone 最后会被清理掉。

但是，如果删除的数据太多，tombstone 就会太多。

scylla 有一个策略是 `gc_grace_seconds` 默认是十天，过了十天就自动删除 tombstone。

按道理，十天内应该早就 compaction 或者 repair 掉了。

但是，如果一个节点如果宕机了十天，那么加回来之后，可能就会误删 tombstone，导致数据又回来了。

> 1. In a 3-node cluster, a deletion with consistency level `QUORUM` is performed.
> 2. One node was down but the other two were up, so the deletion succeeds and **tombstones are written on the two up nodes**.
> 3. Eventually the downed node rejoins the cluster. However, **if it rejoins beyond the Hinted Handoff window, it does not get the message that one of its records was marked for deletion.**
> 4. When `gc_grace_seconds` is exceeded, the two nodes with tombstones do GC, so tombstones and the covered data are gone.
> 5. The node that did not receive the tombstone still has the data that was supposed to be deleted.

大约说 quorum 中，三个节点都有一样的数据。

然后有一个节点宕机了。

另外两个节点成功的删除了数据，增加了 tombstone。

然后，过了 `gc_grace_seconds`，这两个删除数据的节点做了 GC（似乎就是 compaction？），tombstone 和对应的数据就都被删掉了。

那个宕机的节点加回来了，数据又回来了。

## Timeout-based Tombstone GC

上面这个有问题的流程，被称作是 "Timeout-based Tombstone GC".

## Repair-based Tombstone GC

idea 就是只有在 repair 执行时，才执行 tombstone 的删除操作。

确保所有的 replica 都有 tombstone，而不管 repair 是否在 `gc_grace_seconds` 时间范围内。

（听起来相当于一个 quorum = ALL 的策略，只有当所有 quorum 都在线的时候，才执行这个清理操作）