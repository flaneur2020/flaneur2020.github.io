tldr



----

1. 首先，定义一个 reference model 按预期的语义进行实现；它的代码量是真正实现的 1%，比如，reference model 中，LSM 的真正实现是一个 hashmap；在工程师开发新功能时，也会同步更新 reference model；
2. 检查 ShardStore 的实现，确认是否满足 reference model；对于功能正确性，会使用 property based testing；对于崩溃一致性，也会参考 reference model，来定义最近的变更中，哪些修改允许在 crash 中丢失，然后再 apply property based testing；对于 concurrency，使用 stateless model checking 来确认实现是 linearizable 的；

> These checks have prevented 16 issues from reaching production, including subtle crash-consistency and concurrency issues that evaded traditional testing methods, and anecdotally have prevented more issues from even reaching code review

这些 checks 也是 pay-as-you-go 的，可以跑更久来发现更多 issue。

> Formal methods experts wrote the initial validation infrastructure for ShardStore, but today 18% of the total reference model and test harness code has been written by the engineering team to check new features and properties, and we expect this percentage to increase as we continue to adopt formal methods across S3.

## 2 ShardStore

### 2.1 Design Overview

> ShardStore’s key-value store comprises a log-structured merge tree (LSM tree) [41] but with shard data stored outside the tree to reduce write amplification, similar to WiscKey

LSM Tree 会将每个 shard identifier 指向一组 chunks，每个 chunk 会维护在一个 extent 中。

extent 是一个磁盘上连续的 region。一块盘会有几万个 extent。

ShardStore 会要求，每个 extent 中顺序地写入，根据一个 write pointer 来跟踪下一个写入的位置。每个 extent 不能立即被覆写。每个 extent 可以有一个 reset 操作，允许重置 writer pointer 到最开始。

回收一个 extent 似乎需要将之前 extent 里的 chunk 移动到另一个 extent 里。

![[Screenshot 2025-07-12 at 21.28.34.png]]

LSM Tree 本身，也存储于 extent 中的 chunks。

ShardStore 并未将所有的 shard data 往一个统一的 shared log 中存储，这给了 shard store 很大的灵活性，也使得 crash recovery 的一致性变得更复杂。

RPC Interface：每个磁盘有一个单独的 failure domain，并跑一个独立的 kv store。client 可以根据一个 rpc 接口，根据 shard id 路由到目标的盘。

## 2.2 Crash Consistency

shard store 的 crash consistency 受 soft update 的启发。它能够排列写入的顺序，保证任何 crash 时，disk 都是一致的。soft update 不需要 wal 这种重定向写入的开销。

> Soft update 确保：被依赖的结构先写入磁盘;依赖的结构后写入磁盘
>
> 比如 linux 上创建文件的操作顺序：1. 分配 inode；2. 在目录中添加目录项；3. 更新目录的 inode

（大约 soft update 听起来像是没有 WAL 之前的，fs 中 crash recovery 要精细保证几个东西的写入顺序的情况，崩溃了之后 fsck 能够做些恢复。现在的 soft update 会显式跟踪一下 dependency。）

> Correctly implementing soft updates requires global reasoning about all possible orderings of writebacks to disk.

> To reduce this complexity, ShardStore’s implementation specifies crash-consistent orderings declaratively, using a Dependency type to construct dependency graphs at run time that dictate valid write orderings.

shardstore 中，它的 append 操作是唯一往磁盘写入的方法，类型为：

```rust
fn append(&self, ..., dep: Dependency) -> Dependency
```

> every time ShardStore appends to an extent it also updates the corresponding soft write pointer in the superblock (extent 0).


![[Screenshot 2025-07-13 at 09.49.24.png]]
### Why be crash consistent?

作者将 crash consistent 看作一项成本优化，s3 已经有了很强的 replica 恢复能力。

> We instead see crash consistency as reducing the cost and operational impact of storage node failures. Recovering from a crash that loses an entire storage node’s data creates large amounts of repair network traffic and IO load across the storage node fleet

## 3 Validating a Storage System

> intricate on-disk data structures, concurrent accesses and mutations to them, and the need to maintain consistency across crashes

ShardStore 已经 4w+ 行代码，并且经常被修改。

> We chose formal methods because they would allow us to validate deep properties of ShardStore’s implementation that are difficult to test with off-the-shelf tools at S3’s scale and complexityÐfunctional correctness of API-level calls, crash consistency of on-disk data structures, and correctness of concurrent executions including API calls and maintenance tasks like garbage collection.

### 3.1 Correctness Properties

在形式化验证部分，主要考量 consistency 和 durability 的属性，这两项在传统的测试方法中比较难覆盖，而且更难恢复。

> However, this property is too strong in the face of two types of non-determinism: crashes may cause data loss that the reference model does not allow, and concurrency allows operations to overlap, meaning concurrent operations may be in flight when we check equivalence after each API call.

reference model 的问题是有时候语义太强了，在崩溃时，有时是允许丢失数据的，但是 reference model 不会。

作者发现可以将 durability 进行一下拆解：

1. 对于顺序的 crash-free 执行，可以直接检查相等性；
2. 对于顺序的会 crash 的执行，作者扩展了 reference model，告诉它哪些数据在崩溃后允许丢失；
3. 对于并发的 crash-free 执行，作者写了一个单独的 reference model，来检查 linearizability；

## 3.2 Reference Models

> Since the reference models provide the same interface as the implementation, ideally they should give identical results, so that equivalence would be a simple equality check. This is true on the happy path, but we found it very difficult to enforce strict equality for failures.

> The reference models can fail in limited ways (e.g., reads of keys that were never written should fail), but we choose to omit other implementation failures (IO errors, resource exhaustion, etc.) from the models. This choice simplifies the models at the expense of making checking slightly more complex and precluding us from reasoning about most availability or performance properties. ğ4.4 discusses failure testing in more detail

reference model 也可以使用专门的 model 语言，不过作者发现使用同一语言编写 reference model 有好处是，门槛低一些。再就是可以用在 unit test 的 mocking 中。

比如证明目标：LSM-tree 的 reference model 当且仅当收到某个键的删除操作时，才会移除该键值映射。作者在这里有使用 Prusti 验证器。

这里使用 Prusti 验证的是 reference model，通过 Prusti 保障 reference model 满足核心的性质。然后再用 reference model 去验证主实现是否和 reference model 一致。
### 4 Conformance Checking

#### 4.1 Property-Based Testing

Property-based Testing 方法，用来验证实际实现是否正确地**精化**（refines）了 reference model。

#### 4.3 Minimization

作者的 pbt 可以生成随机的操作序列作为输入，当发现 failure 时，可以将这组 sequence 用于重放。

很多 pbt 工具都可以最小化 failing input 来提升 debug 体验。

每当遇到失败的 sequence 时，testing tool 能够自动地应用一些 heuristic 来化简这组 sequence。这个过程可以理解为每次减少 sequence 中的一步，直到没有错误为止。

> They usually do not make any guarantees about finding a minimum failing input, but are effective in practice.

> Although the minimization process is automated, we apply two design techniques to improve its effectiveness. First, we design ShardStore components to be as deterministic as possible, and where non-determinism is required we ensure it can be controlled during property-based testing

完全 determinism 有一定难点，比如 rust 中 HashMap 的默认 hash 算法是随机化的。

> Second, we design our alphabet of operations with minimization heuristics in mind. For example, the property-based testing tool we use [30] minimizes enumeration types by preferring earlier variants in the definition (i.e., it will prefer Get in the IndexOp enumeration in Fig. 3), and so we arrange operation alphabets in increasing order of complexity.

### 5 Checking Crash Consistency

> For ShardStore, reasoning about crash consistency was a primary motivation for introducing formal methods during development, and so it was a focus of our efforts

每个 ShardStore 的 mutating 操作都会返回一个 dependency 对象，可以 poll 它得知是否已经持久。

> 1. persistence: if a dependency says an operation has persisted before a crash, it should be readable after a crash (unless superseded by a later persisted operation) 
> 2. forward progress: after a non-crashing shutdown, every operation’s dependency should indicate it is persisten

在 property test 中增加 `RebootType`参数，控制崩溃时哪些内存数据被持久化到磁盘。

比如是否刷新LSM树的内存部分、是否刷新缓冲区缓存等。

重启和恢复后，test 会遍历每个 mutating operation 返回的 dependency，检查上面两个属性是否满足。

粗粒度方法的局限性是， `RebootType`参数对每个组件（如LSM树）做**单一选择**，要么整个组件状态刷新到磁盘，要么都不刷新。（一次只能测一个组件的持久性？）

潜在问题是可能遗漏某些bug，因为现实中的崩溃可能只影响部分数据块。

在操作中，会多生成一些 flush 交错在执行中，比如：

```
Put(0, 3) → IndexFlush → Put(1, 7) → DirtyReboot(None)
```

