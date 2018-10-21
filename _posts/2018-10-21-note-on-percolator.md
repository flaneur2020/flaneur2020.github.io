---
layout: post
title: "Note on Percolator"
---

percolator 原本是 google 用于增量建索引，以库的形式实现跨行的事务引擎，而有事务引擎在，建索引并使索引一致便简单直接了。按论文里的描述，percolator 在这类半离线的场景下对时延的要求不高，赶上故障恢复的 case 时能接受秒级的延时，不过类似在支持单行事务的 k/v 存储上做跨行事务支持现在貌似蛮有用的，k/v 层面有 ACID 的事务引擎，NewSQL 系数据库的 SQL 层也好做 \[1]，先学习一下思路。

之前了解到 hbase 系同类的事务引擎有 omid 和 Tephra，不过它们都需要有状态的 Transaction Manager 来授时、解冲突、记 REDO Log 做故障恢复。percolator 使用两阶段提交的好处是对外的依赖较少，只依赖中心的授时器，没有单点 Coordinator 的角色，交由所有客户端来协调上锁协议，但是赶上崩溃锁会泄露，那么上锁期间故障恢复就是它的重点。

使用方面论文里贴了这段建索引的例子，允许除了通过 URL 查询 doc，也能够通过 doc 的哈希来查询，做到两个查询维度：

```C++
bool UpdateDocument(Document doc) {
	Transaction t(&cluster);
	t.Set(doc.url(), "contents", "document", doc.contents()); 
	int hash = Hash(doc.contents());
	// dups table maps hash → canonical URL
	string canonical;
	if (!t.Get(hash, "canonical-url", "dups", &canonical)) {
		// No canonical yet; write myself in
		t.Set(hash, "canonical-url", "dups", doc.url());
	}
	// else this document already exists, ignore new copy
	return t.Commit();
}
```

percolator 事务能做到 Snapshot Isolation，需要存储数据的多版本和提交时间，暂不考虑上锁的流程大致是：

- 事务开始时，向授时器请求获取 start_ts；
- 事务期间，Get() 得到 commit_ts 小于当前事务 start_ts 的快照数据，Set() 并不真正发送请求，而是 buffer 在本地 WriteSet；
- 提交事务时，WriteSet 内相关所有数据的最新版本号如果大于当前事务的 start_ts，说明事务开始后有其他人对数据做了有冲突的修改，事务应回滚；

percolator 利用 BigTable Cell 中的时间戳来存储多版本数据：

- write 列：`<commit_ts -> <start_ts>` 保存 commit 时间戳
- data 列： `<star_ts> -> data>`

分布式 2PC 并没有单机环境中真实的锁，而依赖于手工在列中的的标记：

- lock 列： `<start_ts> -> <'I am primary' or 'ID of primary row'>`

lock 列有两种标记值，一种是 `I'm Primary ` 标记，一种是 `'Primary 行的 ID'`。Primary Lock 是 percolator 在 2PC 容错中的重要设计，具体用途先在这里略过。

lock 列与 write 列、data 列均位于一行，后面会利用到行级别事务实现原子提交。

## 2PC

`Commit()`时，会先执行 PreWrite 阶段：

```
1.1 尝试上锁 WriteSet 中的首行数据
  1.1.1 检查该行数据是否存在冲突 (是否存在 commit_ts 大于 start_ts 的数据)
  1.1.2 检查该行数据是否被上锁 (行中是否存在任何 lock 标记)
  1.1.3 写入该行，使 lock 列 start_ts 版本的 Cell 标记为 I'm primary
1.2 遍历 WriteSet 中其他行数据尝试上锁
  1.2.1 检查该行数据是否存在冲突 (是否存在 commit_ts 大于 start_ts 的数据)
  1.2.2 检查该行数据是否被上锁 (行中是否存在任何 lock 标记)
  1.2.3 写入该行，使 lock 列 start_ts 版本的 Cell 标记为 Primary 行的 ID
```

PreWrite 阶段完成后，执行二阶段 Commit：

```
2.1 向授时器请求获取 commit_ts
2.2 对 Primary 行开启行事务，commit 写入并解除锁定
  2.2.1 确认锁是否仍存在
  2.2.2 设置 Primary 行的 write 列为 commit_ts (commit 写入)
  2.2.3 移除 Primary 行的锁标记 (解除锁定)
  2.2.4 提交行事务 (commit point)
2.3 遍历 WriteSet 中其他行数据，commit 写入并解除锁定
```

注意操作 2.2.4，是 2PC 过程里最关键的原子提交。**凡是原子提交之前的任何操作发生意外，皆应回滚事务（Roll Back）。凡是原子提交成功之后发生任何意外，则开弓没有回头箭，皆应继续补完事务（Roll Forward）**。

原子提交前遇到任何意外应回滚整个事务，这点不难理解。但是客户端若在 PreWrite 阶段内崩溃，会导致锁的泄露。谁来回收泄露的锁呢？

Percolator 选择了惰性地回收泄露的锁：其他客户端在 Get() 到这行数据时，如果遇到锁，则选择等待退避重试，或者清理锁。

这里便是 Primary Lock 机制巧妙的地方：他人对锁的清理需要是一个原子操作，与二阶段事务的原子提交操作，两者对齐在 Primary Row 的原子操作上面，通过行事务的原子性使两个操作只能成功其一。

## 崩溃修复

假如 1.2 循环操作中成功锁定并写入了一行数据，而请求者崩溃，其他人 `Get()` 到这行数据时：

```
3.1 发现有锁，但是客户端不能判断是锁泄露，还是正常的锁争用，为此等待退避重试
3.2 等待超时，客户端认为属于锁泄露，查询 lock 列，取到 Primary Row 的 ID
3.3 查询 Primary Row 的锁状态，判断是否已提交
  3.3.1 如果 Primary Row 有锁、未提交、时间戳一致
    3.3.1.1 判断应当 Rollback，清理 Primary Row 的遗留锁
    3.3.1.2 清理该行的锁与遗留数据，返回正确的版本数据
```

假如 2.3 循环操作中未成功 commit 写入并解除锁定，而请求者崩溃，其他人 Get() 到这行数据时：

```
  3.3.2 如果 Primary Row 未上锁、已提交、时间戳一致
    3.3.2.1 判断认为应当 Roll Forward，对该行 commit 写入解除锁定
    3.3.2.2 返回正确的版本数据
```

## References

- Large-scale Incremental Processing Using Distributed Transactions and Notifications
- [Google Percolator 的事务模型](http://andremouche.github.io/transaction/percolator.html)
- 类似 percolator 的 hbase 实现：<https://github.com/VCNC/haeinsa>
- <https://www.slideshare.net/GaoYunzhong/study-notes-google-percolator>
- <https://blog.octo.com/en/my-reading-of-percolator-architecture-a-google-search-engine-component/>

## Footnotes

- \[1]: 听说 TiKV 的事务模型便是与 percolator 相似的。