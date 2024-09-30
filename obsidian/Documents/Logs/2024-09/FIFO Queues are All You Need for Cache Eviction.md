FIFO 作为一个 cache 失效策略有很多吸引人的地方：简单、快速、可扩展、flash 友好；

对它的主要批评集中在 low hit ratio 方面。

作者在这个论文里评估了一把 S3 FIFO（Three Static Queues），发现 S3 FIFO 的性能还挺稳定，跑出来了最低的 miss ratio；相比于 LRU 跑出来了 6x 的吞吐。

作者发现：大多数对象的 workload 都 skewed 的很严重，只在一个 short window 中会被访问到；S3 FIFO 的关键是，一个很小的 FIFO queue 来防止大多数对象进入 main cache。

之前大家普遍的认知中，LRU 是更高效的。但是 LRU 有两个问题：

1. 每个对象有两个指针，有存储开销
2. 每个 cache hit 都需要移动被请求到的对象，需要经过锁

随着 CPU 的多核，cache 的 throughput 和 scability issue 变得更要紧了。

一些研究给出来的 solution 是，直接用个基于 FIFO 的算法。比如 MemC3、Tricache 用的 CLOCK、Segcache 用的 FIFO Merge。

作者的 insight 是，满足 Zipf 分布的请求，只被请求一次的概率大大高于再被请求一次。这导致有限的 cache 空间中净些 one hit wonders。

作者的解决思路是，用一个小的 FIFO Queue 来过滤掉 one hit wonders，保留 cache space 给更常被请求到的对象（这被称作 early eviction or quick demotion）。

从 small FIFO queue 中过期的对象，要么进入 main queue，要么进入 ghost queue，取决于是否又被访问到。

> Many previous works have explored similar ideas to quickly demote some objects [54, 79, 100], especially for scan and streaming workload patterns and in hierarchical caches. However, to the best of our knowledge, this is the first work demonstrating the importance of quick demotion for cache workloads even when there are no scan and streaming patterns. Moreover, this work designs the first FIFO-queue-only algorithm that is more efficient than state-of-the-art algorithms.

很多工作评估过 <mark>quick demote</mark> 在 scan 和 streaming workload 上的效益。不过，这个论文在作者看来是第一个评估对非 scan 场景下的优势。

small FIFO queue 是在内存中的，而从这里 envict 出来的对象多数不需要落到磁盘。

## 2.2 Prevalence of LRU-based cache

> Cache workloads exhibit **temporal locality: recently accessed data are more likely to be re-accessed**. Therefore, Least-Recently-Used (LRU) is more efficient than FIFO and is widely used in DRAM caches [24, 28, 102, 133]. Moreover, advanced eviction algorithms designed to improve efficiency are mostly built upon LRU. For example, ARC [100], SLRU [80], 2Q [79], EELRU [124], LIRS [77], TinyLFU [54], LeCaR [132], and CACHEUS [119] all use one or more LRU queues to order objects.

## 3 Motivation

FIFO 的主要不足是不好 retain 频繁被引用的对象。

最直接的解决办法就是把它再重新插回去。

FIFO-Reinsertion 算法会跟踪对象的访问，在 enviction 时将被访问到的对象再插入回去。

FIFO-Reinsertion 在 cache hit 时的代价较小。但是 reinsertion 的代价仍然不小，导致 FIFO-reinsertion 仍然落后于 SOTA 的 cache enviction 算法。

## 4 Design and implementation

定义：

- LRU queue: 在 cache hit 时将对象移动到头部；
- FIFO Queue：在 cache hit 时不移动对象到头部，但是在 enviction 时有可能 reinsert 回来；

### 4.1 S3-FIFO design

用了三种 queue：

1. small FIFO queue（S）：S 使用 10% 的 cache space；
2. main FIFO queue（M）：M 使用 90% 的 cache space；
3. ghost FIFO queue（G）：保存和 M 同样数量的 ghost entries，不保存真实数据；

S3 FIFO 每个对象平均使用两个 bit 来跟踪访问状态。它相当于一个最大是 3 的 counter；

每个对象在被访问时，如果没有在 G 中，则插入到 S 中；

若 S 满了，如果访问次数超过一，则移动到 M，否则移动到 G；同时清零对象的 access counter；

G 如果满了，则按 FIFO 顺序进行 enviction。

M 则会类似 FIFO-Reinsertion 的做法，参考 access bits 重新插入。在重新插入时，使 access bits 减一。

> The small FIFO queue S can quickly evict these one-hit wonders so they do not occupy the cache for a long time

Ghost queue 在实现中可以保存成哈希。

好像 Ghost Queue 主要起一个跟踪 access counter 的作用，在 Small FIFO queue 在 envict 对象时，如果这个对象在 Ghost FIFO Queue 中存在，则加入到 main queue 中。

## 5.4 Flash-friendliness

> most objects evicted from the S are not worthwhile to be kept in M, we can place S in DRAM and M on flash.