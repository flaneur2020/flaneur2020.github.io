>  The main issue when computing grouping results is that the groups can occur in the input table in any order.

在计算 GROUP BY 期间，原始数据的顺序可能是任意的。

如果原始数据是有序的，那么计算 aggregation 会很轻松。

出于这点，直白的计算 grouped aggregation 的方法就是将原始表按 grouping column 先排序一下。

但是排序不管怎样都是一个昂贵的操作。

## Hash Tables for Aggregation

更好的办法是使用 HashTable。

Hash Table 的条目数量按说和 Group 目标数量一致，远小于排序所需要的开销。

## Collision Handling

两种方式：1. Chaining；2. Linear Probing；

两种方式都会降低 HashTable 理论上的性能上限，所以通行的做法是，在到达一个 fill ratio （比如 75%）之后，resize 一把。

这里的一个 challenge 是 resize 时，需要移动一整个 HashTable 的所有内容，是一个比较昂贵的操作。

![[Pasted image 20240430162530.png]]

所以为了更高效的 resize，支持了 two-part 的 aggregate hash table。

它额外带了一个指针，来只想特定的 grouping values 和 aggregate states。

在需要 resize 时，我们扔掉旧的 pointer array，并申请一个更大的。

读一遍所有的 payload block，计算哈希值，重新插到 pointer array 里。

![[Pasted image 20240430163210.png]]

grouped data 本身不动，所以这一来 resize 的开销就大大降低了。

naive 的 two-part hash table 会对所有的 group value 进行 rehash 计算。

赶上对 string 类型的话，开销仍然不低。

因此可以将原始的 hash 值，也保存在 payload block 中，这样就不用重新计算了：

![[Pasted image 20240430163424.png]]

不过 two-part hash table 在查找时有一个不足：pointer array 和 payload blocks 的 group entries 中都没有顺序。

有内存随机访问。

为了缓解这个情况，我们将哈希值的前一两个 bytes 加入到 pointer array 中。这样跑 linear probe 时，可以只看 pointer array，就能剪掉不必要的对 payload block
的扫描。

这个优化能够显著减少对 payload block 的随机访问。

![[Pasted image 20240430171142.png]]

另外一个相对小的优化点在于 pointer array 中 pointer 的大小。duckdb 支持 4 bytes 和 8 bytes 两种 pointer array entries。

## Parallel Aggregation

哈希表的并发读写性能不大行。

比如，如果一个线程想 resize 这个哈希表，而另一个线程想增加新的 group data 到里面。

有一个可行的办法是，每个线程构造自己的局部哈希表，然后在一个线程中进一步合并成一个大的。

在 group 数量不多时，这个方法是工作良好的。

但是也有一种可能是 group 的数量和原始输入一样多。

这时就需要一个并行的 merge 操作，也就是 parallel hash tables。

我们使用了一个 Leis et al. 的办法：<mark>每个线程并非构造一个哈希表，而是构造多个 partitioned hash table</mark>。对 group hash 按 radix partitioning 进行分区。

![[Pasted image 20240430171853.png]]

（相当于按 thread id 做了一把 shuffle？）

随后，每个线程能够合并自己名下的 partitions。

这里有一个有意思的性质是，我们最终不需要再合并成一个大的完整哈希表。

在 parallel hash table 的基础上，还有两个额外的优化：

1. 只有在超过特定阈值（目前设定是 10000）后，再考虑做 partition 这回事，因为做 partition 也有它的开销存在；
2. 在 pointer table 的数量超过阈值后，就不再往里面插入数据，随后每个线程会构造一组新的 hash table，这一来同一个 group value 可能会重复在多个 hash table 中存在，但是这没关系，最后还是能够 merge 起来；这种优化，对于 group by date 这种 group 值的种类较多，但是分布有规律的数据上很实用；

也有一些聚合操作是不能使用 parallel partitioned hash table 的方法的，比如 median，这种就不像 sum 那样好聚合。不过 duckdb 也支持易于聚合的 `approx_quantile` 作为替代。