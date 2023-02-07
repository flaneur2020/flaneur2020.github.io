---
title: Spilled Hash Aggregation
layout: post
---

之前看 presto 的文章写 facebook 没有在生产开 Spilling 所以没大上心。因为在 fb 的集群体量下，内存基本可以理解为无限，相比于 Spill 得到的好处，不如计算节点还得多加块盘带来的维护成本高。

不过最近遇到一些 OOM 的 case，在没有 fb 那么大规模的常驻集群体量的话，Spill 仍是一项优选。尤其是 duckdb 这种资源受限的场景下，敢说 Never OOM 也是一大卖点，在资源受限的环境里如果捶打多了，搞不好过几年在 ETL 领域能有奇效。因此想了解下 Spill 是怎么工作的。

Spill 并非一个通用的 Buffer Manager 的通用能力，而是需要为不同的算子结合着算子的特性来单独设计，需要 Spill 的场景主要是 Hash Aggregation、Order By 和 Hash Join 这几个。这里先看一下 Hash Aggregation 是怎么做 Spill 的。

## Hash Aggregation

Hash Aggregation 就是跑 SELECT max(revenue) FROM daily_revenues GROUP BY year, city 的查询里，用 (year, city) 做哈希表的键弄一个哈希表，指向一个累加器。当 GROUP BY 的 Key 如果基数太高的话，这个哈希表就会占用太多内存而出现 OOM。

所以就得想办法将占用过多的内存释放出来。

最糙猛的释放内存的做法就是类似 Linux 系统的 Swap，当内存不够的时候，将内存 Swap 到磁盘。当内存够了之后，再 Swap 回来接着处理。

但是先不管性能问题，Swap 回来的时候内存就管够吗？

结合上算子跑聚合的性质这个就好解了。Datafusion 里面有一个讨论：

> **alamb:** I think the current state of the art / best practice for dealing with grouping when the hashtable doesn't fit in memory is to:
> 1. Sort the data by groups and write that data to a spill file (ideally sharing / reusing the code used for externalized sort)
> 2. Create a new hash table, spilling repeatedly if needed
> 3. When all the input has been processed, then the spill files are read back and merged (again sharing code for externalized sort) and a merge group by is performed.

大意是如果在跑 Aggregation 期间发现 Hash Table 用的内存太多了，就给它排个序落盘。

然后开一个新的 Hash Table 接着跑聚合，快跑满了的时候，同样给它排个序落盘。

最后把仍在内存里的哈希表也排序一把，把内存中的哈希表 + 磁盘上的多个有序文件，跑一次多路合并，流式地返回多路合并的结果就可以了。而多路归并需要的内存是非常非常低的。

这里路合并中不只是做 Merge Sort，也会继续做一次聚合，比如：

``` 
## spilled file 1:
(2020, beijing, $10)
(2020, new york, $20)

## in memory hash table
(2020, beijing, $1)
(2020, london, $23)
```

合并后：

``` 
(2020, beijing, $11)
(2020, new york, $20)
(2020, london, $23)
```

这里相比 Swap 的区别是，Spill 出去的内存就不再需要再 Load 回来了，而是通过排序，利用顺序性使聚合成为一个多路合并的流式操作，不再消耗多少内存了。

## Order By

理解了 Hash Aggregation 的 Spill 的做法，Order By 的 Spill 也就容易联想到了。排序期间如果发现内存快不够了，那么可以把手头的有序的部分内容先落盘，再开一个新的 Buffer 来排序。到最后做一次多路归并流式返回结果就可以了。

## References

- [https://docs.google.com/document/d/16rm5VR1nGkY6DedMCh1NUmThwf3RduAweaBH9b1h6AY/edit](https://docs.google.com/document/d/16rm5VR1nGkY6DedMCh1NUmThwf3RduAweaBH9b1h6AY/edit)
- [https://github.com/duckdb/duckdb/pull/4970](https://github.com/duckdb/duckdb/pull/4970)
- [https://github.com/apache/arrow-datafusion/issues/1570](https://github.com/apache/arrow-datafusion/issues/1570)
- [https://deepai.org/publication/sort-based-grouping-and-aggregation](https://deepai.org/publication/sort-based-grouping-and-aggregation)
