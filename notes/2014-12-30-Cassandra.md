---
layout: default
title: Cassandra
---

# 读书笔记: Cassandra


## Performance Tuning

<原文开始>However, it's recommaned that you store that datafiles and the commit logs on sparate hard disks for maximum performance.</原文结束>

commit log 最好写到单独的磁盘，ssd 该最好。

<原文开始>you might wang to update the concurrent_reads setting immediately before you start your server. That's because the concurrent_reads setting is optimal at two threads per processor core. By default, this setting is 8, assuming a fore-core box.</原文结束>

concurrent_reads 最好设置为 CPU 核数 * 2

<原文开始>the concurrent_writes setting behaves somewhat differently. This should match the number of clients that will write concurrently to the server. If cassandra is backing a web application server, you can tune this setting from its default 32 to match the number of threads the application server has available to connect to Cassandra.</原文结束>

concurrent_writes 最好设置为应用程序的连接数。

<原文开始>The keys_cached setting indicates the number of key locations—not key values—that will be saved in memory. This can be specified as a fractional value (a number between 0 and 1) or as an integer. If you use a fraction, you’re indicating a percentage of keys to cache, and an integer value indicates an absolute number of keys whose locations will be cached.

This setting will consume considerable memory, but can be a good trade-off if your locations are not hot already.
</原文结束>

相比 bitcask 把索引整个放在内存里，cassandra 把它视为一个缓存，可以设置缓存键的比率或者数量。

<原文开始>The purpose of disk_access_mode is to enable memory mapped files so that the oper- ating system can cache reads, thus reducing the load on Cassandra’s internal caches. This sounds great, but in practice, disk_access_mode is one of the less-useful settings, and at this point doesn’t work exactly as was originally envisioned. This may be im- proved in the future, but it is just as likely that the setting will be removed. Certainly feel free to play around with it, but you might not see much difference.
</原文结束>

使用 MMap 读取磁盘对性能提升意义不大。

不过 sqlite4 的 lsm 模块说 mmap() 相比 read() 要快的多，可能是因为 sqlite4 lsm 没有自己的缓存管理机制：

 LSM_CONFIG_MMAP

    If LSM is running on a system with a 64-bit address space, this option may be set to either 1 (true) or 0 (false). On a 32-bit platform, it is always set to 0.

    If it is set to true, the entire database file is memory mapped. Or, if it is false, data is accessed using ordinary OS file read and write primitives. Memory mapping the database file can significantly improve the performance of read operations, as database pages do not have to be copied from operating system buffers into user space buffers before they can be examined. 

<原文开始>The rows_cached setting specifies the number of rows that will be cached. ... You’ll want to use this setting carefully, however, as this can easily get out of hand. If your column family gets far more reads than writes, then setting this number very high will needlessly consume considerable server resources. If your column family has a lower ratio of reads to writes, but has rows with lots of data in them (hundreds of columns), then you’ll need to do some math before setting this number very high. And unless you have certain rows that get hit a lot and others that get hit very little, you’re not going to see much of a boost here.
</原文结束>

- rows_cached 缓存行数据本身
- 如果读远多于写，那么如果设置它过高，会消耗过多资源
- 如果读小于写，但行比较大，最好权衡一下
- 除非某些行的读写频率远大于其它行，那么很难得到明显的性能提升

<原文开始>Do not use the Serial GC with Cassandra.</原文结束>

<原文开始>However, do not simply set your JVM to use as much memory as you have available up to 4GB. There are many factors involved here, such as the amount of swap space and memory fragmentation. Simply increasing the size of the heap using -Xmx will not help if you don’t have any swap available.</原文结束>

不要使用物理内存的上限。


## the cassandra elevator pitch

<原文开始>on the contrary, Cassandra requires a shift in how you think about it. Instead of designing a pristine data model and then designing queries around the model as in RDBMS, you are free to think of your queries first, and then provide the data that answers them.</原文结束>

对于迭代快的初创项目不大适合用 Cassandra 的感觉， rdbms 建立一个 schema 可以灵活地更改 query，Cassandra 则需要改 Schema 以适应 query。（不过 rdbms 要性能高也是需要对 query 比较了解？）
## 列式数据库

<原文开始>"Sparse" means that for any given row you can have one or more columns, but each row doesn't need to have all the same columns as other rows like it ( as in a relational  model).</原文结束>