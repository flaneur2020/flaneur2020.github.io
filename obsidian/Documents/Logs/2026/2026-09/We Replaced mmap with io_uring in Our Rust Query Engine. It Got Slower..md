
tldr

- 每次 page fault 不管 major 还是 minor，都会查找 vma，也就意味着有锁；
- page cache 是一个共享资源，在扩展进程之后，抢的不是仅仅 CPU，更是 page cache，相互锁冲突会比较严重；

---


### Our Workload

- 在 conviva 每天分析几 trillion 的 event 在 pinpoint 里面，来分析用户的行为；
- 核心是一个 datafusion、arrow、rust、rayon、tokio 上做的一个引擎；
- raw event 会被转换、编码成一个内部的 mostly numeric 的格式、保存在 cloud；
- 将它们拷贝到 local NVMe，读取大的（3～5GB）的 Arrow IPC 文件；
- 作者考虑 Arrow IPC 是因为它简单而且快速，它的内存和磁盘的 layout 相同，解码几乎零开销，mmap 能够做到 zero-copy 的读取，而且 arrow-rust 已经支持；
- 一个典型的 query 会读取 6 个 column，跨 8 个 batch files，每个 batch 大约 1.6GiB，扫描约 ～13GB 的日内数据；

### The Test Setup

- 192 核、750 GiB 的 RAM；
- 测试中有两种磁盘的配置：
	- 2xNVMe 开 LVM 条带（5.5GB/s）
	- 32xNVMe RAID-0（～21GB/s）

### The Production Symptom

- 在比较小的负载上，mmap 工作的很好；
- 问题发生在 concurrency 上面；
- 一些特定负载的 latency 增加是预期内的，比如更多的 query 竞争 CPU；
- 但是 P95、P99 spike 超过了 linear scaling 能预测的范畴，rows scanned 在有 concurrency 时显著下降：
	- OS page cache 下降 - 每个 pod 消耗了更多的内存用于自己的分配，而非共享的 cache；
	- 大量的 page faults；
	- 在 concurrent load 时 P95 spike 从 30s 到 150s；
	- 增加 pod 使事情变得更坏了而没有更好；
- 问题指向到了 memory pressure 时 page cache 的 thrashing 现象；

### Controlled Benchmark: 1 Pod vs. 4 Pods

- 用一个跨 14 天的 query（能够大于 page cache 的容量）
- 一个 pod 赢了 4 个 pod：41% at max、>20% at p95;
- page cache 在宿主机上共享，所以 4 个 pod 并没有 fight each other for CPU，而是 fight for Page Cache，perf record 显示 100% 的 lock contention 发生在 kernel level；
- 核心问题：mmap 的 page cache 是一个 implicit shared state；
	- Every process on the host shares one cache, one lock hierarchy, one eviction policy. No single pod controls the resource that matters most for read latency, and as concurrency rises, everyone’s slice of it shrinks.

### A Storm of Page Faults

- 在 mmap 时候，发生的事情：
	1. CPU 踩到一块 vma，这时还没有物理页，触发一个 page fault；
	2. kernel 处理 page fault，检查 vma、检查 ownership、获取一个锁；
	3. kernel 6.4+ 有一个 fast per-VMA 的锁，更早的内核会 fallback 到一个更慢的 mmap_lock 上；
	4. kernel 确认了 vma 是 file backed，触发一个 file fault —— 一个 minor fault，如果 bytes 已经在 warm 的 page cache 中，否则一个 major fault 来触发 physical IO；
- 在 heavy 的 page cache contention 时，read-ahead 会争起来，飙起来 major fault；
- 这时 RSS 涨到 98.91% 的 RAM，kernel 没有选择只能 evict page，然后 evicetd 的 page 又被访问到，又产生 majour fault；
- 同时，minor fault 大约保持在每秒几百万次；
- Each minor fault touches a cache line via atomics — at 2 million faults/sec, that’s enough to thrash L1/L2 entirely
- Faults can also trigger TLB shootdowns, and CPUs only hold a few thousand TLB entries
- 每个fault 都是一个 vma lookup，每个 lookup 又持有一个 mmap 的锁；
- 以至于每秒的 context switch：cs = 2,106,576/sec；

### Enter io_uring

- 作者的计划：使用 O_DIRECT 来 bypass 掉 page cache；通过 io_uring 来提交读请求，用 tokio 来协调，并解析 arrow；
- 作者使用了 compio 作为 io_uring 的 wrapper；
- 每个 arrow row 对应一个 future，所有的 40 列并发地提交；
- 一开始的 laptop testing 不是很理想（在 mac os 上，没有真的 io_uring 不过 compio 有个抽象也能跑），整体上执行时间更慢一些，不过 major faults 少了 70 倍；
- linux 上 Total query time **13.6s** 涨到了 **21.8s**；major faults 确实少了，但是 minor faults 多了 8 倍；

### The Batch Materialization Layer

- Arrow IPC 将数据组织为 batches；
- 在作者的 workload 中，每个文件包含一个大的 batch，每天数据大约 8 个文件；
- 扫描到一天数据的 query 就会踩到这 8 个文件，取 5～6 个列；
- 在使用 mmap 时，read 的次数不一定重要，对于 query engine 来讲，数据几乎在内存里；
- 在 io_uring 中，每个读操作都是显式提交的；
- 因此作者在 query engine 和 io_uring 中间做了一个 layer：Batch Materialization layer，有两个简单的 API：
	- prefetch(batch, columns) - 发起一组 io_uring 读请求，将 per-column 的 cache 放到一个 OnceCell 风格的 slot 里；
	- materialize(batch, column) - 返回 cache 的 bytes，或者等待 in-flight 的 read；
	- ![](https://www.conviva.ai/wp-content/uploads/2026/09/BMT_Architecture_Diagram.svg)
- 这个涉及在 whiteboard 上看起来还挺不错；
- prefetching 和 query engine 要 hide IO latency 的出发点很匹配：先发起读取、等待期间做点别的、在需要时继续捞 bytes；
- 这个 cache 将 query engine 和 io_uring 之间做了解耦；
- 作者没有提前预料的是，这个 layer 做了多么多的工作，这些都是在同一个线程里做的：
	- 接收 prefetch/materialize 调用；
	- 为每个 requested column 提交 compio future；
	- 等待 completions；
	- 解析 bytes 懂啊 Arrow buffers；
	- populate 这个 cache；
	- 处理 materialized column 回到 query engine；
- IO coordination、arrow decode、cache management、query facing API、以及 one async runtime；所有这些；
- The first cut fired all ~40 column reads at once — ~40 concurrent compio futures the moment a query wanted a file. That’s what “prefetching” meant to us then: fire everything, let async coordinate, wait for it all.

### The Meandering

- io_uring 的 papers 都强调了 O_DIRECT;作者并没有使用 O_DIRECT 而是还在使用 page cache；
- 因此，作者所做的第一件事就是打开 O_DIRECT：直接 DMA 到作者的 buffer 里、不过 kernel caching，不引入 page-cache；
- 执行时间从 21s 降到了 19s；有效，但是不能达到一开始的预期；
- 第二件事情就是 arrow，Buffer::from_slice_ref 是 arrow-rs 默认的构建一个 buffer 的方法， 它会分配内存，然后 memcpy 给它；每个 4kib 的目标页都需要 kernel 给置零，这就是 minor page fault，对 13GB 的读几乎需要 800w minor faults；
- 作者的 arrow layer 实际上破事 kernel 来做内存管理工作，而这部分工作我们本以为移动到 io-uring 之后没有了；
- 作者将 Buffer 改成了直接从 io_uring owned bytes 来构建来绕过 copy；执行时间从 19s 降到了 16s；但是这个代码很丑陋，任何 reviewer 都接受不了；
	- manual Buffer construction sidesteps arrow-rs invariants in ways that are hard to follow six months later
- 作者将这个看作是一个未来的改进项，弄一个 reusable 的 buffer pool，让 io_uring 写到预分配的内存里，让 Arrow 拷贝它，仍然做 memcpy，但是作用在 pre-faulted 的内存页面上；
- 不过到这里 16s，还是比 mmap 版本慢，作者做了 O_DIRECT，也做了 arrow 的 default copy 的 work around，仍然也就这样；
