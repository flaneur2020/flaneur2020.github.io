### TLDR

- 在一个 warm 的 ring 上，chunk size 影响不大；
- 应当 keep IO thread long lived；
- core breakthrough：针对每个 column 发送一个大的 SQE 是不行的（40 个并发，每个 650MiB）；很多小的 SQE 更适合 RAID 的 stripe；
- 在作者的 workload 下，一个专门的 IO thread（没有 async、future，只做 SQE submit 和 poll）会比 compio 的 async 抽象更好；
- 最大的 single win 不在 io_uring，而是 `MADV_POPULATE_WRITE` 和 HUGETLB，直接减少了 memcpy 的 wall-clock 三四倍；
- baremtal 相比 k8s pod 中快 20～30%；

### Going back to basics: adding real logs

- 卡住在什么地方时，你需要的是更多的数据而不是更多的理论；
- 因此作者在 batch materialization layer 的每个地方都增加了日志：每次 prefetch 的提交、每次 materialize call、每次 io_uring 的完成，也对 column cache 增加了 metrics：hits、misses、in-flights、L1/L2 hits、ring depth、completions；
- 指标显示这个 layer 工作是正常的，但是 trace log 讲了不同的故事；

### What the logs showed: the 2.5-second stalls

- 每 10 个 submissions，会有一个 2.5s 左右的 gap，然后突现一波 completions；
- 阻塞在哪里？两个 prefetch 之间的 submission for columns 都是同样的已经打开的文件，两个连续的 submission 之间不该有需要等待的东西；

### Argument with Claude: the O_DIRECT theory

- claude 的说法是，gap 来自于 ext4 的 O_DIRECT 的 open path，会在并发之下serialize 一把，类似在 inode 的 locking 之类。
- 不过事情不大对劲，openat 是一个 metadata 操作，file 打开是一个 us 级别的操作，即使是很重的 O_DIRECT 并发之下；一个 2.5s 的 file open 不是一个缓慢的 syscall，是一个卡住的；
- 作者做了另一个实验，将 fd.open() 从 event submission loop 移动到了 async worker thread 中；这时 open 降到了 5ms 的水平（虽然 query time 没有变化）；
- 可见 delay 并不是来自 kernel，openat 其实很快就完成了，但是问题是 submission loop 一致在等待，导致所有的 downstream 操作都 stall 在这里了；（协程卡调度了。。。）
- Worth being fair here: this wasn’t Claude being useless. The theory was internally consistent and cited real kernel behavior — if I hadn’t had the intuition that file opens should be fast, I could easily have accepted it. This became the first data point in a rule I kept applying: **if a theory contradicts what you know about the basic cost of an operation, don’t trust the theory. Test it.**

### The stall was fixed. The query wasn’t.

- prefetch 的 stall 在上面的 fix 之后得到了解决，40 个 task 的 spawn 时间降低到了 0.343ms，所有的 40 个 file open 在 5ms 内完成；
- 但是 40 个 column 的 read 操作仍然花了 7.3s；每个 do_column_read 都显示 7.3s elapsed；这些任务都是同样的时间显示，并发的；7.3s 大约是 NVMe 的带宽限制；
- 40 columns across 8 batches, ~13.1 GB total, read in 7.3 seconds = 1.79 GB/s，这个性能比 NVMe 盘的配置低很多；
- 为什么 40 个读在同一个时刻一起完成？
	- 表面上看似乎因为 40 个列一起提交的，流水线要等 40 个全部完成才开始处理；
	- 深层原因是巨型的 SQE，内核会在 block layer 内不拆成很多子请求，但是 io_uring 仍把它看作一个操作，RAID 驱动无法处理多个 SQE 那样高效地流水线化 “单个 SQE 内不的子请求”
- 为什么 1.79 GB/s 远低于 fio 的 20GB/s？
	- fio 每个单次 io 是 4MiB，而本应用是 300～600MB
	- raid 看到的是 fio 是 128 个独立的操作，很容易 32 个盘条带化；本应用是 40 个巨型操作，无法跨 SQE 条带化；
- 因此修复方向可以是把大读拆成小 SQE；去掉 batch barrier 改成列级流水线、提高队列深度（QD）；

### Chunk size doesn’t matter (mostly)

- 不过作者在重写之前，先用 fio 模拟了一下各种 chunk 的大小，发现 chunk size 不是关键变量；
- 关键的区别在于，本应用的并行度来自多个文件、多个列，而 RAID 需要看到跨流的独立请求才能条带化；
	- 单文件顺序读写：chunk size 无所谓
	- 40 个文件并发读、RAID 条带化：SQE 粒度很关键；
		- RAID 要跑满 32 块盘，必须满足一个条件：在任意时刻，有足够多的"独立请求"同时排队，让调度器能把它们分发到 32 块盘上。
- 因此可以理解为，拆小的 SQE 不是关键，而是要让 RAID 看到更多的独立操作；
- 因此作者不再纠结 chunk size，决定把经历放在架构改造上；

### Chunking alone didn’t help

- 作者将 giant 的 column 拆成了很多小的，仍然在 compio 的 async framework 之上，同时也做了很多其他的修改：将每个 chunked read 拆到单独的一个 async 的 compio 的 task 上，为 in-flight 的 task 加了一个 semaphore 来控制 in-flight 数量（后来又拿掉了因为似乎没啥必要）、简化 main event loop（单纯从代码结构上的优化）、增加了 debugging info；
- 并没有很多收益，新的设计在 concurrency 下反而也更差了；
- 必须接收的一个事实是，io_uring 一般来讲是不错的，很多数据库都移动到了上面，但是从 mmap （操作系统管理的）移动到 io_uring 和 direct IO 上，意味着应用程序要管理一切；不只是读取，还有 multithread scheduling、queuing、concurrency，并且要和自己管理的 cache 一起工作；这是一个很大的架构改造，不能指望一次就成功；

### Re-architecture: dedicated I/O thread

- 原因基本确认是因为这一层做了太多事情：coordinating async futures, managing io_uring SQEs, feeding decoded bytes into Arrow, running the cache, coordinating with the query engine.
- 作者的 workload 是一个 query pipeline 协调一个专门的 io 线程，而不是很多独立的不同的异步请求；
- compio 在设计上，是免洗那个很多独立的请求的 case，和作者的这个 workload 不大匹配；
- 新设计的 core idea：一个专门的 thread 用于处理 io，借鉴 fio，fio 里面就是用了一个 spin loop，没有 future、没有 waker、没有 executor，它就直接提交 SQE、poll CQE：
	- 没有异步的原语 - 只使用 compio 的底层的 SQE 提交、polling 接口；
	- 将读取操作拆分为 chunks，按顺序地读各个 region；
	- 使用一个 round robin 的 mmap-backed 的 buffer，在读取时没有内存分配；
	- ![](https://www.conviva.ai/wp-content/uploads/2026/09/Re-Architecture_dedicated-io-thread.svg)
	- （arena ring buffer 好像相当于将这些 buffer 组织成 arena，在请求完成之后全部释放）
- 作者在开发之前，先按做了一个和 fio 相似的 demo，验证它能够跑到 fio 的速率；
- 随后在 32 个 NVMe 的 RAID-0 上，跑出来了 20GB/s 的性能；
- 得到的经验：
	- start simple：不要引进 async，先做一个小的 utility 和 io layer，来模仿 fio；
	- 拆分架构到两层：seperation of concern 让每一层都更容易测试；
	- 在实现真实的 pipeline 之前，写一个工具程序来独立地 benchmark 每一层；

### Parallel decoding on Tokio (Rayon nearly killed us)

- io 层重构之后，cpu 侧的解码工作；
- 作者尝试使用 rayon 做并行的 decode，然而跑出来了死锁，最后该用 tokio 解决；
- rayon 的 fork-join 模型不大适合这个场景，tokio 的 spawn blocking + semaphre 更合适；
- rayon 是一个 fork-join + work stealing 模型，在做嵌套 pool 时无法保证公平性，无法保证长任务先释放资源，短任务会堆在长任务后米娜；
- Tokio is just much better at any non trivial concurrency configuration, which ours was turning into, especially anything nested.

### First real end-to-end numbers

- 第一次超过了 mmap 基线：
	- 旧 mmap：小查询 4s、11 天查询 63s
	- 第一版 io_uring：6s、78s
	- 新 iothread + bmt + tokio 解码：3s、49s
	- 加上 Add HUGETLB + MADV_POPULATE_WRITE（先把页面填上不要惰性踩 page fault）：45s
	- 加上更大的缓存 - 37s

### The second investigation: memory strategy

- **perf 显示 memcpy 是瓶颈，但真正的瓶颈是 memcpy 触发的页错误**

### Argument with Claude, take three: dead-end memory hypotheses

> "**Every time we followed that pattern, it paid off. Every time we tried to reason about a layer while it was tangled up with three others, we got stuck.**"

- 每次测数据时都成功，每次理论架构该怎么设计时就被卡住

### The memory_test results

- Key takeaways：
	- "MADV_POPULATE_WRITE doesn't make memcpy cheaper in absolute terms — it **moves the fault work off the critical path**."
	- "HUGETLB adds another 30–40% on top. 2 MiB pages instead of 4 KiB means **1/512 the faults**, plus a much better TLB hit rate for large accesses."
- Final design：
	- the buffer arena backing the IOThread's ring uses **anonymous mmap with MAP_HUGETLB where available, falling back to 4K pages plus MADV_POPULATE_WRITE**. Prewarming happens on a **decoder-side task issued before the byte slice arrives**, so the memcpy always hits already-faulted pages. **Parallel memcpy via Rayon (row g) adds another 4–5× on top** when the decode workload is large enough to make the parallelism worthwhile."
- 结论：**This — not io_uring itself — turned out to be the single biggest source of end-to-end speedup once the I/O layer was working.**
	- 最大的加速不是来自 io_uring，而是来自内存的策略（HUGETLB + prewarming + parallel memcpy）
	- io_uring 只作用在 IO 层面，真正的加速来自内存

### Debugging with an LLM in the loop

- Three times in this project, Claude was confidently wrong at exactly the moment a plausible answer would have made me stop digging.
- Each time, the theory was **internally consistent**, cited real system behavior, and would have derailed the investigation if accepted.
- What saved us each time was a bit of **engineering intuition** about the baseline cost of an operation.
- When Claude's theory **contradicted a cost we knew**, that was the signal to test rather than accept.
- 作者的建议：
	- Before believing any LLM theory about a performance problem, **ask it to name a metric or log line that would distinguish the theory from the alternatives**, then go collect that data.
	- If it **can't produce something distinguishing**, the theory **hasn't earned belief yet**.
	- Watch for a **fluent explanation carrying an unverified claim** — **fluency isn't correlated with correctness**, and the **most confident-sounding explanations deserve the most skepticism**.