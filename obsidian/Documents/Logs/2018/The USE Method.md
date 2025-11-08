- 三种指标类型+分析复杂系统性能的一种策略，作者发现用它使用 5% 的力气即足够分析 80% 的系统原因；
Summary
- For every resource, check utilization, saturation, and errors.
	- 资源包括 CPU、磁盘、总线...
	- utilization：资源 busy 状态的时间的百分比，比如 "one disk is running at 90% utilization"；
	- saturation：资源超负载工作的程度，通常处于排队状态，比如 "the CPUs have an average run queue length of four"；
	- errors：错误事件的次数，比如 "this network interface has had fifty late collisions"；
Does Low Utilization Mean No Saturation?
- 突发性的高利用率可能导致 saturation 和性能问题，而在一个时间周期看来，它的百分比可能并不高；
- 比如，每分钟衡量的 CPU 利用率可能从未超过 80%，但实际上其中总有几秒钟处于 100%；
Resource List
- CPU：socket、core、hardware threads；
- Memory: Compacity;
- Network interface;
- Storage devices: I/O, compacity;·
- Controllers: storage, network cards;
- Interconnects: CPUs, memory, I/O;
- 有些硬件包括两种资源，比如存储设备包括一个 IO 资源，也包括容量资源；两者都能成为系统瓶颈；
- USE Method 适用于那些高利用率、高饱和环境中容易出现性能下降的资源；
	- Cache 在 utlization 更高的时候有助于提升性能，所以不适用于 USE Method；
	- Cache 的 hit rate 和其他系统指标可以在观察 USE 指标之后再做观察，通过 USE 指标首先判断系统性的瓶颈所在；
Functional Block Diagram
- 另一个了解系统整体资源的方式是画一个 Function Block Diagram；
Interconnects
- CPU, memory 乃至 I/O interconnect 经常被 overlooked；
- 幸运的是，它们并不常常成为瓶颈；而不幸的是，如果它们属于瓶颈，可以做的事情不多，只能升级主板，或者减少 load（比如 zero copy 机制减少 memory bus 负载）；
Metrics
- CPU Utilization: per CPU or per System average;
- CPU Saturation: run-queue 长度、调度器延时；
- Memory Capacity Utilization：available free memory；
- Memory Capacity Saturation: anonymous paging or thread swapping;
- Memory Capacity Errors: failed malloc();
- Network interface Utilization: Rx/Rt throughput / max bandwidth;
- Network Interface Saturation: 丢包？
- Storage device I/O Utilization: deivce busy percent; 当前 IOPS、带宽 / 上限；
- Storage device I/O Saturation: wait queue length;
- Storage device Errors: deivce errors (比如 raid 卡坏了...);
In Practice
- 当系统遭遇性能问题时，USE Method 帮助你手头能有一个可用的 check list；
- 具体实践中的指标可以随着时间逐步改进；
Software Resources
- mutex lock：utilization 可以定义为这把锁的持久时间；saturation 可以定义为排队等待这把锁的线程数；
- thread pool：utilizaion 可以定义为线程忙碌处理的时间；saturation 可以定义为排队等待线程池的任务数；
- process/thread capacity: 系统进程、线程数存在上限，当前的线程数作为 utilization；等待进程创建的时间可以认为是 saturation；"cannot fork" 视为 errors；
- file descriptor capacity: 与上述有限资源类似；
Suggested Interpretations
- Utilization：100% Utilization 通常意味着瓶颈存在；70% 的高 Utilization 可能存在问题：
	- 在较宽的时间周期内的 70% Utilization，往往存在着突发性的 100% Utilization；
	- 部分系统资源，如硬盘，在执行操作期间不能中断，当它们的利用率达到 70% 时，排队延时会变得显著；
- Saturation：任何 saturation 现象都可能成为问题；saturation 可以通过 wait queue、队列等待时间等指标衡量；
- Errors：任何 error 都应该研究原因，尤其当性能下降且 errors 值仍在上升时；
Tools Method
- Tools Method：列出可用的性能工具；列出每个工具中表达的指标；列出每个工具对性能指标的解释规则；
- 不同的是，USE 方法将通过迭代系统资源的方式，形成一个 check list，然后找工具来回答这些问题；