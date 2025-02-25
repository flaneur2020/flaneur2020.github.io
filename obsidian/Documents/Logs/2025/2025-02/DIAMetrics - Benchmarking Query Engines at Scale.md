传统上，通常是 vendor 手工分辨出来客户最重要的 query 来构造针对性的 benchmark。

但是，这个方法通常不那么好使，因为：

1. production workload 是随时变化的；
2. workload 与 query engine 之间有时候有紧密耦合，阻止了用户去比较 query engine；
3. 对于小修改，难以衡量

作者提出来 DiaMetrics，目标是：

1. 能够有一个统一的解决方案，来 end-to-end 地 benchmark，并支持多个 query engine；
2. 覆盖 benchmark 的完整生命周期；
3. 提供 system performance 和 effieciency 的 insight；

之前的 tpch 和 tpcds 主要是静态定义的 benchmark，而且假设对整个 data management stack 有管理能力；

> Not having a way to benchmark a production system in a dynamic and often unpredictable environment may prove detrimental not only to the system developer but also to the user

> an architecture for benchmarking that is capable of generating indicative benchmark workloads over production deployments, executing them, and measuring a system’s performance on that workload

相对于静态的 benchmark，作者认为应该提供一个 benchmark framework。需要有办法生成 benchamark、执行 benchmark 并最后收集数据。

在 google，内部起码有四套 query engine：F1、Dremel、Spanner SQL、Procella；也有专用的存储系统，如 Mesa、Colossus；

（听起来有一些内部不同部门重复造 benchmark 轮子的情况）

> - (a) Instead of focusing on a specific benchmark workload and using that as the means to test performance and efficiency, DIAMetrics provides an endto-end benchmarking framework. The system is capable of generating indicative benchmark workloads over production deployments, executing them, and measuring a system’s performance on that workload.
> - (b) In order to avoid duplicate effort, DIAMetrics is query engine independent and relies on a handful of generic reusable components that can be instantiated with minimal effort for every system that is to be benchmarked.
> - (c) DIAMetrics provides the means to track the performance per indicative benchmark workload and use that historical information to measure improvement over time.

![[Screenshot 2025-02-13 at 11.54.33.png]]
## 2. OVERVIEW

有这几个组件：

1. the workload extractor：挖掘用户的 query log，提取一个 query 的 subset 表示用户的 workload；
2. the data and query scrambler：“The data scrambler anonymizes data in various ways”，在数据脱敏抓取来之后，query 也采用类似的方法，来对 query 中的数据脱敏；
3. the data mover：能够在不同的 storage backend 之间移动数据；
4. the workload runner：根据 workload 规律，发压；
5. system monitoring

（对 data and query scrambler 比较感兴趣...）

## 2.2 Workflow

google 内部的几套系统的 log format 都不大一样。diametrics 做了一个归一化处理；

summarizer 会根据这个收集来的 log 选出来一个子集，能够代表用户的 workload。

diametrics 的 data scrambler 能做脱敏，应用多种脱敏技术。

- 最简单的 case 里，它会重新排列一个列的出现顺序（相当于列的每个值重新映射一下）。
- 它也可以对列的数据做混淆，做哈希，加一个随机的 noise 等等；

## 3. FRAMEWORK COMPONENTS

### 3.1 workload extractor

一个系统在不同的时间段，可能有不同的 work load 特征。

> To process not only standardized benchmarks but also user-specific benchmarks, we developed techniques that compress a user’s workload into a small set of representative queries that can then be used as a benchmark workload [14]. Our framework for workload extraction and summarization roughly undertakes the following tasks:

1. Log canonicalization: 先对用户的请求日志进行统一化处理，包含一系列 features，其中包括 syntactic feature 和 profiling feature。syntactic features 可以通过 parsing 从 query 中提取，比如 query 中有多少个 join、query 中使用的聚合方法；profiling features 包括性能指标，如 cpu usage、query latency 等等；
2. Workload summarization：（Once the workload features have been extracted, we can leverage them to identify a subset of queries for benchmarking this workload.）选择 queyr 的策略有两种参考：代表性（reprsentativity）和覆盖度（coverage）；
#### 3.1.1 Summarization algorithm

summarization 问题可以看做一个针对 representavity 和 coverage 的优化拟合问题。

好像有一个 KL 散度。
#### 3.2 Data and query scrambler

> it provides a simple and efficient way to use production data for query benchmarking.

思想就是用脱敏的生产 query 针对脱敏的数据，来进行 benchmark。

在 workload summarization 摘出来 query 之后，就可以将现在的生产数据打一个 snapshot，并用这个 snapshot 进行 benchmark。

> In the data scrambler we solve that problem by breaking correlations between values; by protecting data through hashing their values to obfuscate them; and by adding small amounts of noise to the data so that their distributions are not significantly altered, whereas their original values airare.

#### 3.2.1 Scrambling techniques
![[Screenshot 2025-02-14 at 19.17.11.png]]
基本款的脱敏是上面这种，将 input table 拆成 chunk，在 chunk 内部将 column 之间的关系打乱。

每个列有不同的 recorder 顺序。

这个办法的优势：

> - The first one is that the value distributions per column in the scrambled table will remain exactly the same as those in the original table.
> - The second advantage is that any correlations between columns will be broken, as the values of the two columns will be permuted independently of one another.

能够保留每个列的统计分布。（第二个优势真的是优势吗？）

> Note that depending on the use case for the benchmarked dataset, this is not necessarily a good idea, as these correlations may be important. If that is the case, the scrambler can be configured so that groups of columns have the same permutation order and correlations are preserved in the output, while it can still guarantee the property of the correlations being split between a group of correlated columns and the rest of the columns of the row.

每个列的顺序完全打乱不一定合适，作者也提供了方法，允许配置相关的列总是按照相同的次序被打乱。

