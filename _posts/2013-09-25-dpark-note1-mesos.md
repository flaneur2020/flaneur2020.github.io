---
layout: post
title: "Dpark Note 1: Mesos"
---

终于花一周时间理了一遍 dpark 的逻辑，原始的笔记放在了 evernote，刚刚放下一天就不忍直视了，在这里尝试提炼一下。目测把内容堆到一篇 blog 里有些吃力，还是计划分成三篇然后祈祷不要坑掉好了。

dpark 在项目开始之初的定位为 spark 的 python 版实现，思路基本一致，读代码时可以相互参考。尤其是刚刚开始阅读代码时，对各逻辑结构心里没底，spark 代码中的类型签名能够起到很棒的引导作用。然而毕竟 spark 的代码规模更大一些， 总体而言仍是 dpark 更容易读完。

spark 将设计构建于 mesos 平台之上，很多代码都是围绕着 Mesos 的接口展开，在这里先简单记录一下 Mesos 的基础概念和基本的接口，在此暂且无意深究 Mesos 的内部实现。

## Background

Mesos 是一个专门为集群计算框架设计的基础设施，就像虚拟化允许大家在一个物理机器上运行多个操作系统， Mesos 允许我们在同一个集群中执行多个计算框架，比如 Hadoop,MPI,Spark,Storm 等等，把它们放在各自的 cgroup 里，实现资源的隔离。

在同一个集群中跑这么多不同的框架有什么用处？比较直接的一个回答是，Map Reduce 其实是个很受限制的计算模型，对于一些特定的情景，可能其它的某些计算模型更为适用，一个例子就是为高实时的流式计算而设计的 Storm。如果有一般的数据分析需求，也有流式计算的需求，那么分别搭建 Hadoop 集群和 Storm 集群明显是不经济的。另外，即使没有特殊的计算需求，使用 Mesos 管理计算集群仍有很大的好处：集群的维护与更新是一件头疼的事情，曾经雅虎有段时间用的 Hadoop 版本都非常老，却不敢升级，担心现有的业务代码坏掉。Mesos 既然支持多个计算框架共存，当然也支持某计算框架的多个版本共存，作为接近 rvm 这种版本管理器的存在。

## Design

Mesos 作为集群计算框架的基础设施，统筹计算资源的分配与管理，它的核心其实是一个调度器。调度的基本单位是 Task，这就引出了一个问题，不同的计算框架对于 Task 的解读并不相同，它们往往也有自己独特的调度需求，实现了自己的Task类、自己的调度器。Mesos 提供的是怎样的一套通用方案，使各位都满意？答案是一个两级调度器，Mesos 仅负责资源的调度，做到对每个任务都公平，至于各任务孰先孰后的调度则交与计算框架去负责。

集群中一旦有节点空闲，Mesos 就会统计这个节点内的资源，包装成 Offer 分发给需要资源的计算框架，然而 Mesos 对计算框架的需求并不太清楚，给出的 Offer 并不一定满足计算框架的需求，比如内存太小、CPU太少，Mesos 允许框架拒绝不合适的 Offer，如果被拒绝，它就继续拿这个 Offer 尝试发给其它计算框架。然而对于计算框架而言，拒绝掉 Offer 就意味着需要继续等待一段时间，作为对这段时间的补偿，Mesos总是优先调度等待时间最长的计算框架。就 "公平" 的意义而言，这与 Linux 的 CFS 调度器的思路有点相似。

Mesos 为计算框架设计了两个接口：Scheduler 和 Executor，前者作用于 Master，即执行代码的那台机器 [1]，既是 Mesos 调度器的接口，也是计算框架的任务调度器；后者作用于 Slave，即计算节点，用于执行来自于 Master 分配的任务，当然任务的分配是经过 Mesos 中转的。它们都是被动地接受来自 Mesos 的回调的接口，至于主动调用 Mesos 的接口，是 MesosSchedulerDriver，它可以用来向 Mesos 申请 Offer、提交 Task、返回计算结果等等。

一个任务的大体生命周期如下：

- 计算框架首先主动地向 Mesos 申请计算资源
- Mesos 的调度器发现了一个节点，有多少内存有多少CPU闲置，反馈给计算框架一个 Offer
- 计算框架收到 Offer，判断这个 Offer 是否满足自己的需求。如果不满意，就拒掉它，直到有满足需求的 Offer 到手。
- 计算框架将 Tasks 打包，带着 Offer 的凭据发给 Mesos。Mesos 收到打包的 Task，会把它发送到 Offer 对应的节点，交付守护的 Executor 进程执行。
- Executor 执行完毕，会把任务的结束状态(成功?失败?)返回给 Mesos，Mesos 继而通知计算框架。

## API

具体到 dpark，与 Mesos 交互的类主要是 MesosScheduler, MyExecutor 和 MesosSchedulerDriver。如果打算山寨一个集群计算框架，那么这三个类就相当于你的基础 API 了。下面仅列出几个代表性的方法：

*dpark.schedule.MesosScheduler*: 向 Mesos 申请资源，并安排任务的先后。

- resourceOffers: 收到 Offer
- offerRescinded: Offer 被拒绝
- statusUpdate: 任务执行状态发生变化，比如有任务执行完毕了，有任务失败了
- registered: 有新增计算节点

*dpark.executor.Executor(MesosExecutor)*: 侦听 Mesos 的事件，接收任务执行真正的计算。此外，每个 Executor 也会开一个简单的 HttpServer，为其它节点传送本地的文件，这就属于 Mesos 调度之外的工作了。

- registered: 连接到 master，在这里可以启动自己的 HttpServer
- launchTask: 收到并执行任务
- killTask: 杀死任务
- shutdown: 关闭守护的进程

*mesos.MesosSchedulerDriver*: Mesos 的通信协议基于 protobuf2，为了方便请求的发送，dpark 中附带了 mesos_pb2.py 这么一个文件，包装了用到的消息。

- reviveOffers(): 申请 Offer
- launchTasks(): 将任务交付给 Mesos
- sendStatusUpdate(): 更新任务的执行状态

## Footnotes

[1]: 用户执行代码的机器其实是集群的客户端，在逻辑上反而扮演者 Master 的角色。
