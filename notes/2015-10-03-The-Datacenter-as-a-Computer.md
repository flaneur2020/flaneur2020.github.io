---
layout: default
title: The Datacenter as a Computer
---

# 读书笔记: The Datacenter as a Computer


## Balanced Designs

<原文开始>There is opportunity here to find solutions by software- hardware co-design, while being careful not to arrive at machines that are too complex to program.</原文结束>

<原文开始>The most cost-efficient and balanced configuration for the hardware may be a match with the combined resource requirements of multiple workloads and not necessarily a perfect fit for any one workload</原文结束>

<原文开始>Fungible resources tend to be more efficiently used. </原文结束>

好像说在存储系统上，网络带宽和磁盘带宽是可以互换的，因而可以利用率更高。
## Workloads and Software Infrastructure

<原文开始>Typical Internet services exhibit a large amount of parallelism stemming from both data- and request-level parallelism. Usually, the problem is not to find parallelism but to manage and efficiently harness the explicit parallelism that is inherent in the application</原文结束>

WSC 中的应用固有无限的数据级、请求级的并行。最困难的问题不再是挖掘并行，而是应对固有的并行。

<原文开始>A beneficial side effect of this aggressive software deployment environment is that hardware architects are not necessarily burdened with having to provide good performance for immutable pieces of code. Instead, architects can consider the possibil- ity of significant software rewrites to take advantage of new hardware capabilities or devices.</原文结束>

<原文开始>Homogeneity within a platform generation simplifies cluster-level scheduling and load balancing and reduces the maintenance burden for platforms software (kernels, drivers, etc.).</原文结束>

<原文开始> Ideally, the cluster-level system software should provide a layer that hides most of that complexity from application-level software, although that goal may be difficult to accomplish for all types of applications.</原文结束>

<原文开始>Although the plentiful thread-level parallelism and a more homogeneous computing platform help reduce software development complexity in Internet services compared to desktop systems, the scale, the need to operate under hardware failures, and the speed of workload churn have the opposite effect.</原文结束>

简化复杂度的地方：1. 不必挖掘并行；2. 基础组件同构；
增加复杂度的地方：1. 需要日常地应对硬件故障；2. 访问模式异构；

<原文开始>Once the diverse requirements of multiple services are considered, it becomes clear that the datacenter must be a general-purpose computing system. </原文结束>


## Introduction

<原文开始>They differ significantly from traditional datacenters: they belong to a single organization, use a relatively homegeneous hardware and system software platform, and share a common systems management layer.</原文结束>

<原文开始>Most importantly, WSCs run a smaller number of very large applications(or Internet Services).</原文结束>

<原文开始>The relentless demand for more computing capabilities makes cost efficiency a primary metric of interest in the design of WSCs.</原文结束>

<原文开始>However, network switches with high port counts, which are needed to tie together WSC clusters, have a much different price structure and are more than 10 times more expensive (per 1Gps port) than comodity switches.</原文结束>

<原文开始>A switch that has 10 times the bi-section bandwidth costs about 100 times as much. As a result of this cost discontinuity, the networking fabric of WSCs is often organized as the two level hierarchy despicted on Figure1.1</原文结束>

每个 rack 一个低端交换机，上面一个高端的核心交换机。

<原文开始>In such a network, programmers must be aware of the relatively scarce cluster-level bandwidth resources and try to exploit rack level networking locality.</原文结束>

<原文开始>one can remove some of the cluster level networking bottlenecks by spending more money on the interconnect fabric. for example, Infiniband interconnects typically scales to a few thousand ports but can cost $500~$2000 per port.</原文结束>

<原文开始>Alternatively, lower-cost fabrics can be formed from commodity Ethernet switches by building "fat tree" Clos network.</原文结束>

现在该 clos 布局弄的比较多，局部性是需要适应的现状，也可以看做一个可以解决的问题。

<原文开始>A large application that requires many more servers than can fit on a single rack must deal effectively with these large discrepancies in latency, bandwidth, and capacity.</原文结束>

<原文开始>these discrepancies are much larger than those seen on a single machine, making it more difficult to program a WSC.</原文结束>

<原文开始>key challenge for architects of WSCs is to smooth out these discrepancies in a cost efficient manner. conversely, a key challenge for software architects is to build cluster infrastructure and services that hide most of this complexity from application developers </原文结束>

存储层次的异构是复杂性的一个来源

<原文开始>Although this breakdown can vary significantly depending on how systems are configured for a given workload domain, the graph indicates that CPUs can no longer be the sole focus of en- ergy efficiency improvements because no one subsystem dominates the overall energy usage profile. .</原文结束>

CPU 不再是耗能的主要来源

