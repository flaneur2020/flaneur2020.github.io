---
layout: post
title: "Quick Note on Shadow Page Table"
---

有关 Shadow Page Table 的一些设定困惑了很长时间，毕竟没有硬件的支持，放出来特别多黑魔法眼花缭乱的坑爹感，印象十分深刻。

昨天把书从旧居搬了过来，躺床上翻 《支撑处理器的技术》 无意翻到影子页表这一节，对这两段颇为疑惑：

*200页, 影子页表 -- x86的 Hardware Table Walk 设施:*

> 此外， Guest OS 更改页表项目时，也必须修改与之对应的影子页表。为解决这个问题，VMM 首先将保存 Guest OS 页表的内存区域设置为只读。

*202页，虚拟TLB -- 更好的影子页表:*

> 虚拟 TLB 方式中， Guest OS 可以随意改变页表内容， 例如， Guest OS 可以改变某个页面对应的虚拟物理内存地址，而不会通知 VMM，这样就与虚拟 TLB 的内容产生矛盾。
>
> 但是，即使没有虚拟化，操作系统像这样改变页表的内容，也会导致硬件的 TLB 内容和被改变的页表内容之间的矛盾， 因此必须清空 TLB 的内容。在虚拟化环境中， 清空 TLB 的特权指令执行时会产生异常，从而控制转移到 VMM，这样就能将 Guest OS 对页表的改变反映到虚拟 TLB 中。

那么既然 Guest OS 会使用 invlpg 同步页表项的改动，那么还为什么必须给 Guest OS 页表的内存区域写保护？

带着这个问题 google 了一番，几处资料给出的解释都不大一样，但都是很宝贵的见解。结合它们，在这里尝试记录一下自己对影子页表在概念上的理解。但真相还是在源码中 ! 其中难免有错误的地方，愿朋友指正。

## The Tables

在没有硬件虚拟化支持的系统上，为实现内存的虚拟化，会牵涉到这几张表：

1. Guest 页表: Guest 虚拟地址 => Guest 物理地址
2. pmap: Guest 物理地址 => Host 物理地址
3. 影子页表: Guest 虚拟地址 => Host 物理地址

影子页表作为内存虚拟化的基础设施，大约相当于 Guest 页表与 pmap 叠加的结果，会被最后交给 cr3 。 但是 Guest 页表会更新，pmap 也会变化，影子页表需要想方设法与它们保持同步，但 "不完整" 的同步也是允许的，影子页表扮演的角色实际上也更接近于 TLB 。比方说在 Guest 页表中存在的一项映射在影子页表中不存在， 这是允许的，到访问这条映射对应的内存时，会发生页面错误, VMM能够捕获它，调整影子页表增加这条映射，这在 Guest 看来仅仅相当于 TLB Miss 而已。

## The Simple Implementation

先从影子页表最简单的实现出发，理一下会出现的问题。

采用 Lazy 的同步策略，每次 `mov cr3, pgd` 切换页目录时，将影子页表中的所有表项初始化为 Non Present 。随后 Guest 访问内存会触发页面错误， VMM 捕获它们， 根据目标地址、 Guest 页表和 pmap 更新相应的影子页表项，退出页面错误处理例程之后即可顺利执行下去。到 Guest OS 切换进程时，则丢弃原先的影子页表，从零重新开始。 Guest 对此可以毫不知情，这也被称作 Hidden Page Fault [2]。

说起 Hidden Page Fault，还有一种情景是 Host 的页面有可能被换出，Guest 踩到时也会触发页面错误， VMM 这时能把换出的页面读回来，可是物理地址变了，这时需要更新 pmap 中的映射，同时更新相应的影子页表项。

除此，Guest 更会主动修改页表项。比如 Guest 进程执行 `mmap()`，或者写时复制、请求调页之类，都会通过修改页表项，所幸在修改页表项之后 Guest OS 一般需要调用 `invlpg` 这种指令使相关的 TLB 条目失效， VMM 可以捕获 `invlpg` 指令，相应地修改影子页表项。

相对于 Hidden Page Fault，Guest OS 的页面也可能会被换出，这点要求 VMM 已捕获了换出页面这一操作对页表的修改，并同步到影子页表，将相关的表项设置为 Non Present 。待 Guest 踩到这张页面时，Host 会发生页面错误并由 VMM 捕获，进而触发虚拟的 Guest 页面错误。这被称作 True Page Fault [2]。

除了每次进程切换都重建影子页表这个比较显眼的性能问题之外，一眼望去 so far so good.

## Too Young Too Simple, Sometimes Naive

但里面有问题，可能不是 VMM 的问题，但它是问题。

Linus 曾在邮件里提到，x86 的 TLB 在遇到页面错误时倾向于进行一次 Page Walk 同步页表项的内容，然后再检查一次，确认确实需要抛错误时再抛错误 [3]。 比如 Guest OS 可能会更新页表项为某页面解开写保护，却忘记了 `invlpg`， TLB 中的表项没有更新，依然觉得这个页面是不可写的，MMU 觉得需要抛错误，但它又想谨慎为上，于是开动 Table Walk 检查了相关的页表项，发现没有写保护了，就开心地吞下了这个错误，得以继续执行。 这个 "feature" 没有任何文档，只是碰巧这么实现了， 合理的 OS 不应该依赖这个 "feature"。

但是依赖这个 "feature" 的 OS 是存在的，还非常流行。这个 "feature" 把问题隐藏了起来，糟糕地十年二十年没出问题。

直到冤大头 VMM 的设计者出现，再试图依赖 `invlpg` 来捕获页表的更新，已然无解。

## The Performance Problem

回到那个显眼的性能问题，每次进程切换都重建影子页表，显然太低效了。那么，为什么不将 Guest 中各进程对应的影子页表缓存起来？

明天继续

## A & D

## Well, I Still Have Questions

[3] 提到使用写保护跟踪所有 Guest 页表的改动的一个原因是，基于 `invlpg` 的跟踪会导致大量的 Hidden Page Fault。这点依然不是很理解，使用写保护的话，代价会是(大量的?) 修改页表的页面错误，跟 Hidden Page Faults 的开销相比孰优孰劣？

## Reference

1. 支撑处理器的技术
2. http://www.scs.stanford.edu/08wi-cs240/notes/vm-techniques.txt
3. http://yarchive.net/comp/linux/x86_tlb.html
4. [KVM 分析报告](http://wenku.baidu.com/view/9e5abf31b90d6c85ec3ac649.html)
5. [KVM: MMU: Cache shadow page tables](http://lwn.net/Articles/216759/)

