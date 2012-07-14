---
layout: post
title: "Personal Notes on Linux Virtual Memory Management"
---

    All memories are created equal, but some are more equal than the others.
    - Geogre Orwell

前几天简单翻了遍《Understanding Linux Kernel Memory Management》，正文很短，读起来并不容易，该属于常读常新的书了。趁印象还热乎先记一点笔记吧，内容比较个人化没有什么条理，见谅了 :p

『内存管理』其实是个很宽泛的词，物理页管理、页表管理、地址空间管理、内存分配等等，单独拿出每一块出来都可以称作『内存管理』，但其间差异何其巨大，用一个词盖过去，就像是一叶障目，不见树林。要理解它的全貌，还是要分开讨论比较好。如果能将每个局部都搞清楚的话，总不至于见不到全局的。

### 系统初始化与内存布局

在引导阶段(`setup.o`)，内核会先映射两个临时页表到页目录`swapper_pg_dir`^1 的前两项，将0~8m, 3g~3g+8m的虚拟地址同时映射到0~8m的物理地址。

内核页表的初始化则是在内核初始化阶段的`kernel_physical_mapping_init()`函数中。如果可以，就利用PSE设置大页；利用PGE设置全局页，将内存映射固定在TLB中，在切换地址空间时不需要重新刷新。

Linux固定映射内核地址空间。在x86的机器上，物理内存分为3个Zone： 

+ `ZONE_DMA`: 0~16m
+ `ZONE_NORMAL`: 16m~896m
+ `ZONE_HIGHMEM`: 896m~

其中`ZONE_DMA`与`ZONE_NORMAL`都是固定映射到0xc000000处，剩余128mb的地址空间用于映射`ZONE_HIGHMEM`中的物理页，抑或FIXMAP。

在x64中，地址空间已足够映射所有的物理内存，`ZONE_HIGHMEM`为空。

### kmap

要在内核中使用`ZONE_HIGHMEM`中的内存，必先通过`kmap()`将它映射到内核地址空间。

但`kmap()`有可能进入睡眠，在中断上下文中必须使用`kmap_atomic()`。

### Buddy Allocator与物理页面管理

在linux中，Buddy Allocator就是作为页级内存分配器了，用于分配2次幂大小的连续物理页面。它同时作为内核内存分配器与Page Cache的后端。

相关的例程为`alloc_pages(gfp_t gfp_mask, unsigned int order)`，它将返回一只page结构组成的链表。

### 资源有限怎么办 

+ bounce buffer: DMA内存有限，空间换生存周期；
+ `kmap_atomic()`: 预留的页表项有限，尽快的用，尽快的释放。

### NUMA与Node

在SMP系统上，当CPU增多时，无差别的内存访问成为了影响伸缩性的一项瓶颈^2 。为此的解决方案是为每块处理器单独提供一块连续的内存，原则上处理器尽可能只访问靠近自己的内存，从而提高内存访问的伸缩。这就是NUMA(Non-Uniform Memory Access，非一致内存访问)，其中的每块处理器与内存被合称为Node。

而内核需要做的，就是在CPU分配内存时尽量只分配本Node中的内存。在Linux中，表示Node的结构为`pg_data_t`。

### Slab与内核内存分配

内核中要使用内存，就用Slab了。它将buddy allocator作为后端，申请来大块的内存分成小块，每个slab中都是等大的内存块。

+ `kmem_cache_alloc()`与`kmem_cache_free()`，专用内存分配器，用于分配内核中常用的结构。
+ `kmalloc()`与`kfree()`，通用内存分配器。它的背后是一组不同大小的slab。

如果结构的分配/释放不是特别频繁，一般只用`kmalloc()`就足够了。

### 页表管理

x86的linux的三级页表其实是为PAE准备的，64位机器的话，得有四级才行。

### misc notes

+ `struct address_space`这个名字有一定误导性，实际上内核中描述用户地址空间的结构为`struct mm_struct`。`struct address_space`这个结构更像是用来描述Page Cache的。
+ GFP为Get Free Page的缩写。

----------------

1. `swapper_pg_dir`这个名字是因为历史原因，0号进程曾经扮演着现在kswapd的功能，因此0号进程又被称作swapper。
2. 只有两三个CPU的话，内存访问的影响就不大了。

