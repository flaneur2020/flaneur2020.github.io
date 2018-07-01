---
layout: post
title: "Note on GHC GC"
---

前段时间看了一下 haskell 的 gc，简单记一下。

比较特色的地方来自对象的不变性。如果所有对象不可变，GC 在设计上是否能有简化？ 其实 haskell 对象也并非完全不可变，一个惰性求值的 thunk 可以求值一次，也仅能求值一次，这便足以简化掉 remember set 的设计。相对于 jvm 那样的 dirty card，ghc 更容易做到这样的不变式：**年轻代单向引用老年代**，老年代的 thunk 如果求值生成的新引用，直接 eager promotion 给提升到老年代即可。这样 minor GC 扫描的范围可以更少。

函数式里的临时对象多，请求生命周期的话，临时对象都在年轻代，拷贝式 gc 相比 mark & sweep 理论上会更轻快。block 的堆布局在这方面能更精细回收掉旧的数据，也会作为并行分工的单位。

与 jvm 年轻代走拷贝 / 老年代走 mark & sweep 不同，ghc 的不同代都是拷贝式。

## 堆布局

ghc 的堆有被分成等大小的 block，并非 CMS 那样连续的堆。堆的每代都由离散的 block 链成，如果空间不足，可以新增分配 block。

对象的年龄信息 "step" 并未存放于对象元信息内，而是划分 generation 为 k 个 step 区域，一次 gc 之后仍存活的对象，会被拷贝到 step + 1 表示寿命提高，当 step 超过 k 之后，将这些对象整体提升到老年代。

## Evacuate & Scavenge

Evacuate 和 Scavenge 拷贝式 GC 中常见的两个词：

- **Evacuate** 是 “拷贝” 的近义词，将对象拷贝到 to-space，同时将 from-space 的旧对象修改成 forwarding-pointer 指向对象的新地址，也返回这个新地址。

- **Scavenge** 是 “scan” 的近义词，遍历对象所引用的所有对象执行 evacuate，并修改对象引用的地址。

当 to-space 的所有对象都已经过 scavenge 之后，整个 gc 过程结束。

## Parallel Copying

一个经过 evacuate 但仍未 scavenge 的对象，视为处于 pending 状态。

多个 gc 线程做的事情，都是原子取一个 pending 状态的对象 `p`，执行 `scavenge(p)`，将它引用的对象列表加入 pending set：

```
while (pending set is non-empty) {
  remove an object p from the pending set
  scavenge(p)
  add any newly-evacuated objects to the pendings set
}
```

用 block 来维护 pending set 这个 "任务队列" 是这里精巧的地方。

![](/images/ghc-gc-block.png)

pending set 中等待 scavenge 的对象，一定会位于 to-space 的 block 之中。在链表相连的 block 之内按顺序遍历这些对象，就形成了一个 “任务队列“ 的形状，两个指针维护已遍历执行 evacuate 的对象的指针 S，加一个 pending 对象末尾的 SL 指针。

每个线程并发拷贝的逻辑相同，但是每个线程有自己私有的 to-space，都会拷贝向自己的 block。拷贝过程中生成的新的 block 可以加入队列的尾部作为新的 work 单位，这里似乎可以选择 work-stealing 的方式去提高并行。

## References

- [Parallel Generational-Copying Garbage Collection with a Block-Structured Heap](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/06/par-gc-ismm08.pdf)
- [GHC Heap Internals](http://www.cse.chalmers.se/edu/year/2016/course/course/pfp/lectures/Frolov14.pdf)
- [The GHC Runtime System](http://www.scs.stanford.edu/16wi-cs240h/slides/rts-lecture-annot.pdf)
- [Commentary/Rts/Storage/GC/Copying – GHC](https://ghc.haskell.org/trac/ghc/wiki/Commentary/Rts/Storage/GC/Copying)
