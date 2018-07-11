---
layout: post
title: "Note on G1GC: Write Barriers"
---

了解一点 GHC GC 之后接着看了一下 G1GC。

G1GC 和 GHC 的 GC 一样，将堆划分为等宽的 region，而 "Garbage First" 的名字的由来，就是出于 G1GC 会优先回收垃圾数更多的 region。regional GC 是相对于分代假设之外，又一个比较重要的设计导向。

region 会作为 G1GC 做增量回收的基本单位，对 region 做拷贝仍然会 STW，但是拷贝一个 region 的时间是固定的，这样控制一轮 GC 中拷贝 region 的个数，可以控制 STW 的时间范围。

G1GC 的 region 分为 Young （包括 Eden 和 Survivor）与 Old 两代，加上 Humongous Region 专门用于存大对象。只回收年轻代的 GC 模式叫做 Young GC，另一个 Mixed GC 模式会在年轻代之余，回收老年代中被认为垃圾较多的部分 region。在执行 GC 时，被选择到做回收一组 Region 称作 Collection Set（简称 CSet）。

Region 内分配内存会十分简单，每个 region 一个 top 指针，分配内存直接跳一下指针即可。

## Remember Set

那么，怎样才能单独回收一个或者一组 region？与 CMS 维护代际间的 Remember Set 不同，G1GC 会为每个 region 维护自己的 "point-into" 的 Remember Set （简称 RSet），每个 region 记录起所有其他 region 指向本 region 的引用。

<img src="/images/g1gc-regions.png"/>

Remember Set 在维护上一定需要 write barrier，而 write barrier 只用于捕捉跨 region 的引用，因而可以使用这样的逻辑做过滤：

```
(&object.field XOR &some_other_object) >> RegionSize
```

write barrier 的更新先存放到本线程的 update buffer 中，到 update buffer 满后，把满的 buffer 挪到公共的队列中供 Refinement Thread 消费。

## SATB

Write Barrier 除了用于维护 Remember Set，还有一种 Concurrent Marking Barrier 用于维护 concurrent mark 的中间状态，比较典型的例子是三色 GC 的 barrier：黑色对象的字段引用到白色对象时，改变黑色对象为灰色。[1]

Mutator Thread 在运行中，对象图不停地变化，那么我们自然能想到的，就是通过 Write Barrier 去跟进对象图的变化，如果一个对象的某字段指向新的对象，把新对象加入 mark stack 等待遍历。

```
write_barrier_ref(Object* src, Object** slot, Object* new_ref)
{   *slot = new_ref;
    if( is_marked(src) )
        enqueue(new_ref);
}
```

这样的 barrier 属于 Increment Update 的方式，简称 INC，表示它会增量同步堆内的对象图。但是 INC barrier 存有一项缺陷：无法发现 concurrent mark 期间堆外根集（寄存器、栈）的变化，为此在 concurrent mark 之后，需要一个 remark 阶段再 STW 扫一遍根集，这是 CMS GC 的特点，有时 remark 阶段时间并不短。

G1GC 用的 Snapshot-At-The-Beginning 的 barrier 另辟蹊径，并不持续跟踪对象图的变化，而是打下 concurrent mark 那一刻的快照，所有新分配的对象统统视为活跃，做到：

> - Anything live at Initial Marking is considered live.
> - Anything allocated since Initial Marking is considered live.

SATB Barrier 不需要最后的 remark，代价就是因为新分配的对象统统视为活跃，有更多的 float garbage。最后回收到的垃圾对象，一定是开始 mark 那一刻之前产生的垃圾。

这时要跟踪的不是新引用的赋值，反而是旧引用的被解除，以维持快照时刻的对象关系：

```
write_barrier_slot(Object* src, Object** slot, Object* new_ref) {
    old_ref = *slot;
    if( !is_marked(old_ref) ){
      enqueue(old_ref);
    }
    *slot = new_ref;
}
```

但是 G1GC 为什么仍有最终标记阶段？G1GC 中 Write Barrier 产生的标记并不是实时更新的，而会记录在本线程的 update buffer 中（它扮演的角色有点类似 golang 里的 chan？），当写满一个 buffer 后，再把整个 buffer 加入到全局的 update buffer 队列中，供 Refinement Thread 消费来真正地做 Mark。到最终标记阶段，需要做的事情就是把这些 buffer 都给 flush 出来，完成所有标记，这点与 CMS 的 Remark 有很大不同。

## TAMS

Initial Mark 时 G1GC 会记录 region 当前的 top 指针，记做 TAMS（Top At Mark Start），而位于 TAMS 之后分配的对象都视为活跃，这也叫做隐式标记。

<img src="/images/g1gc-tams.png"/>

每个 region 有两个 TAMS 指针，表示当前的 nextTAMS 和上一轮标记的 prevTAMS，也有两个记录对象标记的 nextBitmap 和 prevBitmap。为什么要放两个 TAMS 指针和 bitmap 呢？我猜可能是因为 G1GC 的 Evacuation 并不一定等待 Mark 阶段结束才开始，在 Concurrent Mark 期间也可能进入 Evacuation 阶段，这时会选用 prevTAMS 和 prevBitmap 做为标记信息，它们扮演了一个快照，能回收掉上一轮标记时发现的垃圾对象，在 Concurrent Mark 完成之后再进入 Evacuation 阶段的话，再取当前的 nextTAMS 和 nextBitmap。

## References

- [Write Barriers in Garbage First Garbage Collector](https://www.jfokus.se/jfokus17/preso/Write-Barriers-in-Garbage-First-Garbage-Collector.pdf)
- [Garbage-First Garbage Collection](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.6386&rep=rep1&type=pdf)
- [HotSpot VM 请教G1算法的原理 - 资料 - 高级语言虚拟机 - ITeye群组](http://hllvm.group.iteye.com/group/topic/44381)
- [Java Hotspot G1 GC的一些关键技术 -](https://tech.meituan.com/g1.html)
- [Garbage-First Garbage Collection 论文笔记 - 简书](https://www.jianshu.com/p/b1609c81cd5f)
- [Grid Designer's Blog: Understanding GC pauses in JVM, HotSpot's CMS collector.](http://blog-archive.griddynamics.com/2011/06/understanding-gc-pauses-in-jvm-hotspots_02.html)
- Advanced Design and Implementation of Virtual Machines

## Footnotes

- 1: 实际上三色 GC 的术语非常有代表性："All the marking algorithms do is coloring white gray, and then coloring gray black"。
