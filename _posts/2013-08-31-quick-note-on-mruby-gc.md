---
layout: post
title: "Quick note on mruby GC"
---

前段时间看了一遍 mruby GC，基本把逻辑理清楚了，发补丁补了一些注释之余这里简单记一下。

## 三色 GC

搞过 rails 的同学会有印象，传统 Mark-Sweep 的主要问题是收集时间的不可控 [1]，不幸赶上 GC 的话，页面延迟会比较大。mruby 像 lua 一样将目标设定为可嵌入的解释器，对此很自然的考虑是令 GC 分阶段渐进执行，而三色 GC 也是最简单、最正确的渐进式 GC 的实现方式。

三色 GC 的思路来自 Knuth 的三色图遍历算法 [2]，对象分为黑、灰、白三种颜色，分别维护在不同的集合里。一轮 GC 的流程很简单：

1. 在一开始，所有对象都是白色，从根对象开始遍历，标记为灰色
2. 随后只要灰色队列中还存在对象，便从中取一个对象，将它标记为黑色，并将它直接引用的对象标记为灰色，然后就允许暂停了 (当然继续跑下去也不碍事，GC 的暂停与否，一般看某些阙值的设定)
3. 待所有灰色对象消失之后，白色对象就是可回收的死对象了。

为了让 Sweep 阶段渐进化，仍可以引入另一种白色，放到单独的集合里。到下一轮 GC 开始之后，新申请的对象会被标记为这种新的白色，至于旧的白色对象，则在理论上可以随时收集，不会受到新申请的白色对象的干扰。也就是说，mruby GC 在 Sweep 阶段回收的永远是上一轮 GC 留下的死对象。

此外一条铁律是：黑色对象不可以引用白色对象。然而在一轮 GC 期间对象的引用关系难免变化，比如新申请的白色对象，可能需要被黑对象引用。对此，需要用户手工调用 Write Barrier [3]，将黑对象重新标记为灰色。

其间的每一小步遍历的步子迈的都非常小，所以理论上对于延迟敏感的应用，三色 GC 要友好的多。但是因为将 GC 分派到多次执行的同时，也延长了一轮 GC 的周期，期间积累的内存占用会相对多一些。

## mruby 的三色 GC

在 mruby 中，GC 触发的时机有如下几个:

- `mrb_realloc_simple()` 申请内存时内存不足，触发 `mrb_full_gc()`。
- `mrb_obj_alloc()` 申请对象时存活对象的数目超过阙值，触发 `mrb_incremental_gc()`。
- 开启 / 关闭分代模式时，触发 `mrb_full_gc()` 事先执行一次完整的清理。

不管 `mrb_full_gc()` 还是 `mrb_incremental_gc()`，都是对 `incremental_gc(mrb, limit)` 的封装，它也正是 mruby GC 的中心函数，就像一台自动机那样维护着 mruby GC 的状态：

- 如果 `gc_state` 的值为 `GC_STATE_NONE`: 表示前一轮 GC 已经完成，进入 `root_scan_phase()` 将根对象 mark，并 `flip_white_par()`，将两种白色切换，准备进入 Mark 阶段。
- `GC_STATE_MARK`: 执行 `incremental_marking_phase()` ，如果 `gray_list` 已经空了，再去扫描 `atomic_gray_list` 单独处理 Write Brrier 标记出来的灰色对象 [4]，完成 `final_marking_phase()`，准备进入 Sweep 阶段。
- `GC_STATE_SWEEP`: 执行 `incremental_sweep_phase()`，回收上一轮 GC 留下的死对象，之后将黑色对象标记为白色。完成之后将状态恢复为 `GC_STATE_NONE`。

limit 参数在 Mark 阶段和 Sweep 阶段，分别指代一步渐进 GC 中标记或者回收的对象数目。在渐进式 GC 的一步中 (`incremental_gc_step()`)，这个对象数目是跟据用户可设置的 `gc_step_ratio` 字段计算出来的。此外，渐进式 GC 不会立即回收内存，所以在一步走完之后，会相应地提高下次回收的阙值。

`incremental_sweep_phase()` 的实现在直观上有点粗暴，遍历待 Sweep 的堆中的所有对象，如果是死对象就回收。一开始我怀疑这里可能有性能问题，但是画了个火焰图发现瓶颈并没在这里，在 `free()` 面前它的消耗根本不值一提，就不了了之了。

## 分代 GC

除去三色 GC 将一轮 GC 分派到多次执行以外，另一条渐进式 GC 的思路是缩小每一轮 GC 的范围。好处是两方面的：减少了一轮 GC 的时间，更加及时地回收了内存。分代 GC 也正是最经实战考验的 GC 实现。

分代 GC 假定新的对象存活时间要比老对象短，将 GC 分为两种：

- Minor GC: 仅标记/收集新对象，因为需要遍历的对象要相对少许多，从而减少了每一轮 GC 的时间
- Major GC: 将所有老对象恢复成新对象，免死金牌摘去，执行一次完整的收集

mruby 和 lua 都提供了一个分代模式的开关。不同在于 lua 将分代模式视为一个试验 feature，因为它没有性能数据支撑却引入了额外的复杂性而受到质疑，可能会被删除 [5]； 而 mruby 将分代模式默认开启，作为未来主要投入精力的方向，并且将分代模式的 major GC 按三色的方式分成了渐进的几步进行。

然而 mruby 及 lua 的分代 GC 模式并非 "严肃" 的分代实现，而是重用了三色 GC 的基础设施，没有加多少代码就提供了分代的支持，这也正是有趣的地方。在一轮 GC 之后仍存活的黑色对象即被视作老对象。也就是说，象征性地分了两代对象。

## mruby 的分代 GC

分代 GC 的逻辑在重用三色基础设施的同时，也在很大程度上把自己的逻辑和三色 GC 揉在了一起，在一开始理解起来并不是很容易。

在 `mrb_incremental_gc()` 中， mruby 会判断当前是否为 Minor GC，如果是，则调用 `incremental_gc_until(mrb, GC_STATE_NONE)` 一路跑到底，到 `incremental_sweep_phase()` 也有对 Minor GC 的特殊处理，那就是在回收的最后不会把黑色对象再重新标记为白色。

随着一轮又一轮 Minor GC，老对象会留下的越来越多，一旦超过了老对象的阙值 (`majorgc_old_threshold`)，就该触发 Major GC 了。 这时会触发 `clear_all_old()` 将当前的所有对象重置为白色，并设置一个奇怪的 `gc_full = TRUE;`  [6]。但是等等，`mrb_incremental_gc()` 里面对于 `is_major_gc()` 的判断仅仅是对 GC 完成之后的清理工作，这是怎么回事？

这个问题困惑了我好久。想明白 mruby 是把 Major GC 也做了渐进化处理，这就容易理解多了。 Major GC 就等同于一轮普通的三色 GC，所以在 `mrb_incremental_gc()` 只迈了简单的一小步，而不是立即把所有老对象回收。

## Pitfalls

cruby 那种最传统的 Mark-Sweep GC 实现中用户无需手工调用 Write Barrier 的幸福时代已经远去了。 mruby 的起点比 cruby 高得多，然而对于 cruby 的扩展开发者看来，强制使用 Write Barrier 却需要转变一下思路才好接受。

在扩展中每修改对象的引用，都要分外注意调用 Write Barrier，不然可能会有奇怪的 seg fault 出现，靠 core dump 难以定位问题的位置 (但可以定位到漏掉 Write Barrier 的对象类型)。

## Footnotes

- [1]: 或者说可控的粒度太粗
- [2]: 题外话，之前无意搜过一次 Map-Reduce 下遍历图的方法，发现正是三色图遍历算法，Map-Reduce 无副作用的的环境下不得不分成多次才能遍历一个图
- [3]: 此 "Write Barrier" 与形容乱许执行的那个内存栅栏一点关系也没有。
- [4]: 将 Write Barrier 标记成灰色的对象单独维护在 `atomic_gray_list` 里的原因还没有很想明白，哪位同学能帮忙解释下？
- [5]: https://love2d.org/forums/viewtopic.php?f=3&t=10887
- [6]: 怀疑这个 `gc_full` 不是说内存满了，而是说接下来执行的是一轮 *完整的* GC...

## References

- http://wiki.luajit.org/New-Garbage-Collector
