---
date: "2013-08-31T00:00:00Z"
title: Quick note on mruby GC
---

Recently, I went through the mruby GC and basically figured out the logic. Besides patching up some comments, I'll jot down a quick note here.

## Tri-color GC

If you've worked with Rails, you might remember that the main issue with traditional Mark-Sweep is the uncontrollable collection time [1]. If you're unlucky enough to catch a GC, page latency can be pretty high. mruby, like lua, targets an embeddable interpreter, so it naturally considers making GC incremental. Tri-color GC is the simplest and most correct way to implement incremental GC.

The idea of tri-color GC comes from Knuth's tri-color graph traversal algorithm [2]. Objects are divided into black, gray, and white, maintained in different sets. The process of a GC round is simple:

1. At the beginning, all objects are white. Start from the root object and mark it gray.
2. As long as there are objects in the gray queue, take one object, mark it black, and mark its directly referenced objects gray. Then you can pause (though continuing is fine; whether to pause the GC usually depends on certain thresholds).
3. Once all gray objects are gone, the white objects are the dead ones that can be collected.

To make the Sweep phase incremental, you can introduce another type of white and put it in a separate set. When the next GC starts, newly allocated objects are marked as this new white. The old white objects can theoretically be collected at any time without being disturbed by the new white objects. This means mruby GC always collects dead objects left over from the previous GC round during the Sweep phase.

Another ironclad rule is: Black objects cannot reference white objects. However, object reference relationships inevitably change during a GC round, such as newly allocated white objects needing to be referenced by black objects. For this, users need to manually call Write Barrier [3] to re-mark the black object as gray.

Each small step in the traversal is very small, so theoretically, tri-color GC is much friendlier to latency-sensitive applications. However, by distributing GC over multiple executions, the duration of a GC round is extended, and the accumulated memory usage will be relatively higher.

## mruby's Tri-color GC

In mruby, GC is triggered at the following times:

- `mrb_realloc_simple()` runs out of memory and triggers `mrb_full_gc()`.
- `mrb_obj_alloc()` allocates an object when the number of live objects exceeds a threshold, triggering `mrb_incremental_gc()`.
- When enabling/disabling generational mode, `mrb_full_gc()` is triggered to perform a full cleanup beforehand.

Both `mrb_full_gc()` and `mrb_incremental_gc()` wrap `incremental_gc(mrb, limit)`, which is the central function of mruby GC, maintaining the state of mruby GC like an automaton:

- If `gc_state` is `GC_STATE_NONE`: Indicates that the previous GC round is complete. Enter `root_scan_phase()` to mark the root object and `flip_white_par()` to switch the two whites, preparing for the Mark phase.
- `GC_STATE_MARK`: Execute `incremental_marking_phase()`. If the `gray_list` is empty, scan the `atomic_gray_list` to handle gray objects marked by Write Barrier [4], complete `final_marking_phase()`, and prepare for the Sweep phase.
- `GC_STATE_SWEEP`: Execute `incremental_sweep_phase()`, collect dead objects from the previous GC round, and then mark black objects as white. After completion, restore the state to `GC_STATE_NONE`.

The `limit` parameter in the Mark and Sweep phases refers to the number of objects marked or collected in one step of incremental GC. In one step of incremental GC (`incremental_gc_step()`), this number of objects is calculated based on the user-settable `gc_step_ratio` field. Additionally, incremental GC does not immediately reclaim memory, so after one step, the threshold for the next collection is correspondingly increased.

The implementation of `incremental_sweep_phase()` is somewhat brute-force, iterating over all objects in the heap to be swept and collecting dead objects. Initially, I suspected performance issues here, but profiling showed that the bottleneck wasn't here; the cost is negligible compared to `free()`.

## Generational GC

Apart from distributing a GC round over multiple executions with tri-color GC, another approach to incremental GC is to reduce the scope of each GC round. The benefits are twofold: reducing the time of a GC round and more timely memory reclamation. Generational GC is the most battle-tested GC implementation.

Generational GC assumes that new objects have shorter lifespans than old objects, dividing GC into two types:

- Minor GC: Only marks/collects new objects, since there are relatively fewer objects to traverse, reducing the time for each GC round.
- Major GC: Turns all old objects into new ones, removing their get-out-of-jail-free cards, and performs a full collection.

Both mruby and lua offer a generational mode switch. The difference is that lua treats generational mode as an experimental feature, as it lacks performance data support and introduces extra complexity, which may lead to its removal [5]; while mruby defaults to generational mode and sees it as a major focus for future efforts, breaking down the major GC of generational mode into progressive steps using the tri-color method.

However, mruby and lua's generational GC modes are not "serious" generational implementations; they reuse the tri-color GC infrastructure with minimal additional code to support generational features, which is where the fun lies. Black objects that survive a GC round are considered old objects. In other words, they symbolically divide objects into two generations.

## mruby's Generational GC

While reusing the tri-color infrastructure, the logic of generational GC also intertwines its own logic with that of tri-color GC, making it not so easy to understand at first.

In `mrb_incremental_gc()`, mruby checks if it's a Minor GC. If so, it calls `incremental_gc_until(mrb, GC_STATE_NONE)` to run through to the end, and `incremental_sweep_phase()` also has special handling for Minor GC, which is not to re-mark black objects as white after collection.

As Minor GCs continue, more and more old objects accumulate. Once the threshold for old objects (`majorgc_old_threshold`) is exceeded, it's time to trigger Major GC. This triggers `clear_all_old()` to reset all current objects to white and sets a peculiar `gc_full = TRUE;` [6]. But wait, the judgment for `is_major_gc()` in `mrb_incremental_gc()` is only for cleanup after GC completion. What's going on?

This puzzled me for a long time. Realizing that mruby processes Major GC incrementally makes it much easier to understand. Major GC is equivalent to a regular tri-color GC, so `mrb_incremental_gc()` only takes a small step, not immediately recycling all old objects. By memory, this is also a difference from lua's implementation.

## Pitfall 1: Write Barrier

The blissful era of users not needing to manually call Write Barrier in the most traditional Mark-Sweep GC implementation of cruby is long gone. mruby starts higher than cruby, but for cruby extension developers, being forced to use Write Barrier requires a shift in mindset to accept.

Every modification of object references in extensions must be格外注意调用 Write Barrier, otherwise, strange seg faults may occur, and locating the issue solely through core dump is difficult (though it can pinpoint the object type where Write Barrier was missed).

## Pitfall 2: Arena

mruby's GC does not scan the C stack, so for C extension developers, how to avoid tragedies like this?

```
a = mrb_str_new(mrb, "a", 1);
b = mrb_str_new(mrb, "b", 1);
mrb_str_concat(mrb, a, b); // What if memory is tight and either a or b gets recycled?
```

mruby's solution is to add an array in the `mrb_state` structure to store references to recently new objects, pushing each new object into this array. During GC, it first marks objects referenced in this array, thus avoiding the recycling of temporary objects. When a function execution ends, its temporary objects no longer need extra protection, and the array can be reverted to its state before execution. Its name is a bit odd: `arena`, but it has no connection with the `arena` concept in memory allocators like dlmalloc.

However, it brings a new problem: if a function internally allocates many objects, it can cause the arena array to overflow. Therefore, C extension developers need to格外注意,务必配合调用 `mrb_arena_save()` 与 `mrb_arena_restore()`,及时地回退 arena. Issue [#1533](https://github.com/mruby/mruby/issues/1533) is an example of this problem.

## References

- http://wiki.luajit.org/New-Garbage-Collector
