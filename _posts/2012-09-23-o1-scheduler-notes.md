---
layout: post
title: "O(1) Scheduler Notes"
---

上周发觉看文件系统到了一个小瓶颈，不如转移下注意力，先看下内核的调度算法好了。而且手头正好有Robert Love的书，zet君每提到这本书都对它的调度算法部分赞不绝口，值得仔细看看。

但是一下子又没看懂 :( 

不如回头从O(1) Scheduler开看好了，记一点笔记在这里。

需要注意的是，O(1) Scheduler是在2.5中引入，距今已十年多了。

-----

## 调度的不同情景

操作系统提供的多任务支持，很大程度上就是调度算法本身了。然而多个任务并非生而平等：有的任务希望响应时间尽可能短，有的任务希望整体性能尽可能高。有的任务对IO密集(IO bound)，有的任务对计算密集(CPU bound)，也有的任务对实时有所要求。

对调度算法而言，如何才能做到公正，其实是个两难的问题：

+ 如果追求第一时间的响应速度，那么进程切换的开销将降低任务的整体性能。
+ 如果追求任务的整体性能，那么用户可能需要等待很久才能看到任务的结果。

就实际场景而言：

+ 应用程序会更想得到响应速度：用户点击了按钮，应该尽可能地给个反馈，不然用户会觉得"死机"了<sup>1</sup>。至于运算的性能则通常不是大问题，时间长一点没关系，给用户看一个进度条就行了。
+ 超级计算机会想要得到更高的整体性能：尽快的算出来结果就好，几乎没有用户交互。
+ Web Server的期望则介于两者之间：必须重视整体性能，不然会被高负载压垮；用户不会频繁地交互，但是用户依然希望及时得到响应：假如100个用户同时想下载文件，正确的做法该是同时响应这些用户的下载请求，而非为了整体性能而传完一个传下一个<sup>2</sup>。
+ 此外，实时应用比如音乐播放器，则希望定时往声卡的buffer中填上后几秒的音频数据，哪怕系统再忙，音乐被打断终究是不好的。

此外，SMP与NUMA的存在又让问题更加复杂了一分。

在这个时代，Linux已无法打出Keep It Simple的旗号去躲避问题——它必须面对所有的问题：低端Android手机也需要跑计算密集的垃圾收集；超级计算机也需要有一个终端来交互。

哪怕问题之间有矛盾，哪怕舍此不能及彼，Linux也需要有一个四海皆通的方案，给上述的所有情景提供一个较好的支持，允执厥中。Linux不是一个Do one thing and do it well的内核，这点很重要。

## 数据结构

O(1) Scheduler中核心的数据结构只有两个，runqueue与优先级数组(priority array)。

每个CPU对应一个runqueue，每个runqueue包含两个优先级数组，一个是active priority array，装有还没跑完时间片的活跃任务；另一个是expired priority array，装有跑完时间片的任务。当active priority array中的任务跑光了时间片，就把它移动到expired priority，同时重新计算时间片。当active priority array中没有任务了，就简单地将active priority array与expired priority array的指针交换过来。

优先级数组是一组链表的数组，长度为140，每个优先级对应一条链表，也就是最多可以有140条链表了。相同优先级的任务链在同一条链表中，按照Round Robin调度。此外，优先级数组维护着一个bitmap，用以表示某一优先级对应的链表是否为空。这一来查找当前最高优先级的任务时，只需遍历固定大小的bitmap即可，最坏为常数时间。

由此可见，O(1) Scheduler设定了140个优先级。其中的前100个优先级皆为实时任务保留，后40个优先级供一般任务使用，保证实时任务的优先级永远比一般任务更高。

## 优先级

O(1) Scheduler将优先级分为静态优先级(static priority)与动态优先级(或者_有效优先级_，effective priority)，使得时间片的计算与静态优先级相关，而进程的抢占与动态优先级相关。

静态优先级就是平时的nice值了，默认为0，用户可以设置为-20到19之间。保存于`t->static_prio`。

动态优先级则根据nice值以及一些统计信息计算而来，目标是奖励IO密集型任务，而惩罚CPU密集型任务。保存于`t->prio`。

### 动态优先级的计算：启发式方法<sup>3</sup>

O(1) Scheduler会将一个任务平时的睡眠时间与执行时间统计在`sleep_avg`中：

+ 当任务唤醒时，将它刚刚的睡眠时间(纳秒)加在`sleep_avg`中；
+ 当任务放弃CPU控制权时，使`sleep_avg`减去任务刚刚的执行时间；

这一来，理论上任务的`sleep_avg`越高越IO密集，由此即可作为奖励还是惩罚的一个参照。

动态优先级由`effective_prio()`函数计算得出，它会首先判断任务的类型，若为实时任务，则直接返回任务的静态优先级不做处理；否则，依据`sleep_avg`计算bonus值(-5~5)，进而与静态优先级相加得出动态优先级，最后保证动态优先级在-19~20之间，公式如下：

    bonus = CURRENT_BONUS(p) - MAX_BONUS / 2
    prio = p->static_prio - bonus
    prio = max(min(prio, 20), -19)

其中:

    MAX_BONUS = 10 
    MAX_SLEEP_AVG = 1000
    CURRENT_BONUS(p) = NS_TO_JIFFIES(p->sleep_avg) * MAX_BONUS / MAX_SLEEP_AVG

### 反馈：Interactivity Credit

CPU密集的任务偶尔也会产生大量IO，但是据这一段时间的IO就将它当作IO密集型任务就上当了。对此，O(1) Scheduler额外统计了一个`interactive_credit`，作为避免这种误判的一个参照。

平时任务若睡眠了较长时间，则`interactive_credit`加1<sup>4</sup>；任务若执行了较长时间，则使`interactive_credit`减1<sup>5</sup>。

`interactive_credit`的值仅当大于100或者小于-100时才会发挥作用。若大于100，则认为是高Credit，进而被认为是IO密集而获得奖励；若小于-100，则认为是低Credit，进而被认为是CPU密集而得到惩罚。而奖励与惩罚的手段，就是在计算`sleep_avg`时，影响运行时间与睡眠时间的统计：

+ 若为高Credit，将任务的执行时间减少<sup>6</sup>；
+ 若为低Credit，则：

  + 在进程从较长时间的不可中断睡眠中唤醒时，忽略这段睡眠时间；
  + 睡眠时间最多只能计入一个时间片的时间。

## 时间片的计算

时间片的计算只与静态优先级相关：静态优先级越高，时间片越长。

时间片由`time_slice()`函数计算而来，公式大致如下：

    MIN_TIMESLICE = 10ms
    MAX_TIMESLICE = 200ms
    time_slice(p) = (MAX_TIMESLICE - MIN_TIMESLICE) * (MAX_PRIO - 1 - p->static_prio) / (MAX_USER_PRIO - 1) + MIN_TIMESLICE

设`p->static_prio`为0，计算可得时间片的默认长度为100ms。

跑完时间片的任务会被移入expired priority array等待下一次调度。

不过，100ms已经是相当长的一段时间了，对于同一优先级中交互频繁<sup>7</sup>的任务，无端等待100ms是不可接受的。对此，策略是约定任务跑完`TIMESLICE_GRANULARITY`(默认50ms)时进行一次Round Robin<sup>8</sup>，将当前任务移动到队列的尾部。从某种意义上讲，时间片并非O(1) Scheduler中最小的时间单位。

## `scheduler_tick()`

`scheduler_tick()`函数是O(1) Scheduler的心跳，由`do_timer()`触发，每1ms执行一次。它会将当前进程的`time_slice`值减一，并在必要时将进程放入expired priority array或者Round Robin，并设置`TIF_NEED_RESCHED`。

这里有个例外情况，对于交互频繁的任务，跑完时间片就将它放进expired priority array等待猴年马月的再次活跃将是不可接受的。跑完时间片并非意味着它不再活跃、不再关心交互的响应。

这里的Workaround是，如果是交互频繁<sup>7</sup>的任务跑完时间片，则尽可量将它重新插回active priority array。代价是有可能会饿到expired priority array中的任务，对此引入了一个宏`EXPIRED_STARVING(rq)`来判断expired priority array是否感到饥饿，如果已经饿了，再将跑完时间片的任务插入expired priority array。

需要留意的是，`scheduler_tick()`并不会直接调用`schedule()`进行任务切换，`schedule()`将在返回用户态时被调用<sup>9</sup>。

## schedule()

`schedule()`才是O(1) Scheduler中最主要的函数，这里放到最后，但不打算记太多关于它本身行为的内容了。而它所做的也十分简单：选出下一个任务，并切换。

相比之下，它被调用的情景则更值得记一下：

1. 主动放弃控制权，如mutex, semaphore, waitqueue等；
1. 若设置`TIF_NEED_RESCHED`并开启抢占，则在返回用户态时调用`schedule()`发生抢占；

-------------------------

## Footnotes

1. 事实上，我们常说的'死机'正是调度算法失败的场景。
1. 就像12306那个排队一样。
1. 启发式方法，即依照过往的统计数据作为经验，在下次求解中推测出一个较好的结果。
1. 见`recalc_task_prio()`，"较长时间"通过一个宏`INTERACTIVE_SLEEP(p)`来比较
1. 见`schedule()`
1. 见`schedule()` [[ref](http://lxr.oss.org.cn/source/kernel/sched.c?v=2.6.8;a=arm#L2225)]
1. "交互是否频繁"通过一个宏`TASK_INTERACTIVE(p)`来判断
1. [LWN: sched-2.5.66-A2, scheduler enhancements](http://lwn.net/Articles/26586/)
1. `scheduler_tick()`正位于中断上下文，如果发现有更高优先级的进程，一般都会立即执行`schedule()`，除非内核关闭了抢占

-------------------------

## References

+ Understanding the Linux 2.6.8.1 CPU Scheduler
+ Understanding Linux Kernel
+ Linux Kernel Development
+ LWN
+ http://permalink.gmane.org/gmane.linux.kernel/1337629 (or http://wangcong.org/blog/archives/2076)
