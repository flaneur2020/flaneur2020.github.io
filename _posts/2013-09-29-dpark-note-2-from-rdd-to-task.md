---
layout: post
title: "Dpark Note 2: from RDD to Task"
---

为生产环境设计的 dpark 虽然构建于 Mesos，然而它并非完全绑定于 Mesos。除了 Mesos ，dpark 也提供了多线程与多进程作为后端。这样做很有意义，最起码使代码更加容易测试。

dpark 与 Mesos 交互的逻辑比较复杂，不如放到下一篇单独记录。在这篇 blog 里，只简单记录一下 dpark 与 Mesos 无关的部分。


## RDD

RDD 可以说是 spark 那篇论文的主题。Hadoop 总是将有依赖关系的计算之间的临时数据存储于分布式文件系统，利用文件系统的 Replication 提高数据的可靠性。然而分布式文件系统是一个共享内存模型，较强的一致性使得同步成本也相对更大。对此 spark 提供了更加讨巧的设计：会丢数据是既成事实，但是我们对数据的计算规则一清二楚，当丢数据时，只消重新计算再传一遍就好了，没必要维护临时数据的一致性。效果惊人的好。[^1]

对应的实现就是 RDD (Resilient Distributed Datasets)。dpark 官方文档的描述非常好，就直接贴过来吧：

> * 有两种方式可以产生 RDD，通过特定函数从存储设备（内存或硬盘）创建，或者由其他 RDD 生成
> * RDD 带有它所依赖的其他 RDD 的信息，以备计算失败的时候能够重新计算
> * RDD 是只读的，这样从拓扑中恢复这个 RDD 的工作就能简单很多
> * RDD 可以被重复使用
> * 一个 RDD 由多个 Split 组成，Split 是执行并行计算的基本单位
> * RDD 支持两类操作，窄依赖的 map 和 宽依赖的 reduce

使用的注意事项在官方文档中也很详细，在此掠过了。

不过，虽说 RDD 是核心概念，但是从理解代码的角度，还是顺着代码的执行路径走更加容易。接下来顺着代码的执行路径，尝试整理一下从 RDD 到 Task 的转换流程。

## 样例

就拿这段代码作为例子，修改自 dpark 的 README:

```
from dpark import DparkContext
dpark = DparkContext()

wc = dpark.textFile("/tmp/words.txt") \
    .flatMap(lambda x:x.split()) \
    .map(lambda x:(x,1)) \
    .reduceByKey(lambda x,y:x+y) \
    .collectAsMap()
print wc
```

执行:

```
$ python wc.py # 单进程执行
$ python wc.py -m process # 多进程执行
$ python wc.py -m host[:port] # mesos 集群执行
```

## context.py: 启动

DparkContext 正是面向用户的接口类，在文档中见到的函数比如 textFile(), parallelize(), accumulator(), broadcast() 等等，都是在这里面定义的。它是一个单例，在创建 RDD 之余，也会在这里解析命令行的参数，初始化对应的后端，并维护整个计算过程的生命周期。

RDD 是惰性的，在 DparkContext 中创建出来了 RDD ，随后一系列变形，比如 map(), flatMap(), filter()，都只是生成新的 RDD 而已，只有必须返回结果的操作才会真正执行计算，比如 count(), toList(), take(), collect(), reduce()。

对比一下 map 和 count：

```
def map(self, f):
    return MappedRDD(self, f)

def count(self):
    return sum(self.ctx.runJob(self, lambda x: sum(1 for i in x)))
```

可见真正的执行还是调用的 DparkContext 对象的 runJob() 方法，而 DparkContext#runJob() 仅仅是对 Scheduler 对象的 runJob() 方法的一层薄薄包装。

## rdd.py: RDD, Split, Dependency

RDD 的子类一般覆盖这几个方法：

```
class RDD:
     def compute(self, split): pass
     def preferredLocations(self, split): pass
     def splits(self): pass
     def __setstate__(self, state): pass # 反序列化
     def __getstate__(self): pass # 序列化
```

一个 RDD 可能会在多台机器上并行执行，计算划分的基本粒度就是 Split。每个 Split 可以有自己的 preferredLocations()， 表示优先考虑的节点。比如带有 Replication 的分布式文件系统会把一个数据块冗余地存储在多个节点上，在拥有数据块的节点上执行相关的计算显然最优。因变换产生的 RDD 可以继承父 RDD 的 Split，比如 TextFileRDD 按照文件的块划分了多个 Split 对应多个存储节点，到 MappedRDD 按照相同的 Split 划分就地执行计算是合理的。

RDD 的父 RDD 是谁显而易见，但是经过复杂的变换之后，RDD 中 Split 的 "父 Split" 的是谁就不能一下子说清楚了。 Spark 为此设计了 NarrowDependency 和 ShuffleDependency，并基于它们派生了多种 Dependency 类，分起类来主要是两类情景：

NarrowDependency 约定相互依赖的两个任务允许在同一个节点上执行。它的子类都实现一个 getParents(self, pid) 方法，其中pid 的全称是 partition id, 基本是 RDD 中 splits 数组的下标。比如最简单的 MappedRDD 派生自 DerivedRDD，而 DrivedRDD 与父 RDD 的依赖关系 OneToOneDependency 仅仅将父 RDD 中对应 Split 的下标原样返回：

```
class OneToOneDependency(NarrowDependency):
    def getParents(self, pid):
        return [pid]
```

ShuffleDependency 会打乱数据的分布。它并没有提供 getParents() 方法的实现，到后面不乏针对它的特殊处理。比如 reduceByKey() 操作会按照 key 将数据重新分布，最初的来自于分布式文件系统的数据分布就不再奏效了。

这要求子 RDD 主动向父 RDD 寻求计算的结果，就引出了一个问题：RDD 是静态的，对未来将在哪个节点上执行一无所知，那么子 RDD 中派生出来的任务该怎样寻找目标节点？Spark 的解决方案是给每个 ShuffleDependency 一个唯一的 shuffleId，在 Master 节点开一个简单的 key-value 存储，即 mapOutputTracker，到任务执行完毕之后，会把 (shuffleId, 所在的节点) 注册进去，子 RDD 的任务在获取中间数据之前凭 shuffleId 为 key 查询 Master 即可获得目标节点。

## DAGScheduler

RDD 定义了执行的逻辑和依赖，但最终交付到计算阶段的是 Task 。从 RDD 到 Stage 到 Task，就像编译器的多层中间表示，生成最终结果之前经过多轮转换。而这些转换，几乎都是 DAGScheduler 的工作。

DAGScheduler 的父类 Scheduler 类只起到一个接口的作用:

```
class Scheduler:
    def start(self): pass
    def runJob(self, rdd, func, partitions, allowLocal): pass
    def clear(self): pass
    def stop(self): pass
    def defaultParallelism(self):
        return 2
```

此外，各 Scheduler 的继承结构如下：


* Scheduler
  * DAGScheduler
    * LocalScheduler
      - MultiProcessScheduler
    * MesosScheduler

DAGScheduler 实现了任务调度的通用逻辑，将 RDD 拆成多个 Stage，使每个 Stage 中的计算各不依赖，并按照依赖顺序依次发送相应 Stage 中的所有 Tasks。其间它会多次调用 submitTasks(self, tasks) 方法，这也正是 LocalScheduler, MultiProcesScheduler 和 MesosScheduler 给出不同实现的主要地方。

DAGScheduler 中最主要的方法是 runJob(self, finalRdd, func, partitions, allowLocal)，它的方法体有我两个头那么长，不过简单来说做的事情就是将 RDD 转换为多个 Stage，并在 Stage 的层面维护计算的生命周期，最终返回一个 generator。各个参数含义如下：

* finalRdd: 特指最终触发求值的 RDD
* func: 取 iterator 为参数的函数
* partitions: 当前关心的 Partition/Split 的下标集合，默认为 RDD 的 Split 数，即 len(rdd)
* allowLocal: 是否允许本地执行

runJob() 首先依据 finalRdd 初始化 finalStage，并且定义了一些变量用于统计执行进度以及返回结果，比如：

* finished: 执行完毕的 Stage
* waiting: 等待中的 Stage
* failed: 执行失败的 Stage
* pendingTasks: 内容为 Stage => [Task.id] 的字典
* numFinished: 已执行完毕的 Stage 数
* lastFetchFailureTime: 网络失败数

然后调用 updateCacheLocs()，清理 preferredLocations 的缓存，先不理会它。重点看后面主要的两步：submitStage() 和事件循环。

## Stage 的提交与生命周期

submitStage(stage) 会递归地检查 Stage 的父 Stage，并将其加入 waiting 状态。直到发现没有 Parent 或者没有执行完毕的叶子 Stage，调用 submitMissingTasks(stage) 为 RDD 的每个 split 创建单独的 Task，并一股脑提交出去，这便是第一批任务。随后就都是在它们执行完毕之后，再提交依赖于它们的子任务了。这些任务的类型会是 ShuffledMapTask，到最后发送的来自 finalStage 的任务则属于例外情况，类型为 ResultTask，它仅仅起到一个标记的作用，方便到后面对最终计算结果做特殊处理。

然后就进入了一个以 numFinished != numOutputParts 为不变式、侦听 self.completionEvents 这个消息队列的事件循环，用来维护 Stage 的生命周期。其中关心的事件仅有 Success 和 FetchFailed，前者来自 self.taskEnded(task, reason, result, update)，为 Scheduler 后端触发；后者来自网络异常，处理必要的重试或者报错。事件中带有 task 与 reason 两条信息，通过 task 的 stageId 字段，可以查表得到对应的 Stage 对象。

收到成功的事件 [^3]，如果任务的类型为 ResultTask，就有可能返回 evt.result 作为结果了：

```
if isinstance(task, ResultTask):
    finished[task.outputId] = True
    numFinished += 1
    results[task.outputId] = evt.result
    while lastFinished < numOutputParts and finished[lastFinished]:
        yield results[lastFinished]
        results[lastFinished] = None
        lastFinished += 1
```

这段逻辑有点绕，我猜测是出于保序以及排重的需求：将各 partition [^2] 的计算结果按照按顺序依次 yield，同时排除因重试导致的多次返回。

如果任务类型为 ShuffleMapTask，做的事情更多一些：

```
elif isinstance(task, ShuffleMapTask):
    stage = self.idToStage[task.stageId]
    stage.addOutputLoc(task.partition, evt.result)
    if not pendingTasks[stage] and all(stage.outputLocs):
        logger.debug("%s finished; looking for newly runnable stages", stage)
        running.remove(stage)
        if stage.shuffleDep != None:
            self.mapOutputTracker.registerMapOutputs(
                    stage.shuffleDep.shuffleId,
                    [l[-1] for l in stage.outputLocs])
        self.updateCacheLocs()
        newlyRunnable = set(stage for stage in waiting if not self.getMissingParentStages(stage))
        waiting -= newlyRunnable
        running |= newlyRunnable
        logger.debug("newly runnable: %s, %s", waiting, newlyRunnable)
        for stage in newlyRunnable:
            submitMissingTasks(stage)
```

首先登记计算结果所在的节点名，设置 outputLocs[task.partition] = evt.result。待 all(stage.outputLocs) 为真，这个 stage 就执行完毕了，stage.isAvailable() 会变为真，getMissingParentStages(stage) 返回的结果会相应地有所变化。执行完毕之后，会将 stage 移除 running 状态，登记 (shuffleId, 结果所在的节点) 到 mapOutputTracker，以 not getMissingParentStages(stage) 为条件在 waiting 集合选出下一轮待执行的 stage，并提交。

留意与 ResultTask 的 evt.result 表示最终计算结果的值不同，在这里 evt.result 中保存的是计算结果所在的节点，配合 shuffleId 可获得文件完整的 URL。因为中间的结果比较大，会在该计算节点上就地保存成文件，在需要时由子任务去主动访问。留意这里的文件并不会保存到分布式文件系统，而是放在 dpark 自己维护的一个 HTTP File Server 里面，允许文件通过全局的 URL 标识。这可能是性能的关键。

有点困了，FetchFailed 事件的处理就先略过，毕竟属于异常流程，看代码先重点关注正常流程。

## Footnotes

1. 大家都说 spark 比 hadoop 快 15 倍? 挺想了解下快了这么多的代价是什么
2. Partition 和 Split 的概念在某种程度上等价
3. 先忽略 Accumulator 的逻辑，不难理解，但需要占一自然段的篇幅

## Reference

* https://github.com/douban/dpark/blob/master/docs/guide_full.rst
