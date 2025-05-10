作者在 openai 工作， 前几个月一直在开发一个 k8s 的 scheduler plugin，来自定义抢占，使之更适合 ML 的 workload。

作者感到这个工作非常困难，每当作者觉得自己已经理解了这些之后，都在随后发现了更多的未知的细节。

scheduling framework 某种意义上是一个 leaky abstraction，需要了解 scheduler 内部的细节，才能很好的写插件。

作者会关注在 preemption 上，因为这是文档没有很好的描述的部分。

## Prerequisite Knowledge

- **Nominated Pod**: nominated Pod `p` 是一个抢占了更低优先级的一组 pod `v` 的 pod。经过抢占后，pod `p` 会在所有的 `v` pod 退出后，重新进入调度队列。A “nominated node” is the node on which the preemption took place. scheduler 会优先安置 `p` 到 "nominated node" 上，但并不保证；
-  **Pod Disruption Budget**: 一个 feature 让 k8s 去尽可能地避免抢占某些 pod；PDB object 有一个 label selector，允许标注一些 pod；

## Scheduling Framework Overview

![[Pasted image 20250505215850.png]]

每个 stage 的行为可以被插件和 k8s native plugin 修改。而且每个 stage 可以注册多个插件。

当前要调度的 pod `p` 在 scheduler 的 pod queue 中取出之后，会进入到 **scheduling cycle**。这个队列会按照 pod 的优先级进行调度，每次调度只关注一个 pod，最后将这个 pod 绑定到某一个 node 上。

### CycleState Object

CycleState 是一个线程安全的 key-value store。每个 scheduling cycle 之后，会创建一个空的 CycleState 对象。

每个调度周期的 CycleState 是被多个插件共享的，插件可以用它来共享内部状态，并用于多个阶段之间的通信。

### Filter Stage

每个 Filter Plugin 能够决定一个特定的 Pod 能否调度到一个特定的 node 上。

一个 pod 只能调度到一个 “feasible” 的 node 上，也就是通过了所有的 Filter plugin 的 node。

每个 Filter 插件相当于提供一个函数，取如下参数：

- `node` 对象
- pod 对象
- CycleState 对象

<mark>如果没有 feasible 的 node，那么 scheduler 会尝试执行 PostFilter 插件来执行抢占</mark>。

### PreFilter stage

**Note: This is the most confusing stage of the scheduling framework. It is also key to truly understanding preemption.**

PreFilter stage 会在 Filter stage 之前执行。PreFilter 插件的主要用途是，初始化需要的状态信息，并持久到 CycleState 中。

相应的 Filter 插件稍后会获取并使用此自定义状态来做决策。

在抢占过程中，当调度器搜索可以合法抢占哪些 Pod 时，此自定义状态也可以被修改。

如果你的 Filter 插件使用了任何依赖于 Pod 信息的自定义状态，则需要实现 AddPod 和 RemovePod PreFilter 回调。

> This ensures that preemption respects the restrictions imposed by your Filter plugins and won’t make a preemption decision that will lead to a violation of your constraints.

### PostFilter Stage

在 Filter Stage 失败之后进入。一般用于抢占，但是你实际上可以跑任意代码进来。

k8s 默认的抢占算法是 DefaultPreemption 这个 PostFilter。

### Binding Cycle

当 scheduler 认为一个 pod 可以调度到某一个 node 时，会进入到一个 “binding“ 状态。

Pod binding 是一个异步操作，主的调度循环并不会阻塞在这里。scheduler 会假设 Pod 在进入 binding cycle 之后即已开始 schedule 了。如果 pod 最后因为任意原因还是 binding 失败了，所有的 plugin 的 Unreserve 这个 rollback hook 都会被触发，来清理各自的自定义状态信息。

### Eating Own Dog Food

1. Resource requests in Kubernetes work via the [Fit](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/noderesources/fit.go) plugin which [implements](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/noderesources/fit.go#L252-L255) a Filter plugin to exclude nodes that lack the pod’s requested resources
2. Taints and tolerations functionality comes from the [TaintToleration](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/tainttoleration/taint_toleration.go) plugin which also uses a [Filter](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/tainttoleration/taint_toleration.go#L63-L74) plugin to ensure node taints are respected
3. Standard pod preemption comes from the [DefaultPreemption](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/defaultpreemption/default_preemption.go) plugin which is implemented using a [PostFilter](https://github.com/kubernetes/kubernetes/blob/89ab733760ac26f0c4c620f8c3d07103f02cefd2/pkg/scheduler/framework/plugins/defaultpreemption/default_preemption.go#L84-L105) plugin

k8s 内部的 requests 匹配来自 Fit 插件、taint/tolerantions 来自 TaintToleration 插件、抢占来自一个实现了 PostFilter 的 Default Preemption 插件。

## Deep Dive: Default Preemption

抢占可能是 scheduler 中最复杂的部分，因为它涉及多个调度 strage，并涉及多个没有文档的执行细节。

> The goal of default preemption is to find the optimal set of victim pods located on the same node that need to be removed in order for a higher priority pod `p` to use that node. This minimal set of victims must also preferably not violate PDBs, have minimal pod priority, and also create minimal pod churn once evicted

决定哪个 pod 可以抢占，可能是一个复杂的问题。

因为 Pod 之间可能存在复杂的依赖关系，比如一个 Pod `server` 和另一个共存的 Pod `database` 具有必须的亲和性。那么默认的抢占策略除了抢占该 Pod `database`，也需要同时移除另一个 Pod `server` 。

此外，任何自定义的插件都可以引入任意的调度依赖关系。

在执行抢占时，scheduler 必须尊重所有的这种野生依赖关系、只抢占最小集合的 Pod、尊重 Pod Disruption Budget、最小化被抢占的 Pod 的优先级。

#### Preemption Control Flow

1. scheduler 从优先队列中，pop 出来 `p`，开始调度周期；
2. scheduler 对所有的 node objects 和 pod objects 打一个快照
3. PreFilter stage
	1. 执行所有的 PreFilter plugin，一部分可以决定持久它们的状态到 `CycleState` 中；
	2. 任意的 PreFilter plugin 也可以决定 fail，使该 pod 的整个 scheduling cycle 退出；
4. Filter stage
	1. 并行执行所有针对 `p` 的 Filter plugin
	2. 只要有一个 Filter plugin 针对特定 node 表示通过，那么就认为可以执行
5. <mark>如果没有可用的节点，那么执行 `DefaultPreemption` 的 PostFilter</mark> 
	1. 首先 `findCandidates`搜索合法的 eviction candidates（Eviction candidates 意思是一组位于同一个节点上的 victim Pod，如果移除掉，那么会允许 `p` 在这个节点上执行）
		1. `DryRunPreemption` 会尝试在整个集群中搜索 eviction candidates；会并行地对每个节点执行 SelectVictimsOnNode；
		2. 每个对特定 node 的 `SelectVictimsOnNode` 调用，如果成功，均会返回一个 eviction candidate C；
	2. `SelectCandidate` 会从 `findCandidates` 的列表中，选择出来最好的 eviction candidate `B`，这个有最少的 PDB violation、更低的 priority pods 等等；
	3. `prepareCandidate` 针对 `B` 执行真正的抢占；
		1. 删除 B 中的 victim pods，清理
		2. 发出一个 `Preempted` event
6. 如果有成功的抢占，PostFilter plugin 返回 nominated node `nm`；
7. scheduler 设置 Pod `p` 的 `.status.norminatedNodeName` 为 `nn`，同时也在 scheduler 的 local cache 中跟踪这个 normination 关系；
8. Pod `p` 重新入队，因为在 preempted victim 完成退出并释放资源之前，它并不能成功调度；

