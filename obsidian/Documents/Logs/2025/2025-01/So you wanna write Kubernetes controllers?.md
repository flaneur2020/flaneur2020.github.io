## Design CRDs like Kubernetes APIs

如果希望开发出来一套生产级别的 controllers，那么需要深入理解 k8s 的 API Convention。

没有对 API Convention 有体感的新人容易犯如下错误：

1. 不理解 spec 和 status 的区别，应该在什么时候修改谁的字段；
2. 不理解怎样 embed 一个子对象到 parent object 中，比如（Deployment.spec.template 怎么成为一个 Pod 的），因此会在父对象中重复子对象的字段
3. 对字段的语义理解不大够，比如 zero values、defaulting、validation 等

可以通过学习 Knative、Istio 以及其他的流行的 controller，来学习怎样组织字段，以及怎样重用 k8s 已经提供的原语（比如 ControllerRevision、PodSpecTemplate）

## Single-responsibility controllers

> I recommend studying the common [controller shapes](https://youtu.be/zCXiXKMqnuE?t=539) (great talk by Daniel Smith, one of the architects of kube-apiserver) and the core Kubernetes controllers. You’ll notice that each core controller in Kubernetes core has a very clear job and inputs/outputs that can be explained in a small diagram. If your controller isn’t like this, you’re probably misarchitecting either your controller, or your CRDs.

## Reconcile() method shape

假设你在使用 kubebuilder（它使用了 controller-runtime 来定义 controller），通常会有一个 `Reconcile()` 函数会在每次 input 改动之后调用。

在大型项目比如 Knative 中，他们通常定义了自己的 common controller shape，其中每个 controller 都按同样的顺序执行同样的 step；

kubebuilder 对这个 shape 并没有很 opinionated，不过作者发现，几乎所有的 controller 都有类似的形状：

```go
func (m *FooController) Reconcile(..., req ctrl.Request) (ctrl.Result, error) {
    log := ctrl.LoggerFrom(ctx)
    obj := new(apiv1.Foo)
    // 1. Fetch the resource from cache: r.Client.Get(ctx, req.NamespacedName, obj)
    // 2. Finalize object if it's being deleted
    // 3. Add finalizers + r.Client.Update() if missing

    orig := foo.DeepCopy() // take a copy of the object since we'll update it below
    foo.InitializeConditions() // set all conditions to "Unknown", if missing

   // 4. Reconcile the resource and its child resources (the magic happens here).
   //    Calculate the conditions and other "status" fields along the way.
   reconcileErr := r.reconcileKind(ctx, kcp) // magic happens here!

    // 5. if (orig.Status != foo.Status): client.Status.Patch(obj)
}
```

作者比较建议在公司内约定好一个统一的 shape。
## Report `status` and `conditions`

作者几乎没有在任何一个新人那里看到正确设计的 `status`。

在 linkedin 中，custom API 都有一个 status.conditions 字段，会使用一个类似 Knative condition set manager 的东西来提供统一的 accessor 函数包装来设置 conditions。

效果类似：

```go
func (r *FooReconciler) reconcileKind(obj *Foo) errror {
    // Create/configure a Bar object, wait for it to get Ready
    if err := r.reconcileBar(obj); if err != nil {
        obj.MarkBarNotReady("couldn't configure the Bar resource: %w", err)
        return fmt.Errorf("failed to reconcile Bar: %w", err)
    }
    obj.MarkBarReady()
}
```

每次 mark 一次 condition，condition manager 就会重新计算一次，top-level 的 `Ready` condition 是否成立。

## Learn to use `observedGeneration`

condition 中有一个 `observedGeneration` 字段。

这个字段能够告诉我们这个 condition 是不是基于对象的最新 generation 而计算出来的。

因此，单纯 `Ready` 为 true 还不行，还要判断 `cond.observedGeneration == metadata.generation`。不然，`Ready` 的状态可能是 stale 的。

## Understand the cached clients

> However, if you try to make queries with `client.{Get,List}` on resources that you haven’t declared upfront in your controller setup, controller-runtime will initialize an informer on-the-fly and block on warming up its cache. This leads to issues like:
>
> - Controller-runtime starting a watch for a resource type and start caching all its objects in memory (even if you were trying to query only one resource), potentially leading to the process running out of memory.
> - Unpredictable reconciliation times while the informer cache is syncing, during which your worker goroutine will be blocked from reconciling other resources.
> 
> That’s why I recommend setting `ReaderFailOnMissingInformer: true` and disabling this behavior so you’re fully aware of what kinds your controller is maintaining watches/caches on. Otherwise, controller-runtime doesn’t provide any observability on what informers it’s maintaining in the process.

## Fast and offline reconciliation

Reconcile 一个最新的对象，应当足够快且离线，不产生任何 API Call。

作者有见过一些 controller，在没有变化发生的情况下，仍然会产生 APi 调用。这是一个 anti pattern。

> - They bombarded the external APIs with unnecessary calls during controller startup (or full resyncs, or when they had bugs causing infinite requeue loops)
> - When the external API was down, reconciliation would fail even though nothing has changed in the object. Depending on the implementation, this can block the next steps in the reconciliation flow even though those steps don’t depend on this external API call.
> - Logic that takes long to execute in a reconciliation loop will hog the worker goroutine, and cause workqueue depth to increase, and reduce the throughput/responsiveness of the controller as the worker goroutine is occupied with the slow task.

如果你有一个 s3 bucket controller，如果每次 reconcile 都产生 s3 api 调用，那么一定是做错了。正确的做法应该是带一个类似的 `status.observedGeneration` 并只有在这个数字不匹配时，才更新。

> Let’s go through a concrete example: Assume you have an S3Bucket controller that creates and manages S3 buckets using AWS S3 API. If you make a query to S3 API on every reconciliation, you’re doing it wrong. Instead, you should store the result of the S3 API calls you made, in a field like `status.observedGeneration`, to reflect what’s the last generation of the object that was successfully conveyed to S3 API.

## Reconcile return values

`Reconcile` 会在每次 `builder.For,Owns,Watches` 更新时调用。关于返回值，作者一般的建议是：

1. 如果在 reconcile 中有遇到错误，那么就返回 error，而不要 `Requeue: true`；
2. 只有当没有错误，但 `something you started is still in progress` 时，开启 `Requeue: true`；
3. 只有在做基于时间的调度时，才使用 `RequeueAfter: <Time>`，比如做 `CronJob`；

在每次 Reconcile 中，只执行一小步，还是执行尽可能多的修改，都取决于偏好。但是你可能发现前者可能对 unit test 更友好；

## Workqueue/resync mechanics

## Expectations pattern

> Cached clients don’t offer [read-your-writes consistency](https://arpitbhayani.me/blogs/read-your-write-consistency/).

> In this case, controllers need to do in memory bookkeeping of their expectations that resulted from the successful writes they made. Once an expectation is recorded, the controller knows it needs to wait for the cache to catch up (which will trigger another reconciliation), and not do its job based on the stale result it sees from the cache.
> 
> You can see many core controllers [use this pattern](https://github.com/kubernetes/kubernetes/blob/a882a2bf50e630a9ffccbd02b8f759ea51de1c8f/pkg/controller/controller_utils.go#L119-L132), and Elastic operator also has a [great explanation](https://github.com/elastic/cloud-on-k8s/blob/6c1bf954555e5a65a18a17450ccddab46ed7e5a5/pkg/controller/common/expectations/expectations.go#L16-L78) alongside their implementation. We implemented a couple of variants of these at LinkedIn ourselves.

controller 需要记住预期的修改内容，然后等待 cache 追上，这是一个很常见的 pattern；