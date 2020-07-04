---
layout: post
title: "Kotlin Coroutine 的 CPS 变换"
---

一般 coroutine 的实现大多基于某种运行时的 yield 原语，然而 kotlin 并没有这种下层支持，在没有 yield 指令支持的 jvm 上仍基于 CPS 变换实现出了 coroutine，其实很厉害，想了解一下是怎样做到的。

## CPS Transformation

CPS 是 Continuation Passing Style 的缩写，大致上是将原本长这样的代码：

```
function auth(token, resourceName) {
  let userId = loginUser(token)
  let ok = checkPermission(userId, resourceName)
  return ok ? 'success': 'failed'
}
```

转换为这个画风：

```javascript
function auth(token, resourceName, callback) {
  loginUser(token, function(userId) {
    checkPermission(userId, resourceName, function(ok) {
      callback(ok ? 'success': 'failed')
    })
  })
}
```

也就是俗称的「callback hell」，这种风格一度曾在 node js 中发扬光大了一把。大致上是将本来函数该返回的值通过参数中的回调函数进行返回。其中「continuation」 词义较晦涩，实际上它大约就是「callback」的同义词。

流程上大致是：

1. 调用异步方法，注册回调函数到 io loop；
2. io loop 确认 io 事件（如收到服务端响应）后，回调注册在 io loop 中的回调函数，在参数中传递异步返回的结果，如果有异常，也同样在回调函数的参数中；
3. 回调函数继续调别的回调函数，调用到新的异步方法时注册新的回调函数到 io loop。

真正驱动整个控制流的是 io loop，即调度器。

回调函数的问题是缺少结构化，每一次异步调用，都必然涉及到异常处理流程，这代码就没法看了。Future / Promise 在结构化方面能稍微进步一些，对 onSuccess(result) 与 onError(err) 进行标准化的包装，这一来代码结构方面能相对漂亮一点：

```javascript
function Promise auth(token, resourceName) {
  loginUser(token)
    .then(userId => checkPermission(userId, resourceName))
    .then(ok => ok ? 'success': 'failed')
    .catch(err => console.log(err))
  })
}
```

到这里朝 async / await 式语法的演进还差一步，便是 yield。与回调函数风格异步方法的不同在于，async / await 模式下将函数上次执行的点位，作为回调入口登记到 io loop。io loop 调度器发现有收到响应，便找出对应的函数，按 resume(result) 推动后续的执行。

但是 jvm 明显没有 yield 支持，那么 kotlin 是怎样做到的？

## CPS in kotlin
回来看 kotlin。异步方法在 kotlin 中增加 suspend 关键字做标记。

```kotlin
suspend fun auth(token: String, resourceName: String): String {
  val userId = loginUser(token)  // suspending fun
  val ok = checkPermission(userId, resourceName)  // suspending fun
  return ok ? 'success': 'failed'
}
```

在编译时，大约会将 suspend 方法做 CPS 转换，增加一个 cont 参数，用于回调返回值或者异常:

```
suspend fun auth(token: String, resourceName: String, cont: Continuation<Any?>) {
  ...
}
```

其中的 Continuation 便是个类似 Promise/Future 的接口：

```
public interface Continuation<in T> {
  public val context: CoroutineContext
  public fun resume(value: T)
  public fun resumeWithException(exception: Throwable)
}
```

其中 context 表示协程上下文，这里先略过。resume 与 resumeWithException 与 js 中 Promise 的 then() 与 catch() 完全一致。到这里缺少的就是 yield 的替代方案了。

## StateMachine
替代 yield 的方案便是 StateMachine，它扮演两个功能：1. 暂停并记住执行点位；2. 记住函数暂停时刻的局部变量上下文。

每个 suspend 函数都会生成出一个内部类 StateMachine，用于保存函数的局部变量与执行点位。

怎样做到记录执行点位的？

首先所有的暂停点位，都必然是调用 suspend 函数的点位。kotlin 这里的做法很有意思，使用了一个 switch case，每个 suspend 执行的位置对应一个 label，下次恢复执行时，按递增的 label 找到下一个执行位置，大约的伪代码会长这样：

```kotlin
suspend fun auth(token: String, resourceName: String, cont: Continuation<Any?>) {
  val sm = cont as? AuthSM ?: AuthSM(cont)

  when (sm.label) {
    0 -> {
      sm.cont = cont
      sm.label = 1
      loginUser(token, sm)
      return
    }
    1 -> {
      throwOnFailure(sm.result)
      sm.userId = sm.result as Int
      sm.label = 2
      checkPermission(sm.userId, sm)
      return
    }
    2 -> {
      throwOnFailure(sm.result)
      sm.ok = sm.result as boolean
      sm.cont.resume(sm.ok ? "success": "failed")
    }
    else -> throw IllegalStateException(...)
  }
}
```

每个 StateMachine 对象也是 Continuation 接口的实现。每当 suspend 函数执行到 suspend 点位时，实际上会退出执行，函数的执行上下文会完整记录在 StateMachine 中，调用的异步方法有响应时，会回调 StateMachine 的 resume 方法，而 resume 方法的执行，相当于使用记录在 StateMachine 中的上下文作为参数，再执行一次该函数。

画图整理一下大致的流程：

![](/images/kotlin-coroutine-cps.png)

## 总结

- kotlin 的协程支持大约是 CPS 变换和 StateMachine 两部分，CPS 变换使同步代码异步化，增加额外的 Continuation 类型的参数，用于函数结果值的返回；
- 通过 switch / case 配合 label，做到执行点位的记录与暂停；
- 每个 suspend 方法都会生成一个内部的 StateMachine 类，StateMachine 类中包含函数的所有局部变量，以及暂定点的返回值与异常值，扮演了 yield 语句的功能，即暂停函数的执行，为此需要记录下函数当前的局部变量上下文与执行点位。
- 被暂停函数的恢复执行，实现上等于将函数的局部变量上下文与点位作为参数，重新调用一次这个函数。

## References

- https://medium.com/androiddevelopers/the-suspend-modifier-under-the-hood-b7ce46af624f
