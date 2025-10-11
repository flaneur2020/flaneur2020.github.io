
在 Sync Rust 中，要 cancellation 只能设置一个 bit flag 不停地去查，很麻烦；

在 async Rust 中，取消有点过于简单了：只要不 poll 一个 Future，就取消掉了。

但是，取消了父 Future，所有的子 Future 多会跟随被 cancel。要看这个子 future 被 cancel 是否会导致问题，你需要看它关联的那些 Future 的情况。

去理解 cancel 导致的问题，不能在局部地分析，而需要关注全局。

> To figure out whether a child future’s cancellation can cause issues, you have to look at its parent, and grandparent, and so on. Reasoning about cancellation becomes a very complicated _non-local operation_.

## Analyzing cancellations

### Cancel safety and cancel correctness

cancel safety 的意思是你可以随意 cancel 一个 future，不会有任何 side effect；

比如说，sleep 是 cancel safe 的。

而 mpsc channel 的 send 方法并不 cancel safe。

```rust
let message = /* ... */;
let future = sender.send(message);
drop(future); // message is lost!
```

如果这个 future 被 cancel，那么这个消息就永远丢失了。

如果你的程序中，丢失一个 message 也没有关系，那么也不见得是一个 bug，并不影响程序的 correctness。

那么什么时候会违反 cancel correctness？

1. 系统中有一个 cancel-unsafe 在里面；已知，很多 API 都不能满足 cancel safe；如果系统中没有任何 cancel unsafe 的 future，那么系统就能满足 cancel correctness；
2. cancel-unsafe 的 future 被 cancel 了；如果 cancel unsafe 的 future 总是能够跑到结束，那么这个系统就不会有 cancel correctness 的 bug；
3. cancel 这个 future，会影响系统中的一些属性；比如 `Sender::send` 会导致数据丢失，或者有某些必须要执行的 cleanup 逻辑却没有执行到；

### The pain of Tokio mutexes

tokio 的 mutex 是一个典型的非常容易产生 cancellation correctness bug 的例子。

它的文档中有写：

> This method uses a queue to fairly distribute locks in the order they were requested. Cancelling a call to lock makes you lose your place in the queue.

单看文档，它并不会影响彻底的 unsafe，只是影响 fairness，但是真正的问题在于我们在 mutex 中做什么。

主要的问题是，我们使用 mutex 时，大多数情况下，我们都是在假设这是一个原子的临界区。

但是 future 的 cancel，会破坏这个假设，导致一个临界区执行不完整。

```rust
let guard = mutex.lock().await;
// guard.data is Option<T>: Some to begin with
let data = guard.data.take(); // guard.data is now None

// DANGER: cancellation here leaves data in None state!
let new_data = process_data(data).await;
guard.data = Some(new_data); // guard.data is Some again
```

### Cancellation patterns

#### try join

```rust
async fn do_stuff_async() -> Result<(), &'static str> {
    // async work
}

async fn more_async_work() -> Result<(), &'static str> {
    // more here
}

let res = tokio::try_join!(
    do_stuff_async(),
    more_async_work(),
);

// ...
```

如果其中的一个 future 返回 error，其他的所有 future 就都会被 cancel 掉。

#### select!

最有名的 cancel 来源，可能是 rust 的 `select!` macro。

```rust
tokio::select! {
    result1 = future1 => handle_result1(result1),
    result2 = future2 => handle_result2(result2),
}
```

如果一个 future 返回，其他的所有 future 默认就会被 cancel 掉。

## What can be done?

目前么有通用的解法。

回到前面的三条：

1. 系统中有 cancel unsafe 的 future 存在；
2. 该 cancel unsafe 的 future 真的被 cancel 了；
3. 这个 cancellation 破坏了系统的某个属性；

要解决自己的问题，就要消除掉这三者之一。

### Making futures cancel-safe

#### select!

怎样使 future 变得 cancel safe？有些时候确实可以做到。

```rust
loop {
    let msg = next_message();
    match timeout(Duration::from_secs(5), tx.send(msg)).await {
        Ok(Ok(_)) => println!("sent successfully"),
        Ok(Err(_)) => return,
        Err(_) => println!("no space for 5 seconds"),
    }
```

在这个 case，可以使用 cancel safe 的 `reserve()` 方法来替代：

```rust
loop {
    let msg = next_message();
    loop {
        match timeout(Duration::from_secs(5), tx.reserve()).await {
            Ok(Ok(permit)) => { permit.send(msg); break; }
            Ok(Err(_)) => return,
            Err(_) => println!("no space for 5 seconds"),
        }
    }
}
```

（不过 `.reserve()` 也并不是完全的 cancel safe，如果 cancel 掉它，也会丢掉 mpsc channel 的 FIFO 属性）

#### AsyncWrite

另一个例子是 `AsyncWrite` 这个 trait。

```rust
use tokio::io::AsyncWriteExt;

let buffer: &[u8] = /* ... */;
writer.write_all(buffer).await?; // Not cancel-safe!
```

这个 `write_all` 方法不是 cancel safe 的！

这意味着如果 cancel，会导致 buffer 中的数据写了一半，后面的丢了。

要 fix 这个问题，可以使用一个经过仔细设计的 `write_all_buf` 方法，它能够记录下写入的位置，配合一个 loop，你可以在上次写入的位置继续写。

```rust
use tokio::io::AsyncWriteExt;

let mut buffer: io::Cursor<&[u8]> = /* ... */;
writer.write_all_buf(&mut buffer).await?;
```

### Not cancelling futures

#### pin

在 select 中，你可以通过 `pin!` 来固定住一个 future，每次 poll 的都是 future 的一个 mutable reference。

这样做，可以避免 future 被 cancel 掉。而是从上次的位置，继续 poll。

```rust
let mut future = Box::pin(channel.reserve());
loop {
    tokio::select! {
        result = &mut future => break result,
        _ = other_condition => continue,
    }
}
```

#### using tasks

另一个办法就是使用 tasks。

与 future 被 caller driven 不同，task 是被 async runtime 驱动的。

在 tokio 中，drop 一个 task 的 handle 并不会使它被 drop 掉，这意味着它是一个很好的跑 cancel unsafe 代码的地方。

## Conclusion

Some of the recommendations are:

- Avoid Tokio mutexes
- Rewrite APIs to make futures cancel-safe
- Find ways to ensure that cancel-unsafe futures are driven to completion


