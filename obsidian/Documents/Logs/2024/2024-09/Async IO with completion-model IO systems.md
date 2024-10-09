
TLDR;

- Completion IO 的问题：你需要传递一个 buffer 给操作系统，但是跑到一半 future 可能会被 cancel，如果你的 buffer 跟着被释放，可能会出现一个情况是这块内存被别人申请了，但是操作系统往里写了东西，导致产生脏数据；
- 解决办法可能是弄一个 OwnedBuffer，进入 IO 时，将 buffer 转移给 OS，IO 结束时拿回来，不过这也需要有一个大于 IO 生命周期的对象来 hold 这个 IO 过程。
- 也有解决办法是 IO 框架将所有的 buffer 托管起来（比如 Glommio），统一管理生命周期（个人感觉这个路线可能更靠谱些）

---

> Completion-model IO systems don't work naturally with the `Read` trait.
## Completion model reads

Completion based IO 下，读取相当于给 os 发一个 buffer，等待 IO 完成，最后读取这个 buffer 中的数据。

这个 buffer 必须保持存活，并且在 IO 完成前不能修改。在 rust 的语义来看，在 IO 期间 ownership 等于转移给了 OS。

这对 async read 函数来讲还比较直接，除了 concellation 的问题。

```rust
async fn foo(stream: impl async::Read) {
    let mut buf = [0; 1024];
    stream.read(&mut buf).await?;
    // Use buf
}
```

如果这个 future 被 drop，`buf` 也会被删除。

<mark>不幸的是，这个 buf 在操作系统的视角上来看是没有暂停的。在它写入 buf 时，就会产生 use-after free 的问题</mark>。

在类型系统的视角来看，这个 error 来自于 future `foo` 有 `buf` 的 ownership，OS 只是 mut borrow 走了它。

实际上应当是 OS 拿走了 ownership 才对。

> I see two ways to accomplish that, either the user passes an owned buffer to `read`, or the buffer is managed by the reader and reading is done through a `BufRead` trait. `BufRead` assumes the reader owns the buffer and that lets us handle buffer ownership independently of reading.

## Reads with owned buffers

```rust
async fn foo(stream: impl async::Read) {
    let buf = vec![0; 1024];
    let buf = stream.read(buf).await?;
    // Use buf
}
```

<mark>reader 必须保证 IO 期间这个 buffer 仍然是存活的，即使 IO 被取消。</mark>

作者认为可以定义一个类似这样的 trait：

```rust
trait OwnedRead {
    async fn read(&mut self, buffer: impl OwnedReadBuf) -> Result<OwnedReadBuf>;
}

trait OwnedReadBuf {
    fn as_mut_slice(&mut self) -> *mut [u8];
    unsafe fn assume_init(&mut self, size: usize);
}

// An implementation using the initialised part of a Vector.
impl OwnedReadBuf for Vec<u8> {
    fn as_mut_slice(&mut self) -> *mut [u8] {
        &mut **self
    }

    unsafe fn assume_init(&mut self, size: usize) {
        self.truncate(size);
    }
}
```

## Reads with `BufRead`

```rust
async fn foo(stream: impl Read) {
    let mut buf = vec![0; 1024];
    let mut reader = Reader(stream, buf);
    reader.read().await?;
    let buf = reader.buffer();
    // Use buf
}
```

这个代码中，buffer 没有传递给 `read` 而是被 `Reader` 拥有所有权。

这里也需要考虑一个问题是，如果 `reader` 在 IO 完成前中间被 drop 了，仍然需要保持 `buf` 存活足够长时间。
### Flexibility of buffer management

## Specialist solution

Programmer is responsible for keeping the buffer alive：让程序员手工管理 buffer；
## Possible future solutions

Non-cancellable futures：应该不大可行。

Structured (async) concurrency：可行但是好像改动比较多。
## Possible solutions

### Pass owned buffers

接受一个 owned 的而不是 borrowed 的 buffer，在 IO complete 时把这个 buffer 退回来。这个 buffer 可以是一个具体的类型，比如 `Vec<u8>` 或者也可以是一个 trait。当 IO 被取消时，这个 buffer 会被 release 掉。

这个方法需要加一个 `ReadOwned` 的 trait 或者向 `Read` trait 中加一个 `read_owned` 方法。这个方法不好处就是 trait 的复杂性，好处就是还比较简单。
### Fully managed buffer pool

IO library 会管理一个 buffer pool。library 拥有这些 buffer。

（个人感觉这可能是最靠谱的？）
### Graveyard

> the IO library is not responsible for managing buffers entirely, but offers a way for the library to take ownership of the buffer if the IO is cancelled.

在 cancel IO 时，不是简单地 drop 这个 buffer，而是移动到一个 grave yard 去，直到 IO cancel/compelte 为止。

## Existing solutions

- Glommio 在用 managed buffer pool 方案，说未来也会考虑 passing owned buffer 方案；
- Ringbahn 定义了一个 Ring 类型，使用自己的 IO buffer（类似 managed buffer pool 方案），下层，Event 类型提供了一个 unsafe 方法和 cancellation callback（programmer is responsible 方案）
- tokio-uring 使用了一个 owned buffer trait（IoBuf），这个 buffer 会传递给 IO library，还给 user（程序员也需要关注这个 reader 的生命周期的吧，要是跟 future 一起被 drop 了感觉也不大行）
- Monoio 使用了一个 owned buffer trait `IoBufMut`，基于 passing owened buffer design