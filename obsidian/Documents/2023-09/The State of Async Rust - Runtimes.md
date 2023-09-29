https://corrode.dev/blog/async/

- 文章的目标受众：初学者、参与 async 生态的库维护者
- 比较糟糕的现实是：库维护者需要对接不同的 runtime；
- 如果希望编写与 runtime 无关的 async 逻辑，那么往往需要 conditional compilation、compatibility layer 并处理 corner case；
- Executor coupling 是 async Rust 的一个大问题，这导致了生态与 runtime 太耦合；
- 对于库作者来说，这是一个很昂贵的负担，这并不意外使得有名的库 reqwest 坚持只支持 tokio 作为 runtime；

## The case of `async-std`

- async-std 是一个贴近 rust 标准库的 async 生态的尝试；
- 在目前有 1754 个 public crate 在依赖 async-std，但是它好像不大更新了；

## Can't we just embrace Tokio?

- tokio 更像一个 async programming 的 framework 而非简简单单一个 runtime；
- tokio 默认的文档中提示大家可以默认打开 features = full
- 但是 full 的话就会创建出来一个 multi-threaded 的 runtime，这要求类型要么 Send 且 'static。
- 使简单的应用也得用 Arc 和 Mutex；
- **Multi-threaded-by-default runtimes cause accidental complexity completely unrelated to the task of writing async code.**
- Maciej suggested to use a [local async runtime](https://maciej.codes/2022-06-09-local-async.html) which is single-threaded by default and does **not** require types to be `Send` and `'static`.

## Other Runtimes

- smol：整个 executor 只有 1000 行代码；
- embassy：适用于嵌入式环境中的 runtime；
- glommio：适用于 IO bound 的任务，在 io_uring 上跑 thread per core 架构；
- Especially, iterating on smaller runtimes that are less invasive and single-threaded by default can help improve Rust's async story.

## Async vs Threads

- [Greenspun's tenth rule](https://en.wikipedia.org/wiki/Greenspun%27s_tenth_rule): Any sufficiently advanced async Rust program contains an ad hoc, informally-specified, potentially bug-ridden implementation of half of an operating system's scheduler.
- Thread-based frameworks, like the now-inactive [iron](https://github.com/iron/iron), showcased the capability to effortlessly handle [tens of thousands of requests per second](https://github.com/iron/iron/wiki/How-to-Benchmark-hello.rs-Example). This is further complemented by the fact modern Linux systems can manage [tens of thousands of threads](https://thetechsolo.wordpress.com/2016/08/28/scaling-to-thousands-of-threads/).
- 传统上有 gc 的语言遇到的一些问题比如 stop the world 在 rust 中也是不存在的；

## Summary
- Use Async Rust Sparingly
	- 劝新人一开始可以先绕开 async，但是后来发现并不一定可行，因为很多 library 都是 async first 的设计；
	- 作者会更建议当你真正需要使用它的时候再用；
	- 如果不得不用 async，那么就用在 tokio 生态中的一些优秀的库比如 reqwest、sqlx 等；
- Consider The Alternatives
	- In binary crates, think twice if you really need to use async. It's probably easier to just spawn a thread and get away with blocking I/O. In case you have a CPU-bound workload, you can use [rayon](https://github.com/rayon-rs/rayon) to parallelize your code.
	- If you don't need async for performance reasons, threads can often be the simpler alternative. — [the Async Book](https://rust-lang.github.io/async-book/01_getting_started/02_why_async.html#async-vs-threads-in-rust)
- Isolate Async Code
	- the error messages of sync Rust are much easier to reason about than those of async Rust.
- Keep It Simple