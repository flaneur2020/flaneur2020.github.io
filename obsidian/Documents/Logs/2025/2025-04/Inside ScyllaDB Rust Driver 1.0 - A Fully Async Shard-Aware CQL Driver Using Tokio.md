是一个 client-side、shard aware 的 driver，用 tokio 写的 full async client。

## What’s New in ScyllaDB Rust Driver 1.0?

### Refactored Error Types

过去的 error type 比较 adhoc，不是很 type safe，一些错误会把别的 error 吞掉，没有为诊断 root cause 提供足够的信息。

有些错误会被滥用，比如 ParseError；也有一个 One-To-Rule-Them-All 的错误类型叫做 QueryError，会在 user facing API 中返回。

之前长这样：

![[Pasted image 20250402110824.png]]

问题：

1. 这个结构很痛地扁平，有很多 niche 小 error 种类（比如 UnableToAllocStreamId）；
2. 很多 variant 只是 string，很多 error 只是简单地包在了 IoError 中；
3. 有这些 variant 在，matching error kind 几乎是不可能的；
4. 这个 error type 是 public 的，没有包装为 `#[non_exhaustive]`，每加一个 error kind 都会 break api；

在 1.0.0 中，error type 更加的清晰，并反应代码的结构；

![[Pasted image 20250402111226.png]]

- 这个错误类型更能反映 driver 的 module；
- stringification 没有了
- error type 加了 `#[non_exhaustive]` attribute

### Refactored Module Structure

### Removed Unstable Dependencies From the Public API