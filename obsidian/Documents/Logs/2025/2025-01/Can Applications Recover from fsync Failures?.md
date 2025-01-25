
tldr
- fsync() 如果遇到失败，几个主流文件系统竟然会将相关的 dirty page 标记为 clean？？？
- 这导致简单的重试，会和预期不符合
- PG 这个 bug 已经持续了二十年，有人报出来之后，改成了立即崩溃重启的形式，MySQL、WiredTiger 也做了类似的跟进；
- 在 crash recovery 时，程序需要注意不要被 page cache 欺骗，有可能 page cache 里是好的但是其实并没有成功写入到磁盘；<mark>在 crash recovery 时，应当总是读取磁盘上的数据</mark>；

---

## Introduction

作者做了一个 CuttleFS 来复现这类问题。

发现 redis 甚至没有检查 `fsync()` 的返回值。没有一个程序正确地处理了 `fsync()` 的错误。

作者发现 CoW 的文件系统似乎比基于 WAL 的文件系统更稳健，能够更容易恢复到一个一致的状态。

## Motivation

PG 处理 fsync 的 bug 已经持续了 20 年。

之后 PG 修改为了 fsync 如果有错误则立即崩溃
