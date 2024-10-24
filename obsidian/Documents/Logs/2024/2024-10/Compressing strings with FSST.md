https://blog.spiraldb.com/compressing-strings-with-fsst/

votex 是个用来压缩 Arrow 的库。提供了一系列 lightweight 的压缩算法，允许 reader 随用随解压；

针对 string，parquet 支持字典压缩以及一些通用压缩算法。

> FSST is fast–the reference implementation achieves gigabytes-per-second decoding–and has shown to compress on average by a factor of 2-3x on many real-world datasets.

好像说 FSST 事目前最快的算法，能实现 gb/s 的解码速度，能对真实世界数据集压缩 2～3 倍。
## String compression, in brief

zip、lz4、zstd 这些算法都是作用到 block 级别的。

要访问其中的几个 bytes，需要解压整个块。有研究显示，读写这种按块压缩的数据，CPU 很容易成为瓶颈。

如果希望能够 random access 被压缩的数据，这样就不大可行。

## Dictionary Encoding

要高效的 random access，Dictionary Encoding 可能是这方面最广为人知的压缩算法。

![[Pasted image 20241020210000.png]]

不过 Dictionary Encoding 有如下局限：

1. 不能处理字符串内部的重复；
2. 只有当数据 low-cardinality 时才工作良好；
3. 不能压缩不在 dictonary 中的数据；

## FSST has entered the chat

CWI 和 TUM 在 2020 年放出来一个论文《[_FSST: Fast Random Access String Compression_](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf?ref=blog.spiraldb.com)》

FSST 的意思是 <mark>“Fast Static Symbol Table”</mark>。

## Putting the ST in FSST

在 FSST 中压缩文本，首先需要构造一个 symbol table。

与 dictionary 不同，symbol table 中的 symbol 是一些 short substring。

一个 symbol 可以在 1～8 bytes 之间，这样可以 fit 到一个 64bit 主机的寄存器中。

为了便于 pack 表示 symbol，可以将 symbol table 限制为 256 个项。

压缩按 string-by-string 的方式进行，贪婪地匹配最长的 symbol table，将输出记录到 output array 中。

![[Pasted image 20241020211341.png]]

在 symbol table 中，保留一个 code `0xff` 作为 escape char。当匹配不到 symbol table 时，fallback 回 escape char + 原始字节值。

在最坏的情况下，原始输入会变为两倍大小：

```rust
[0xFF, b'h', 0xFF, b'e', 0xFF, b'l', 0xFF, b'l', 0xFF, b'o']
```

基于这个观察，可以发现，理论上的压缩比，从 8 （一个 symbol 的最大长度）到 0.5。

所以，构造这个 symbol table，使尽量长的 sequence 记录进来、尽量少出现 escape，是压缩比的关键。
## A seat at the (symbol) table

（好像有点像 LLM 的分词算法 🤔）

在每轮迭代，它使用当前代的 symbol table 来压缩一段 sample text。然后根据当前的编码，创建一组新的潜在的 symbols。到最后，保留一组最好的 symbols，用于下一轮迭代。

在 paper 中，作者定义了一个指标叫做“effective gain”。每个 symbol 的 “effective gain” 等于它的长度 * 频率。

在每轮迭代中：

1. 使用当前的 symbol table 对目标 string 进行压缩，并统计输出中各个 symbol 出现的次数；同时，我们将每个 symbol 前后相邻的 symbol 拼在一起，当作新 symbol 的 candidate；
2. 对于当前的 symbol 和 candidate symbols，赋予一个 gain（count * length），保留 gain 最高的 255 个 symbol；
3. 选择最高的 255 个 symbol，再跑下一轮迭代；

在第一次迭代，symbol table 就是单纯的每个 byte 原本的值。

上面是一个简化版，正式的版本还要考虑更多：

> 1. Extending support from single-strings to string arrays with many values. The naive way to do this is to call the compression/decompression kernels in a loop. A more advanced variant would compress the entire array range as a single mega-string to avoid branching.
> 2. Training the symbol table on a whole array forces us to compress the full-size array 5 times, which is slow. <mark>Instead, we choose a small fixed-size sample of 16KB, drawn as 512B chunks sampled uniformly at random across all elements of the array </mark>. Both 16KB and 512 were chose empirically by the authors as good tradeoffs between locality (longer runs are more realistic representations of the source) and performance (getting a diverse enough sample to train a table that compresses the whole array well).
> 3. The authors of the paper implemented an AVX-512 compression kernel, which we have not implemented in the Rust code but can lead to better performance over the scalar variant. You can learn more about their “meatgrinder” in Section 5 of the paper.