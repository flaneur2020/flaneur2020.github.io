https://eli.thegreenplace.net/2024/tokens-for-llms-byte-pair-encoding-in-go/

openai 的文章中介绍说 token 是 word 的一段 piece；

这位作者的文章中介绍了 tokenizer 的实现，它性能不一定最优但是主打一个完整、可读、兼容 tiktoken；

## Byte pair encoding - introduction

BPE 的输入：包含单词、数字、空格、标点符号

输出是一组数值类型的 token 列表，每个 token 对应一个唯一的 id，可以通过它还原成为原文。

BPE 有一个重要的预处理：将输入拆分为单词。

拆分单词这步操作在不同的模型中，有不同的规则，用不同的正则表达式。

一般是采用基于空格的 split（同时保留空格）

比如，"i'm blue dabadee dabadam" 可能被拆分为：

```
i
'm
_blue
_dabadee
_dabadam
```

值得注意的是：

1. `'m`、`'ll`、`'re` 通常被视为单独的 word。
2. whitespace 被看作单词的开头
## Training

BPE training 首先假设每个 byte 对应一个单独的 token

然后它会持续地合并 token 对成为一个更长的 token，并加入到单词表中，直到满足了特定的词表大小为止。

BPE 的训练从 256 大小的词表开始，

然后重复下述过程：

1. 计算 input 中每对 ordered bytes 出现的次数
2. 找出最长的 count，创建一个新的 token，对应一个新的 token id
3. 替换 input 中，这个 ordered bytes 的内容

GPT-4 的 token 表有 100,000 左右。
## Encoding

BPE 是一个贪婪策略。

token 在 encoding 的时候，和 training 阶段时的顺序一致。

在一开始，将输入按字节划分 token，

然后再遍历所有相邻的两个 token，寻找词表中是否有匹配项，有多个匹配项时，选择 token id 最小的。

## Realistic vocabulary and splitting

GPT4 使用的词表叫做 cl100k_base，包含 100k 个 token。

这也是 tiktoken 库使用的词表，可以直接下载。

openai 的 tokenization 还有一个很重要的地方是这个拆分用的正则表达式：

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

这个正则的功能有：

1. 拆分 space-delimited words，并保留空格在 word 最前面
2. 英文中常见的 `'s` `'t` 这些省略词，单独拆分
3. 对于比较长的数字，按 3 拆分

> For Go programmers, it's important to note that this pattern uses ?! - negative lookahead - which the standard regexp package doesn't support. Therefore, we'll have to reach for the 3rd party [regexp2](https://github.com/dlclark/regexp2) to implement this

