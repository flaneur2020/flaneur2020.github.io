---
layout: post
title: "Mining Massive Datasets 学习笔记: LSH"
---


怎样在大规模数据集中发现相似的项对列表？

最糙的解法是嵌套两层 for 循环遍历所有的项对，计算彼此的相似性再按相似性排序，这一计算量显然是不可接受的。而 k-d tree 之类的多维搜索树，面向高维度的数据集性能也退化的厉害，几乎与嵌套两层 for 循环无异。

最近在看《大数据：互联网大规模数据挖掘与分布式处理》介绍了 LSH (Locality Sensitive Hashing, 距离敏感哈希) 算法，它的思路是对数据项计算哈希，想办法将相似的项对分到同一个哈希桶中，从而简化大大计算过程。过程很有意思，在这里记一下。

## 集合的 Jaccard 相似度

集合 S 和 T 的 Jaccard 相似度表示为 S 和 T 的交集与并集之间的差：

$$
Jaccard(S, T) = | S \cup T | \div | S \cap T |
$$

集合相似度可以用于衡量文本文档的相似性，比如论文抄袭、重复新闻的检测；可以应用于协同过滤的场景，如一位用户喜欢的书籍集合和我的书籍集合相似，那么可以认为我们喜好相似；也可以将 uber 的驾车行驶路线表示为集合，继而可以通过 LSH 算法找出相似路线。

## 将文本文档表示为集合：Shingling

将文本文档表示为集合的 Shingling 方法和搜索引擎的 N-gram 是同义词，如 k-shingle 等于长度为 k 的任意子串。

对于重复新闻检测，可以将 shingle 定义为停用词加上后续的两个词形成 shingle 集，这一做法基于假定：新闻正文中富含停用词，而正文内混插的广告语通常是简短的口号性的文本，停用词少。这样做有助于减少同一新闻内容受站点插入的广告造成的干扰。

## 集合的摘要表示: MinHash

shingle 集合非常大，远远大于原始文档的体积。幸运的是可以通过 SimHash 将集合压缩为摘要，而摘要仍能用于相似性的计算。

$$
minHash(S) = min \{ h(x) \ for \  all \ x \ in \ S \}
$$

$$
h(x) = (ax + b) \  \% \  m
$$

minHash 之所以有用，在于两个集合的在经过 minHash 转换之后相等的概率恰好等于两者的 Jaccard 相似度。 

设 shingle 集中有两个 shingle 集合 S 和 T，分别求取 minHash 等价于在两个集合中随机抽取一个元素：

- 总样本空间为 $$ S \cup T $$
- 两个元素相等的样本空间为 $$ S \cap T $$

易知随机抽取的两个元素相等的概率等于 Jaccard 相似度。

类似 bloom filter，使用的哈希函数越多，Jaccard 相似度的估算也就越精确。生成 100 个哈希函数，分别应用于一个集合所得的 minHash 列表可作为一个集合的 minHash 签名，对两个集合的签名做概率计算，即可估算出彼此的 Jaccard 相似度。

## LSH 的哈希桶

对每个集合计算 minHash，并将 minHash 分别划分到哈希桶中，将分到同一桶中的集合视为可能相似的候选集。为了提高精确性，计算多个 minHash 并分桶，同一对集合可能存在于多个桶中，通过一次排重扫描，可找出哈希桶重复度最高的一批集合匹配，随后可以在它们中间计算 Jaccard 距离，确认出最相似的集合。

## Reference

- <http://www.mmds.org/mmds/v2.1/ch03-lsh.pdf>
- <https://www.slideshare.net/SparkSummit/locality-sensitive-hashing-by-spark>