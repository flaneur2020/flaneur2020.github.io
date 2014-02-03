---
layout: post
title: "BigTable Note 1: Basic"
---

最近用 Evernote 明显多于纸质笔记和 blog，有些笔记就留在了里面长草，没有进一步整理。放假在家正好整理一下前段时间看 BigTable 论文的笔记。

传统关系式数据库很难做到水平扩展，BigTable 是 Google 三大件之一的海量数据库方案。它基于 GFS 之上，在 GFS 的海量存储之上提供结构化的表存储。然而相对于面向高吞吐优化的 GFS，BigTable 的应用场景更加广泛，既适用于高吞吐的数据分析应用，也适用于面向用户的低延时应用。

论文中对 BigTable 的定义为： "A sparse, distributed, persistent multi dimension sorted map"。此 map 基于行键、列键和时间戳进行索引，其中的键与值为裸的字节序列。

为了方便讲解，论文中将 WebTable 作为例子，假定为爬虫爬来的网页数据表：

![](/img/webtable.png)

## 行与 Tablet

BigTable 不支持跨行事务，但是每行的数据读写是原子的，因而得以安全高效地基于行做数据分割。每个行键对应一行数据，行键按字典序排列。每张表按一定行键的范围划分为 Tablet，Tablet 也正是数据分割与负载均衡的基本单位。每张 Tablet 大约为 100mb，当数据增大会自动分裂；当一台节点上的 Tablet 数量增多时，会尝试将 Tablet 的所有权 [1] 迁移到其它节点上。

基于行键按字典序排列并按范围存储的性质，应用得以组织数据的局部性。比如 WebTable 将倒序的域名作为行键（像 com.cnn.www/index.html），这一来分析同一域名之下的地址更加高效。

[1]: Tablet 的迁移并不一定发生数据的迁移，因为数据总是持久化地保存在 GFS 上为所有节点共享，Tablet 的迁移只需要迁移 Tablet 相关的元信息。

## 列与 Column Family

列键会组织在 Column Family 中，作为基本的访问控制单元。往列中存储数据之前，该列必先加入某个 Column Family。Column Family 中的数据一般类型相同，会被一同压缩。每张表中 Column Family 的数量是有限的，但每个 Column Family 中可以存储无限数量的列。

列键的命名规范为 family:qualifier，其中 family 必须为人类可读的字符，而 qualifier 可以为任意数据。比如 WebTable 中有个 Column Family 为 language，作为该网页唯一的语言标识；另一个有用的 Column Family 为 anchor，用以表示网页中的一个超链接地址，像 anchor:cnnsi.com, anchor:my.look.ca 。直白地说，一个 Column Family 可以在某种程度上视为表中的一个 Key-Value 类型字段。


## 时间戳

时间戳允许每个单元格存储多个版本的数据。它的值可以作为默认的毫秒级的系统时间，也可以由客户端手工指定。多版本的单元格数据，按照时间倒序排列，因为近期的数据更容易放问到。客户端可以设置仅保留最近的 n 个版本的数据。

比如 WebTable 中的 contents 列，用于存储爬虫爬下来的同一网页的多个版本的内容。从这个例子也可以看出，BigTable 的设计是完全本着谷歌的业务出发的。

## API

接口的设计非常漂亮，偷懒从论文里贴过来：

```
// Open the table
Table *T = OpenOrDie("/bigtable/web/webtable");
// Write a new anchor and delete an old anchor
RowMutation r1(T, "com.cnn.www");
r1.Set("anchor:www.c-span.org", "CNN");
r1.Delete("anchor:www.abc.com");
Operation op;
Apply(&op, &r1);
```

RowMutation 对象允许录制针对某行的一系列操作，到 Apply() 时原子执行。虽说不支持多行事务，但可以基于单行事务实现 Read-Modify-Write 式的原子同步。

Scanner 对象允许扫描某行数据特定 Column Family 中所有的列：

```
Scanner scanner(T);
ScanStream *stream;
stream = scanner.FetchColumnFamily("anchor");
stream->SetReturnAllVersions();
scanner.Lookup("com.cnn.www");
for (; !stream->Done(); stream->Next()) {
    printf("%s %s %lld %s\n",
        scanner.RowName(),
        stream->ColumnName(),
        stream->MicroTimestamp(),
        stream->Value());
}
```

遍历 Column Family 时甚至支持正则表达式，比如 `anchor:*.cnn.com`。

此外也有一系列"高级"接口，比如支持客户端批处理加速多行数据处理；允许将单元格中的数据作为数值处理；支持内嵌 Sawzall 脚本对数据进行读取和转换等。

## References

* http://static.googleusercontent.com/media/research.google.com/zh-CN//archive/bigtable-osdi06.pdf
* http://www.cs.utexas.edu/~dahlin/Classes/439/lectures/bigtable.txt

