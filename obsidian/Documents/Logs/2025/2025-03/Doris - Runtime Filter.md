有两种 Runtime Filter：Join Runtime Filter 和 TopN Runtime Filter。
## Join Runtime Filter

基于 runtime data 结合 join condition，动态生成 filter。

可以减小 Join probe，也可以减少网络 IO。

假设有两张表：

1. Orders table：包含一亿行数据，有 order key（`o_orderkey`）、customer key（`o_custkey`）和 order 信息；
2. Customer table：包含 10w 行数据，有 customer key（`c_custkey`）、customer 国家（`c_nation`）和其他 customer 信息；包含 25 个国家的 customer，每个国家平均有 4000 个 customer；

比如要统计中国的 order 数量：

```sql
select count(*)  
from orders join customer on o_custkey = c_custkey  
where c_nation = "china"
```

![[Pasted image 20250305113546.png]]

如果没有 Join Runtime Filter 能力，会扫描整个 orders 表，然后针对 1 亿行数据执行 hash probe 来得到结果。

优化：

- `c_nation = 'China'` 会过滤掉所有非中国的客户，因此只有少部分 customer table 的数据（1/25）涉及在这个 join 中。
- 给定 Join 条件：`o_custkey = c_custkey`，我们需要关注过滤相关的 `c_custkey`，假设这组集合为 A
- 如果将集合 A 下推给 orders table 的过滤条件，那么 Scan 可以更高效地过滤

```sql
select count(*)  
from orders join customer on o_custkey = c_custkey  
where c_nation = "china" and o_custkey in (c001, c003)
```

![[Pasted image 20250305114706.png]]

这一来，<mark>通过给 orders table 增加一个过滤条件，orders 表中需要关注的行数从 1 亿下降到了 40w</mark>，大幅减少了扫描的行数。

实现形式上，可以实现为一个 bloomfilter。

## TopN Runtime Filter

```sql
select o_orderkey from orders order by o_orderdate limit 5;
```

如果没有 topn filter，那么这个 scan 会扫遍整个 orders 表。（minmax 索引会不会有帮助？）

因为一个 data block 一般包含有 1024 行，topn node 可以从第一个 data block 中找出来排第五的行。

假设第五行是 `1995-01-01`，那么余下的扫描中，就不需要扫描小于 `1995-01-01` 的数据了。