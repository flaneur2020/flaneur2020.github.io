很多时候 query optimizer 会被看做 black magic，因为：

1. 必须在完成数据库系统的其他部分（比如 data storage、transactions、sql parser、expression evaluation、plan execution 等）之后，optimizer 才会变得 critical；
2. optimizer 的设计与系统的其他部分（比如 storage 或索引）关联密切，所以一些经典的 optmizer 和特定的系统的术语绑定的比较紧；
3. 一些 optimizer task，比如 access path selection、join order 等，都有一些没有最优解的情况，因此真的有 black magic；

但是作者认为，optimizer 也没有比数据库的其他部分更困难。因此写了两篇博文，希望读者能够：

1. 根据自己的需求，去基于 datafusion 去设计自己的 optimizer；
2. 做一些真实的 query optimization 的学术研究；

## Query Optimizer background

![[Pasted image 20250406145058.png]]

不管 DataFrame 还是 SQL，都会有一个 initial 的 plan，Query Optimizer 能够重写 initial plan 成为一个 Optimized Plan。

优化器对 SQL 和 DataFrame 都同样有效。传统的 pandas 和 Polars 等 DataFrame 库默认没有做什么优化，不过更现代的 API 比如 Polar 的 lazy API、spark DataFrame、DataFusion Dataframe 等，都有应用 optimization。

## Example of Query Optimizer

```sql
SELECT location, AVG(population)
FROM observations
WHERE species = ‘contrarian spider’ AND 
  observation_time >= now() - interval '1 month'
GROUP BY location
```

![[Pasted image 20250406145713.png]]

优化后：

![[Pasted image 20250406145720.png]]

## Query Optimizer implementation

工业的 optimizer 的话，通常会实现一系列的 pass of rules 来重写一个 query plan。

multi-pass optimizer 通常是一个标准，因为:

1. Understand, implement, and test each pass in isolation
2. Easily extend the optimizer by adding new passes

工业优化器中有三种不同的优化：

1. Always Optimization：只要做了就没什么坏处，比如 expression 简化、predicate pushdown、limit pushdown 等，它们的理论也比较简单，当然也需要非常多的代码量和测试；
2. Engine Specific Optimization：比如 expression 的执行、使用哪一种 Hash 或者 join 实现；
3. Access Path and Join Order Selection：这些通常使用 heuristic 或者 cost model；数据库通常有多种不同的访问数据的方式（比如 index scan 或者 full table scan），也有多种不同的 join 顺序来访问多张表；