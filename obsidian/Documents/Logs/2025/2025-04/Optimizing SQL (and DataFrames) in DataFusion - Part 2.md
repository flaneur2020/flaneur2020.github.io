optimizer 实现为在 logical optimizer 中 rewrite LogicalPlan；或者在 physical optimizer 中 rewrite ExecutionPlan。

## Always optimizations

一些优化是属于所有的 query engine 都会先实现一下，有最高的性价比。

### Predicate/Filter Pushdown

大多数成熟的数据库都会积极地使用 filter pushdown、early filtering，配合一些 partition pruning 之类的技术（比如 parquet row group pruning）。

### Projection pushdown

Avoids carrying unneeded _columns_ as soon as possible.

### Limit Pushdown

The earlier the plan stops generating data, the less overall work it does, and some operators have more efficient limited implementations.

### Expression Simplification / Constant Folding

Evaluating the same expression for each row when the value doesn’t change is wasteful

如果一个表达式的数值和 row 的取值无关，那么可以提前算出来。比如这个 SQL 中的 `extract(year from now())`：

```sql
SELECT … WHERE extract(year from time_column) = extract(year from now())
```

### Rewriting `OUTER JOIN` → `INNER JOIN`

INNER JOIN 一般总是比 outer join 的实现更快而且更简单。而且，`INNER JOIN` 在优化 pass 中的限制比其他的 join 类型少，比如 join reordering 和额外的 filter pushdown。

## Engine specific optimizations

### Subquery Rewrites

按 "a row at a time" 处理的性能比较低。这时可以将它重写成 join，这样可以快 100 ~1000 倍。

```sql
SELECT customer.name 
FROM customer 
WHERE (SELECT sum(value) 
       FROM orders WHERE
       orders.cid = customer.id) > 10;
```

```sql
SELECT customer.name 
FROM customer 
JOIN (
  SELECT customer.id as cid_inner, sum(value) s 
  FROM orders 
  GROUP BY customer.id
 ) ON (customer.id = cid_inner AND s > 10);
```

### Optimized expression evaluation

