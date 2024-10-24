
Write Skew： 当一个竞争条件出现时，违反了基于多个记录的状态约束，但是仍允许了写入。

比较典型是 doctor oncall rotation 这个例子。

## 阻止 Write Skew

- Atomic 的 single object lock 是不管用的，因为涉及有多个对象。
- Snapshot Isolation 也不够用。
- 多数数据库并不支持针对多个对象设置约束；

## Phantoms causing write skew

1. 针对某种业务层的要求进行查询，比如当前没有其他医生在 oncall。
2. 程序基于这个查询，决定后续要 INSERT、UPDATE 或者 DELETE 等操作，而这部分修改，有可能影响到 （1）中的条件检查；
3. 在检查这里的条件时，你可以选择 SELECT FOR UPDATE 来锁定这些行；
4. 如果 query 的条件中，会受到一些新插入行的影响，而 SELECT FOR UPDATE 锁不上这些行，也会起不到作用；

