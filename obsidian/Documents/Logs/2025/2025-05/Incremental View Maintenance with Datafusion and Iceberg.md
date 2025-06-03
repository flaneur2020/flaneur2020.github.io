## TLDR

好像也是根据查询的 plan 加上 delta 的数据（新增的、删除的），<mark>找出来输出 V 的哪些行需要 Refresh 掉</mark>

输入是 Plan、相关表中新增/删除的行，输出是目标的 V 表中，要 Refresh 的行所对应的查询条件。

相关的链接：<https://github.com/JanKaul/iceberg-rust/tree/main/datafusion_iceberg/src/materialized_view>

## Incremental View Maintenance

$$
V_{t+1} = (V_t - \nabla(V)) \uplus \Delta(V)
$$

其中：

- $\Delta(V_t)$：要添加的行
- $\nabla(V_t)$：要删除的行
- $\uplus$ 表示并集操作

### Projection propagation (SELECT)

![[Screenshot 2025-05-13 at 21.59.12.png]]
## Filter propagation (WHERE)

![[Screenshot 2025-05-13 at 21.59.41.png]]

## Join propagation
![[Screenshot 2025-05-13 at 21.26.39.png]]
好像 S 这边有修改 $\Delta S$，V 这边有关的行都要修改。

S 这边增量的行 $\Delta S$，V 中需要把 $\Delta S$ 中关联的 R 中的行，都带上。

R 这边增量的行 $\Delta R$，V 中需要把 $\Delta R$ 中关联的 S 的行，都带上。

## Aggregate propagation
![[Screenshot 2025-05-13 at 21.26.29.png]]

找出 $\Delta R$ 相关的，哪些现存的目标聚合键需要刷新，如果是新的聚合键，那么直接计算它的聚合值。