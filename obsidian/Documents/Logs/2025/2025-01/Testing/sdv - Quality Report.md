quality report 中包含有 Column shapes、Column Pair Trends、Cardinality 等信息。

## Column Shapes

shape 反映的是每个列的分布。

![[Pasted image 20250103142027.png]]
每个列单独统计 shape 的相似度，最后取一个平均值。

## Column Pair Trends

![[Pasted image 20250103142536.png]]

表示生成的数据，能多大程度上还原数据之间的趋势关联。

## Cardinality

（parent 和 child tables 怎么理解？）

![[Pasted image 20250103142817.png]]

会计算一个 CardinalityShapeSimilarity 指标。

## Intertable Trends

![[Pasted image 20250103143141.png]]

与 Column Pair Trend 相似，不过是跨表之间的列关系。