tldr:

sqlancer 的思路是：找一个随机的行作为 pivot row，然后生成总是为 true 的 where 条件，检查 query 的结果中包含有这行 pivot row。

仍属于 differential testing，传统的 differential testing 的问题是，不同的 database 之间的语法语义有差异，导致很难作为靠谱的 oracle。

这个 pivot row 的做法也称作 “containment oracle”



