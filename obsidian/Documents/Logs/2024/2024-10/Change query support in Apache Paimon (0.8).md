## Some relevant internals background

### Incremental table scans

paimon 和 iceberg 之间最大的不同在于，paimon 的每个 snapshot 包含两个 manifest 文件：base manifest file（修改开始时刻的完整）、delta manifest file（本次修改涉及的文件）；

一个 paimon 的 reader 可以遍历 snapshots 的历史，根据这里面的 metadata 来做到 incremental query。

### Sharding and LSM trees

paimon 会把每个 partition 中的数据 shard 到一组 bucket 中。bucket 的数量可以固定，也可以是动态的。

read、compactions、change queries 均发生在 bucket 级别。

每个 bucket 就是一个自洽的 LSM Tree。


