
Chunk storage system 的设计目标：即使存储介质有故障，也能跑到最高带宽。

3FS 的读写吞吐量应随 SSD 数量以及客户端与存储服务之间的对分网络带宽线性扩展。

### Data placement

每个 file cunk 会对一组 storage 节点按 CRAQ 进行 replication。

CRAQ 中，写操作会发送给 head 节点，并按 chain 进行传播。

读操作可以发送给任意一个 storage target。每个 storage target 可以创建在单独的一个 SSD 上。

假设有 6 个节点 A、B、C、D、E、F，每个节点单独一个 SSD，每个 SSD 可以创建 5 个 storage targets：1、2、3、4、5。那么可以提前创建出来一个 chain table。

|Chain|Version|Target 1 (head)|Target 2|Target 3 (tail)|
|:-:|:-:|:-:|:-:|:-:|
|1|1|`A1`|`B1`|`C1`|
|2|1|`D1`|`E1`|`F1`|
|3|1|`A2`|`B2`|`C2`|
|4|1|`D2`|`E2`|`F2`|
|5|1|`A3`|`B3`|`C3`|
|6|1|`D3`|`E3`|`F3`|
|7|1|`A4`|`B4`|`C4`|
|8|1|`D4`|`E4`|`F4`|
|9|1|`A5`|`B5`|`C5`|
|10|1|`D5`|`E5`|`F5`|

每个 chain 有一个 version number。在 chain 修改之后，会递增 chain 的版本号。

只有 primary cluster manager 才能够对 chain 进行修改。

可以创建多个不同的 chain table 用于不同的场景，比如创建两个，一个用于 batch/offline 任务，另一个用于 online service。这两个 chain table 的 storage targets 需要隔离开。

### Balanced traffic during recovery

假如在上述 chain table 中，如果流量均匀地打过去。如果 A 失败，读请求会打到 B 或者 C。这时容易打满 B 和 C。将 A 的失败的 SSD 替换，同步数据回来可能需要数个小时，这期间 B 和 C 的压力会比较大。

为了减少这方面的影响，我们可以有更多 SSD 来共享这些 redirected 的流量。

在如下的 chain table 中，A 和所有的其他 SSD 均做了配对，当 A 失败时，每个其他的 SSD 都能接受 1/5 的 A 的读流量。

| Chain | Version | Target 1 (head) | Target 2 | Target 3 (tail) |
| :---: | :-----: | :-------------: | :------: | :-------------: |
|   1   |    1    |      `B1`       |   `E1`   |      `F1`       |
|   2   |    1    |      `A1`       |   `B2`   |      `D1`       |
|   3   |    1    |      `A2`       |   `D2`   |      `F2`       |
|   4   |    1    |      `C1`       |   `D3`   |      `E2`       |
|   5   |    1    |      `A3`       |   `C2`   |      `F3`       |
|   6   |    1    |      `A4`       |   `B3`   |      `E3`       |
|   7   |    1    |      `B4`       |   `C3`   |      `F4`       |
|   8   |    1    |      `B5`       |   `C4`   |      `E4`       |
|   9   |    1    |      `A5`       |   `C5`   |      `D4`       |
|  10   |    1    |      `D5`       |   `E5`   |      `F5`       |
为了在恢复期间实现最大读取吞吐量，可以将负载均衡问题建模为平衡不完全区组设计（BIBD）。通过使用整数规划求解器，可以得到最优解。

（似乎相当于通过 chain table 提前规划好，每个盘的流量分担好）

## Data replication

CRAQ 是一个 write-all-read-any 的 replication protocol，为读 heavy 的场景进行优化，能够利用起来所有 replica 的读带宽。

当收到一个写请求时：

1. 检查写操作的 chain version 是否匹配最新的 version，如果不匹配就报错；
2. 发送 RDMA Read 操作来拉取 write data。如果 client/predecessor 失败，RDMA read 操作会 timeout，abort 该写入操作；
3. 当 write data 拉到 local memory buffer 后，会从 lock manager 获取该 chunk 的 lock，其他的并发写入会阻塞；所有写入都会在 head target 这里序列化；
4. service 将该 chunk 的 committed version 读到内存，apply update，将 updated chunk 作为 pending version 保存；一个 storage target 可能保存同一个 chunk 的两个版本：一个 committed 版本、一个 pending 版本；每个版本有一个递增的版本号；
5. 如果 service 是 tail 节点，则 committed version 原子性地替换为 pending version，发送 ack 给 predecessor；否则，写操作发送给后继者；committed version 更新后，当前的 chain version 会保存为 chunk 的 metadata；
6. 当 ack 消息到达 storage service 时，service 会替换 committed version 为 pending version，继续传播 message 给前继者；local chunk lock 的锁也会被释放；

当收到读请求时：

1. 如果有 committed 版本的 chunk，则返回 client；
2. 如果同时有 committed 版本和 pending 版本，则返回一个特定的 status code，client 能够稍后重试；

### Failure detection

