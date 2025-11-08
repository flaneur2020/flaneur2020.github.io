Introduction
- 场景：基于 Omid 开发了 Sieve，作为 Yahoo 搜索、个人页的内容管理平台；每天聚合几十亿内容 push 到一个实时索引；
- 目标：对 battle tested 的 key / value 存储增加事务支持；设计上与存储无关，实现上只有 hbase；
- 第二个主要考虑是简单，选择了中心化的 transaction manager 方案；
- Scalability：
	- 可以在 pb 级的数据集上跑 100,000 tps 级的事务；
	- 性能上得益于： Snapshot Isolation 事务，数据管理与控制分离的设计；
	- 1. 单 TM 设计，无协调开销；2. scale up TM 节点的内存做冲突探测；3. scale out Metadata（hbase）；
- High availability
	- TM primary-backup 设计，共享元信息访问；（TM is implemented asa primary-backup process pair with shared accessto critical metadata.）
	- 允许一定的事务重叠，绕过昂贵的协调；
2 Design Principles and Architecture
- 理论上可以支持所有支持多版本+ 原子 putIfAbsent 插入的 k/v 存储；
- 下层存储满足持久性（有 WAL log）、扩展性（sharding）和高可用（通过 replication）的前提下，留给 Omid 只做事务管理即可；
- metadata 包括一张专用的表为每条事务保存一行记录，和每行数据的 metadata；
- Omid 依赖下层存储的 multi-version 机制；
	- 支持原子的 put(key, val, ver), putIfAbsent(key, val, ver) 操作，加上 get(key, ver) 允许读到不高于 ver 的最新版本；
- 旧版本数据需要保留，在 active transaction 仍在访问它期间；(An old version might be required as long as there is some active transaction that had begun before the transaction that overwrote the version has commited.)
- 过期的数据在 hbase 中可以通过 coprocessor 进行清理；
- TM 有三种功能：
	- 1. 分配版本号；
	- 2. commit 时探测冲突（conflict detection）；
	- 3. 持久化 commit log；
- TM 元信息存储层面在下层存储做 scale out，冲突探测在 TM 节点内存中 scale up；
- Omid 使用基于时间的租约做 primary/backup 选举，租约信息保存于 zk；
3 Related work
- 学术界方案大多是整体方案，部分方案选用了 RDMA 和 HTM；不过这些方案没有利用软件栈的分层设计，大多实现了 Serializability 级别的事务隔离；
- 另一面，google spanner、megastore、percolator、雅虎的 Omid1、Cask 的 Tephra 更多地做分布式系统的分层设计，重用现有的高可用存储层，比如 megastore 建立在 bigtable 之上，Cockroach 在 RocksDB 之上；
- Omid1 与 Tephra
	- Omid1 与 Tephra 将所有 committed transaction 与 aborted transaction 信息存储于 TM 的内存，乃至传播给客户端；这一方案不能很好 scale，传播给客户端的数据量经常在几 mb 级别；
	- omid 通过将元信息存储在 metadata table，能节约掉这部分带宽开销；
	- 性能行 omid 显著优于 omid1，而 omid1 的设计与 tephra 很相似；
	- 为了故障恢复，Tephra 与 omid1 写入 wal 日志，增长了故障恢复时间（需要应用日志）；而 Omid 直接利用下层存储的高可用机制，减少恢复时间；
- Percolator 也是用了中心化的 timestamp 分配，但是通过两阶段提交解冲突；
- spanner 与 cockroachDB 通过同步的时钟服务分配 timestamp；spanner 同样使用两阶段提交，而 cockroach 使用分布式的 conflict resolution 算法，可能导致只读的事务被 abort；omid 绝对不会中断只读事务；
- 这些线上系统一般提供 SI 隔离，不过 omid 可以增加 serializability 支持，类似 omid1 的 serializability 插件；本质上它属于覆盖到 read set 的 conflict analysis，可能影响到性能；
- 近期有一些工作尝试绕过 2pc 的开销，通过一个全局的 serialization service 比如高可用的日志服务、total ordered multicast；
4 Service Semantics and Interface
- Semantics
	- SI 隔离级别下，仅当同时更新同一个条目时，才会出现冲突；
	- Serializability 隔离级别下，可能因为读取一个条目而发生冲突；
	- 对于只读场景，SI 隔离级别有更好的扩展性；
- API
	- 客户端 API 包括事务控制接口（begin、commit、abort） 和数据访问接口（put、get）；
5 Transaction Processing
5.1 Data and metadata
- Omid 通过在 commit 时做冲突探测实现乐观并发控制；
- Omid 为每个事务记录两个 timestamp：1. read timestamp：事务开始的时间；2. commit timestamp：事务 commit 的时间；
- 每个事务有唯一的 tx id，取事务开始的时间戳；
- 事务 commit 需要保证原子性，omid 在一个 commit table 中跟踪 commited transaction；
	- 每当 commit 一个 transaction，TM 在 CT 中写入 (txid, TSc)
	- get 操作根据 txid 查询 CT，确认读到的数据是否已 commit；
- 每次读取操作查询 CT 可以保证正确性，然而存在查询开销，为了避免这部分开销，omid 在每行记录中冗余的 commit timestamp 列 (CF)
	- 默认 cf 列为 nil，表示这行数据为 tentative 状态；
	- commit 时， TM 会更新行数据的 cf 列，并删除 CT 中的记录；
	- 后台清理进程会负责已过期数据的清理；
	- 读取 tentative value 时，如果发现 CT 中已经 commit，则更新到行数据的 cf 中（类似设置缓存，减少 CT 查询）
5.2 Client-side operation
- 事务乐观并发控制体现在 commit 阶段；
- Snapshot 隔离级别只需要探测写冲突，因此只需要跟踪事务的 write-set；
- Begin(): 客户端从 TM 得到 read timestamp 作为 txid;
- Put(key, val)：客户端将 tentative 记录写入下层存储，跟踪它的 key 在本地 write-set 中；
- Get(key)：(似乎没有考虑事务内读到本事务 put 的数据，只是打了一个 begin 时刻的快照)
	- 从下层存储中遍历 key 的所有小于等于 read timestamp (TSr)版本的记录，从最新到最老；返回第一个 TSc 小于 TSr 的版本数据；
	- 如果读取到 tentative value（cf 列值为 nil），则调用 GetTentativeValue() 在 CT 中查找 TSc；
	- race condition：在更新记录的 cf 与删除 CT 中的记录之间，可能有其他事务在 CT 删除记录那一刻读取过，为了避免这一场景，在 CT 中不存在记录时，选择重新读取一次下层存储取当前 cf 值；
	- 所有的场景中，都是返回从新到老遍历所有小于 read timestamp 满足 TSc 小于 TSr 的第一条记录；
- Commit()：
	- 客户端携带 txid、write-set 请求 TM 的 commit() RPC；
	- TM 返回一个 commit timestamp (TSc)，并探测冲突；
	- 如果没有冲突，则将 (txid, TSc) 写入 CT，随后客户端将 write-set 的行记录更新到 cf，最终删除 CT 中的记录；
	- 后台进程持续清理数据，用以应对客户端故障场景；
5.3 TM Operation
- Begin returns once all transactions with smaller commit timestamps are finalized, (i.e., written to the CT or aborted).
- CT 的写入操作是批量的；begin() 与 commit() 操作都需要等待批量写入操作的成功：begin 等待更早的事务的持久化；而 commit 等待提交中的事务；
- Conflict Detection 函数通过内存中的一个哈希表做冲突检测；
- Conflict Detection 需要：
	- 1. 确保 write-set 中的 key 对应的版本没有大于表中的行数据的 TSc；
	- 2. 如果成功，则更新表数据，
tbd