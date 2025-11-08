https://www.usenix.org/sites/default/files/conference/protected-files/nishtala_nsdi13_slides.pdf
http://www.quora.com/How-does-the-lease-token-solve-the-stale-sets-problem-in-Facebooks-memcached-servers

- 基础需求：1. 接近实时的通信；2. 多数据源的实时聚合；3. 允许访问修改高访问量的内容；4. 扩展到百万 qps；
- 设计层面的目标：1. 需要支撑海量读压力；2. 异地分布；3. 支持产品的快速变动；4. 外部持久化；
- 每个修改总会或多或少影响到用户，如果不是显著的优化，就不多考虑；将读到过期数据的可能性视为可以调适的参数；
Reducing Latency
- 每个请求往往包含几百个 mc 请求；缓存条目按照一致性哈希进行分布；结果是，所有的 web 服务器会访问到所有的 mc 服务器；这种 all-to-all 的访问模式，很容易导致 incast congestion，也容易使单机成为瓶颈；Replication 可以缓解单机瓶颈问题，但会牺牲过多的内存；
- 在优化延时问题上，作者更多地关注于客户端；它部署于每个应用服务器上，会负责序列化、压缩、request routing、故障处理、request batching 等工作；客户端无状态，可以作为内嵌在应用中的库，也可以作为一个独立的代理，即 mcrouter；客户端会把请求组织为 DAG，按照数据的依赖关系 batch 起来；平均每个 batch 约 24 个 key；
- 客户端对 get 请求使用 UDP 来减少延时和开销；对于 UDP 的乱序、丢包，客户端不尝试做任何恢复，只视为缓存 miss，这时不会读后端存储的内容不会写入 mc，以减轻 mc 负担；在负载峰值（peak load），0.25% 的 get 请求会因为 UDP 而 miss；其中又有 80% 是因为超时或者乱序；统计显示使用 UDP 使延时降低了 20%；
- 客户端对 set/delete 请求会经本地的 mcrouter 走 TCP；TCP 连接的成本较高，如果每个 web 线程都对应一个 mc 连接，那么 TCP 连接开销会比较昂贵，为此 mcrouter 会将连接合并；
- 为了缓解 incast congestion 问题，client 实现了类似滑动窗口的机制，控制并发请求的数量；当探测到拥塞时，缩小窗口；
Reducing Load
- Lease 机制：
	- 普通的缓存 miss 更新存在 stale set 问题，当并发地读写同一条缓存时，写入请求可能被乱序，而导致最终的缓存与数据库不一致；
	- 也存在惊群问题，当一条缓存读写都很频繁时，读操作频繁地 miss 导致频繁地访问数据库；
	- 为此引入 lease 机制：
		- 当缓存 miss 时，mc 返回给客户端一个 lease，到再次写入时携带该 lease；
		- 期间其它 get 请求将得到一个 hot miss 报错，但不能得到 lease，只能稍后重试，从而限制同一时刻只有一个数据库请求；
		- 期间如果收到 delete 请求，那么会使该 lease 失效；持有该失效 lease 的 set 请求仍将成功，但后来的 get 请求将得到 hot miss 报错，并携带一个新的 lease；留意这里的 hot miss 报错中带有最后的值，但认为它处于 stale 状态，留给客户端去判断是否采用它，在一致性要求不严格的场景中可以进一步减少数据库请求；
	- 之前峰值的数据库请求量有 17k/s；应用 lease 机制之后，数据库的峰值请求量降到了 1.3k/s；
- Memcache Pools：将机器分割到不同的 pool 里以服务不同的访问模式；
- Replication Within Pools
Handling Failures
- 两种故障：1. 少数机器宕机或者网络故障，则尽力恢复；2. 整个集群宕机，则将流量切到其它机房；
- 对于小规模故障，依靠自动恢复系统；恢复过程有几分钟，而几分钟已足够产生级连故障；为了缓解故障恢复期间的压力，作者预留了少数称为 Gutter 的机器（1% 左右）；若 get 请求超时，客户端将把查询数据库的结果写入 gutter；
- gutter 方案和重平衡一致性哈希的方案有所不同，在于重平衡方案在键访问不均匀时存在级联故障的风险；
- 在实战中，gutter 方案使客户端可见的故障减少了 99%；如果一台 mc 服务器完全宕机，4 分钟内 gutter 即可承担到原本 35% 甚至 50% 的负载；
In a Region: Replication
- 将 web server 和 mc 服务器分割为多个 frontend cluster；多个 frontend cluster 共享同一个 storage cluster；分割允许更小的故障范围、更精细的网络配置、更小的 incast congestion；
- Region 内 invalidation：
	- 多 frontend cluster 共享同一个 storage cluster 意味着数据库的同一份数据可能在多个 frontend cluster 中存在多个副本；
	- 修改关键数据的 SQL 语句会携带要在 commit 之后失效的 mc 键列表；fb 对每个数据库部署了一个 invalidation daemon（即 mcsqueal），它会解析 binlog，将失效请求广播到 region 内所有的 frontend cluster；实际的运行中，来自 mcsqueal 的失效请求只有 4% 真正失效了缓存；
	- mcsqueal 会将多个失效请求打包在一起交付给每个 frontend cluster 中的 mcrouter；这里的批量使每个 tcp 请求中键数量的中位数提升了 18 倍；
- Regional Pools：
	- 每个 frontend-cluster 独立的缓存，导致多个 web 服务可能过多地重复缓存；为了缓解重复缓存，引入 Regional Pool，允许多个 frontend cluster 共享同一批 mc 服务器；
	- 跨 frontend cluster 通信的带宽较小，延时更高；这是一个 mc 网络开销和内存开销之间的 trade off；
- Cold Cluster Warmup：
	- 上线新集群时，Cold Cluster Warmup 系统会通过客户端从 warm cluster 抓取数据到 cold cluster，允许 cold cluster 几个小时即可上线（而不是几天）；
	- 代价是部分缓存失效无法及时反映到 warm cluster；但是运维上的便利性更高；
Across Regions: Consistency
- 每个 region 有一个 storage cluster 和多个 frontend cluster；设定只有一个 region 有 master 数据库，而其它 region 都是 slave；这里的挑战在于主从同步的延时；
- 面向 non-master 机房的写请求，会向 master 数据库写，但是主从同步存在延时，再从 non-master 机房读到的将不是最新数据；
- 为了避免这个问题，会在本机房的 mc 集群中设置 stale marker，标记一条数据处于 stale 状态，下次读取将打回 master 机房，写入本机房缓存之后删除 stale marker；
Single Server Improvement
- 添加了多线程支持、哈希表的动态 rehash、给每个线程自己的 UDP 端口；
- 实现 Adaptie Slab Allocator；
- 实现 Transient Item Cache；