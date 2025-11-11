
- 短时间内可以在数据中心中实现 5~10us 级别的 rpc
- 系统设计者长期以来都更多地在容忍延时问题，而不是改进它，宁愿牺牲延时换取吞吐等其它目标；
- 现在（2011年）HPC 领域已经利用专有的硬件和软件实现高速的通信；

2 A History of High Latency

- 30 年里延时只降低了 30 倍，而带宽已增长了 3000 多倍
- 对于几千台机器的大集群而言，Round-trip 时间大约在 200~500us 级别，而拥塞导致的 spike 可以高到几十 ms；
- 实际上高延时并不是固有的，开发者往往选择牺牲延时去达成其他目标；
- 比如交换机往往会开一个大 buffer，因为：1. 网络总是 oversubscribed；2. tcp 在丢包时表现较差；悲剧是 buffer 用的越多延时也就越高（排队？）；
- 操作系统和 NIC 都倾向于优化吞吐而非延时，比如很多 NIC 会等待 30us 攒足够多 packet 之后再触发中断；

3 Impact on Applications

- 虽然系统开发者容易接受牺牲延时换取吞吐的现状，但对应用开发者来讲延时仍是无法回避的负担；
- 比如， facebook 的每张页面后端只能发 100~150 次顺序网络请求；
- 为此，facebook 选择并发地发送请求、读取解范式的数据，这增大了应用开发的复杂度；
- 鉴于应对延时问题的困难，近年开发了许多框架来照顾对延时不敏感的应用，如 MapReduce；
- 因为磁盘 IO 的瓶颈，过去网络延时并不为人所重视；但是现在 SSD 乃至内存存储的普及，网络已成为了首要的延时来源；
- facebook 的 memcache 延时超过 80% 来源于网络；
- 如果网络延时可以优化 10 倍，那么应用的开发方式可以有革命性的换代；应用程序将更容易开发，省去周旋在延时问题上的精力；
- 作者认为，更重要的是延时的降低可以促使诞生一批新的、更具交互性的数据密集型应用；
- “It is hard to pre- dict the nature of these applications, since they could not exist today, but one possible example is collaboration at the level of large crowds (e.g., in massive virtual worlds); another is very large-scale machine learning applications and other large graph algorithms.”
- 存储系统中的一些固有问题也可以得到改善，如强一致性在并发执行的昂贵，使很多 NoSQL 系统作出牺牲一致性的设计；如果系统的延时足够小，事务执行得更快，事务重叠执行的概率就会降低，从而减少冲突的可能性；
- “We speculate that low-latency sys- tems may be able to provide stronger consistency guar- antees at much larger system scale.”
- 低延时也可以降低网络的 incast 问题；因为延时高，所以应用程序倾向于多路复用地发请求；然而，这使响应同时返回，可能导致客户端拥塞或者交换机 buffer 溢出而丢包；
    

4 Low Latency is Within Reach

- HPC 社区已验证可以通过专有的 interconnect 实现低延时，到现在 commodity 硬件也足够达到同样的程度了；
- 以 100m 的铜线作为介质，光速极限的延时可以在 1us 左右；这意味着可以实现大规模的 ms 级低延时；而数据中心更是得天独厚；
- 新的 10g 以太网交换机芯片已显著降低延时和带宽；比如 Arista 基于 Fulcrum Microsystem 芯片的交换机，可以将交换的延时降到 1us 以下；
- 新的交换芯片也使带宽变得充裕；目前的多数数据中心网络的带宽往往不能负担所有节点随机地以全线路速率通信；造成的结果是，上层交换机严重地 oversubscribed；
- 上层交换机的总带宽通常要小于各服务器总带宽 100~500 倍；
- Oversubscription, in a SAN (storage area network) switching environment, is the practice of connecting multiple devices to the same switch port to optimize switch use.
- Mellanox 等公司 NIC 在两个方面显著地降低了延时：1. 通过 NIC 本身的优化降低延时；2. 允许用户态直接访问，减少途径内核的开销（Bypass kernel），支持 polling 模式，可以避免中断和上下文切换开销；
- HPC 社区已经在专有系统中综合地利用并验证了上述改进；通过 infiniband 交换和 Mellanox 的 NIC，在小规模网络中使用类似 TCP 的可靠传输协议的往返时间可以低于 5us；上述改进有望进入主流网络，然而不幸的是，这些系统的 interconnect、协议乃至 API 都与目前广泛部署的 commodity Ethernet/IP/TCP 方案差异较大；
- HPC 方案大规模推广的可能性并不大：1. Ethernet 的市场支配地位，其 economies of scale 难以与之竞争；2. HPC 将逻辑转移到 NIC 允许更复杂 offloading 的策略，使得 NIC 更加昂贵，更重要的是使网络的适应性变差；反观 Ethernet 因为简单，鼓励了推广和 innovation；
    

5 The OS Community’s Role

- 直到不久前，外部因素如长距离传输的光速限制、数据中心较慢的交换性能，使操作系统内的延时问题并不显著；然而再过几年，操作系统将成为数据中心内 RPC 延时的主要来源；
- 操作系统社区应该设置一个目标，使主流数据中心在几年内实现 5~10us 的 RPC 延时成为可能；
- 首先要做的是建立一个新的 vision 重新规划 NIC、操作系统、应用之间的职责；
- 应该允许用户态直接操作 NIC，将网络操作视为内存操作的类似物，操作系统提供基本的隔离支持，用户态对硬件的操作不必经过操作系统中转；
- 新的网络架构应该基于 polling 而非中断；
- 然而，当多个应用同时 polling 时不容易 scale；
- 前几年 NIC 供应商在尝试尽可能地从 CPU 负载 offload 到 NIC，但是本文作者认为这条路线是错误的；
- 烧进 NIC 的功能很难变更；
- NIC 应该尽可能地保持功能精简，专注于与 CPU 交换数据；
- 低延时的实现也需要新的网络协议；
- TCP 为单向数据流优化，而非对双向的 RPC 场景优化；目前实现的往返延时有 25~50us；
- TCP 对数据中心的 switch fabric 适应较差；如果开启为了降低 congestion 的 randomized routing，那么 TCP 的表现将变差；TCP 面向 incast 场景的表现也不是很好；
- 数据中心网络作为一个封闭网络，比较容易试验新的网络技术；

6 Pushing the Envelope: Integrated NICs

- 1us 的延时要求将 NIC 功能集成进 CPU 芯片；
- 而且 NIC 必需有能力直接访问 CPU 缓存，在 hot path 必需不能有内存访问；
- 作者认为快速的网络通信之于数据中心就像快速的浮点运算器之于科学计算；