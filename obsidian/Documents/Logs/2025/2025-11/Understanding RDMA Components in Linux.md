在 linux 系统中，RDMA 涉及以下组件：

1. RDMA 设备（Host Channel Adapter - HCA）：比如 Mellanox ConnectX，通常会命名为 mlx5_0, mlx4_0, hfi1_0，会处理 kernel bypass、zero copy 的数据移动、硬件抽象；
2. **/dev/infiniband/uverbsN**（Users Verb Device）：是一个特殊的字符设备，比如 /dev/infiniband/uverbs0，通过 ib_verbs 模块进行暴露，它们扮演 RDMA 设备的 user-space 接口；它们会负责扮演控制面、Memory Pinning 管理（Pin 了之后才能交给 HCA 来操作）、Enable Kernel Bypass 控制；
3. RDMA Network Device（Standard Linux Network Interface）：表示在 HCA 基础之上的，Linux 标准网络设备；这个设备有 IP 地址，位于标准的 TCP/IP 栈中；对于 IB，通常叫做 ibX 之类的名字，会通过 IP over InfiniBand 进行创建，能够允许 IP 包在 IB 网络中传输；对于 RoCE 或者 iWARP，通常就是现有的网络设备比如 eth0、enp1s0f0；
4. RDMA Link device：执行 `rdma link` 时，会显示出来 HCA 和标准网络设备之间的“link”；表示底层的 RDMA 网络设备和上层网络设备之间的关联关系；它并不是真正的一个设备类型，而是一个关系的表示；

## In summary

- RDMA 设备（HCA）是专用的硬件；
- /dev/infiniband/uverbsN 是 Linux 内核提供的用户态接口，用来控制这块硬件；
- RDMA 网络设备是基于 HCA 构建的标准 IP 可寻址接口，用于通用网络通信；
- 而 RDMA 链路设备（rdma link show 可见）描述了 RDMA 设备与其网络接口之间的直接关联。

所有这些组件协同工作，依赖一个正常的网络 fabric，才能实现 RDMA 所特有的高性能、低延迟数据传输。