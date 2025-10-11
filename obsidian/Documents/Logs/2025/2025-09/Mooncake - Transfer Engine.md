两个抽象：

1. Segment：表示一段连续的内存空间，允许远程读写，比如 DRAM、VRAM，或者持久的 NVMeOf 存储；
2. BatchTransfer：封装了操作的请求，比如从 Segment 中一组非连续的数据，到另一个 Segment，允许异步传输，以及更灵活的 AllScatter/AllGather；

![[Pasted image 20250907200336.png]]

每个 Segment 下面可以有多个 Buffer，不同的 Buffer 可以有不同的网卡 Affinity。

TransferEngine 接口都来自类 `TransferEngine`，底下有 `TcpTransport`、`RdmaTransport`、`NVMeoFTransport`。

## Segment

Segment 表示一组 source address ranges 以及传输期间可用的 target address ranges。

### 1. RAM Segment in Memory Address Space (DRAM, VRAM)

在进程启动之后，Transfer Engine 会按自己的 `local_hostname` 自动创建一个 segment。这个 segment 中会包含整个的内存地址空间，包括 DRAM 和显存。

在使用 BatchTransfer 方法时，Transfer Engine 能够根据硬件条件，自动选择最优的传输方法。

每个进程只有一个这样的 Segment。

其他进程可以通过 `openSegment` 来引用这个 Segment 并进行读写操作。

在真实的部署中，应用一般只会使用一部分内存用于数据传输。Transfer Engine 会将 Segment 进一步拆分为 Buffer。每个 Buffer 指向同一个设备中的一段连续内存。

在使用 `BatchTransfer` 方法时，每个读写任务必须指向合法的 Buffer。

### 2. NVMeof Segment

直接从 NVMe 通过 PCIe 将数据传输到 DRAM/VRAM。

## BatchTransfer

支持这些渠道：

1. 本地的 memcpy：如果是 DRAM 到 VRAM，会用到 cudaMemcpy
2. TCP：支持 DRAM 到远程 DRAM；
3. RDMA：支持本地 DRAM/VRAM 到远程 DRAM；支持 multi-NIC pooling 和 retry；
4. cuFile（GPUDirect Storage）：支持本地 DRAM/VRAM 到本地/远程的 NVMeOf。

BatchTransfer API 接受一组 request 组成的 array，每个 request 有自己的操作类型（read / write），data length、local / remote memory addresses。

这些 transfer 的完成状态，可以通过一个异步的 `getTransferStatus` API 进行获取。

## Topology Aware Path Selection

现代推理机上通常有很多 CPU 槽、DRAM、GPU、RDMA NIC 设备等。

理论上可以通过任何 RDMA NIC 设备，但是这些 transfer 往往会受限于 Ultra Path Interconnect 或者 PCIe Switch 的带宽。

为了克服这些困难，Transfer Engine 实现了一个拓扑感知的 path selection 算法。

在发送请求之前，每个 server 会生成一个 topology matrix，并广播给整个 cluster。

这个 matrix 中会针对不同的内存，将 NIC 分发为 preferred 和 secondary list。

> Under normal conditions, a NIC from the preferred list is selected for transfers, facilitating RDMA operations within the local NUMA or GPU Direct RDMA through the local PCIe switch only. In case of failures, NICs from both lists may be utilized.

![[Pasted image 20250907211547.png]]

> To further maximize bandwidth utilization, if a single request’s transfer is internally divided into multiple slices if its length exceeds 64KB. Each slice might use a different path, enabling collaborative work among all RDMA NICs.

如果请求的 size 超过 64kb，可以拆解为多个 slice，那么每个不同的 slice 可以使用一个不同的 path，从而利用起多个 RDMA NIC。

## Endpoint Management

> Mooncake Store employs a pair of end- points to represent the connection between a local RDMA NIC and a remote RDMA NIC. In practice, each endpoint includes one or more RDMA queue pair objects. Connections in Mooncake Store are established in an on demand manner; endpoints remain unpaired until the first request is made. To prevent a large number of endpoints from slowing down request processing, Mooncake Store employs endpoint pooling, which caps the maximum number of active connections. We use the SIEVE algorithm to manage endpoint eviction. If a connection fails due to link errors, it is removed from the endpoint pools on both sides and re-established during the next data transfer attempt.

没有请求时，endpoint 之间默认先不配对。为了避免大量 endpoint 影响 request 处理，mooncake 做了一套 endpoint pooling，使用 SIEVE 算法来管理 endpoint 的 eviction。

## Fault Handling

当 mooncake 发现一个连接不好使之后，会自动找另一条可靠的 path。