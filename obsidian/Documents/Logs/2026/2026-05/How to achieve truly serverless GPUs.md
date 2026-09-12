> There are four key ingredients:
>
> - **Cloud buffers**: maintain a small buffer of healthy, idle GPUs to take on new load
> - **Custom filesystem**: serve container images lazily out of a content-addressed, multi-tier cloud-native cache
> - **Checkpoint/restore**: fast-forward through CPU-side initialization by directly restoring processes into memory
> - **CUDA checkpoint/restore**: fast-forward through GPU-side initialization by directly restoring CUDA contexts into memory

## 影响 scale 的因素：启动时间

启动涉及的步骤：

1. 启动新的 VM/Pod 并进行健康检查（分钟～数十分钟）
2. 加载应用程序和文件系统（分钟级）
3. 在主机上启动应用程序并准备好处理请求（数十s）
4. 在 GPU 上启动程序并准备好请求（分钟～数十分钟）

Modal 的优化：将上述各步骤从数十分钟压缩到几s到几十s。

从原来的约 2000 秒降至约 50 秒。

## You can remove tens of minutes of latency by taking instance allocation and health checks out of the hot path.

维护一个资源池。

## caching

使用 libfuse 开发了一个 ImageFS，实现 lazy loading 和多级的内容寻址的 cache。

本地 SSD 可以在 4GiB/s，AZ 的缓存服务器在 10GiB/s、region 内的 CDN 在 3～10GiB/s。

readahead_kb 从 128 提升到 32x1024。

跳过 gzip 解压（单线程瓶颈在 100MiB/s）。

## Checkpointing

使用 gVisor 的 runsc 运行时，使用它内置的 CPU checkpoint 机制。

`enable_memory_snapshot=True` 之后，import torch 可以从几s优化到几乎瞬时。

需为每种硬件生成独立快照。

## 设备快照

驱动将设备内存检查点到主机内存，再由主机端快照系统（如runsc/CRIU）存入磁盘；恢复时反向操作。也同样通过 image fs 加速分发。

典型加速比为 **4-10倍**，启动时间从数分钟降至数十秒。

实测数据（Qwen 3 0.6B模型）：

- vLLM：平均启动时间从95.7秒降至13.8秒
- SGLang：平均启动时间从83.7秒降至17.5秒

<mark>多GPU程序快照较困难</mark>（NCCL 通信会死）。

大多数应用需小幅调整：如开启权重卸载（快照前将权重移回主机）、跳过KV缓存快照（重新创建更快）。
