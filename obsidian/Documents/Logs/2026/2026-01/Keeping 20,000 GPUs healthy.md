## Instance type testing and selection

作者维护了一个 benchmark `model-host-bench` 来衡量不同云厂商上实例的性能的影响因素。

很令人意外的是，不同云厂商上差异意外的明显。

| **Category**                    | **Cloud D H100 SXM** | **Cloud B H100 NVL (PCIe)** | **% diff** |
| ------------------------------- | -------------------- | --------------------------- | ---------- |
| `torch_matmul_duration_seconds` | 1.62                 | 2.72                        | 67.5%      |
| `torch_matmul_flops`            | 678 TF/s             | 405 TF/s                    | -40.3%     |
| `h2d_bw_pageable_1024`          | 7.68 GiB/s           | 21.0 GiB/s                  | 174%       |
| `h2d_bw_pinned_1024`            | 49.1 GiB/s           | 51.2 GiB/s                  | 4.40%      |
| `d2h_bw_pageable_1024`          | 14.3 GiB/s           | 20.9 GiB/s                  | 46.0%      |
| `d2h_bw_pinned_1024`            | 50.7 GiB/s           | 53.4 GiB/s                  | 5.30%      |

## Machine images

Our images keep up with the latest production NVIDIA driver version ([580.95.05](https://www.nvidia.com/en-us/drivers/details/250991/)) for security, performance, and new features.

在构建镜像之后，会跑 system tool tests （比如 DCGM 的检查）以及自定义的测试脚本：

```
provisioner "shell" {
  script = "./setup/check_nvidia_ctk.sh"
}

provisioner "file" {
  destination = "/tmp/modal/"
  source      = "./.bin/modal-healthcheck"
}
```

Cloud C is the fastest to boot a new VM with our machine image, <mark>averaging just under 2 minutes</mark>. 一些别的云厂商做到 5 分钟内启动都比较难。

Although the hyperscalers are not significantly differentiated in their machine image feature and reliability, cloud D has _extremely_ slow regional image replication, taking <mark>3 hours</mark> to replicate to 10 regions.

## Instance boot

如果在一个不健康的 GPU 的主机上启动，或者我们的 cloud-init 脚本有bug，我们需要在客户使用这些 GPU 之前了解这一点并进行干预。

这里有一个权衡是，增加检查可能会将启动时间变慢。变慢会增加客户的调度开销，更糟的是，增加启动时间的延迟，实际上会降低 failover 时的可靠性。

在机器上可以跑的最深度的检查是 `dcgmi diag --run 4`，它能够找到很多长尾问题，但是需要跑一个小时。

最浅层的检查 `dcgmi diag --run 1` 也需要一分钟。

启动时的这部分检查可能会跟云厂商做的检查重叠。

在实例启动时，作者最终一般只采取轻量级的检查，比如 systemctl 查询、nvidia-smi 查询、随机选择 GPU (0~7) 进行基本的读写操作。

> Today, we almost never have GPU problems slip through and hit user containers. The one irksome issue we have in production is that Cloud C’s [L4s flake at CUDA initialization](https://modal.com/docs/guide/troubleshooting#cuda-driver-initialization-failed-on-l4-gpu-type) in 0.1% of cases. Application code targeting those cards must use `cuInit` retries.

如今，作者可以做到几乎不会使有问题的 GPU 漏网；唯一一个令人烦恼的问题是，cloud C 的 L4 卡，在 0.1% 的情况下在 CUDA 初始化时会遇到问题，针对这些卡的程序必须使用 `cuInit` 进行重试。

## Lifetime management

在机器启动之后，就开始在上面跑工作负载了，在跑工作负载的期间，会有两种层面的检查：

- **被动健康检查**不在GPU上运行，是非侵入式的、只读的。被动数据流包括 `dmesg` 和 `dcgmi health`。
- **主动健康检查**独占一个GPU设备，并通过读写操作来获取健康数据。`dcgmi check` 和 `nvbandwidth` 就是例子。

### Passive healthchecking

用20%的工作量，你可以获得80%的被动健康检查收益。

定期检查 dcgmi 和 dmesg 中最常见的问题。比如，dcgmi 可以告诉你，特定的 GPU 上不可纠正的 ECC 错误。我们也可以被动地发现 GPU thermal violations, sync boost violations, hardware slowdowns, and excessive temperatures (> 88°C) 等问题。

Cloud C 直到几个月前还存在严重的冷却问题。作者见过 Cloud C 的GPU达到 94°C。

### Active healthchecking

主动检查需要 GPU 的独占锁定，因此调度起来更加复杂。

根据SemiAnalysis的ClusterMAX预期，我们确保每个GPU节点至少每周进行一次深度的主动检查。

虽然我们已确认底层云提供商执行自己的深度主动健康检查，但他们显然无法在我们占用实例时进行他们的检查。

我们的大量实例容量来自短期租赁（<24小时），所以我们不会像租赁数月机器的平台那样经常遇到这种情况。

不过，我们确实有一些长期存在的 capacity。每周我们保有一个实例时，我们运行以下主动检查：

- NVIDIA DCGM 诊断级别2
- GPUBurn/GPU-fryer - 用于验证GPU在负载下不会失败
- 本地 NCCL allreduce 测试，用于验证 NVLink/NVSwitch/NVLink SHARP 性能

如果这些检查失败，我们会收到警报，该实例不被允许继续接受任务，有时我们会"隔离"该实例供我们或底层提供商进行分析。

作者也在添加这些面向网络的主动检查：

- 本地 InfiniBand allreduce 测试，用于验证 InfiniBand 性能和连接（通过强制禁用NVLink/p2p/SHM）
- 成对的 CPU 和 GPU ib_write_bw及ib_write_latency双向测试，以验证网络是否符合参考数字的规格。

### Taking action

In theory it’s possible to recover from some unhealthy GPU states by isolating and resetting the GPU. In practice, for us, this is overcomplicated and no guarantee of recovery. **So instead we automatically mark the entire host unhealthy, drain it, and then either dispose of it or reinstall.**

## Observability

![[Pasted image 20260111173919.png]]

Going beyond metrics, we also pipe abnormal GPU health events into dashboard container logs. See the informational “gpu-health” lines in the screenshot below (indicated with purple).

Our guide documentation maintains [a detailed Xid and sXid dictionary](https://modal.com/docs/guide/gpu-health) for understanding errors. We think it’s the best GPU error resource on the internet. 