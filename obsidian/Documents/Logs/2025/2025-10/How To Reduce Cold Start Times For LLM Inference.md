tldr：

- 作者将一个 llama 7b 模型的拉取时间从 6min 下降到了 40s 水平
- 两个思路：1. 预热下载 TGI 这种常用的基础镜像到宿主机；2. 使用 s5cmd 来加速镜像参数的下载；
- ==将 llama2 7b 模型（12.6GB）从 85s 下降到了 7s==，能跑到 14.4Gbps，能够跑到和 EBS volume 相近（16Gbps）的水平
- 也在调研使用 stargz 来加速基础镜像的下载；

---

如果 cold start 足够快，需要的 warm pod 就可以减少；

大部分时间都在 pull docker image 和下载权重。

**Faster Pod Initialization**  

> We tried a few ideas and eventually optimized away this portion of time by caching images onto the nodes using Kubernetes daemonsets.

最后的办法是通过 daemonset 将 image 缓存在 k8s 的 node 上。

一个 cacher 会定期扫描所有高优的模型，然后会构造一个 daemonset，更早地拉取到本地。

在 image caching 之外，也会使用 ballon deployment 来 prewarm 节点。

**Faster Model Weights Loading**

作者使用 s5cmd 来下载模型权重，会比 aws-cli 快的多。

作者也曾尝试将所有文件放在一个 tar ball 里，但是发现这样对 concurrent download 不友好。

作为替代，作者将模型文件按 2gb 分块，使用 s5cmd 来并行下载。—numworkers 选择了 512，—concurrency 配置了 5。

在这个优化之后，将 llama2 7b 模型（12.6GB）从 85s 下降到了 7s，能跑到 14.4Gbps，能够跑到和 EBS volume 相近（16Gbps）的水平。

**Summary**

在这些优化之后，作者成功将冷启动时间从 6min 下降到了 40s。因为下降到了 40s，这使得作者可以在不稳定的 workload 时，仍保持 0 worker。

> To better generalize our image cache beyond just “high-priority” endpoints, we are investigating lazy-loading of images with projects like [stargz](https://github.com/containerd/stargz-snapshotter/tree/main).

目前只有常用的 cache 在里面，作者也在调研使用 stargz 的 lazy-loading 能力来加速镜像拉取。