
## 2015-2017: Laying the CNI foundation with kubenet

一开始使用 kubenet，一个基础的网络插件。

它会在每个机器上开一个 bridge，分配一个 CIDR，在 CIDR 里为 pod 分配 IP。

但是随着 k8s 规模的扩张，遇到了管理 IP 的问题和在 VPC 中高性能通信的问题。

因此，作者开始搞了一套和底层的 cloud 网络集成度更高的方案。

## 2018-2019: Embracing VPC-native networking

提供了 vpc native networking 能力。能够允许 Pod 直接获得 VPC 网络中的 IP。

VPC native networking 成为了 GKE 上默认的网络方案，允许将 GKE cluster 扩展到 15k 级别。

优势：

1. 简化 IP 管理：直接从 VPC 中申请 IP；
2. 利用 VPC 基础，提高安全性：可以对 Pod 直接应用 VPC 网络的防火墙规则；
3. 性能提高、扩展性提高
4. 为更高级的 CNI 特性打好了基础

这个阶段，网络方案也被称作 GKE Standard Networking with Dataplane v1（DPv1）。

## 2020 and beyond: The eBPF revolution

GCP 利用起来了 Cilium，做了 GKE Dataplane V2 架构。现在是 GKE 上默认的网络架构。

## 2024: Scaling new heights for AI Training and Inference

GKE 最大跑到了 65000 个节点。

> For AI/ML workloads, GKE Dataplane v2 also supports ever-increasing bandwidth requirements such as in our recently announced [A4 instance](https://cloud.google.com/blog/products/compute/new-a4x-vms-powered-by-nvidia-gb200-gpus?e=48754805). GKE Dataplane v2 also supports a variety of compute and AI/ML accelerators such the latest [GB200](https://cloud.google.com/blog/products/compute/new-a4x-vms-powered-by-nvidia-gb200-gpus?e=48754805) GPUs and [Ironwood](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/), [Trillium](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus?e=48754805) TPUs.

现代的网络挑战：

1. Extreme Throughput
2. Ultra-low latency
3. Multi-NIC 能力：需要允许为 pod 提供多个 NIC，来提高网络的吞吐能力；

## 2025 - Beyond CNI: addressing next gen Pod Networking challenges

针对网络，提供 Dynamic Resource Allocation 特性。

DRA 一开始是针对 GPU 做的抽象，不过也能应用在网络上。

KND 通过 DRA，将宿主机上的网络设备透露给 Pod。

## Looking ahead: Innovations shaping the future

KND 有一个 reference 实现的项目叫做 DRANET。