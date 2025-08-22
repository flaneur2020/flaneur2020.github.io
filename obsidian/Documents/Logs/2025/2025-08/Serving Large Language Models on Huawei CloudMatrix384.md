## 3.2 CloudMatrix384 Overview: A Fully Peer-to-Peer Hardware Architecture

集成 384 个 Ascend 910 的 NPU。

通过 UB 协议，链接所有的 NPU 和 CPU。384 个 NPU 和 192 个 CPU 通过 UB switch 连接。

跨节点通信的效率可以和节点内的通信效率媲美。

inter-node 的 latency 只增长 1us。跨节点的带宽开销低于 3%。

matrix384 有三种 network plane：

1. UB Plane：用于 supernode 内部通信，non-blocking all-to-all 拓扑；可以用于高性能的 fine-grained 并行策略，比如 TP 和 EP；高速访问池化的内存，即 KV cache 和模型权重；
2. RDMA Plane：快速的 KV cache transfer、分布式训练中利用 RDMA 兼容的技术栈、与其他超节点之间低延迟的通信
3. VPC Plane：高速 NIC，提供标准的以太网和 IP 协议。

## Hardware Component

### 3.3.1 Ascend 910 Chip

每个芯片是一个两个 die 的封装，两个 die 之间通过片上互联通信。
 
每个 Ascend 910 die 直接集成 UB Plane 和 RDMA Plane。

### 3.3.2 Ascend 910 Node

CPU 节点，四个 Kunpeng CPU socket 组全互联的 NUMA 拓扑。

其中一个 CPU 带着整个节点的 Qingtian 卡，一个专门的 DPU。Qingtian 卡，可以作为节点主要的 南北向的 egress 入口，和 VPC plane 交互。

### 3.3.3 UB Switch System

超节点拆成 16 个 rack，12 个 compute rack，包括 48 个 Ascend 910 node（总共 384 个 NPU），4 个通信 rack。

通信 rack 负责 L2 的 UB switch，对所有的节点做交换。

## 3.4 Software Stack

CANN（Compute Architecture for Neural Networks）。

CANN 可以和 pytorch 之类的框架集成，也负责 Ascend NPU 的底层硬件交互。

可以将抽象的计算图，翻译成优化的指令。

#### CANN 架构

CANN 软件架构主要有三层，driver、运行时、library。整体上会和 Nvidia 的 CUDA 生态比较对齐。

1）Driver Layer：kernel 模块、固件，组成一个操作系统和 Ascend NPU 之间的底层的接口。负责设备初始化、资源分配、命令调度、跨 NPU 通信等。

2）Runtime Layer：是程序在 Ascend NPU 上跑的核心执行引擎。管理程序的执行生命周期，协调 model 的计算、提供设备的控制、内存管理、model 的执行管理以等等。

3）Library Layer：提供高度优化的软件栈来加速 AI workload。包括专门的加速库（AOL）、Huawei Collective Communication Library（HCCL）、一个可扩展的算子包（OPP）、神经网络加速引擎（NNAE）、离线推理（NNRT），以及支持自定义算子的开发的 Ascend C，乃至和三方库的对接。

在核心的 layers 之外，graph engine（GE）的编译并优化来自 pytorch 等框架的 computation graphs。

将高级的 model 和底层的图优化，比如 operator fusion、memory planning、dynamic shape handling、scheduling 等工作。

### 3.4.2 Infra Software for Cloud Deployment



