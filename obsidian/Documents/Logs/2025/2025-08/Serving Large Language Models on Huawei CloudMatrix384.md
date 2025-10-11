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

- MatrixResource：管理在 supernode 内的资源 provisioning，包括结合拓扑的计算实例分配。provision 任务的执行，通过一个跑在 qingtian 卡里的 MatrixResource Agent 进行。
- MatrixLink：QoS、dynamic routing。管理 link level 的配置，管理 network aware 的 workload placement，也是在 qingtian 卡里跑的。
- MatrixCompute：管理 cloudmatrix 实例的生命周期，包括 baremental provisioning 到 auto-scaling 乃至 fault recovery。
- MatrixContainer：提供基于 k8s 的容器服务，增强了 topology-aware scheduling。
- ModelArt：在 infra 栈的最上方，提供 end-to-end 的 AI 平台服务。

## 3.5 Suitabiliby Analysis of DeepSeek Models

### 3.5.1 DeepSeek Models and their Deployment on Nvidia H800

Deepseek 模型的性质：

1. 671B 的 MoE 架构，37B 的激活参数。在 256 个 router export 中选择 top 8 的 expert。
2. 利用 MLA 来压缩 KV cache 到 93.3%。
3. Multi-token Prediction 支持 decode time validation 的多个 token 生成。
4. FP8 量化训练。

Deepseek 将他们的模型部署在 H800 的集群中，每个卡有 80GB 的内存，卡之间用 NVLink 相连，400 Gbps InfiniBand 跨节点通信。

整个部署是 PD 分离的形式。

在 prefill 阶段：

- deepseek 将 4 个 h800 节点（总共 32 张卡）组成一个单独的部署单元。
- 每个单元内，256 个 router expert 分布在这些 GPU 中，每个 GPU 装 9 个 router expert，一个 shared expert。
- 这个配置，被称为 DP32 + EP32，在 32 个 GPU 中实现 Expert Parallelism。其中 shared expert 和 MLA 机制在同样的一组中按 Data Parallelism 进行 replicate。

在 Decode 阶段：

- Deeseek 将 parallelism 扩展到 DP144+EP144。组了 18 个节点，总共 144 个 GPU。
- 每个 GPU 管理两个 router expert，一个 shared expert。maintaining a system-wide redundancy of 32 router expert replicas。

为了优化吞吐和延迟，deepseek 将一个 dual-microbatch pipeline 策略，将计算和 all-to-all 通信 overlap 在一起。

当一个 microbatch 在 MoE 相关的 dispatch 和 combination 中，下一个 microbatch 会并行地在 local attention 或者 MLP 计算。

这套仔细协同的部署，显著的提高了吞吐。在 prefill 阶段，每个 H800 GOU 可以在 56.3% 的 context caching hit rate 上，跑到 9123 token/s。在忽略 cache 时，可以跑到 4029 tok/s。

在 decoding 阶段，每个 GPU 平均可以跑到 1850 tok/s。

### 3.5.2 Architectural Synergy between CloudMatrix and Deepseek Models

#### MoE Communication Synergy: Efficient Dispatch and Combinations

MoE 架构需要在 token dispatch 和 expert output combination 中涉及很重的跨 NPU 通信。

CloudMatrix384 的高吞吐、低延迟 UB interconnect 对这些需求适配的很好。

在 token dispatch 中，token 必须从 router 到 selected expert 上，可能涉及几百个 NPU。

在 combination phase，多个 expert 的输出必须及时地合并到一起，UB 的高吞吐能够很好的解决这个需求。

#### Memory Capacity and Management：Accomodating Large Models and KV Caches

> CloudMatrix384's generous memory footprint supports these scenarios, but efficient partitioning and synchronization of KV cache across NPUs remain essential。

#### Context Cache Reuse: Accelerating Cache Access

deepseek 官方说法是 cache hit rate 能达到 56% 以上。

NPU 能够允许通过 UB plane 来访问 disaggregated 的 CPU 的 DRAM 池。能够针对远端的 kv cache 访问，提供内存级别的带宽和 latency。

#### Quantization for Efficientcy：Int8 Support

## 4 DeepSeek Serving on Huawei CloudMatrix384

PDC 分离（Prefill、Decoding、Caching 分离）。

### 4.1 Overview: A Peer-to-Peer Serving Architecture with PDC Disaggregation
