

## 关于我  

- Github：flaneur2020
- Databend Cloud 工程师
- pua-lang 之父、《Haskell 趣学指南》译者

## Agenda

- 介绍下 LLM 和推理的常识
- 推理的瓶颈分析与基本的优化技术
- 对目前推理生态的理解
- 值得关注的推理相关技术

## Crabml

- 业余项目，出发点用于学习 LLM/AI
- 可以认为是 ggml 的 rust 复现
- 兼容 GGUF 格式的模型，目前已支持 llama2、gemma 两种模型，Q8_0、Q8_1 两种量化模式，SIMD、WebGPU 加速
- 近期目标是使推理性能接近追上 GGML，做一个家庭智能中心，解决一下自己的日常需求，走平民推理路线

## Motivation

- “这次不一样” 毫无疑问 AI/LLM 是一定需要参与或者学习的
- 但是这个领域日新月异 目不暇接
	- 比如看到很火的帖子，随手一个术语，比如 attention、kv cache、MoE、AWQ、RAG 等等，这些都是什么？
	- 打开了一个面向小白的软件，随手一个参数，比如 temprature、topp，这些又是什么？


  ![[the-fomo-is-real-v0-lczlol5tf64b1.png]]


### 理由一：学习青蛙的最好方式是构造一只青蛙


- LLM 这个领域非常的资金密集，不是我们穷人能玩的
- 但是穷人可以玩推理和 RAG

### 理由二：看好平民推理


- 开源模型已经可以打平 ChatGPT-3.5 的性能，甚至也有多模态的能力
- 工业推理（vLLM 为代表）和平民推理（llama.cpp 为代表）有一定差异性：优化目标有差异、社区运转有差异
	- 工业推理技术更先进、也能利用更先进的硬件优势，成本驱动希望挖到性能和利用率的极致
	- 平民推理主要瓶颈在于内存容量，追求性能够用体验良好即可，不看重吞吐量或卡的利用率
- 硬件厂商针对性的提高平民硬件推理的能力，也有助于增强品牌的优势，再过两年一定有更廉价的平民推理方案进入千万家
- worse is better：技术的发展规律就是，高精尖的专有硬件和专有技术几年容易被人遗忘，看起来糙猛的廉价硬件和廉价技术因为有更广大的用户群体和社区，活得往往更长久
- 劳模风范的 llama.cpp：来者不拒地给各种杂七杂八的硬件做手写优化，来了都能跑，/r/locallama 是 llama.cpp 的后花园

## 什么是推理

推理可以大致理解为，拿到一个已经训练、微调好的大语言模型的参数，对外提供服务的过程。


![[Screenshot 2024-03-03 at 10.18.45.png]]

### 推理的输入与输出

实际上里面，是一个 token 一个 token 蹦的：


![[Screenshot 2024-03-03 at 13.28.51.png]]

每追加一个 token，会将这个 token 转化为 embedding，在每层 decoder 中，均进行 attention 后 ffn，最后输出 token 表的每个 token 的概率。

在选择输出时并不是概率越高越好，好像说加一点随机性在里面，更像真实的人类语言，temprature 就是这里的配置，可以在采样时增加一定的不确定性在里面。

到这里我们有一些陌生的术语，比如 kv cache、attention，不用担心，我们会单独介绍。
### Llama.c

https://github.com/karpathy/llama2.c

做自己的轮子，Karpathy 的 llama.c 是最好的 lab 项目。

只有 600 行！通过它，可以自己动手快速对推理的过程有一个体感。

不过 llama.c 作为 lab 项目，仍不能满足日常的实用性：

1. 没有标准的量化格式，不容易找些社区里的模型直接跑，还需要转码
2. 没有 tensor 抽象，不容易对接各种各样的野生模型
3. 没有 GPU 加速
4. 固定的 llama 架构，没有其他变种架构支持

这些就是 crabml 在 llama.c 的基础上需要一个一个补齐的。

### Not actually the original Transformer

顺着 llama.c 走下来，你会发现计算的流程相比 Transformer 原始论文中这个还挺复杂的架构会简单很多。

![[Screenshot 2024-03-03 at 11.10.26.png]]


### Decoder Only Transformer

现在工业界流行的 Decoder Only 架构要简单很多：

![[Screenshot 2024-03-03 at 11.13.12.png]]

在推理框架中，我们需要将 Llama 上面的架构和公式，用 tensor 表达式来写出来，然后让推理框架的 各种 tensor 算子来加速。

### 先有 Decoder only 架构一统天下，后有通用推理框架

现在架构比较大同小异，少量修改就可以支持很多衍生模型，在 transformer 大一统时代之前也并不存在通用的推理引擎的土壤。


![[Screenshot 2024-03-03 at 12.36.07.png]]

### LLM 推理和传统推理技术的区别

传统上模型推理的难度并不高，甚至不见得能成为课题：1. 参数量并不那么多，在 CPU 也能跑；2. 推理的成本是固定的。

但是 LLM 推理有两个不同：

1. 参数量巨大，一台普通的 PC 甚至都放不下
2. 随着聊天上下文的增长，kv cache 占用的内存会越来越多，计算量也越来越大

（有点像自动机和图灵机的区别？）

![[Pasted image 20240303113639.png]]

（这也是前两天贾扬清手撕 groq 芯片的原因）

### Attention 的朴素理解

KVCache 就是 Attention 作用的文本序列的本体。

在看什么是 KV Cache 之前，我们可以先对 Attention 有一个朴素的理解：Self Attention 可以解决词汇的歧义问题。

![[Pasted image 20240303121426.png]]

### 为什么是 QKV 这个形式？

Attention 可以帮助 transformer 找到理解当前 token 的语义时，需要「关注」序列中的哪些其他的 token。

Attention 机制相当于为之前的序列打一个相关性的评分。

QKV 不是唯一的 Attention 形式，更早时还有直接用一个神经网络来拟合 Attention Score 的做法，据说效果和 QKV 是等效的。不过 QKV 更省电。

可以将矩阵乘法看作一个投影，我可以找到一个角度，比如下面这个三维的点投影到二维上，我总可以找到一个角度，能让任意两个点距离最近。

投影后的向量相乘，可以看作是投影后的点的 cosine 距离，距离越近，也就相关性越高。

![[Pasted image 20240303124133.png]]

## 推理框架
### 模型架构的表示：mlx 为例

以苹果的 mlx 为例，模型架构的表示大致长这样：

![[Screenshot 2024-03-03 at 11.18.04.png]]

### 模型架构的表示：llama.cpp 为例

![[Screenshot 2024-03-03 at 11.22.27.png]]

### 模型架构的表示：crabml 为例

![[Screenshot 2024-03-03 at 11.25.00.png]]
### 推理框架里包含什么

具体到实现层面，推理框架需要内置一些流行的模型代码在里面。

现在的 Self-Recursive Decoder 大都长的很像，兼容了一个 Llama 之后，其他的都很容易对接。

其他可以大约理解为一个深度学习框架的子集，也可以直接利用 pytorch 生态比如 vLLM。

深度学习框架可以做推理，推理框架不需要关注训练。

![[Screenshot 2024-03-03 at 11.00.51.png]]

### Tensor

抽象的 Tensor 中封装了一系列算子（比如 matmul、silu、rms_norm 等等）和维度操作（reshape、transpose 等），在书写模型代码时，不需要关注具体的硬件细节，也不需要关注模型的存储格式。

也是一种「一次编写、到处运行」。

LLM 推理涉及的算子的数量很少，这也允许我们做更简单的设计，能够比通用的 Tensor 库简单很多。

crabml 目前支持 CPU 和 WebGPU 两种后端，后续也有增加 CUDA 支持的想法，但是需要先攒点钱买卡。


![[Screenshot 2024-03-03 at 12.47.28.png]]


### WebGPU Support

为什么优先选择 WebGPU 的支持？

因为浏览器周边的生态有最好的兼容性，而且基于浏览器用户的庞大基数，也有比较高的概率能够把性能优化的足够好。

而且 WebGPU 和 rust 好像也还挺契合。


![[7IShIg3VYP.png]]
### Quantization

量化的做法类似于，将每 32 个浮点数字可以组成一个 block，在这个 block 中找出来最大最小值，然后把里面的所有数字映射到 [-128, +127] 的范围中。

量化是民间推理的一门显学，因为大语言模型的内存占用量，直接制约了玩家能不能跑起来。

学术界也很关注量化技术，怎样量化能使智力下降的越少，也可以大幅降低工业推理的成本（类似于超卖术之于云厂商，也必然会是一门显学）。

怎样量化不降智力？有很多研究发现在 LLM 架构下，激活值通常有少数的 outlier，就是特别大或者特别小的数，这些 outlier 对 LLM 的推理性能有决定性的影响，而这部分决定性作用的数值的绝对数量又很小，1% 这个水平，就给了不降智力提高量化比例的做法给了一定的理论基础。

最近一个 1.58bit 的量化方法很热门，据说智力基本不下降的前提下，让每个参数从 32bit 压缩到 1.58 bit。

![[Pasted image 20240303130222.png]]

## 推理性能优化

### 推理的时间消耗的构成

在推上看到这个贴写的很贴切，这两行代码对应了 90% 的推理时间消耗（训练也是），其余 9.9% 发生于 attention。

优化推理速度 约等于 优化矩阵-向量相乘，也就是 GEMV 操作。

![[Pasted image 20240303135351.png]]

GEMV 操作一般会遍历这层模型的所有参数。所以模型推理的总时间消耗约等于 load 所有模型参数的时间，加上针对这些参数做矩阵乘法计算的时间。
### 计算还是内存

比较反直觉的是，这种密集的纯计算操作中，其实 GPU 的计算单元都在摸鱼。

> The A100 features high-speed HBM2 (High Bandwidth Memory) with a total memory bandwidth of up to 1.6 TB/s (terabytes per second).

> For deep learning tasks using Tensor Float 32 (TF32), the A100 can deliver up to 156 TFLOPS.
> For AI inference tasks using INT8 precision, the A100 can reach up to 1,248 TOPS (tera operations per second).

如果用 f32，完全把 156 TFLOPS 跑满需要 624tb/s 的吞吐，是实际吞吐的 390 倍。

如果用 int8，相当于接近 800 倍。

所以不管优化推理还是优化训练，基本思想都是在一轮内存访问中多加些计算，如果计算量翻几倍但是可以换来更好的内存访问模式，这个优化就是值得的。

因此量化也是对推理性能提升最为有效的手段。

（这也是为什么 kernel fusion 有助于提高性能的原因，在遍历所有参数的一轮内存循环中，尽量多塞更多计算进来）
### Batching

给予增加一轮内存访问中增加计算的思想，最能有效提高计算量的思路就是增加每一轮请求的批次大小。

一轮推理下来，携带的不是一个人的请求，而是塔 10 个人的并发请求，搭 10 个人的一批请求的开销，和一个人的一个请求的开销区别不大。

所以，Batching 也是工业推理要降低服务成本的一项关键。我们有时候觉得 chatgpt 响应慢，不一定是因为 openai 缺卡，而很可能是 openai 在攒火车。

## 貌似比较有潜力的技术

### Sparsity

在苹果的 LLM in a Flash 有基于利用 Sparsity 这个思路， 说一般的激活函数是平滑的，出来的激活值没有零值，而 ReLU 做激活函数有一个好处是激活值中有多达 95% 的零值。

这意味着在计算矩阵相乘时，可以只把非零值对应的参数捞出来，这样需要传输、计算的量就特别少了，也就是利用 sparsity 的性质，这样现场从 SSD 读参数也来得及。

直观上也合理，比如我在马路上看到一个猫，不至于在脑子里把所有的童年经历都遍历一遍，肯定只是唤醒脑子里和猫相关的一部分参数。

直观上这个术跟工业推理需要的 Batching 可能有一定冲突，不过做民间推理应该是很合适的。

（MoE 也是利用 Sparsity 有助于更快速地推理的机制，8x7B 的模型每次推理的性能开销理论上和 7B 模型区别不大）
### Sepculative Decoding

Auto Recursive Decoder 的架构决定了一次出一个 token，不能直接增加并行。

Speculative Decoding 的思想是，弄一个小模型来猜 N 个 token，然后把这 N 个 token 给更大的模型来判断能不能用，在 sampling 的数学上可以做到完全无损。

在较大模型的推理时，N 个 token 同时 fill，和推理一个 token 所花费的时间相差无几。
### AWQ

看起来是很 make sense 的量化不降智的方法，也和 outlier 的假设有关系。

在量化时，不会像 32 个数直接量化，而是结合一个数据集，根据这个数据集产生的激活值的分布，找一个量化参数，能够更少地减少激活值的总统计误差。

直观上这个术应该挺有助于减少量化后的误差，不过不大清楚会不会使得量化后的模型和校对数据集产生什么奇怪的拟合。
### 1.58bit

还没看这个论文，不过这个看起来非常 promising，把一个参数压到平均 1.58bit 而且智力不怎么掉。

700_0000_0000 * 1.58 / 8/ 1024 / 1024 / 1024
=> 12.87553459405899

70B 的模型只需要 12.8gb 内存？仿佛 Too good to be true，密切关注。

## 阅读推荐

- 这就是 ChatGPT
- AWQ - Activation-aware Weight Quantization for LLM Compression and Acceleration
- [Towards 100x Speedup: Full Stack Transformer Inference Optimization](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Efficient Memory Management for Large Language Model Serving with PagedAttention
- LLM in a Flash


