https://finbarr.ca/how-is-llama-cpp-possible/

- llama.cpp 项目用 C++ 重写了 LLama 的 inference 逻辑；
- 在一些优化和量化权重之后，允许在一些设备上本地跑起来 LLM：
	- 在 Pixel5 上，可以 1tok/s 的速度跑一个 7B 的模型；
	- 在一个 M2 Macbook Pro 上，可以 ~16token/s 的速度跑一个 7B 的模型；
	- 甚至可以在 4GB memory 的树莓派上 0.1 token/s
- GPU 在 deep learning 领域有两大优势：
	- 有极高的内存带宽（A100：1935GB/s、4090：1008GB/s）
	- 有大量的计算力（A100：312 TFLOPS of FP16、4090：82.6 TFLOPS of FP16）；
- 在谈及内存带宽时，指的是从 HBM 到 on-chip memory 移动的速度；
	- 而 on-chip memory 很小（A100 只有 400mb，而它的 HBM 有 40~80GB）；
	- 内存带宽相比算力的吞吐，小两个量级；
- 这点对 LLaMa 的影响是什么？可以做一些 [inference arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
	- $Q$、$K$、$V$ 乃至 attention output 矩阵的形状都是 $[d_{model}, d_{head}]$，其中 $d_{model} = d_{head} \times n_{heads}$；
	- 每层 layer 有 $n_{heads}$ 个 $Q$、$K$、$V$ 矩阵，加一个 attention output 矩阵，每层的参数量为 $4 \times [d_{model}, n_{heads} \times d_{head}]$；
	- MLP 层有两个 weight 矩阵，形状为 $[d_{model}, 4 \times d_{model}]$ 和 $[4 \times d_{model}, d_{model}]$；
	- embedding 矩阵的参数量为 $[d_{vocab}, d_{model}]$；
- 汇总一下得出总的参数数量：$P = n_{blocks}(4 \times d_{model}^2 + 2 \times 4 \times d^2_{model}) + n_{vocab} \times d_{model}$
- 为了高效的 inference，需要将 KV Cache 保存在内存中，KV Cache 需要将每层的 KV 值保存下来
	- 参数量相当于：$n_{bytes} \times 2 \times d_{model}$
	- 其中 $n_{bytes}$ 等于每个浮点参数的 bytes 数；
	- 总共 n 层的话，KV Cache 的大小：$n_{blocks} \times n_{bytes} \times 2 \times d_{model}$
- 可以得出来一个表：![[Pasted image 20230817223037.png]]
	- **可见量化的优势**！
	- 通过牺牲精度，可以显著减少内存；
	- 经过 int4 的量化后，所有这些模型都可以装进 A100 中；除了 65B 的模型之外，其他所有模型也都可以装进消费 GPU 中（3090/4090 有 24gb 内存）；
- 大约每个 Token 需要 $2P$ FLOPS 的算力；
	- 会对 $P$ 个参数做一系列 matmul 计算；
	- $(m, n)$ 的矩阵乘以 $(n, )$ 的成本为 $2mn$；
- 到这里可以统计一把 LLama 的成本：
	- 在内存中保存 KV cache，加上所有的参数；
	- 读取 HBM 的所有参数到 on-chip memory 中；Because we sample auto-regressively, we have to repeat this for each token we sample.
	- 执行真正的 matmul 计算，来生成结果；
- latency 取决于计算或者内存的延时，读取参数到 on-chip memory 和计算是异步的：
	- $\text{latency}_{model} = max(\text{latency}_{compute}, \text{latency}_{memory})$
	- $\text{latency}_{memory}=\frac{2 P \times n_{bytes} \times B}{n_{memory \space bandwidth}}$
	- $\text{latency}_{compute}=\frac{2P}{n_{flops}}$
- 其中 $B$ 为 batch size，设 $n_{memory \space bandwidth}=1.935e12$，$n_{flops} = 3.12e14$，而 batch size 小于 161，因此<mark>model 属于 memory bound</mark>；
- 若 batch size 为 1，在大多数硬件下，<mark>在降低精度时可以得到线性的性能提升</mark>；
- LLaMA.cpp 使用了 int4s，RAM 需求可以降到 1.33GB 的 KV Cache、16.25 GB 的 vram 的模型参数；
- 因为内存带宽远小于 FLOPS，因此内存带宽是性能的约束；![[Pasted image 20230817225026.png]]

## Running LLaMa on an A100

A100（80GB PCIe）内存带宽为 1935GB/s。int4 算力是 1248 TOPS。65B 大约能做到 30 token/s，7B 可以做到 277 token/s。

## Running LLaMa on a M1 Macbook Air

M1 GPU 的带宽为 68.25GB/s，M1 GPU 可以对 fp16 跑 5.5 TFLOPS。对 65B 的 int4s 可以跑出来 1token/s，7B 可以跑出来 10 token/s。

M2 Pro 的带宽有 200GB/s，M2 Max 有 400 GB/s，M2 Max 跑 65B 可以搞到 6 token/s。

## Running LLaMa on a Raspberry Pi 4

Respberry Pi 4 有 13.5 GFLOPS 的算力，~4GB/s 的带宽。理论上跑 7B 可以出来 2 token/s。但是实际上得到的是 0.1 token/s，作者怀疑这里是计算有瓶颈。
