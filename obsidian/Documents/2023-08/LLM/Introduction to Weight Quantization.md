https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c

浮点数中的格式中包括：

- Sign：使用 0 或者 1 表示正数、负数；
- Exponent：用于表示指数，一般是 2 作为底数；
- Significand/Mantissa：浮点数的精度与 Mantissa 的长度高度相关；

![[Pasted image 20230820100543.png]]

深度学习中常用的浮点类型：

| type  | total | sign bit | exponent | Significand | Notes  |
| ----  | ----- | -------- | -------- | ----------- | -----  |
| FP32  | 32    | 1        | 8        | 23          | 有比较高的精度，但是有比较高的计算和内存开销 |
| FP16  | 16    | 1        | 5        | 10          | 内存占用更少，但是牺牲了表达的范围和精度，可能有数值不稳定性，影响性能|
| BF16  | 16    | 1        | 8        | 7           | 相比 FP16 有更大的表示范围，减少 underflow 和 overflow 风险，虽然牺牲了精度，但是对深度学习任务性能的影响较小|

在深度学习中，FP32 通畅称作”full precision“（4 bytes），而 BF16 和 FP16 被称作”半精度“（2 bytes）。

能不能进一步用 INT8 来表示？

# 🔰 Naïve 8-bit Quantization

量化技术的目标：将一个 FP32 的向量 $X$ 映射到一个 INT8 的向量 $X_{quant}$。

先介绍 absmax 和 zero point 两种量化技术。
### absolute maximum (absmax) quantization

求出向量中的最大值，使每个数值除以最大值后，乘以 127，使之分布在 `[-127, 127]` 范围内：

![[Pasted image 20230820102618.png]]

 假如向量中最大值为 $3.2$，对 $0.1$ 这个数会量化为 $round(0.1 \times 127 / 3.2) = 4$，反量化回来：$4 \times 3.2 / 127 = 0.1008$，产生了 $0.0008$ 的误差。
 
### zero-point quantization

在 ReLU 等函数中只会输出正数值。输入的数值会在向量的最大值与最小值之间，分 255 个块。然后选择中间的第 128 个数值为零点，转换到 `[-128, 127]` 范围内。

![[Pasted image 20230820103145.png]]
![[Pasted image 20230820103151.png]]

### benchmarks

![[Pasted image 20230820104025.png]]

Both plots are quite similar, with a surprising spike around 0. This spike shows that our quantization is quite lossy since reversing the process doesn’t output the original values. This is particularly true for the absmax model, which displays both a lower valley and a higher spike around 0.

对整个模型做量化会比较显著地降低模型的性能，在实践中，往往采取 vector-wise quantization 的办法，会对 tensor 中的一组行或者列数据做单独的量化。

但是，即使 vector-wise 的量化也不能解决 outlier feature 的问题。transformer 模型只要足够大（>6.7B），极大或者极小的 outlier-feature 值会在每一层都存在。一个单独的 outlier 值便可以降低所有其他数值的精度。但是移除这些 outlier feature 数值也会显著降低模型的性能。

# 8-bit Quantization with LLM.int8()

LLM.int8 是 [Dettmers et al. (2022)](https://arxiv.org/abs/2208.07339) 中引进的，用于解决 outlier 问题。

它在一个 vector-wise 的 absmax 量化技术之上，引进了一个混合精度量化。让 outlier features在一个 FP16 格式中来保持精度，其他值通过 INT8 来处理。

outlier 大约只占 0.1% 的总数，总的来说可以减少 50% 的内存占用。

![[Pasted image 20230820110252.png]]
LLM.int8() 之后，矩阵乘法分三步：

1. 对于 input hidden state，按列取出超过 threshold 的 outlier，拆成两个矩阵；
2. 针对 FP16 的 outlier 矩阵，和非 outlier 的 INT8 矩阵，分开分别执行矩阵乘法；
3. 将两个结果矩阵再回来；

![[Pasted image 20230820110835.png]]