https://huggingface.co/blog/hf-bitsandbytes-integration

理想的情况下，训练和 inference 都最好用 FP32 精度，但是这样计算比 FP16 慢两倍。在前向和后向传播中，一般转化为 FP16/BF16 来加速训练，FP16/BF16 的 gradients 用来更新 FP32 的 ”main weights“。在训练中，main weights 总是保存为 FP32 的全精度。

![[Pasted image 20230820120348.png]]

## A gentle summary of LLM.int8(): zero degradation matrix multiplication for Large Language Models

作者发现在量化后 transformer 的性能退化很大程度上来自于 outlier features 的影响。

基于 LLM.int8() 算法的 matmul 可以概括为：

1. 对于 input hidden state，按列取出超过 threshold 的 outlier；
2. 针对 FP16 的 outlier 矩阵，和非 outlier 的 INT8 矩阵，分别相乘；
3. 将两个矩阵再拼回来；

## The importance of outlier features

作者发现传统的量化方法在 > 6B 的 transformer 上就不大好使了，每一层都有 outlier 存在。

transformer 架构下倾向于将所有元素 link 在一起，误差会随着 layer 递增指数传播。因此作者的思路是引进 mixed precision 来容纳这些极端的 outlier。

## Inside the MatMul

作者发现取出来 magnitude 为 6 的 outlier 能够复原完整的解析性能。

![[Pasted image 20230820121916.png]]

outlier 部分是标准的 FP16，因此普通的 matmul 计算就够了。

8-bit 部分需要将权重和 hidden state 量化到 8-bit 精度。对 hidden state 做 row-wise 的量化，对 weight matrix 做 column-wise 的量化。随后将计算的结果还原为 FP16 精度。

## What does 0 degradation mean?

作者对一系列的 LLM 做了 benchmark，主要的指标上都没有超过标准差的性能下降，可以认为没有下降。

## Is it faster than native models?

we could improve inference per token from 312 ms to 173 ms for T5-3B and from 45 ms to 25 ms for T5-11B.