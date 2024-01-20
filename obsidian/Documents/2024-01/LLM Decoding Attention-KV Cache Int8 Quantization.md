https://bruce-lee-ly.medium.com/llm-decoding-attention-kv-cache-int8-quantization-b5292aa6af49


> KV cache quantification is very different from model quantification (weight quantification, etc.), and the numerical distribution of the two is very different. KV cache quantification can easily lead to model drop points, so some solutions with higher quantification accuracy need to be adopted.

## KV Cache Int8 Quantization

kv cache 有下面四种量化模式：

1. per tensor: 针对所有 token 的 K 和 V 进行统一量化
2. per token：针对特定 token 的 K 和 V 进行量化
3. per Head：对特定 head 的特定 token 进行统一量化
4. per Group：对特定 head 内的特定 token 进行按 group 做量化

> The Per Tensor solution has high GPU memory benefits, but the accuracy may drop significantly; the Per Group solution has high accuracy, but the scale storage capacity is large, and the GPU memory benefits are not high. Here, in order to take into account both quantization accuracy and memory gain, **Per Head’s quantization scheme was chosen.**




