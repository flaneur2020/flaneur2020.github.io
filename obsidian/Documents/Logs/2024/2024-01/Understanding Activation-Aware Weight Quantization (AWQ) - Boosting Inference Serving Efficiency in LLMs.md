https://friendli.ai/blog/activation-aware-weight-quantization-llm/

## The Basics of Weight Quantization

> Weight quantization is the process of reducing the precision of the parameters (weights) in a neural network. In typical neural networks, weights are represented as floating-point numbers with relatively high precision, often 16 bits (e.g., fp16 and bf16 formats). However, this level of precision requires significant GPU memory resources.

## The Role of Activation in Weight Quantization

在量化时，把激活值也考虑在内。

在传统的量化方法中，权重会单独地进行量化。

AWQ，会将激活值的分布纳入考虑。

AWQ 分这几步：

1) **Collect Activation Statistics**：拿一部分数据，来统计 activation 的分布，
2) **Search Weight Quantization Parameters**：Concretely, we perform a space search for quantization parameters (e.g., scales and zeropoints), to minimize the distortions incurred by quantization on output activations. 
3) **Quantize** : With the quantization parameters in place, the model weights are quantized using a reduced number of bits.

大约说对 weights 做量化的一些参数包括 scales、zeropoints 之类，将它们当作一个搜索空间，寻找到一组参数能够最小化 output 激活值的损失。


## Benefits of AWQ

- 准确性更高
- 效率：使用 AWQ，可以在不损失 accuracy 的前提下，将权重压缩到更小，比如 4bit。压的越小，跑推理的性能也就越高。
- Robustness：面对不同的输入，有更高的稳定性
- 不需要参与训练


