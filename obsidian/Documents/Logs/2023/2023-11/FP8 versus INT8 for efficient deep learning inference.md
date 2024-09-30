
先看结论：

- FP8 还不如 INT8
- FP8 在硬件电路的层面远不如 INT8 高效
- 一般神经网络的层之间 outlier 不多，因为都有 normalization，INT8 对 uniform 分布比 FP8 的精度更高，效果也更好
- Transformer 的 outlier 主要发生在 attention 之后和 ffn 的 residual connection 这里
- 不过 Transformer 的 outlier 分布有规律，比如分布在特定层的特定块里比较密集，针对这几个层开 F32 全精度，或者 W8A16，都有不错的效果
- 与其无脑全换 FP8，不如应用这些适应 outlier 的 tricks

---

> Overarchingly, we have seen that the floating-point formats FP8-E4 and FP8-E5 are not replacements for INT8 for deep learning inference in terms of performance and accuracy. For most well-behaved layers and networks, <mark>these formats are worse than INT8</mark>, especially when networks are trained with either format in the loop. For corner-case scenarios in PTQ, where layers have significant outliers, the floating-point formats can be better in terms of accuracy. However, there are more efficient solutions for the problematic transformer layers. You can either run these layers in W8A16 with mixed-precision or apply quantization-aware training.

> We have also seen that implementing the FP8 formats in hardware for inference is not efficient and incurs significant overhead. Depending on the accumulator size, the FP8 MAC units are 50% to 180% less efficient than their INT8 counterparts. This would make a dedicated chip significantly slower if the workloads are compute-bound.
> 
> Lastly, many networks can easily be quantized into INT8 or even pushing INT4 for even further improved efficiency. The tools for this have been available for many years, and the leap to 4-bit weights is something the floating-point formats do not do as of this writing.

> Because of these reasons, implementing floating point formats for edge use-case scenarios is suboptimal compared to the standard stack of integer solutions available today. If you want the best accuracy and efficiency trade-off for your models, quantizing them to INT4-INT8-INT16 is the best solution.


---

the FP8 format is at least 50% less efficient in terms of area and energy usage than INT8. This means that FP8 will have to be significantly more accurate than INT8 to be worthwhile from a hardware-efficiency perspective

Based on our research and a read of the research field, <mark>we conclude that although the proposed FP8 format is potentially a good match for gradients during training (although comparative evidence to other formats is sparse), the results for inference do not warrant a dedicated implementation of FP8 in favor of INT8</mark>.

## 3 Hardware Considerations

3.1 FP8 is an ambiguous term

好像说 FP8 里面有多少个 bit 用来存储 exp 位还没有标准。

The proposed FP8 implementation is indeed faster compared to FP16 for networks that are weight, memory, or calculation speed-dominated. However, for networks with large activation tensors, the FP16 activations will still be a bottleneck, and speed-ups will be greatly reduced. The actual speed-ups will also depend on how many layers can actually be executed in FP8 instead of FP16 or FP32.


## 4 Deep Learning Network Accuracy Comparison

the only difference between INT8 and FP8-EX is that an X number of exponent bits are used for an increased dynamic range instead of accuracy.

This leads us to a very simple conclusion: It’s all about the outliers. If a distribution has very significant outliers, the FP8-E4/FP8-E5 format is more accurate, but if the distributions are well-behaved and more Gaussian-shaped, then INT8 or FP8-E2/FP8-E3 is expected to perform better

Thus, to optimize your network for quantization, it is best to have a network without any outliers at all. For such a network, the INT8 and FP8-E2 formats would perform best. This will comes back in Section 4.5 on QAT. In training networks for quantization, it is easy to train outliers away, resulting in better results overall, and in many cases, the INT8 format then outperforms the floating-point formats.

训练时候可以把 outlier 给训练没？

4.4 Quantization-Aware Training

## 4.6 Transformers

![[Screenshot 2023-11-21 at 22.02.15.png]]

![[Screenshot 2023-11-21 at 22.02.44.png]]

![[Screenshot 2023-11-21 at 22.05.11.png]]