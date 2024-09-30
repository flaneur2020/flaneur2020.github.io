
TLDR

_为什么对 $w$ 乘以系数，并对 $x$ 除以系数能够起到减少误差的效果？

> 把权重量化难度转移到激活，因为激活不量化，或者比特数更多

$$Q(w) = \Delta \cdot Round(\frac{w}{\Delta}), \Delta = \frac{max(|w|)}{2^{N-1}} $$

在量化中，$Round()$ 会消去浮点数值的小数部分。

如果乘以系数后，再计算 $Round()$，损失的小数部分会减少，精度也就越高。

相应的，对 $x$ 除以系数，会导致 $x$ 的精度减小，但是激活值 $x$ 的精度往往更高。

相当于牺牲 $x$ 的精度，提高了 $w$ 的精度。

---

> Our method is based on the observation that weights are not equally important: **protecting only 1% of salient weights can greatly reduce quantization error**. We then propose to search for the optimal per- channel scaling that protects the salient weights by observing the activation, not weights. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs’ generalization ability on different domains and modalities, without overfitting to the calibration set.

salient：突出的

## Introduction

Quantization-aware training 不那么实际，因为训练的成本太高。

Post-training Quantization 容易损失精确性。最接近的工作是 GPTQ，会使用 second order 信息来修正误差。不过：

> It may over-fit the calibration set during reconstruction, distorting the learned features on out-of-distribution domains (Figure 6), which could be problematic since LLMs are generalist models.

calibration set 应该是量化时用的一个数据集。

GPTQ 会产生针对 calibration set 的 over-fit。

> Our method is based on the observation that weights are not equally important for LLMs’ performance. There is a small fraction (0.1%-1%) of salient weights; skipping the quantization of these salient weights will significantly reduce the quantization loss

观察 LLM 的性能受少数 salient weights 影响比较大，跳过这些 weight 的量化，可以显著减少 quantization loss。

> To avoid the hardware-inefficient mixed-precision implementation, we analyze the error from weight quantization and derive that scaling up the salient channels can reduce their relative quantization error. Following the intuition, we designed a per-channel scaling method to automatically search for the optimal scaling that minimizes the quantization error under full-weight quantization.

没有使用硬件效率低的 mixed-precision 实现。

做的 per channel 量化。

AWQ 在 FastChat, vLLM, HuggingFace TGI, LMDeploy 等框架中落地了应用。

## AWQ: Activation-aware Weight Quantization

### 2.1 Improving LLM Quantization by Preserving 1% Salient Weights

> Interestingly, selecting weights based on activation magnitude can significantly improve the performance: keeping only 0.1%-1% of the channels corresponding to larger activation significantly improves the quantized performance, even matching a strong reconstruction- based method GPTQ.

> Limitations: Despite keeping 0.1% of weights in FP16 can improve the quantized performance without a noticeable increase in model size (measured in total bits), such a mixed-precision data type will make the system implementation difficult. We need to come up with a method to protect the important weights without actually keeping them as FP16.

保留 0.1% 的 weights 为 FP16 就可以使量化后的 LLM 的性能大幅提高。

但是混合精度实现起来复杂，需要出一个办法使它们不必按 FP16 进行保存。

### 2.2 Protecting Salient Weights by Activation-aware Scaling

作者提出了通过 per-channel scaling 来做量化。
 
 **Analyzing the quantization error.**

首先分析一把 weight-only quantization 的 error。

设一个 group、block 的 weight 是 $w$，设 linear operation 为 $y=wx$，量化后的 linear operation 为 $y = Q(w)x$

$Q(w)$ 为：

$$Q(w) = \Delta \cdot Round(\frac{w}{\Delta}), \Delta = \frac{max(|w|)}{2^{N-1}} $$

其中 $N$ 是 quantization bits，$\Delta$ 是绝对最大值决定的 quantization scaler。

尝试给 $w$ 乘以一个 >1 的 scaler $s$，并给 $x$ 除以 $s$：

$$ Q(w \cdot s) \cdot \frac{x}{s} = \Delta^{'} \cdot Round(\frac{ws}{\Delta}) \cdot x \cdot \frac{1}{s} $$

其中 $\Delta^{'}$ 是乘以 $s$ 之后的 quantization scaler。

作者发现 $\Delta^{'} \cong \Delta$

> to verify the idea, we multiply the 1% salient channels with s > 1 for the OPT-6.7B model, and measure the change in ∆ for each group in Table 2. We find that scaling up the salient channels is quite effective: the perplexity improves from 23.54 for s = 1 (simply RTN) to 11.92 for s = 2. As s goes larger, the percentage of changed ∆ generally gets larger, but the proportion is still quite small for s < 2; the relative error for the salient channels continues to go smaller as s increases.

作者对 1% 的 salient channel 乘以了大于 1 的系数 $s$。




**Searching to scale**

寻找这样的参数 $s^{*}$ 使得 $\mathcal{L}$ 最小：

$$ s^{*} = \underset{s}{argmin} \space \mathcal{L}(s), \mathcal{L} = || Q(W \cdot s)(s^{-1} \cdot X) - WX|| $$

> To make the process more stable, we define a search space for the optimal scale by analyzing the factors that will affect the choice of scaling factor. As shown in the last section, the saliency of weight channels is actually determined by the activation scale (thus “activation-awareness”). Therefore, we simply use a very simple search space:

