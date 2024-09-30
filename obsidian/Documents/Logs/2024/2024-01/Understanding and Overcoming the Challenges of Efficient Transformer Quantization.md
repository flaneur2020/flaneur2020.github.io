https://arxiv.org/pdf/2109.12948.pdf

TLDR:

- 比较早期的论文，证明发现了 bert 这种模型可以跑比较低 bit 量化不影响什么性能
- 但是 FFN 的激活值 + residual connection 会有 outlier

---

作者发现标准的 8-bit post-training 量化技术在 bert 这种 encoder 模型下可能导致显著的性能下降。

> We find that the main bottleneck is a considerable mismatch between the different dynamic ranges of activation tensors in the residual connections.

在 residual connections 中有不匹配的 dynamic range。

会包含 structured outlier，比如针对 `SEP` 这种 token 加过多的 attention。

作者提出了一个 per-embedding group 的量化方法。

> A vital step in the PTQ (Post-train quantization) process is finding good quantization ranges for each quantizer. One way of doing this is static range estimation, which determines quantization parameters for the network by passing a few batches of calibration data through the model before inference

## 3 Problem investigation

> the smallest performance drop is when we do not quantize the residual sum after the feed-forward network

不对 FFN 的 residual sum 进行量化的效果是最好的。

> Further analysis suggests that structured outliers in the FFN’s residual connections lead to structured outliers in query-key multiplications in specific attention heads in the next attention layer, causing most of the tokens to attend to the special `[SEP]` token.


