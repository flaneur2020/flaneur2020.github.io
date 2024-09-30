针对一个 pre-trained 模型进行低 bit 量化是一个很有挑战的工作。

作者在 mixtral 上较大的模型上实验做 2bit 量化的效果还是不错的。但是针对小一点的模型比如 llama7b，2bit 量化就很难。

作者发现，fine tune 只影响少量参数（比如 0.65%）能够显著提升质量。

此外发现：

1. 1-bit： 对 7b 的模型做 1bit 直接的量化不是很好。但是经过 fine-tune 之后，输出的质量明显提升。fine-tune 过的 1-bit 模型能够超过 Quip#2-bit（只有 2.8k sample 的 1024 的 context-window）。
2. 2-bit：给出更好的数据，2-bit 的模型可以表现的很好。llama2 7b 2-bit 加上 HQQ+ 在 wikitext 评测上能够超过全精度模型。chat model 能够超过 GSM8K 的全精度模型，在给出足够的 math 和 reasoning data 之后。

## Efficient Matmul for Low-Bit Quantization

HQQ 的反量化是一个 linear operation，需要一个 scaling 和 zero-point parameter。