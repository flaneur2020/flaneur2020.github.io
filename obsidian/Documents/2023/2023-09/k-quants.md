https://github.com/ggerganov/llama.cpp/pull/1684
https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML

增加了好几种 2~6 bit 的量化方法，也有 quantization mixes。PR 中提供了 Scalar, `AVX2`, `ARM_NEON` 和 `CUDA` 的实现。

## Why

![[Pasted image 20230917141643.png]]

需要注意 x 轴是指数增长的。

30B Model 经过 2-bit 量化版后可以在 16gb 的 RTX 4080 GPU 上跑。

经过 6-bit 量化后的版本，甚至比原本的 fp16 model 表现还要更好。

## How

之前 ggml 有 type-0 （Q4_0, Q5_0）和 type-1 （Q4_1, Q5_1）两种量化。

在 type-0 中，权重 `w` 通过量化后的 `q` 乘以 block scale 值 `d` 来得出：`w = d * q`。

在 type-1 中，权重 `w` 除了乘以 block scale 值 `d`，还需要加一个 block minimum 值 `m`，公式为：`w = d * q + m`。

在作者这个 PR 中，又增加了新的量化类型：

**GGML_TYPE_Q2_K**

type-1 的 2 bit 量化，super-block 中包含 16 个 block，每个 block 有 16 个 weight。block scale 和 min 按 4 bit 进行量化，最终可以做到每个权重平均 2.5625 bit。

**GGML_TYPE_Q3_K** 

"type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using `3.4375` bpw.

**GGML_TYPE_Q4_K**

"type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using `4.5` bpw.

**GGML_TYPE_Q5_K** 

"type-1" 5-bit quantization. Same super-block structure as `GGML_TYPE_Q4_K` resulting in `5.5` bpw

**GGML_TYPE_Q6_K** 

"type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using `6.5625` bpw

"type-0" 的 6-bit 量化。super-blocks 有 16 个块，每个 block 中包含 16 个权重。scale 按 8 bit 进行量化。最终平均 6.5625 bpw。

**GGML_TYPE_Q8_K** 

"type-0" 的 8 bit 量化，只用于量化中间的结果。和 `Q8_0` 的区别在于 block size 为 256 。**所有的 2-6 bit 的 dot product 都是通过该量化类型实现的**。

## Code

```
[GGML_TYPE_Q2_K] = {
  .dequantize_row_q = (dequantize_row_q_t) dequantize_row_q2_k,
  .quantize_row_q = quantize_row_q2_k,
  .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q2_k_reference,
  .quantize_row_q_dot = quantize_row_q8_k,
  .vec_dot_q = ggml_vec_dot_q2_k_q8_k,
  .vec_dot_type = GGML_TYPE_Q8_K,
},
```