https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172

# How to quantize LLMs with GGML?

以仓库 [TheBloke/Llama-2–13B-chat-GGML](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main) 为例，可以看到 14 个不同的 GGML Model，对应不同级别的 quantization。

这些文件的命名规范为 q + bit 精度 + 变种类型：

- q2_k：对 attention.vw 和 feed_forward.w2 使用 Q4_K，对于其他的 tensor 使用 q2_k；
- q3_k_l：对 attention.wv, attention.wo 和 feed_forward.w2 使用 Q5_k，其他使用 Q3_K
- q3_k_m：对 attention.wv, attention.wo 和 feed_forward.w2 使用 Q4_k，其他使用 Q3_K
- `q3_k_s`: Uses Q3_K for all tensors
- `q4_0`: 原始的 4-bit 量化方法；
- `q4_1`: Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.
- `q4_k_m`: Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K
- `q4_k_s`: Uses Q4_K for all tensors
- `q5_0`: Higher accuracy, higher resource usage and slower inference.
- `q5_1`: Even higher accuracy, resource usage and slower inference.
- `q5_k_m`: Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K
- `q5_k_s`: Uses Q5_K for all tensors
- `q6_k`: Uses Q8_K for all tensors
- `q8_0`: Almost indistinguishable from float16. High resource use and slow. Not recommended for most users.

作者一般推荐 Q5_K_M 对原始模型的性能保存的会比较好。如果希望省更多内存，可以使用 Q4_K_M。一般来讲 `_K_M` 系列比 `_K_S` 系列表现要好些。作者不推荐 Q2 和 Q3 的版本，性能下降会比较严重。

## Quantization with GGML

GGML 的量化方法没有 GPTQ 那么理想。它会将一组权重组织为 block，并约到一个更低的精度上。Q4_K_M 和 Q5_K_M 等模式允许为一些特别关键的层单独配置更高的精度。除了 attention.wv 和 feed_forward.w2 的<mark>一半</mark>的权重之外，其他权重都是 4 bit 精度。

以 block_q4_0 为例，是这样定义的：

```
#define QK4_0 32  
typedef struct {  
ggml_fp16_t d; // delta  
uint8_t qs[QK4_0 / 2]; // nibbles / quants  
} block_q4_0;
```

每个 block 有 32 个值，两个 4 bit 攒到一个 u8 中。

## NF4 vs. GGML vs. GPTQ

那种 4 bit 量化方法最好？GPTQ 有两种模式，AutoGPTQ 和 ExLlama。

![[Pasted image 20230917150649.png]]

perplexity 值越低越好。

GGML 的办法看起来 perplexity 方面有轻微优势。

作者的建议是：如果你有足够大的 VRAM 能够装下整个模型，那么 GPTQ + ExLLAMA 有最好的性能。不然，可以使用 llama.cpp 来 offload 一部分 layer 到 GPU 的模式来跑。