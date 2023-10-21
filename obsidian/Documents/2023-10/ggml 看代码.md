## matmul 与 quant

matmul 之前会将 x 去做 quantize 后再和参数矩阵相乘。

（不过自己试下来精度很差不知道为什么，有没有办法比较下计算的参数？）

ggml_compute_params 中有一个 work buffers 字段，似乎用于保存 matmul 中 x 做 quantize 的 buffer 重用。

每个 quant 类型的 vec dot 最后返回的是 f32 类型。

## KVCache

llama_kv_cache_init 中，kvcache 可以选择 f16 或者 f32。