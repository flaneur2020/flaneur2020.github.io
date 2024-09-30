LLMs are often bandwidth bound and have a large weight memory footprint. 

## Key results – LLMs on CPUs

关注 Llama2 7B 4bit 量化后的版本，作为 benchmark 依据。

llama.cpp 能够代表 arm 平台当下的表现，但是不能反应真正的潜力。

针对 Graviton3 做特化优化后：

- 35 tok/s 8 线程，相比 GGML 快 3.15x 倍。
- batched inference：设置 batch size = 8，可以跑到 200+ tokens/s ，相比优化的 GEMV 快 2.03x 倍，比 GGML 快 4.37 倍

## Non-batched inference of LLMs – generative phase

- 对单次推理，Projection 和 FFN 是 GEMV, MHA 是 GEMM
- 在 batched inference 中，所有的都是 GEMM
- memory-bound problem 主要受 weights 参数的内存访问所局限

## LLaMA inference – CPU runtime on Graviton3 - Neoverse V1

### GEMM/GEMV background
![[Screenshot 2024-04-02 at 12.05.28.png]]

在 GGML 中，一个 dot-product kernel 会拿来计算一整个 result，每个计算核对应 C 矩阵的一个结果。

![[Screenshot 2024-04-02 at 12.06.21.png]]

### Block Processing steps - original GGML

![[Screenshot 2024-04-02 at 12.09.54.png]]

- 没有 reuse activation 会产生重复的内存访问
- scale 参数在内存上不连续，不能在 SIMD 中一次拿到

## Avoiding pseudo-scalar operations

ggml 的内存布局很多时候都是 “pseudo-scalar”：vector value 没有完美地对齐在 lane 上。

使用真正的 vector operation 能够提升 60% 性能。

> Vector lanes need to accumulate different results rather than multiple parts of the same result


> Compute more than one result at once – for non-batched case this must be different output points
## Transformed block layout

![[Screenshot 2024-04-02 at 12.11.22.png]]

把 scale 值单独捞出来，weights 对齐。

## Block processing steps – 4 simultaneous blocks

> 38 operations, computing 4 blocks => 9.5 operations per block (21% MAC)

一次计算 4 个 block？

## Optimizing in-memory format

- 将 weights 放在连续的内存里
	- 优化了对齐性质
	- 更容易处理 scale factor（连续了，可以 simd 一次处理）
	- Could go full “structure of arrays”; we just went for ”array of more useful structures”.
- Extra saving available on 4->8 bit unpacking:

## Batched inference of LLMs

- 更多的用户请求，可以攒在 batch 中一起处理
- 对于单个用户的请求，可以通过 speculative decoding 技术来提高 batch 度
- batch 是一个 “A throughput optimisation problem within constraints of latencies and limited resources”

## Batching of operators

- linear projection 和 FFN 都很容易 batch 处理
	- GEMV 可以变为 GEMM 算子
- 但是 attention 不那么容易 batch 处理
	- Even if multi-query attention & grouped-query attention can be processed with GEMM operations

## Optimized GEMM for batched inference

- 可以使用 GEMV 同样的概念：
	- weights 做 rearrange 处理
	- 应用给多行的 activations
- 使用 SMMLA 指令
	- 双倍的 MAC 个数的 SDOT（MADC 是啥？）
- 每个输入都通过多个 SMMLA 指令处理，
- Batch 8: 79 vector ops, 32 output points => 2.5 ops/point (40% MAC)
	- 能够比 GGML 快 4.8 倍


![[Screenshot 2024-04-02 at 12.34.24.png]]

## Limitations and future work

CPUs are a viable platform for LLM inference.

Llama.cpp has reasonable implementations for Arm CPUs but further optimization is possible. • 2.1x speedup per thread on non-batched case.

High scaling of overall throughput with batch size shows promise for future platforms with more lpddr bandwidth