https://arxiv.org/pdf/2212.09720.pdf

- 做量化会遇到一个问题是，accuracy 和 total model bits 之间怎样权衡；
	- 比如，有一个 60B 的 4 bit 量化的 LLM 和一个 30B 的 8 bit 量化 LLM，怎样选择；
	- 作者的发现是，这两种情况下，总是选 4 bit 量化的 LLM；
	- For 4-bit precision, data types and a small quantization block size are the best ways to enhance bit-level scaling trends
	- For most 4-bit models, a block size between 64 to 128 is optimal in our study
- From our results, we can make straightforward recommendations for using zero-shot quantized models for inference: **always use 4-bit models with a small block size and <mark>with a float data type</mark>**.
	- with a float data type 大概是指不用等分块的整数类型做分桶，而是在 4 bit 中留几位作为指数值
- Outlier-dependent quantization improves stability, but not scaling.
	- outlier dependent quantization 大意是指将 outlier 单独拎出来单独量化。
	- 好像意思是指 proxy quantization 虽然有助于提高性能的稳定性，但是对性能的帮助不算大（计算量反而多了？）

## Background

> reduction in the total model bits is strongly correlated with inference latency for small inference batch size

### 2.3. Blocking / grouping

grouping 是将一个 tensor 拆分为多个子 tensor，这称作 group。

blocking 将 tensor 看做一个一维的序列，拆分为 n 个 block。

> In our work, we use blocking because, unlike grouping, it provides a measure of additional bits per parameter independent of the hidden dimension


## 3. Outlier-dependent quantization through proxy quantization

> Proxy quantization is input-independent and, therefore task-independent, as it uses the standard deviation of each layer’s hidden unit weights as a proxy for which dimensions have outlier features.

from GPT4:

> 1. **Proxy Quantization:** Once outliers are identified, proxy quantization could involve representing these outlier values using a different method or set of bits compared to the non-outlier values. Proxy quantization seems to be a specific method or approach within outlier-dependent quantization, but without more context from the document, we can't be certain about the specifics of this technique.

大意是指把 outlier 值拆出来？