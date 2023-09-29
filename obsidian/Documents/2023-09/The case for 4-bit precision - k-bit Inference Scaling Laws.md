https://arxiv.org/pdf/2212.09720.pdf

- 做量化会遇到一个问题是，accuracy 和 total model bits 之间怎样权衡；
- 比如，有一个 60B 的 4 bit 量化的 LLM 和一个 30B 的 8 bit 量化 LLM，怎样选择；
- 作者的发现是，这两种情况下，总是选 4 bit 量化的 LLM；
- For 4-bit precision, data types and a small quantization block size are the best ways to enhance bit-level scaling trends
- For most 4-bit models, a block size between 64 to 128 is optimal in our study
- From our results, we can make straightforward recommendations for using zero-shot quantized models for inference: **always use 4-bit models with a small block size and <mark>with a float data type</mark>**.
	- with a float data type 大概是指不用等分块的整数类型做分桶，而是在 4 bit 中留几位作为指数值
- Outlier-dependent quantization improves stability, but not scaling.