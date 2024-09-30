## **Tactical**

### prompting

- 作者建议对于新的应用总是从 prompting 开始入手；
- prompting 经常被低估，好的 prompting 可以走的很远；
- 但是即使是基于 prompt，也需要付出大量的工程工作；
- Focus on getting the most out of fundamental prompting techniques
	- n-shot prompts, in context learning, chain-of-thought 等等
- 通过 n-shot 为 LLM 提供 in-context learning 能力，有一些 tips：
	- 如果 n 太小，模型可能会对这几个 case 过于 over-anchor；作为 rule-of-thumb，n 最好大于 5；
	- 例子应当对于输入的范围有所体现，如果你在做一个电影总结，最好整理几个不同的流派的数据作为 sample；
	- 不一定需要提供完整的 input- output pair，很多时候 output 的例子就足够了；
	- 如果你想让 LLM 支持使用工具，你的 n-shot 中最好包含对使用工具的例子；
- 在 CoT prompting 中，我们鼓励 LLM 在给出最后的回答前，解释它思考的过程；
	- 在早期的 “let's think step by step” 之外，作者发现给出更具体的引导，也能显著减少胡说八道；
	- 首先，列出来关键的点；
	- 然后在草稿上检查细节；
	- 最终根据这些汇总出来结论；
- 向模型提供相关的材料，也是扩充模型知识、减少胡说八道、提高用户信任的关键；这通常被视为 RAG；
- Structure your inputs and outputs
- Craft your context tokens