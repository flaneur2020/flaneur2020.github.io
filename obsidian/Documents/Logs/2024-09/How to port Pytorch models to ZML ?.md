TLDR；

- 大约说是 pytorch 有个 forward hook 能捕获每个 module 的输入输出
- ZML 给了一个 python 脚本，能够自动捕获 huggingface 的模型中每层的输入输出来简化移植

---

大致上按这个流程：

1. 针对已知的输入，执行 reference implementation ，记下 sample layer activations
2. 开始一个 zml 项目，装载 sampled reference activation
3. 一层一层地 port layer，对每个层进行测试
4. end-to-end test the model
## Sampling reference activations

pytorch 有 "forward hooks" 的功能，能够允许 inspect 每个 torch.nn.Module 的输入、输出。

这使得可以捕获每层的输入输出，记录下来。

最容易的办法是从 huggingface 给的 snippet 开始，比如：

```
import torch
import transformers

model_path = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.float16},
    # device="cuda",
    token=token,
)

prompt = "Q: What is the largest animal?\nA:"
output = pipeline(prompt)
print(output)
```

然后可以引进 `zml_utils`。

`zml_utils.py` 目前还不是一个 python 包，拷贝过来就行。

它里面有一个 `zml_utils.ActivationCollector`，可以包裹上原来的那个 pipeline：

```python
prompt = "Q: What is the largest animal?\nA:"
# Wrap the pipeline, and extract activations.
# Activations files can be huge for big models,
# so let's stop collecting after 1000 layers.
pipeline = zml_utils.ActivationCollector(pipeline, max_layers=1000, stop_after_first_step=True)
output, activations = pipeline(prompt)
print(output)
```
