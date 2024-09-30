
https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5

ngmi 是 "Not Gonna Make It" 的缩写？

> This is one more of quantized-aware training rather than something radically new. There is room for kernel fusion and CUDA magic optimization because the hypothesis is storing weight matrix contain only 1,0,-1 that means we only need ADD,SUB operation which are better than expensive MUL. While at inference type we can achieve crazy speed up I am not sure that can be done at training time. Because we need smooth changes in optimizer gradient momentum.

是一个 quantized aware training 机制。

在推理时候加速会很强，只需要 ADD、SUB，不需要 MUL。

但是不知道训练时候怎么样，因为训练时候需要 smooth 的 change。

## Implementation Details

![[Pasted image 20240402134605.png]]

## Weight Quantization

![[Pasted image 20240402134636.png]]
所有的权重存储在 fp32/16 中，在 forward process 中量化一把。

```python
from torch import Tensor

def weight_quant(w: Tensor) -> tuple[Tensor, Tensor]:
    scale: Tensor = 1.0 / w.abs().mean().clamp(min=1e-5)
    quant: Tensor = (w * scale).round().clamp(-1, 1) / scale
    w_quant: Tensor = w + (quant - w).detach()
    scale = abs(w_quant).max().detach()
    w_quant = w_quant / scale
    return w_quant, scale 
```


w_quant is matrices of 1,0,-1

## Training code

![[Pasted image 20240402135239.png]]

作者看起来它的 training loss 不如 llama。
## Conclusion


> Right now, <mark>BitNet is pretty useless for inference on current hardware</mark>; it's much better to train the model in bfloat16 and use the quantized version if you need it. To achieve a significant speed-up,

> the prerequisites for BitNet 1.5 are substantial. We need someone to create a special chip that supports mixed precision with 2-bit. So, the big assumption is we will use a 2-bit model for inference, meaning someone will have to spend a lot (a lot) of money to build the chip, software, and train quantization-aware LLM. Not to mention the fact that we don't know if scaling laws hold the same for 1.5bit (it's 2 bit) quantization training as they do for normal. Unless someone is ready to spend a ton of money.

还没有信心 1.5bit 能不能满足 scaling law。