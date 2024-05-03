https://huggingface.co/blog/gigant/vlm-design

## Vision and language in a shared latent space

CLIP 是一个简单高效的框架，能够同时学习 vision 和 text 的 encoder。

![[Pasted image 20240427010758.png]]

> A key advantage is that the latent tokens representations in CLIP's ViT might have some sort of a [cross-modal](https://arxiv.org/abs/2403.09037) / _[synaesthetic](https://arxiv.org/abs/2306.03678)_ ability, by already being _mostly_ aligned with their captions.

> _"Mostly"_, because the latent representation of the image is aligned to the latent representation of the text, that went through the tokenizer and the transformer-encoder, while in most scenarios the encoded image is fed to a language model along freshly-embedded text tokens.

## Leveraging and aligning pretrained models with a "Visual Abstractor"

使用一个很小的 projection layer，就可以直接把 CLIP 的 latent 当作 word embedding。

projection 和 language model 会在一起经过 “visual instruction tuning”，经过第二个更昂贵的训练阶段，来调教模型来跟进视觉的指令。

在第一版 Llava 中，abstractor 只是一个简单的线性映射，在后续的版本，变为了一个更有表达力的 multi layer perceptron。

encoded image 对应的 token 时固定的，比如 256 个 token。不同的 abstractor 算法，可以根据不同的取舍来选择关键的特征。

> Depending on the choice, images can be seen as an additional information which can be referred to by text tokens, or as a bunch of tokens that can be concatenated with, and processed similarly to, text tokens. When taken to the extreme, the latter is similar to modeling images as a foreign language.
## Are images a foreign language?

在 [_Dosovitskiy et al_](https://arxiv.org/abs/2010.11929) 的 ViT model 研究中，图片可以按照文本一样的方式来处理；

图片可以拆分为 patches，就像 embedding 那样经过一个 language model 来处理。

这样相当于将图片看作一个独立的外语，。

Beit 3 模型基于这个 ViT 架构，图片和文本 token 在同一个 model 中处理，但是经过不同的 expert。

介于对不同模态的预训练模型进行对齐，还是在同一个模型中训练多模态之间的做法是 Fuyu 框架。

> They simplified both the architecture and training procedure by feeding the image patch embeddings as is to a language model. With that framework, there is no need to think about how to scale the vision encoder vs the language model, or what training stages to do and in what order, and the model is able to work with images of varying resolutions.

![[Pasted image 20240427091744.png]]

## _ARE_ images a foreign language? The argument of granularity

> Audio and vision are treated as fine-grained, while text is more coarse-grained.

> This strategy is based on the observation that the visual and audio spaces are fine-grained (there are many visual or sounds of guitars that might be really different to each other) while the textual domain is more coarse as its goal is to abstract away details (e.g. a single “guitar” word).

