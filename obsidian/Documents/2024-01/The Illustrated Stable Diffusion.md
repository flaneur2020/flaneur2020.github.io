
## UNet


UNET 是一个 noise predictor。


![[Pasted image 20240110121639.png]]


> The sampled noise is predicted so that if we subtract it from the image, we get an image that’s closer to the images the model was trained on (not the exact images themselves, but the _distribution_ - the world of pixel arrangements where the sky is usually blue and above the ground, people have two eyes, cats look a certain way – pointy ears and clearly unimpressed).


预测出来的 noise 并不是直接的目标图像，而是这些像素排列的「概率分布」。比如看到天空大概率是蓝色的，人的两个眼睛的位置大概率是深色，等等。


![[Pasted image 20240110124219.png]]

## The Text Encoder: A Transformer Language Model

这里使用了 transformer model 来从 text prompt 中，生成 token embeddings。

早期的 StableDiffusion 使用的预训练的 ClipText 模型，Stable Diffusion v2 使用了更大的 OpenClip。

![[Pasted image 20240110125109.png]]

![[Pasted image 20240110125425.png]]
Clip 基于网上爬的图片和它的 caption 训练的。

经过 Image Encoder 和 Text Encoder 最后编码出来两个向量，可以用于判定它俩是否相似。

## Feeding Text Information Into The Image Generation Process


![[Pasted image 20240110130025.png]]

input image 和 predicted noise 都存在于 latent space。

## Layers of the Unet Noise predictor (without text)

> - The Unet is a series of layers that work on transforming the latents array
> - Each layer operates on the output of the previous layer
> - Some of the outputs are fed (via residual connections) into the processing later in the network
> - The timestep is transformed into a time step embedding vector, and that’s what gets used in the layers

![[Pasted image 20240110130240.png]]
这个是没有 text conditioning 的 unet。

## Layers of the Unet Noise predictor WITH text

![[Pasted image 20240110130317.png]]

这个是加上 text embedding 的 unet。

attention 模块，会对每个像素，加上对 text embedding 的注意力。