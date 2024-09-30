## Decoding the standard autoencoder

Autoencoder 网络是一对网络，包括一个 encoder 和一个 decoder。

encoder network 取一个输入，转换为一个较小的、dense 的表示（这称作“encoding”），这个表示中包含有还原回原始输入的信息，decoder network 能够将这个表示转换为原始的输入。

![[Pasted image 20240608123559.png]]

整个网络是整体训练的，它的 loss function 通常是输入输出之间的 mean-square error 或者 cross entropy，这被称作是 "reconstruction loss"。

encoding 的 size 远小于原始的输入，因此 encoder 必须学会筛减信息，encoder 需要保留用于还原图像尽可能多的信息。而 decoder 学习如何还原回原始图像。

（类似我们理解数字识别中，一部分浅层神经擅长分辨“撇”，一部分浅层神经擅长分辨“捺”，经过 Encoder，encoding 中的信息可以理解为，我怎样通过撇捺横竖等等，来画出来原本的 “0” 这个数字）。

## The problem with standard autoencoders

标准的 autoencoder 能够很好地学习怎样 compact 地表示信息，也可以很好地还原出图像。但是它的应用场景很有限，比如降噪。

> The fundamental problem with autoencoders, for generation, is that the latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation.

autoencoder 用于生成的问题是，它的 latent space 是不连续的，不容易解析。

![[Pasted image 20240608124502.png]]

上面是一个 mnist 的 autoencoder，将它的 latent space 投影到 2D 之后的样子。

> This makes sense, as distinct encodings for each image type makes it far easier for the decoder to decode them. This is fine if you’re just _replicating_ the same images.

对于*还原*图像来讲，它的聚类分布是很合理的，为了便于 decoder 来 decode 它，不同的类型数据之间有比较远的分隔。

但是用于**生成**，你希望在 latent space 中采样，或者希望基于输入的图像生成变体，你希望有一个 continuous 的 latent space。

如果按上面这个分布，在采样时，很可能得到的全是噪音。decoder 面对没有见过的向量，就完全不知道怎样还原。

## Variational Autoencoders

VAE 相比原始的 autoencoder 有一个本质的不同，就是它的 latent space 是 by design 地 continuous 的，允许随机的取样与干预。

> It achieves this by doing something that seems rather surprising at first: making its encoder not output an encoding vector of size _n,_ rather, outputting two vectors of size n: a vector of means **μ**, and another vector of standard deviations, **σ**.

它的思路有点反直觉：它使 encoder 编码出来的不是直接的长度为 n 的向量表示，而是两个向量，对应平均值的向量 $\mu$ 和对应标准差的向量 $\sigma$ 。

![[Pasted image 20240608125256.png]]

这意味着，即使是同一个输入，也会因为 sampling 的存在，对应到一块连续的 latent 空间中。

![[Pasted image 20240608125943.png]]

decoder 学习到的，不再是从一个具体的点还原图像，而是一个附近的范围。

现在模型在局部尺度上，对相似的样本能有一个比较平滑的 latent space。理想的情况下，我们希望不太相似的样本之间也有重叠，从而便于在相邻的类目中插值。

然而向量 μ 和 σ 的取值没有限制，编码器可以为不同的类生成非常不同的 μ，将它们分开聚类，并最小化 σ，确保相同样本的编码本身变化不大，这一来 decoder 的不确定性较小。

最后聚类的样子仍与原先的点状区别不大，与我们的初衷不符合。

![[Pasted image 20240608131522.png]]

我们希望得到的 encoding 是，所有的都比较相邻，从而允许在平滑的空间中插值，并允许生成新的 sample。

为了达到这样的效果，需要引入 KL 散度作为 loss function。

KL 散度有助于比较两个分布之间的差异程度。最小化 KL 散度，就意味着优化概率分布（$\mu$ 和 $\sigma$）尽可能接近。

（优化 KL 散度效果上相当于可以使多个分布的距离更近？）

对于 VAE，KL 散度 Loss 等于 $X$ 中各个部分 $X_i \sim N(\mu_i, \sigma^2_i)$ 的和。在 $\mu_i = 0, \sigma_i = 1$ 时，该值最小。

$$
\text{KL Loss} = \frac{1}{2} \sum_{i=1}^{d} \left( \mu_i^2 + \sigma_i^2 - \log(\sigma_i^2) - 1 \right)
$$

只使用这个 loss 函数的话，会鼓励 encoder 将所有的 encoding 均匀地分布在 latent space 的中心位置（$\mu$ 都是零）。

decoder 完全无法从这里面解码出来任何有用的图。

![[Pasted image 20240608133945.png]]

只有同时优化这两个目标：输入输出的差异 + KL 散度，才可以得到预期的效果，使局部尺度上，相邻的 encoding 有相似性，在全局尺度上，latent space 的所有的点都是密集的。

（KL 散度在这里有点像正则化的作用）

（套上 KL 散度之后，latent space 中的向量满足正态分布 $Normal(0, 1)$，均值为 0，方差为 1）

![[Pasted image 20240608134135.png]]