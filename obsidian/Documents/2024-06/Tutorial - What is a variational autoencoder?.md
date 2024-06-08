## The neural net perspective

在神经网络的语言中，VAE 包含一个 encoder、一个 decoder 和一个 loss function。

encoder 是一个神经网络，它的输入是 $x$，输出一个 hidden representation $z$ ，它的参数是 $\theta$。

$x$ 可以是一个 28x28 的手写数字对应的像素，有 784 个维度。$z$ 对应一个 latent space 中的向量，维度数远小于 784。

这通常被称作一个 “bottleneck”，encoder 必须学会一个压缩的表示方法。

encoder 可以表示为 $q_{\theta}(z | x)$，这是一个高斯概率密度函数，我们可以从这个分布中，采样得到 $z$ 的值。

decoder 是另一个神经网络，它的输入是 $z$，它能够输出原始数据的概率密度，它表示为 $q_{\phi}(x|z)$。

decoder 还原原始图像中肯定会有一定损失，怎样衡量这部分损失？可以通过 $log \space p_{\phi}(x|z)$ 来衡量。

VAE 的 loss function 相当于一个 <mark>negative log-likelihood 加上一个 regularizer</mark>。

（重温下 negative log-likelihood 的损失函数：$l(\theta) = - \sum_{i=1}^{n} \left( y_i \log \hat{y}_{\theta, i} + (1 - y_i) \log (1 - \hat{y}_{\theta, i}) \right)$，手写字母的话每个像素的训练集中好像只有 0 和 1 两个取值)

每个数据点的 loss function：

$$
\begin{equation}
l_i(\theta, \phi) = -\mathbb{E}_{z \sim q_\theta(z|x_i)} \left[ \log p_\phi(x_i | z) \right] + \text{KL}(q_\theta(z | x_i) \parallel p(z))
\end{equation}
$$
公式的第一部分是 reconstruction loss，也就是第 $i$ 个训练数据的 negative log-likelyhood。这一部分会鼓励 VAE 能够重建图像。

公式的第二个部分是 regularizer。这是 KL 散度，衡量 $q_{\theta}(z|x)$ 和 $p(z)$ 的差异。这里衡量的是， $q$ 这个分布和 $p$ 分布的相似性。

在 VAE 中，$p(z)$ 满足正态分布，平均值为 0，方差为 1。

如果 encoder 输出的 $z$ 在总体上不满足这样的正态分布，就会受到 loss function 惩罚。

> This regularizer term means ‘keep the representations 𝑧z of each digit sufficiently diverse’

> If we didn’t include the regularizer, the encoder could learn to cheat and give each datapoint a representation in a different region of Euclidean space.

如果不包含这个 regularizer，encoder 容易 cheat，取巧给每个 datapoint 一个单独的表示位置。比如同样一个 “2”，两个人写的 2 在 $z$ 空间中会距离很远。我们希望这个语义空间更加有意义，因此希望不同人写的 “2” 在 latent space 中都尽量靠近。

## The probability model perspective

在概率模型中，VAE 包含一个特定的概率模型，针对数据 $x$ 和 latent variable $z$，$p(x, z) = p(x | z) p(z)$。

对于每个数据点 $i$：

1. 出 latent variable：$z_i \sim p(z)$
2. 恢复原始数据：$x_i \sim p(x|z)$

根据贝叶斯公式：

$$
p(z|x) = \frac{p(x|z) p(z)}{p(x)}
$$

$p(x)$ 被称作先验。