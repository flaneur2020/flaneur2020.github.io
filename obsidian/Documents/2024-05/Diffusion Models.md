给出一个未知的分布 $x ∼ p(x)$，在这个分布中生成新的样本。

GAN 将这个过程看作一个游戏：generator 模型选择一个随机的 seed 训练出来为了欺骗一个 discriminator 模型。

GAN 没有显式地对 $p(x)$ 进行建模。

另一个路线是，从 sample 中学习一个确定性的、可以回归映射的分布。

然后我们可以从一个已知的分布中，sample 得到一个点。

## 1.1 Denoising diffusion models