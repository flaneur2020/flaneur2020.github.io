## 1 Foundation of Diffusion

> The goal of generative modeling is: given i.i.d. samples from some unknown distribution $p_∗(x)$, construct a sampler for (approximately) the same distribution.

比如，给出一组来自 $p_{\text{dog}}$ 分布的狗图，我们想在这组分布中，生成新的狗图。

解决这个问题的方法之一是，学习一个从一个 easy-to-sample 的分布（比如高斯分布）到目标分布 $p_*$ 的转换方法。

Diffusion model 实现了一个这种转换的通用框架。

它聪明的地方在于，将直接从 $p_*$ 中采样，拆解为一组更容易 sample 的子问题。

### 1.1 Gaussian Diffusion

设 $x_0$ 是 $\mathbb{R}^d$ 分布（比如狗图），然后构造一个序列 $x_1, x_2, ..., x_t$，使它增加随机的高斯噪声：

$$
\begin{align}
x_{t+1} &= x_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma^2) \tag{1} \\
\end{align}
$$

当 $t$ 足够大时，最终的分布会接近于平均值为 0 的高斯噪声。这被称作正向过程（forward process）。

反过来，生成狗图，就是先从均值为 0、方差为 $\sigma$ 的高斯分布中采样。

reverse sampler：给定一个边缘分布为 $p_t$ 的样本，生成一个边缘分布为 $p_{t-1}$ 的样本。

有了 reverse sampler 之后，我们可以简单地从 $p_t$ 中采样一个高斯样本，并持续通过 reverse sampler 采样得到 $p_{t-1}$、$p_{t-2}$ 的样本，最终实现从 $p_0$ 也就是 $p_*$ 中采样。

diffusion 的思想是，学习 reverse 每一步中间步骤，能够比学习整个 $p_*$ 分布更简单。

理想的 DDPM Sampler 使用一个明显的策略：在时间 $t$，给出输入 $z$（来自 $p_t$ 的采样），我们输出一个如下分布的采样：

$$
p(x_{t-1}|x_t = z) \tag{2}
$$

$\text{Fact 1: Diffusion Reverse Process}$. 对于很小的 $\sigma$，按公式 $(1)$ 的高斯分布，条件分布 $p(x_{t-1}|x_t)$ 本身接近于高斯分布。对于每个时间步 $t$ 和满足 $z \in \mathbb{R}^d$ 的 $z$，那么存在一个特定的平均数参数 $\mu \in \mathbb{R}^d$ 满足：

$$
p(x_{t-1}|x_t = z) \cong \mathcal{N}(x_{t-1}; \mu, \sigma^2) \tag{3}
$$

只有在 $\sigma$ 比较小的时候成立。这个能够大幅简化式子，我们只有 $\mu_{t-1}(x_t)$ 这种均值不知道。

对于任意时间 $t$ 和 conditioning value $x_t$，学习 $p(x_{t-1}|x_t)$ 的均值就能够学到 $p(x_{t-1}|x_t)$ 这个分布本身。

$p(x_{t-1}|x_t)$ 可以通过 regression 来学习，会是一个简单得多的问题：

$$
\begin{align}
\mu_{t-1}(z) &= \mathbb{E}[x_{t-1} | x_t = z] \tag{4} \\
\mu_{t-1} &= \min \mathbb{E}|f(x_t) - x_{t-1}|^2 \\
\end{align}
$$

## 2 Stochastic Sampling: DDPM

Denoising Diffusion Probabilistic Models (DDPM)




