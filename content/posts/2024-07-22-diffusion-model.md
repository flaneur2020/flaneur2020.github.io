---
date: "2024-07-14"
title: Intuitions on Diffusion Model
draft: true
---

前面已经学习了 VAE，它的数学推导很完整，但是在实践中 VAE 并没那么好用。Diffusion Model 毫无疑问是现在图片生成领域的 SOTA。这篇文章尝试用直观的方式解释 Diffusion Model。

大多数找到的关于 Diffusion Model 的文章都是用 ELBO 等数学概念来解释的，这篇文章尝试避免这些概念，单从它的 Intuition 层面来看。

## Intuition

在 VAE 中，图片生成被看做是从一个高斯分布中采样。而在 Diffusion Model 中，图片生成被看作是通过一系列逐步去噪的过程。相比于一步到位，单次降噪操作在神经网络看来是个更容易学习的任务。

Diffusion 模型会分为 **Forward** 和 **Reverse** 两个过程。

在 Forward 过程中，我们从一个真实图片开始，逐步添加噪声。随着噪声的增加，图片变得越来越模糊，最终变成了一个高斯分布的白噪声。

在 Reverse 过程中，我们从一个白噪声开始，通过神经网络来逐步去除噪声。随着噪声的减少，图片变得越来越清晰，最终变成了一个真实图片。

## 前置知识

在看数学公式之前先列一下推导中会用到的数学知识，这里只有一个「两个高斯分布相加」需要了解：

两个高斯分布相加的结果仍然是一个高斯分布。设两个高斯分布为 $\mathcal{N}(0, \sigma_a^2)$ 和 $\mathcal{N}(0, \sigma_b^2)$，则：

$$
\mathcal{N}(0,\sigma_a^2) + \mathcal{N}(0, \sigma_b^2) = \mathcal{N}(0, (\sigma_a+\sigma_b)^2)
$$

这个式子在 Diffusion Model 的 Forward Process 中会用到。

## Forward Process

Forward Process 也被称作 "Diffusion Process"。设原始图片是 $x_0$，我们逐步添加噪声，得到 $x_1, x_2, ..., x_t$。最终 $x_t$ 是一个高斯分布的白噪声。

$$
x_t = x_{t-1} + \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, \bf{I}^2)
$$

不过这样直白地加噪声，到最终彻底的白噪声不知道要多久。

我们希望能在固定的步数下，得到一个高斯分布的白噪声。Diffusion Model 通过下面的式子来加噪声：

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, \bf{I}^2)
$$

其中 $a_1, a_2, ... a_t$ 均为小于 1 的系数。

每步加噪声中，除了加高斯噪声，也对原本的图像乘以系数弱化，慢慢图像本体的占比就会越来越少，高斯噪声占比越来越高。

从 $x_0$ 开始走一下加噪声的过程，可以得到：

$$
\begin{align}
x_1 &= \sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1 
&\text{; where}\quad \epsilon_1, \epsilon_2..\epsilon_t \sim \mathcal{N}(0, \textbf{I}) \\
x_2 &= \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2}\epsilon_2  \\
&= \sqrt{\alpha_2}(\sqrt{\alpha_1}x_0 + \sqrt{1-\alpha_1}\epsilon_1 ) + \sqrt{1-\alpha_2}\epsilon_2 \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \textcolor{red}{\sqrt{\alpha_2 - \alpha_1\alpha_2} \epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2} \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \textcolor{red}{\sqrt{1 - \alpha_2 + \alpha_2 - \alpha_1\alpha_2} \overline{\epsilon}_2}  \\
&= \sqrt{\alpha_1 \alpha_2 } x_0 + \sqrt{1 - \alpha_1\alpha_2} \overline{\epsilon_2}  \\
x_3 &= \sqrt{\alpha_1\alpha_2\alpha_3} x_2 + \sqrt{1-\alpha_1\alpha_2\alpha_3}\overline{\epsilon}_3 \\
x_4 &= ... \\
x_t &= \sqrt{a_1..a_{t}} x_0 + \sqrt{1-\alpha_1...\alpha_{t}} \overline{\epsilon}_{t} \\
&= \sqrt{\overline{a_{t}}} x_0 + \sqrt{1-\overline{\alpha}_{t}} \overline{\epsilon}_{t}
&\text{; where} \quad &\overline{a}_{t-1} = \prod_{t=1}^{T} a_{t} \ \\
\end{align}
$$

易知 $\overline{a}_{t}$ 经过一堆 $<1$ 的系数相乘最终接近于 0，$x_t$ 会无限接近于 $\overline{\epsilon}_t$ 本身，也就是一个高斯分布的白噪声。

这段推导中比较神奇的是 $(4)$ 到 $(5)$ 这里。

$\sqrt{\alpha_2 - \alpha_1\alpha_2}\epsilon_1$ 和 $\sqrt{1-\alpha_2}\epsilon_2$ 可以看做是一个 Reparameter Trick，它men等于在正态分布 $\mathcal{N}(0, \alpha_2 - \alpha_1\alpha_2)$ 和 $\mathcal{N}(0, 1-\alpha_2)$ 中的采样。

代入前面的「两个高斯分布相加」的式子，可以得到：

$$
\sqrt{\alpha_2 - \alpha_1\alpha_2}\epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2 = \sqrt{1 - \alpha_1\alpha_2} \overline{\epsilon}_2 \quad \text{where} \quad \overline{\epsilon}_2 \sim \mathcal{N}(0, 1)
$$