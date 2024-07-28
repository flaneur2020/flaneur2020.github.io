---
date: "2024-07-22"
title: "Notes on Diffusion Model: Intuition"
---

前面已经学习了 VAE，它的数学推导很完整，但是在实践中 VAE 似乎并没那么好用，而 Diffusion Model 毫无疑问是现在图片生成领域的 SOTA。在找学习 Diffusion Model 的资料时，大多数找到的文章都带着很厚的 ELBO 之类的数学推导，对我来讲还是太难了，这篇文章尝试避免这些概念，单从它的 Intuition 层面来看。

## 思路

在 VAE 中，图片生成被看做是从一个高斯分布中采样。而在 Diffusion Model 中，图片生成被看作是通过一系列逐步去噪的过程。相比于一步到位，单次降噪操作在神经网络看来是个更容易学习的任务。

Diffusion 模型会分为 **Forward** 和 **Reverse** 两个过程。

![Diffusion Model](/images/2024-07-28-diffusion.webp)

在 Forward 过程中，我们从一个真实图片开始，逐步添加噪声。随着噪声的增加，图片变得越来越模糊，最终变成了一个高斯分布的白噪声。

在 Reverse 过程中，我们从一个白噪声开始，通过神经网络来逐步去除噪声。随着噪声的减少，图片变得越来越清晰，最终变成了一个真实图片。

## 前置知识

在看数学公式之前先列一下推导中会用到的数学知识。

这里只有一个「两个高斯分布相加」需要了解：

两个高斯分布相加的结果仍然是一个高斯分布。设两个高斯分布为 $\mathcal{N}(0, \sigma_a^2)$ 和 $\mathcal{N}(0, \sigma_b^2)$，则：

$$
\mathcal{N}(0,\sigma_a^2) + \mathcal{N}(0, \sigma_b^2) = \mathcal{N}(0, \sigma_a^2+\sigma_b^2)
$$

这个式子在 Diffusion Model 的 Forward Process 中会用到。

## Forward Process

Forward Process 也被称作 "Diffusion Process"。设原始图片是 $x_0$，我们逐步添加噪声，得到 $x_1, x_2, ..., x_t$。最终 $x_t$ 会成为一个高斯分布的白噪声。

$$
x_t = x_{t-1} + \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, \bf{I})
$$

不过这样直白地加噪声，到最终成为彻底的白噪声不知道要多久。

我们希望能在固定的步数 $t$ 下得到一个高斯分布的白噪声。Diffusion Model 通过下面的式子来加噪声：

$$
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, \bf{I})
$$

其中 $a_1, a_2, ... a_t$ 均为小于 1 的系数。

每步加噪声中，除了加高斯噪声，也对原本的图像乘以系数弱化，慢慢图像本体的占比就会越来越少，高斯噪声占比越来越高。到第 $t$ 步时，就能比较确定地成为一个高斯分布的白噪声。

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

有了这个式子，我们在训练神经网络时不需要 for 循环 $t$ 步去加噪音，直接把 $x_0$ 和 $t$ 代入这个式子，按约定好的 $\alpha$ 序列，就可以直接得到第 $t$ 步加噪音后的值 $x_t$。

这段推导中比较神奇的是 $(4)$ 到 $(5)$ 这里。

$\sqrt{\alpha_2 - \alpha_1\alpha_2}\epsilon_1$ 和 $\sqrt{1-\alpha_2}\epsilon_2$ 可以看做是一个 Reparameter Trick，它们分别等于在正态分布 $\mathcal{N}(0, \alpha_2 - \alpha_1\alpha_2)$ 和 $\mathcal{N}(0, 1-\alpha_2)$ 中的采样。

代入前面的「两个高斯分布相加」的式子，可以得到：

$$
\mathcal{N}(0, \alpha_2 - \alpha_1\alpha_2) + \mathcal{N}(0, 1-\alpha_2) = \mathcal{N}(0, 1 - \alpha_1\alpha_2) 
$$

也就是：

$$
\sqrt{\alpha_2 - \alpha_1\alpha_2}\epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2 = \sqrt{1 - \alpha_1\alpha_2} \overline{\epsilon}_2 \quad \text{; where} \quad \overline{\epsilon}_2 \sim \mathcal{N}(0, 1)
$$

## Backward Process

在 Forward Process 中，我们从 $x_0$ 开始逐步加噪点到 $x_t$。而在 Reverse Process 中，我们借助一个降噪神经网络，从 $x_t$ 开始逐步去噪音 $\overline{\epsilon}_t$，从而最终得到 $x_0$。

比较直白的理解就是，我们通过神经网络来预测噪音，训练神经网络的目标就是让预测的噪音和实际的噪音尽可能接近：

$$
\begin{align}
\mathcal{L} &= || \overline{\epsilon}_t - \epsilon_\theta(x_t, t) ||^2 \\
&= || \overline{\epsilon}_t - \epsilon_\theta(\sqrt{\overline{a}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\overline{\epsilon}_t, t) ||^2 \\
\end{align}
$$

其中 $\epsilon_\theta$ 就是我们的神经网络，它的输入是 $x_t$ 和当前的步数 $t$，输出是预测的噪音。

到这里就跟 Diffusion Model 原始论文最后的损失函数一样了。

不过按这个损失函数来看，这个降噪神经网络优化的目标是从 $0$ 到 $t$ 步的所有噪音的总和，而并非只针对单独第 $t$ 步的降噪。如果训练的猛了，一步到位全给降噪了也不是不可能？不过实际训练中训练数据到位的话，我猜在这个优化目标之下模型也能够同样泛化出来一个逐步降噪的能力。而按目前的效果来看显然是的。

## Training

到这里可以组装一下训练的流程大约是：

1. 从训练集中采样一个图片作为 $x_0$
2. 随机采样一个 $t$，按照约定好的 $\alpha$ 序列，计算得出 $x_t$ 与噪音 $\overline{\epsilon}_t$
3. 用 $x_t$ 和 $t$ 作为输入，通过神经网络预测噪音 $\epsilon_\theta(x_t, t)$，与实际的噪音 $\overline{\epsilon}_t$ 计算损失
4. 得到梯度，更新神经网络参数
5. 回到 1. 进行下一轮训练

## Conclusion

可见 Diffusion Model 前后训练相关的逻辑并不复杂，这头加噪声，那头用神经网络预测噪声，that's all。

但是 ELBO 那部分推导仍对我来讲挺难，后面有时间再单独记录一下。个人理解是 ELBO 这系列推导起到一个证明正确性的作用，真正工程中的 Diffusion Model 并没有完全 follow 这个数学上最优的式子，而是在之后做了一点简化，并实验发现效果更好，实验还是最重要的。