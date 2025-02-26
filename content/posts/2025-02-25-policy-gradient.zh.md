---
date: "2025-02-25"
title: "Notes on RL: Policy Gradient & Log Derivative Trick"
language: "zh"
---

前段时间照着[这个教程](https://sarvagyavaish.github.io/FlappyBirdRL/) 用最基础的 Q-Learning 做了一个 Flappy Bird 的强化学习，还意外的挺好使，跑到了一万多分。

Q-Learning 相当于为每个状态-动作对 $(s, a)$ 都估计了一个价值 $Q(s, a)$，然后根据 $Q$ 值来选择动作。Deep Q-Network 相当于在基础版 Q-Learning 的基础上加一个神经网络来估计 $Q$ 值，像是把 Q Table 用神经网络压缩了一把。

Q-Learning 和 DQN 都是对 Reward 进行估值，然后根据 Reward 的 Value 来选择动作。

Policy Gradient 则属于另一个流派，直接根据状态 $s$ 来选择动作 $a$，输入当前状态 $s$，神经网络直接吐出来各个动作的概率，也就是 $\pi_\theta(a|s)$，其中 $\theta$ 是神经网络的参数。

后来大语言模型的 PPO、GRPO 等算法，也都属于 Policy Gradient 这个流派。

不过 Policy Gradient 的数学推导对我来讲有点难理解，在这里记一下。

跟之前学习一样，先列一下推导中会用到的前置数学知识。到后面推导的时候，就无脑代入这些公式。

## 期望和蒙特卡洛

在之前 VAE 的笔记中有提到过期望的计算。对于连续的随机变量，期望的计算式是：

$$
\mathbb{E}[x] = \int x p(x) dx
$$

在训练中，我们得到的都是采样的集合，比如我们有一个样本集合 $X = \{x_1, x_2, \ldots, x_n\}$，那么期望可以这样估算：

$$
\mathbb{E}[x] \approx \frac{1}{n} \sum_{i=1}^{n} x_i, \,\,\,\, x_i \sim p(x)
$$

用采样来估算期望的方法就叫做**蒙特卡洛方法**。

推广到函数 $f(x)$ 的期望：

$$
\begin{align}
\mathbb{E}[f(x)] & = \int f(x) p(x) dx \\
& \approx \frac{1}{n} \sum_{i=1}^{n} f(x_i), \,\,\,\, x_i \sim p(x)
\end{align}
$$

如果可以在连续的期望式子中凑出来一个 $p(x)$，就可以用蒙特卡洛方法来估算期望，转换为离散的采样的式子。

## Log Derivative Trick

假设我们有个函数 $p(x;\theta)$，我们希望得到它的对数求取关于 $\theta$ 的梯度：

$$
\nabla_{\theta} \log p(x;\theta)
$$

应用 chain rule （链式法则）可以得到：

$$
\nabla_{\theta} \log p(x;\theta) =
\frac{\nabla_{\theta} p (x; \theta)}{p(x; \theta)}
$$

反过来，可以得到：

$$
\nabla_{\theta}p(x;\theta) = 
p(x;\theta) \nabla_{\theta} \log p(x;\theta)
$$

这个技巧就叫做 **Log Derivative Trick**，它可以用在下面这种 “Score Function Estimator” 的场景中。

## Score Function Estimator

有些时候，我们想估算函数 $f$ 的期望针对 $\theta$ 的梯度：

$$\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]$$

展开一下：

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) d(x) \qquad \text{(Leibniz rule)}\\
\end{align}
$$

然而，$\nabla_\theta p(x;\theta)$ 并不是一个合法的概率函数，因此不能代入蒙特卡洛来这样估算期望：

$$
\nabla_\theta \mathbb{E}_{p(x; \theta)} \left[ f(x) \right] \approx
\frac{1}{n} \sum^{n}_{i=1}\nabla_\theta p(x_i; \theta) f(x_i) \qquad \leftarrow\text{ 不可以}
$$

这时就是 log derivative trick 的应用了，将 $\nabla_\theta p(x;\theta)$ 转换为 $p(x;\theta)\nabla_\theta \log p(x;\theta)$，从而在式子中凑出来一个 $p(x;\theta)$：

$$
\begin{align}
\nabla_{\theta} \mathbb{E}_{p(x;\theta)} \left[ f(x) \right]
& = \nabla_{\theta} \int p(x; \theta) f(x) dx \\
& = \int \nabla_{\theta} p(x; \theta) f(x) dx &\quad& \text{(Leibniz rule)} \\
& = \textcolor{blue}{\int p(x;\theta)} \nabla_{\theta} \log p(x;\theta) f(x) \textcolor{blue}{dx} &\quad& \text{(Log derivative trick)} \\
& = \mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x;\theta) f(x)\right]
\end{align}
$$

到这里，就可以使用<mark>蒙特卡洛方法</mark>来采样估算期望了：

$$
\mathbb{E}_{p(x; \theta)} \left[ \nabla_\theta \log p(x; \theta) f(x) \right]
\approx \frac{1}{n} \sum^{n}_{i=1} \nabla_\theta \log p(x_i; \theta) f(x_i)
$$

到这里估算出来了梯度。

这就是 Log Derivative Trick 的应用：将原本不能直接用蒙特卡洛估算的梯度 $\nabla_\theta p(x;\theta)$ 转换为 $p(x;\theta)\nabla_\theta \log p(x;\theta)$，从而在式子中凑出一个概率函数 $p(x;\theta)$，使得可以用蒙特卡洛方法来采样估算期望。在推导 Policy Gradient 的时候就主要用到了这个技巧。

## Policy Gradient

在 Policy Gradient 的设定中，我们希望得到一个神经网络参数为 $\theta$ 的策略 $\pi(a|s;\theta)$，输入当前状态 $s$，输出各个动作 $a$ 的概率。

玩完一局游戏，一套策略走下来对应一个轨迹 $\tau$：

$$
\tau = (s_0, a_0, s_1, a_1, \ldots, s_t, a_t)
$$

这个轨迹 $\tau$ 最后得到一个总计的 Reward 值 $R(\tau)$，也就是每步的 Reward 值 $r(s_t, a_t)$ 加起来：

$$
R(\tau) = \sum_{t=0}^{T} r(s_t, a_t)
$$

我们希望最大化整个轨迹的 Reward 的期望 $J(\theta)$：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

要优化它，就是对 $\theta$ 求梯度走梯度上升：

$$
\nabla_{\theta} J(\theta) = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]
$$

展开一下，应用上 Leibniz rule 和 Log derivative trick：

$$
\begin{align}
\nabla_{\theta} J(\theta) 
& = \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] \\
& = \nabla_{\theta} \int \pi_\theta(\tau) R(\tau) d\tau \\
& = \int \nabla_{\theta} \pi_\theta(\tau) R(\tau) d\tau &\qquad& \text{(Leibniz rule)}\\
& = \textcolor{blue}{\int \pi_\theta(\tau)} \nabla_{\theta} \log \pi_\theta(\tau) R(\tau) \textcolor{blue}{d\tau} &\qquad& \text{(Log derivative trick)}\\
& = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_{\theta} \log \pi_\theta(\tau) R(\tau) \right] &\qquad& \\
\end{align}
$$

应用上 Log derivative trick 之后，得到一个期望公式，到这里就可以用蒙特卡洛方法来估算期望了：

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{n} \sum^{n}_{i=1} \nabla_{\theta} \log \pi_\theta(\tau_i) R(\tau_i)
$$

到这里和上面 Score Function Estimator 的推导是一样的，不过神经网络的输出是每一步动作的概率，仍需要把 $\pi_\theta(\tau)$ 展开一下：

$$
\pi_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) p(s_{t+1}|s_t, a_t)
$$

其中 $p(s_0)$ 是初始状态的概率，$\pi_\theta(a_t|s_t)$ 是每一步动作的概率，$p(s_{t+1}|s_t, a_t)$ 是状态转移的概率。

$$
\nabla_{\theta} \log \pi_\theta(\tau) = \textcolor{blue}{\nabla_{\theta} \log p(s_0)} + \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t) + \textcolor{blue}{\nabla_{\theta} \log p(s_{t+1}|s_t, a_t)}
$$

这个式子有两个有意思的地方：

1. 利用 log 的性质，乘法变成了加法，计算起来更方便；
2. $\log p(s_0)$ 和 $\log p(s_{t+1}|s_t, a_t)$ 都跟 $\theta$ 无关，所以梯度为 0；

所以可以简化为：

$$
\nabla_{\theta} \log \pi_\theta(\tau) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t)
$$

把这个式子代入回期望公式中：

$$
\nabla_{\theta} J(\theta) \approx \frac{1}{n} \sum^{n}_{i=1} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t|s_t) R(\tau_i)
$$

到这里就得到了 Policy Gradient 的梯度公式。

不过在工程中一般不会直接操作梯度。在 auto grad 框架下，直接写梯度下降的 Loss 函数就可以优化了：

$$
\begin{align}
Loss(\theta) & = - J(\theta) \\
& \approx - \frac{1}{n} \sum^{n}_{i=1} \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) R(\tau_i) \\
\end{align}
$$

## 总结

总的来看 Policy Gradient 的公式的推导不算长，主要就是应用了 Log Derivative Trick 来凑出概率函数，然后用蒙特卡洛方法来估算期望。

不过这种最简单的 Vanilla Policy Gradient 听说训练效果比较捉急，玩游戏的轨迹按说是很稀疏的，而且 Reward 的方差很大，用蒙卡特罗估算期望并不怎么稳。后续就接着看下它上面做改进的算法，比如 PPO、GRPO 等。

## References

- https://andrewcharlesjones.github.io/journal/log-derivative.html
- https://davidmeyer.github.io/ml/log_derivative_trick.pdf