## References

- https://zhuanlan.zhihu.com/p/566618077

## 相关的基础

### 两个均值为零的高斯分布相加

$$
\mathcal{N}(0,\sigma_a^2) + \mathcal{N}(0, \sigma_b^2) = \mathcal{N}(0, (\sigma_a+\sigma_b)^2)
$$

## Forward diffusion process

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_{t} \textbf{I})
$$
$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$
$$
\begin{align}
x_1 &= \sqrt{\alpha_0}x_0 + \sqrt{1-\alpha_0}\epsilon_0 
&\text{; where}\quad \epsilon_0 \sim \mathcal{N}(0, \textbf{I}) \tag{1} \\
x_2 &= \sqrt{\alpha_1} x_1 + \sqrt{1-\alpha_1}\epsilon_1 \tag{2} \\
&= \sqrt{\alpha_1}(\sqrt{\alpha_0}x_0 + \sqrt{1-\alpha_0}\epsilon_0 ) + \sqrt{1-\alpha_1}\epsilon_1 \tag{3} \\
&= \sqrt{\alpha_0 \alpha_1 } x_0 + \textcolor{blue}{\sqrt{\alpha_1 - \alpha_0\alpha_1} \epsilon_0 + \sqrt{1-\alpha_1}\epsilon_1} \tag{4} \\
&= \sqrt{\alpha_0 \alpha_1 } x_0 + \textcolor{blue}{\sqrt{1 - \alpha_1 + \alpha_1 - \alpha_0\alpha_1} \overline{\epsilon}_1} \tag{5} \\
&= \sqrt{\alpha_0 \alpha_1 } x_0 + \sqrt{1 - \alpha_0\alpha_1} \overline{\epsilon_1} \tag{6} \\
&\,... \\
x_t &= \sqrt{a_0..a_{t-1}} x_0 + \sqrt{1-\alpha_0...\alpha_{t-1}} \overline{\epsilon_{t-1}} \tag{7} \\
&= \sqrt{\overline{a_{t-1}}} x_0 + \sqrt{1-\overline{\alpha_{t-1}}} \overline{\epsilon_{t-1}}
&\text{; where} \quad &\overline{a_{t-1}} = \prod_{t=0}^{T-1} a_{t} \tag{8} \\
\end{align}
$$

其中 $\epsilon_0, \epsilon_1, ... \epsilon_{t-1}$ 都是采样自 $\mathcal{N}(0, \textbf{I})$。

公式 $(5)$ 相当于相加两个均值都为 0、方差分别为 $\alpha_1-\alpha_0\alpha_1$ 和 $1-\alpha_1$ 的正态分布。

$\alpha_0, \alpha_1, ...\alpha_{t-1}$ 都是小于 1 的， $\sqrt{\overline{a_{t-1}}}$ 最后会接近 0。

$$
\begin{align}
x_t &= \sqrt{\alpha_{t-1}} x_{t-1} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-1} \\
\epsilon_{t-1} &= \frac{1}{\sqrt{1-\alpha_{t-1}}} (x_t - \sqrt{\alpha_{t-1}} x_{t-1}) \\
\end{align}
$$
## Loss Function

$$
\mathcal{L} = ||\epsilon - \epsilon_\theta(x_t, t) ||^2
$$

其中 $\epsilon$ 是采样得到的 noise，$\epsilon_\theta(x_t, t)$ 是模型预测得到的 noise 值。
